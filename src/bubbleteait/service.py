"""One local, authenticated coordinator per data directory; no inference on status.

The first watcher uses polling of the real source selection, not OS events.
Immutable snapshots and the existing backend remain the source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import weakref

from .config import Config, ConfigError, ModelConfig, CocoIndexConfig, QwenConfig, SymbolsConfig, load_config
from .project import load_project, project_info, sync_project
from .registry import Registry, _atomic_json, _digest, _matches, _IgnorePolicy
from .qwen_runtime import file_lock, NoRedirect


def now():
    return datetime.now(timezone.utc).isoformat()


def private_directory(path):
    """Protect loopback bearer tokens even when data_dir is outside the user profile."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt':
        path.chmod(0o700)
        return
    import ctypes
    from ctypes import wintypes
    advapi = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    descriptor = ctypes.c_void_p()
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    advapi.SetFileSecurityW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p)
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    # Owner rights, inheritable to files/subdirectories; protected from a public parent DACL.
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW('D:P(A;OICI;FA;;;OW)', 1,
                                                                      ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not advapi.SetFileSecurityW(str(path), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


def config_payload(config):
    """Only coordinator settings; credentials and non-service configuration are not exported."""
    keys = ('project_dir', 'data_dir', 'models_dir', 'models', 'loaded_files', 'cocoindex',
            'qwen', 'symbols', 'project_config', 'service_file', 'sync', 'workspaces', 'embedding_profile',
            'embedding_override', 'sync_overrides', 'embedding_profiles')
    raw = asdict(config)
    return json.loads(json.dumps({k: raw[k] for k in keys}, default=str))


def config_from_payload(raw):
    raw = copy.deepcopy(raw)
    for key in ('project_dir', 'data_dir', 'models_dir', 'project_config', 'service_file'):
        if raw.get(key) is not None:
            raw[key] = Path(raw[key])
    raw['loaded_files'] = tuple(map(Path, raw.get('loaded_files', [])))
    raw['models'] = tuple(ModelConfig(**(m | {'path': Path(m['path']), 'files': tuple(m['files'])}))
                          for m in raw.get('models', []))
    for key, cls in [('qwen', QwenConfig), ('cocoindex', CocoIndexConfig), ('symbols', SymbolsConfig)]:
        if raw.get(key) is not None:
            if key == 'symbols':
                raw[key]['clangd_path'] = Path(raw[key]['clangd_path'])
            raw[key] = cls(**raw[key])
    for item in raw.get('workspaces', {}).values():
        item['path'] = Path(item['path'])
    for item in raw.get('embedding_profiles', {}).values():
        item['model_path'] = Path(item['model_path'])
    return Config(**raw)


def select_profile(config, profile):
    if profile is None:
        return config
    if profile not in {item.name for item in config.models if item.enabled}:
        raise ConfigError(f'Unknown enabled embedding profile: {profile}')
    if profile in config.embedding_profiles:
        item = config.embedding_profiles[profile]
        key, cls = ('qwen', QwenConfig) if item['backend'] == 'qwen-dense' else ('cocoindex', CocoIndexConfig)
        config = replace(config, **{key: cls(item['python_executable'], profile)})
    # A profile must be bound to a supported adapter, not merely listed as weights.
    if not any(settings and settings.model_profile == profile for settings in (config.qwen, config.cocoindex)):
        raise ConfigError(f'Embedding profile has no configured adapter: {profile}')
    return replace(config, embedding_profile=profile, embedding_override=profile)


def discover_workspace(settings):
    root = Path(settings['path']).resolve()
    if not root.is_dir():
        raise ConfigError(f'Workspace directory is unavailable: {root}')
    found = []
    policy = _IgnorePolicy(root)
    try:
        for directory, folders, files in os.walk(root, followlinks=False):
            directory = Path(directory)
            ignored = policy.ignored(directory, [(n, True) for n in folders] + [(n, False) for n in files])
            def allowed(name, is_dir):
                path = directory / name
                rel = path.relative_to(root).as_posix()
                return (name not in ignored and name != '.git' and not path.is_symlink()
                        and not path.is_junction()
                        and not any(_matches(rel, p) or is_dir and _matches(rel + '/_', p)
                                    for p in settings.get('exclude', [])))
            folders[:] = [n for n in folders if allowed(n, True)]
            if 'bti.toml' in files and allowed('bti.toml', False):
                candidate = directory / 'bti.toml'
                if any(_matches(candidate.relative_to(root).as_posix(), p)
                       for p in settings.get('discover', ['*/bti.toml'])):
                    found.append(candidate)
        policy.verify()
    finally:
        policy.close()
    return sorted(found)


class Coordinator:
    def __init__(self, config, *, recipe_factory=None, backend_factory=None):
        self.config = config
        self.recipe_factory = recipe_factory
        self.backend_factory = backend_factory
        self.directory = config.data_dir / 'runtime/service'
        self.registrations = self.directory / 'registrations.json'
        self.cv = threading.Condition(threading.RLock())
        self.stop = threading.Event()
        self.entries = {}
        self.workspace_errors = {}
        self.config_error = None
        self.last_discovery = 0.0
        self.last_config_bytes = None
        self.service_candidate = self._service_candidate(config)
        self.profile_override = config.embedding_override
        self.sync_overrides = dict(config.sync_overrides)
        self.thread = None
        if self.registrations.exists():
            saved = json.loads(self.registrations.read_text(encoding='utf-8'))
            for path in saved.get('manual', []):
                entry = self._entry(path)
                entry['manual'] = True
                try:
                    entry['project'] = load_project(Path(path) / 'bti.toml').name
                except ValueError as exc:
                    entry.update(state='error', error=str(exc))

    def _entry(self, path):
        key = str(Path(path).resolve())
        if key not in self.entries:
            self.entries[key] = {'path': key, 'project': None, 'manual': False, 'workspaces': [],
                'leases': {}, 'state': 'initializing', 'error': None, 'composition_id': None,
                'composition_hash': None, 'requested_revision': 0, 'published_revision': -1,
                'fingerprint': None, 'pending': None, 'changed_at': 0.0, 'first_change': 0.0,
                'next_check': 0.0, 'last_checked_at': None, 'sources': [], 'config_hash': None}
        return self.entries[key]

    @staticmethod
    def _service_candidate(config):
        if config.service_file is not None:
            return config.service_file
        # A legacy user config keeps working; if the machine config appears
        # next to it later, the coordinator starts using it without a restart.
        for path in config.loaded_files:
            if path.name == 'config.toml':
                return path.with_name('config-service.toml')
        return None

    def _save(self):
        _atomic_json(self.registrations, {'manual': [e['path'] for e in self.entries.values() if e['manual']]})

    @staticmethod
    def active(entry):
        return entry['manual'] or entry['workspaces'] or entry['leases']

    def register(self, path, *, manual=False, lease=None, workspace=None):
        root = Path(path).resolve()
        # Exact directory selection; deliberately do not use discover_project here.
        project = load_project(root / 'bti.toml')
        with self.cv:
            for e in self.entries.values():
                if e['path'] != str(root) and e['project'] == project.name and self.active(e):
                    raise ConfigError(f'Duplicate project name {project.name}: {e["path"]} and {root}')
            e = self._entry(root)
            e['project'] = project.name
            e['manual'] |= manual
            if workspace and workspace not in e['workspaces']:
                e['workspaces'].append(workspace)
            if lease:
                e['leases'][lease] = time.monotonic() + 120
            needs_composition = e['composition_id'] is None
        # Verification reads the whole snapshot and must not hold the global
        # condition lock: status/heartbeat of other clients would stall behind it.
        if needs_composition:
            runtime = replace(self.config, project_dir=root, project_config=root / 'bti.toml')
            try:
                info = project_info(runtime)
                if info.get('ready'):
                    from .search import SearchService
                    pinned = SearchService(runtime.data_dir).load_composition(info['composition_id'])
                    with self.cv:
                        if e['composition_id'] is None:
                            e.update(composition_id=info['composition_id'], composition_hash=info['composition_hash'],
                                     config_hash=pinned['configuration'].get('bti', {}).get('config_hash'))
            except ValueError:
                pass  # A missing/unready previous generation is built by reconcile.
        with self.cv:
            if manual:
                self._save()
            self.cv.notify_all()
            return self._public(e)

    def unwatch(self, path):
        with self.cv:
            e = self.entries.get(str(Path(path).resolve()))
            if not e:
                return {'state': 'not_registered'}
            e['manual'] = False
            self._save()
            return self._public(e)

    def release(self, lease):
        with self.cv:
            for e in self.entries.values():
                e['leases'].pop(lease, None)

    def heartbeat(self, lease):
        found = False
        with self.cv:
            for e in self.entries.values():
                if lease in e['leases']:
                    e['leases'][lease] = time.monotonic() + 120
                    found = True
        return found

    def _public(self, entry):
        keys = ('path', 'project', 'manual', 'workspaces', 'state', 'error', 'composition_id',
                'composition_hash', 'requested_revision', 'published_revision', 'last_checked_at', 'sources')
        result = {k: copy.deepcopy(entry[k]) for k in keys}
        result['temporary_clients'] = len(entry['leases'])
        result['watching'] = bool(self.active(entry))
        result['freshness'] = ('current' if entry['state'] == 'ready' else
                               'stale' if entry['composition_id'] else 'unknown')
        result['embedding_profile'] = self.config.embedding_profile or (
            self.config.qwen.model_profile if self.config.qwen else
            self.config.cocoindex.model_profile if self.config.cocoindex else None)
        return result

    def status(self):
        with self.cv:
            return {'state': 'running', 'projects': [self._public(e) for e in self.entries.values() if self.active(e)],
                    'workspaces': {k: str(v['path']) for k, v in self.config.workspaces.items()},
                    'workspace_errors': dict(self.workspace_errors), 'config_error': self.config_error}

    def workspace_projects(self, name):
        if name not in self.config.workspaces:
            raise ConfigError(f'Unknown workspace: {name}')
        projects, errors, names = [], [], {}
        for path in discover_workspace(self.config.workspaces[name]):
            try:
                project = load_project(path)
                if project.name in names:
                    projects = [p for p in projects if p['project'] != project.name]
                    raise ConfigError(f'Duplicate project name {project.name}: {names[project.name]} and {path}')
                names[project.name] = str(path)
                projects.append({'project': project.name, 'path': str(path.parent)})
            except ValueError as exc:
                errors.append({'path': str(path), 'error': str(exc)})
        return {'projects': projects, 'errors': errors}

    def refresh_settings(self):
        filename = self.config.service_file or self.service_candidate
        if filename is None or not filename.exists():
            return
        try:
            content = filename.read_bytes()
            if content == self.last_config_bytes:
                return
            # Record the attempted revision before parsing: a broken file must
            # not be re-read and re-parsed on every 250 ms coordinator tick.
            self.last_config_bytes = content
            updated = load_config(self.config.project_dir, config_path=filename)
            if updated.data_dir != self.config.data_dir:
                raise ConfigError('Changing service data_dir requires restarting the service')
            with self.cv:
                # Explicit launch overrides remain effective during subsequent file reloads.
                override = getattr(self, 'profile_override', None)
                self.config = select_profile(updated, override)
                self.config = replace(self.config, sync=self.config.sync | self.sync_overrides,
                                       sync_overrides=self.sync_overrides)
                self.service_candidate = filename
                self.config_error = None
                self.last_discovery = 0
                for e in self.entries.values():
                    e['next_check'] = 0
        except (OSError, ValueError) as exc:
            with self.cv:
                self.config_error = str(exc)

    def refresh_workspaces(self):
        memberships = {}
        for name, settings in self.config.workspaces.items():
            if not settings.get('watch', True):
                continue
            try:
                for error_key in list(self.workspace_errors):
                    if error_key.startswith(name + ':'):
                        self.workspace_errors.pop(error_key)
                for file in discover_workspace(settings):
                    try:
                        self.register(file.parent, workspace=name)
                        memberships.setdefault(str(file.parent), []).append(name)
                    except ValueError as exc:
                        self.workspace_errors[name + ':' + str(file)] = str(exc)
                self.workspace_errors.pop(name, None)
            except (OSError, ValueError) as exc:
                self.workspace_errors[name] = str(exc)
                # Preserve previous memberships during a transient discovery failure.
                with self.cv:
                    for key, e in self.entries.items():
                        if name in e['workspaces']:
                            memberships.setdefault(key, []).append(name)
        with self.cv:
            for key, e in self.entries.items():
                e['workspaces'] = memberships.get(key, [])
        self.last_discovery = time.monotonic()

    def _composition_ready(self, composition_id):
        """Cheap readiness check: status files only, no source-byte verification."""
        if not composition_id:
            return False
        try:
            path = self.config.data_dir / 'metadata/compositions' / f'{composition_id}.json'
            composition = json.loads(path.read_text(encoding='utf-8'))
            if composition.get('ready') is not True or composition.get('composition_id') != composition_id:
                return False
            registry = Registry(self.config.data_dir)
            for reference in composition['units']:
                if registry.index_status(reference['unit_id'], verify_sources=False)['status'] != 'ready':
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _confirm_unchanged(self, key, target):
        """Confirm a queued revision without waiting for another project's build.

        The full check is intentionally serialized, but a waiter for an
        unchanged project must not time out because a different project is
        indexing. This path only hashes the current selection and republishes
        the already pinned composition.
        """
        from .cli import _recipe
        try:
            config = replace(self.config, project_dir=Path(key), project_config=Path(key) / 'bti.toml')
            project = load_project(config.project_config)
            plan = project.plan(config.data_dir)
            if not plan['ready']:
                return False
            recipe = (self.recipe_factory or _recipe)(config)
            fingerprint = _digest({'config': project.config_hash, 'recipe': asdict(recipe),
                                   'files': [s['files'] for s in plan['sources']]})
        except (OSError, ValueError):
            return False
        with self.cv:
            e = self.entries.get(key)
            if e is None or fingerprint != e['fingerprint'] or not self._composition_ready(e['composition_id']):
                return False
            e.update(state='ready', error=None, last_checked_at=now(),
                     published_revision=max(e['published_revision'], target), pending=None,
                     next_check=time.monotonic() + self.config.sync.get('poll_interval_seconds', 1))
            self.cv.notify_all()
            return True

    def check(self, key, *, force=False):
        """Run one serialized reconcile. Injectable factories allow model-free behavioral tests."""
        from .application import make_backend
        from .cli import _recipe
        with self.cv:
            e = self.entries[key]
            revision = e['requested_revision']
            config = replace(self.config, project_dir=Path(key), project_config=Path(key) / 'bti.toml')
        try:
            project = load_project(config.project_config)
            with self.cv:
                for other in self.entries.values():
                    if other['path'] != key and other['project'] == project.name and self.active(other):
                        raise ConfigError(f'Duplicate project name {project.name}: {other["path"]} and {key}')
                e['project'] = project.name
            plan = project.plan(config.data_dir)
            if not plan['ready']:
                raise ConfigError('Project preflight failed; inspect bti plan: ' +
                                  json.dumps(plan, ensure_ascii=False, default=str)[:3000])
            recipe = (self.recipe_factory or _recipe)(config)
            fingerprint = _digest({'config': project.config_hash, 'recipe': asdict(recipe),
                                   'files': [s['files'] for s in plan['sources']]})
            moment = time.monotonic()
            with self.cv:
                e['last_checked_at'] = now()
                e['sources'] = [s.source_id for s in project.sources]
                if fingerprint == e['fingerprint'] and self._composition_ready(e['composition_id']):
                    e.update(state='ready', error=None, published_revision=revision,
                             next_check=moment + config.sync.get('poll_interval_seconds', 1),
                             pending=None)
                    self.cv.notify_all()
                    return
                if e['pending'] != fingerprint:
                    e['first_change'] = e['first_change'] or moment
                    e.update(pending=fingerprint, changed_at=moment)
                debounce = config.sync.get('debounce_ms', 2000) / 1000
                if not force and moment - e['changed_at'] < debounce and moment - e['first_change'] < 10:
                    e.update(state='pending', next_check=moment + min(1, debounce))
                    self.cv.notify_all()
                    return
                e.update(state='indexing', error=None)
            backend = (self.backend_factory or make_backend)(config)
            result = sync_project(project, config, recipe, backend, plan=plan)
            with self.cv:
                e.update(composition_id=result['pinned_composition_id'], composition_hash=result['composition_hash'],
                         config_hash=project.config_hash, fingerprint=fingerprint, pending=None, first_change=0.0,
                         state='ready', error=None, published_revision=revision,
                         next_check=time.monotonic() + config.sync.get('poll_interval_seconds', 1))
                self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                e.update(state='error', error=str(exc), failed_revision=revision, next_check=time.monotonic() + 5)
                self.cv.notify_all()

    def select(self, path, *, lease, freshness='available', timeout_ms=30000, manual_sync=False):
        if freshness not in {'available', 'wait'} or type(timeout_ms) is not int or not 0 <= timeout_ms <= 60000:
            raise ConfigError('Invalid freshness or timeout_ms (0..60000)')
        self.register(path, lease=lease)
        key = str(Path(path).resolve())
        target = manual_target = None
        with self.cv:
            e = self.entries[key]
            if manual_sync:
                e['manual_sync'] = True
                e['next_check'] = 0
                if freshness != 'wait':
                    e['requested_revision'] += 1
                    manual_target = e['requested_revision']
                self.cv.notify_all()
            if not self.config.sync.get('enabled', True) and not manual_sync and freshness == 'wait':
                return {'state': 'sync_disabled', 'sync': self._public(e)}
            if freshness == 'wait':
                e['requested_revision'] += 1
                target = e['requested_revision']
                e['next_check'] = 0
                self.cv.notify_all()
        # This hashes the current selection, so it must run without the global
        # condition lock. Otherwise a waiter for project B would block every
        # status/heartbeat while project A is indexing.
        confirm_target = target if target is not None else manual_target
        if confirm_target is not None:
            self._confirm_unchanged(key, confirm_target)
        if target is not None:
            with self.cv:
                e = self.entries[key]
                deadline = time.monotonic() + timeout_ms / 1000
                while e['published_revision'] < target and not self.stop.is_set():
                    if e['state'] == 'error' and e['next_check'] > time.monotonic():
                        # Give the explicitly requested retry a chance to run.
                        if e.get('failed_revision', -1) >= target:
                            break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.cv.wait(min(remaining, .25))
                if e['published_revision'] < target:
                    return {'state': 'error' if e['state'] == 'error' else 'updating', 'sync': self._public(e)}
        with self.cv:
            e = self.entries[key]
            # A manual sync without wait is queued, not complete: never report the
            # previous composition as a successful synchronization.
            if manual_target is not None and e['published_revision'] < manual_target:
                return {'state': 'error' if e['state'] == 'error' else 'updating', 'sync': self._public(e)}
            info = self._public(e)
            # Config boundary edits cannot silently keep excluded sources in implicit results.
            current_hash = load_project(Path(key) / 'bti.toml').config_hash
            if e['composition_id'] and current_hash != e['config_hash']:
                return {'state': 'updating', 'sync': info}
            if not e['composition_id']:
                return {'state': 'error' if e['state'] == 'error' else 'initializing', 'sync': info}
            return {'state': 'ready', 'composition_id': e['composition_id'], 'sync': info}

    def run(self):
        while not self.stop.is_set():
            self.refresh_settings()
            if time.monotonic() - self.last_discovery >= self.config.sync.get('reconcile_interval_seconds', 60):
                self.refresh_workspaces()
            with self.cv:
                moment = time.monotonic()
                for e in self.entries.values():
                    e['leases'] = {k: v for k, v in e['leases'].items() if v > moment}
                pending = [(k, e['requested_revision'] > e['published_revision']) for k, e in self.entries.items()
                           if self.active(e) and moment >= e['next_check']
                           and (self.config.sync.get('enabled', True) or e.get('manual_sync'))]
            for key, forced in pending:
                if self.stop.is_set():
                    break
                self.check(key, force=forced)
                with self.cv:
                    e = self.entries[key]
                    newer_request = e['requested_revision'] > max(e['published_revision'], e.get('failed_revision', -1))
                    if newer_request:
                        e['next_check'] = 0
                    else:
                        e['manual_sync'] = False
            self.stop.wait(.25)

    def start(self):
        self.thread = threading.Thread(target=self.run, name='bti-coordinator', daemon=True)
        self.thread.start()


class ServiceClient:
    def __init__(self, config):
        self.config = config
        self.directory = config.data_dir / 'runtime/service'
        self.state_path = self.directory / 'state.json'
        self.lease = secrets.token_hex(16)
        self.spawned = None
        self.last_path = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

    def _quarantine_state(self):
        target = self.state_path.with_name(f'state.invalid-{os.getpid()}-{int(time.time())}.json')
        try:
            os.replace(self.state_path, target)
        except OSError:
            pass

    def state(self):
        try:
            value = json.loads(self.state_path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # A truncated/foreign state must degrade to "stopped", not break
            # status/ensure with JSONDecodeError on every call.
            self._quarantine_state()
            return None
        if (not isinstance(value, dict) or type(value.get('port')) is not int or not 1 <= value['port'] <= 65535
                or not isinstance(value.get('token'), str) or len(value['token']) != 64
                or value.get('data_dir') != str(self.config.data_dir.resolve())):
            self._quarantine_state()
            return None
        return value

    def _call(self, state, action, payload):
        request = urllib.request.Request(f'http://127.0.0.1:{state["port"]}/rpc',
            data=json.dumps({'action': action, **payload}, allow_nan=False).encode(),
            headers={'Authorization': 'Bearer ' + state['token'], 'Content-Type': 'application/json'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=65 if action == 'select' else 5) as response:
            value = json.loads(response.read(8 * 1024 * 1024))
        if not value.get('ok'):
            raise ConfigError(value.get('error', 'Service request failed'))
        return value['result']

    def probe(self):
        state = self.state()
        if not state:
            return None
        try:
            return self._call(state, 'status', {})
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return None

    def ensure(self):
        private_directory(self.directory)
        with file_lock(self.directory / 'startup.lock', 30):
            if self.probe():
                return
            # Wait for a stopping worker to release the writer lock. A zero
            # timeout here made an immediate restart fail with a misleading
            # "busy" error while the previous build was still finishing.
            with file_lock(self.directory / 'service.lock', 30):
                launch = self.directory / 'launch.json'
                _atomic_json(launch, {'config': config_payload(self.config), 'token': secrets.token_hex(32)})
                os.chmod(launch, 0o600)
            # Spawn only after releasing the lock: the child acquires the same
            # lock during startup, so holding it across WMI/cscript created a
            # race where the new service died immediately.
            if os.name == 'nt':
                from .qwen_windows import spawn_worker
                spawn_worker(sys.executable, Path(__file__).resolve().parents[2] / 'tools/service_worker.py',
                             launch, self.config.data_dir)
            else:
                environment = os.environ.copy()
                environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
                with (self.directory / 'service.log').open('ab') as log:
                    self.spawned = subprocess.Popen([sys.executable, '-m', 'bubbleteait.service_worker', '--config', str(launch)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=environment, start_new_session=True)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.probe():
                    return
                if self.spawned is not None and self.spawned.poll() is not None:
                    raise ConfigError('BTI service startup failed; inspect runtime/service/service.log')
                time.sleep(.05)
            raise ConfigError('BTI service startup timed out; inspect runtime/service/service.log')

    def request(self, action, **payload):
        if action in {'status', 'stop'} and not self.probe():
            saved = self.directory / 'registrations.json'
            return {'state': 'stopped', 'projects': [],
                    'saved_registrations': json.loads(saved.read_text()) if saved.exists() else {'manual': []}}
        if action in {'heartbeat', 'release'} and not self.probe():
            return {'state': 'stopped', 'found': False, 'released': False} if action == 'heartbeat' else {'state': 'stopped', 'released': False}
        if action not in {'status', 'stop', 'heartbeat', 'release'}:
            self.ensure()
        state = self.state()
        result = self._call(state, action, {'lease': self.lease, **payload})
        if action == 'stop':
            self._wait_stopped(state.get('pid'), 5000)
            # The worker removes state.json only after the coordinator has
            # finished its current build. Reporting "stopped" earlier was a
            # false success that made the next start hit the writer lock.
            result = dict(result)
            result['state'] = 'stopped' if not self.state_path.exists() else 'stopping'
        if action == 'select' and self._heartbeat_thread is None:
            ref, stop = weakref.ref(self), self._heartbeat_stop
            def heartbeat():
                while not stop.wait(30):
                    client = ref()
                    if client is None:
                        return
                    try:
                        renewed = client.request('heartbeat')
                        if isinstance(renewed, dict) and not renewed.get('found') and client.last_path:
                            # The service restarted and lost its lease table.
                            # Re-register the last selected project instead of
                            # silently renewing a lease that no longer exists.
                            client.select(client.last_path)
                    except (OSError, ValueError):
                        pass
                    finally:
                        del client
            self._heartbeat_thread = threading.Thread(target=heartbeat, name='bti-lease', daemon=True)
            self._heartbeat_thread.start()
        return result

    def _wait_stopped(self, pid, timeout_ms):
        if type(pid) is not int:
            return
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE only, never terminate.
            if handle:
                try:
                    kernel.WaitForSingleObject(handle, timeout_ms)
                finally:
                    kernel.CloseHandle(handle)
        elif self.spawned is not None:
            try:
                self.spawned.wait(timeout=timeout_ms / 1000)
            except subprocess.TimeoutExpired:
                pass

    def select(self, path, **kwargs):
        self.last_path = str(path)
        return self.request('select', path=str(path), **kwargs)

    def close(self):
        self._heartbeat_stop.set()
        if self.probe():
            self.request('release')
