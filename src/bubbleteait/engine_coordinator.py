"""Live-engine coordination using the existing service lifecycle and watch queue."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import threading
import time

from .config import ConfigError, load_config
from .engine_index import CocoEngine, atomic_bytes
from .engine_projects import selected_project, workspace_catalog
from .engine_read_context import function_ranges, page_end, pinned_read, read_context
from .registry import _digest, _matches
from .service import Coordinator, now


def bounded(result, budget):
    if type(budget) is not int or not 512 <= budget <= 64000:
        raise ConfigError('max_chars must be an integer in [512, 64000]')
    result['truncated'] = False
    removed = 0
    while len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))) > budget:
        if not result.get('excerpts'):
            raise ConfigError('max_chars is too small for result metadata')
        result['excerpts'].pop()
        removed += 1
        result['truncated'] = True
        result['omitted_count'] = removed
    return result


class EngineCoordinator(Coordinator):
    def __init__(self, config, *, engine_factory=CocoEngine):
        super().__init__(config)
        self.engine = engine_factory(config)
        self.project_locks = defaultdict(threading.RLock)
        self.evidence = {}
        self.evidence_lock = threading.Lock()
        self.symbol_sessions = {}
        self.symbol_lock = threading.Lock()
        self.requests_cv = threading.Condition()
        self.active_requests = 0
        self.symbol_project_locks = defaultdict(threading.RLock)

    def register(self, path, *, manual=False, lease=None, workspace=None):
        root, project, _ = selected_project(self.config, path)
        with self.cv:
            if any(e['project'] == project.name and e['path'] != str(root) and self.active(e)
                   for e in self.entries.values()):
                raise ConfigError('Duplicate active project name: ' + project.name)
            entry = self._entry(root)
            entry['project'] = project.name
            entry['manual'] |= manual
            if workspace and workspace not in entry['workspaces']:
                entry['workspaces'].append(workspace)
            if lease:
                entry['leases'][lease] = time.monotonic() + 120
            if manual:
                self._save()
            self.cv.notify_all()
            return self._public(entry)

    def _public(self, entry):
        result = super()._public(entry)
        result['generation'] = result.pop('composition_id')
        result.pop('composition_hash')
        result['backend'] = 'cocoindex-code'
        result['embedding_profile'] = self.config.engines['cocoindex']['model_revision']
        return result

    def status(self):
        value = super().status()
        value['engines'] = {'cocoindex': {'fingerprint': self.engine.fingerprint,
            'bridge_running': self.engine.process.process is not None and self.engine.process.process.poll() is None,
            'bridge_pid': self.engine.process.process.pid if self.engine.process.process is not None else None},
            'serena': {'sessions': len(self.symbol_sessions), 'configured': 'serena' in self.config.engines}}
        return value

    def workspace_projects(self, name):
        return workspace_catalog(self.config, name)

    def refresh_workspaces(self):
        memberships = defaultdict(list)
        for name, settings in self.config.workspaces.items():
            if not settings.get('watch', True):
                continue
            catalog = self.workspace_projects(name)
            if catalog['errors']:
                self.workspace_errors[name] = str(catalog['errors'])
            else:
                self.workspace_errors.pop(name, None)
            for item in catalog['projects']:
                self.register(item['path'], workspace=name)
                memberships[item['path']].append(name)
        with self.cv:
            for key, entry in self.entries.items():
                entry['workspaces'] = memberships.get(key, [])
        self.last_discovery = time.monotonic()

    def refresh_settings(self):
        # A live model/index namespace is process-owned. Do not partially apply a new recipe.
        filename = self.config.service_file
        if filename is None or not filename.exists():
            return
        content = filename.read_bytes()
        if content == self.last_config_bytes:
            return
        try:
            candidate = load_config(self.config.project_dir, config_path=filename)
            from dataclasses import replace
            engines = copy.deepcopy(candidate.engines)
            for name, values in self.config.engine_overrides.items():
                engines.setdefault(name, {}).update(values)
            if 'cocoindex' in engines:
                engines['cocoindex']['model_path'] = Path(engines['cocoindex']['model_path'])
            candidate = replace(candidate, engines=engines, engine_overrides=self.config.engine_overrides,
                                sync=candidate.sync | self.sync_overrides, sync_overrides=self.sync_overrides)
            from .service import config_payload
            before, after = config_payload(self.config), config_payload(candidate)
            keys = ('engines', 'projects', 'workspaces', 'data_dir', 'models_dir')
            if any(before[k] != after[k] for k in keys):
                raise ConfigError('Engine/project settings changed: restart BTI service to apply them')
            self.config = candidate
            self.config_error = None
            self.last_config_bytes = content
        except (ValueError, OSError) as exc:
            self.config_error = str(exc)

    @staticmethod
    def signature(project, plan):
        return _digest({'config': project.config_hash, 'sources': [
            {'selection': p['selection'], 'files': p['files']} for p in plan['sources']]})

    def check(self, key, *, force=False, retry=True):
        with self.project_locks[key]:
            try:
                root, project, _ = selected_project(self.config, key)
                plan = project.plan(self.config.data_dir, refresh=force)
                signature = self.signature(project, plan)
                with self.cv:
                    entry = self.entries[key]
                    entry['next_check'] = time.monotonic() + self.config.sync.get('poll_interval_seconds', 10)
                    if entry['detected_fingerprint'] != signature:
                        entry['detected_fingerprint'] = signature
                        entry['requested_revision'] += 1
                        entry['changed_at'] = time.monotonic()
                    target = entry['requested_revision']
                    if entry['fingerprint'] == signature and self.engine.available(root):
                        entry.update(state='ready', error=None, published_revision=target, last_checked_at=now())
                        self.cv.notify_all()
                        return
                    delay = self.config.sync.get('debounce_ms', 2000) / 1000
                    if not force and entry['composition_id'] and time.monotonic() - entry['changed_at'] < delay:
                        entry['state'] = 'updating'
                        entry['next_check'] = time.monotonic() + .25
                        return
                    entry.update(state='updating', error=None)
                def verify():
                    _, current, _ = selected_project(self.config, key)
                    if self.signature(current, current.plan(self.config.data_dir, refresh=True)) != signature:
                        raise ConfigError('Source changed during indexing; synchronization will retry')
                manifest = self.engine.synchronize(root, project, plan, verify=verify)
                with self.cv:
                    entry.update(state='ready', error=None, fingerprint=signature, config_hash=project.config_hash,
                        composition_id=manifest['generation'], composition_hash=signature,
                        published_revision=target, last_checked_at=now(), sources=[
                            {'source_id': s['selection']['source_id'], 'files': s['file_count']} for s in plan['sources']])
                    self.cv.notify_all()
            except Exception as exc:
                with self.cv:
                    self.entries[key].update(state='error', error=f'{type(exc).__name__}: {exc}', last_checked_at=now(),
                        next_check=time.monotonic() + self.config.sync.get('poll_interval_seconds', 10))
                    self.cv.notify_all()

    def select(self, path, *, lease, freshness='available', timeout_ms=30000, manual_sync=False):
        if freshness not in {'available', 'wait'} or type(timeout_ms) is not int or not 0 <= timeout_ms <= 60000:
            raise ConfigError('Invalid freshness or timeout_ms')
        self.register(path, lease=lease)
        key = str(Path(path).resolve())
        with self.cv:
            entry = self.entries[key]
            if freshness == 'wait' or manual_sync:
                entry['explicit_revision'] += 1
                entry['requested_revision'] += 1
                entry['force_check'] = True
                entry['manual_sync'] = True
                entry['next_check'] = 0
            target = entry['requested_revision']
            self.cv.notify_all()
            deadline = time.monotonic() + timeout_ms / 1000
            while freshness == 'wait' and (entry['published_revision'] < target or entry['state'] != 'ready'):
                if entry['state'] == 'error' and not entry['force_check']:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.stop.is_set():
                    break
                self.cv.wait(min(remaining, .25))
            public = self._public(entry)
            ready = entry['composition_id'] is not None and self.engine.available(Path(key))
            if freshness == 'wait':
                ready &= entry['published_revision'] >= target and entry['state'] == 'ready'
            return {'state': 'ready' if ready else 'error' if entry['state'] == 'error' else 'updating',
                    'generation': entry['composition_id'], 'sync': public}

    def scopes(self, request):
        workspace, name, names = request.get('workspace'), request.get('project'), request.get('projects')
        if names is not None and (not isinstance(names, list) or not 1 <= len(names) <= 32
                                  or any(not isinstance(n, str) or not n for n in names)
                                  or len(set(names)) != len(names)):
            raise ConfigError('projects must contain 1..32 unique project names')
        if name and names is not None:
            raise ConfigError('project and projects are mutually exclusive')
        if workspace:
            catalog = self.workspace_projects(workspace)
            candidates = catalog['projects']
            for error in catalog['errors']:
                alias = error['path']
                if alias in self.config.projects and alias in self.config.workspaces[workspace].get('projects', []):
                    candidates.append({'project': alias, 'path': str(self.config.projects[alias]['path'])})
            if names is None and not name:
                name = self.config.workspaces[workspace].get('default_project')
                if name is None:
                    configured = [p for p in candidates if Path(p['path']).resolve() == self.config.project_dir.resolve()]
                    name = configured[0]['project'] if configured else candidates[0]['project'] if len(candidates) == 1 else None
                if name is None:
                    raise ConfigError('Select project/projects or configure workspace.default_project')
            wanted = [name] if name else names
            missing = set(wanted) - {p['project'] for p in candidates}
            if missing:
                raise ConfigError('Unknown workspace projects: ' + ', '.join(sorted(missing)))
            return [p for p in candidates if p['project'] in wanted]
        if names is not None:
            raise ConfigError('projects requires workspace')
        root, project, item = selected_project(self.config, name)
        return [{'project': project.name, 'path': str(root), 'description': item.get('description', '')}]

    def excerpt(self, project, manifest, record, start, end):
        lines = self.engine.text(record).splitlines(keepends=True)
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ConfigError('Invalid source line range')
        value = {'project': project, 'generation': manifest['generation'], 'source_id': record['source_id'],
                 'path': record['path'], 'source_path': record['source_path'], 'absolute_path': record['absolute_path'],
                 'lines': [start, end], 'sha256': record['sha256'], 'text': ''.join(lines[start-1:end])}
        value['id'] = 'e-' + _digest({k: v for k, v in value.items() if k != 'text'})[:32]
        try:
            value['source_changed'] = hashlib.sha256(Path(record['absolute_path']).read_bytes()).hexdigest() != record['sha256']
        except OSError:
            value['source_changed'] = True
        with self.evidence_lock:
            if len(self.evidence) >= 20000:
                self.evidence.pop(next(iter(self.evidence)))
            self.evidence[value['id']] = (copy.deepcopy(value), copy.deepcopy(record))
        return value

    def engine_call(self, request):
        with self.requests_cv:
            if self.stop.is_set():
                raise ConfigError('BTI service is stopping')
            self.active_requests += 1
        try:
            return self._engine_call(request)
        finally:
            with self.requests_cv:
                self.active_requests -= 1
                self.requests_cv.notify_all()

    def _engine_call(self, request):
        operation = request.get('operation')
        budget = request.get('max_chars', 12000)
        if operation in {'read', 'resolve'} and request.get('ids'):
            if len(request['ids']) > 80:
                raise ConfigError('At most 80 evidence IDs are allowed')
            with self.evidence_lock:
                try:
                    records = [copy.deepcopy(self.evidence[i]) for i in request['ids']]
                except KeyError as exc:
                    raise ConfigError('Unknown or expired evidence ID; repeat search') from exc
            if operation == 'resolve':
                return {'evidence': [{k: v for k, v in r.items() if k != 'text'} for r, _ in records]}
            excerpts = []
            anchors = request.get('anchors')
            if anchors is not None and len(anchors) != len(records):
                raise ConfigError('Each pinned read requires its original evidence ID')
            if type(budget) is not int or not 512 <= budget <= 64000:
                raise ConfigError('max_chars must be an integer in [512, 64000]')
            for index, (value, record) in enumerate(records):
                start, end = value['lines']
                continuation = None
                if anchors is not None:
                    read = anchors[index]
                    if (request.get('project') != value['project'] or
                            any(read.get(key) != value[key] for key in ('path', 'source_id', 'sha256'))):
                        raise ConfigError('Pinned source locator differs from the issued evidence')
                    start, end = read.get('start'), read.get('end')
                    lines = self.engine.text(record).splitlines(keepends=True)
                    stop = page_end(value, lines, start, end, budget)
                    if stop < end:
                        continuation = pinned_read(value, stop + 1, end)
                        end = stop
                excerpt = self.excerpt(value['project'], {'generation': value['generation']}, record, start, end)
                if continuation:
                    excerpt['next_read'] = continuation
                excerpts.append(excerpt)
            return bounded({'state': 'ready', 'excerpts': excerpts}, budget)
        scopes = self.scopes(request)
        if operation == 'info':
            return {'mode': 'coordinated-live', 'projects': scopes,
                    'engines': list(self.config.engines), 'workspace': request.get('workspace')}
        if operation == 'plan':
            return {'projects': [selected_project(self.config, s['path'])[1].plan(self.config.data_dir, refresh=True) for s in scopes]}
        if operation == 'symbols':
            if len(scopes) != 1:
                raise ConfigError('Select one project for symbols')
            return self.symbols(scopes[0], request)
        if operation in {'read', 'files'}:
            return self.read_or_files(scopes, request)
        if operation not in {'search', 'sync'}:
            raise ConfigError('Unknown engine operation')
        query, limit = request.get('query', ''), request.get('limit', 6)
        if operation == 'search' and (not isinstance(query, str) or not query.strip() or len(query) > 16000):
            raise ConfigError('query must contain 1..16000 characters')
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ConfigError('limit must be in [1, 100]')
        groups, omitted, rows = [], [], []
        timeout = request.get('timeout_ms', 30000)
        if type(timeout) is not int or not 0 <= timeout <= 60000:
            raise ConfigError('timeout_ms must be in [0, 60000]')
        deadline = time.monotonic() + timeout / 1000
        for scope in scopes:
            root = Path(scope['path'])
            try:
                selected = self.select(root, lease=request['lease'], freshness=request.get('freshness', 'available'),
                    timeout_ms=max(0, int((deadline-time.monotonic())*1000)), manual_sync=operation == 'sync')
            except (ValueError, OSError) as exc:
                error = {'project': scope['project'], 'state': 'error', 'error': str(exc)}
                groups.append(error)
                omitted.append(error)
                continue
            groups.append({'project': scope['project'], **selected})
            if selected['state'] != 'ready':
                omitted.append({'project': scope['project'], **selected})
                continue
            if operation == 'sync':
                continue
            try:
                rows.append(self.search_scope(scope, root, query, limit, request))
            except (ValueError, OSError) as exc:
                error = {'project': scope['project'], 'state': 'error', 'error': str(exc)}
                groups[-1] = error
                omitted.append(error)
        if operation == 'sync':
            return {'state': 'ready' if not omitted else 'updating', 'projects': groups, 'omitted_projects': omitted}
        # Preserve per-project ranking; incomparable scores never determine cross-project order.
        combined = [row[i] for i in range(limit) for row in rows if i < len(row)][:limit]
        state = 'partial' if omitted and combined else 'error' if omitted and any(o['state'] == 'error' for o in omitted) else 'updating' if omitted else 'ready'
        return bounded({'state': state, 'excerpts': combined, 'projects': groups, 'omitted_projects': omitted,
                        'freshness': 'unknown' if not combined else 'stale' if any(e['source_changed'] for e in combined) else 'indexed'}, budget)

    def search_scope(self, scope, root, query, limit, request):
        with self.project_locks[str(root)]:
            manifest = self.engine.manifest(root)
            if manifest['generation'] != self.entries[str(root)]['composition_id']:
                raise ConfigError('Native manifest generation changed outside the coordinator')
            paths = [key for key, record in manifest['files'].items() if self.matches(record, request)]
            hits = self.engine.search(root, query, limit, paths)['results'] if paths else []
            excerpts = []
            sources = {}
            for hit in hits:
                key = hit['file_path'].replace('\\', '/')
                if Path(hit['file_path']).is_absolute():
                    key = Path(hit['file_path']).relative_to(self.engine.project_directory(root) / 'view').as_posix()
                if key not in paths:
                    raise ConfigError('Engine returned a path outside the selected sources')
                record = manifest['files'][key]
                start, end = hit['start_line'], hit['end_line']
                # Validate before issuing an evidence ID.
                if key not in sources:
                    source = self.engine.text(record)
                    sources[key] = source.splitlines(keepends=True), function_ranges(record['path'], source)
                lines, ranges = sources[key]
                text = ''.join(lines[start-1:end])
                if hit['content'].replace('\r\n', '\n') not in text.replace('\r\n', '\n'):
                    raise ConfigError('Engine text does not match the indexed source version')
                excerpt = self.excerpt(scope['project'], manifest, record, start, end)
                kind, first, last = read_context(ranges, start, end, len(lines))
                excerpt.update(context_kind=kind, next_read=pinned_read(excerpt, first, last))
                excerpts.append(excerpt | {'score': hit['score'], 'backend': 'cocoindex-code'})
            return excerpts

    @staticmethod
    def matches(record, request):
        return ((not request.get('roles') or record['role'] in request['roles'])
            and (not request.get('source_ids') or record['source_id'] in request['source_ids'])
            and (not request.get('paths') or any(_matches(record['path'], p) or _matches(record['source_path'], p) for p in request['paths'])))

    def read_or_files(self, scopes, request):
        if len(scopes) != 1:
            raise ConfigError('Select one project for reads/files')
        root, project, _ = selected_project(self.config, scopes[0]['path'])
        plan = project.plan(self.config.data_dir, refresh=True)
        if not plan['ready']:
            raise ConfigError('Source preflight failed; inspect bti plan')
        records = self.engine.records(root, plan)
        if request['operation'] == 'files':
            return {'project': project.name, 'files': [v for v in records.values() if self.matches(v, request)]}
        reads = request.get('requests', [])
        if not isinstance(reads, list) or not 1 <= len(reads) <= 20:
            raise ConfigError('requests must contain 1..20 path reads')
        excerpts = []
        for read in reads:
            if not isinstance(read, dict) or not isinstance(read.get('path'), str):
                raise ConfigError('Each read requires a path')
            candidates = [r for r in records.values() if read['path'] in (r['path'], r['absolute_path'], r['source_path'])
                          and (not read.get('source_id') or r['source_id'] == read['source_id'])]
            if len(candidates) != 1:
                raise ConfigError('Path is outside selected sources, excluded, absent or ambiguous: ' + read['path'])
            record = candidates[0]
            if read.get('sha256') and read['sha256'] != record['sha256']:
                raise ConfigError('Stale source locator; source SHA changed')
            content = Path(record['absolute_path']).read_bytes()
            if hashlib.sha256(content).hexdigest() != record['sha256']:
                raise ConfigError('Source changed during read; retry')
            blob = self.engine.blob(record['sha256'])
            if not blob.exists():
                atomic_bytes(blob, content)
            count = len(content.decode('utf-8-sig').splitlines())
            start = read.get('start', 1)
            end = read.get('end', min(count, start + 119))
            excerpts.append(self.excerpt(project.name, {'generation': 'w-' + record['sha256']}, record, start, end))
        return bounded({'state': 'ready', 'excerpts': excerpts}, request.get('max_chars', 16000))

    def symbols(self, scope, request):
        from .engine_symbols import SymbolSession
        root, project, settings = selected_project(self.config, scope['path'])
        if 'serena' not in self.config.engines:
            raise ConfigError('Configure engines.serena before navigating symbols')
        with self.symbol_project_locks[str(root)]:
            plan = project.plan(self.config.data_dir, refresh=True)
            source_signature = self.signature(project, plan)
            database = settings.get('compile_commands')
            signature = _digest(json.loads(json.dumps({'sources': source_signature, 'database':
                hashlib.sha256(Path(database).read_bytes()).hexdigest() if database else None,
                'settings': settings, 'engine': self.config.engines['serena']}, default=str)))
            key = str(root)
            with self.symbol_lock:
                session = self.symbol_sessions.get(key)
                if session is None or session.signature != signature:
                    if session:
                        session.close()
                    session = SymbolSession(self.config, root, project, settings, plan, self.engine, signature)
                    self.symbol_sessions[key] = session
            result = session.query(request)
            _, current, _ = selected_project(self.config, scope['path'])
            if (self.signature(current, current.plan(self.config.data_dir, refresh=True)) != source_signature
                    or database and hashlib.sha256(Path(database).read_bytes()).hexdigest() != session.database_sha):
                raise ConfigError('Source changed during symbol query; retry navigation')
            return result

    def close(self):
        self.stop.set()
        with self.requests_cv:
            while self.active_requests:
                self.requests_cv.wait(.25)
        with self.symbol_lock:
            for session in self.symbol_sessions.values():
                session.close()
            self.symbol_sessions.clear()
        self.engine.close()
