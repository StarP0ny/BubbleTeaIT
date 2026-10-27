"""Project/workspace selection and session-owned evidence across publications."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

from .config import ConfigError
from .direct import DirectService
from .search import SearchError
from .service import ServiceClient, config_from_payload


class DirectSession:
    def __init__(self, config, explicit_service, workspace=None):
        self.config = config
        self.explicit_service = explicit_service
        self.workspace = workspace
        self.client = ServiceClient(config)
        self.services = {}
        self.ledger = {}
        self.content_ids = {}
        if workspace is not None and workspace not in config.workspaces:
            raise ConfigError(f'Unknown workspace: {workspace}')

    def scopes(self, composition_id=None, *, project=None, workspace=None, projects=None,
               compositions=None, freshness='available', timeout_ms=30000):
        if composition_id is not None:
            if project or workspace or projects is not None or compositions is not None or freshness != 'available':
                raise ConfigError('Explicit composition_id conflicts with live selectors/freshness')
            return [(None, self.explicit_service(composition_id), None)], None
        workspace = workspace or self.workspace
        if compositions is not None:
            if project or projects is not None or freshness != 'available':
                raise ConfigError('Pinned compositions conflict with live selectors/freshness')
            values = []
            names = set()
            for item in compositions:
                if (not {'project', 'composition_id'} <= set(item)
                        or not set(item) <= {'project', 'composition_id', 'composition_hash'} or item['project'] in names):
                    raise ConfigError('Pinned compositions need unique project/composition_id pairs')
                names.add(item['project'])
                # Pinning does not create registrations or launch the service.
                direct = self.explicit_service(item['composition_id'])
                if not item['composition_id'].endswith('-' + direct.composition['composition_hash']):
                    raise ConfigError('compositions requires immutable composition IDs, not live aliases')
                components = direct.composition.get('configuration', {}).get('bti', {}).get('components', [])
                if components and components[0]['name'] != item['project']:
                    raise ConfigError('Pinned composition belongs to a different project')
                if 'composition_hash' in item and item['composition_hash'] != direct.composition['composition_hash']:
                    raise ConfigError('Pinned composition_hash does not match its composition_id')
                values.append((item['project'], direct, None))
            if not values:
                raise ConfigError('compositions must not be empty')
            return values, None
        if project and projects is not None:
            raise ConfigError('project and projects are mutually exclusive')
        if workspace:
            if workspace not in self.config.workspaces or self.workspace and workspace != self.workspace:
                raise ConfigError('Workspace is outside this MCP connection scope')
            self.client.ensure()
            catalog = self.client.request('workspace_projects', workspace=workspace)
            candidates = catalog['projects']
            if catalog.get('errors') and freshness == 'wait':
                return [], {'state': 'error', 'workspace_errors': catalog['errors']}
            wanted = [project] if project else projects
            if wanted is not None:
                missing = set(wanted) - {p['project'] for p in candidates}
                if missing:
                    raise ConfigError('Unknown workspace projects: ' + ', '.join(sorted(missing)))
                candidates = [p for p in candidates if p['project'] in wanted]
            if not candidates:
                return [], {'state': 'initializing', 'workspace': workspace, 'reason': 'No discovered projects yet'}
        else:
            if projects is not None:
                raise ConfigError('projects requires a workspace')
            if self.config.project_config is None:
                raise ConfigError('No project configured; use --project-dir or mcp --workspace')
            from .project import load_project
            configured = load_project(self.config.project_config)
            if project and project != configured.name:
                raise ConfigError('Project is outside this MCP connection scope')
            candidates = [{'project': configured.name, 'path': str(self.config.project_config.parent)}]
        import time
        deadline = time.monotonic() + timeout_ms / 1000
        chosen, omitted = [], []
        if workspace and catalog.get('errors'):
            omitted.extend({'state': 'error', **entry} for entry in catalog['errors'])
        for entry in candidates:
            selected = self.client.select(entry['path'], freshness=freshness,
                                          timeout_ms=max(0, int((deadline - time.monotonic()) * 1000)))
            if selected['state'] != 'ready':
                omitted.append({'project': entry['project'], **selected})
                continue
            cid = selected['composition_id']
            key = (cid, entry['path'])
            if key not in self.services:
                settings = self.client.request('config')
                runtime = config_from_payload(settings['settings']) if 'settings' in settings else self.config
                self.services[key] = DirectService(replace(runtime, project_dir=Path(entry['path'])), cid)
            chosen.append((entry['project'], self.services[key], selected['sync']))
        if omitted and (freshness == 'wait' or not chosen):
            state = 'error' if any(o['state'] == 'error' for o in omitted) else 'updating'
            return [], {'state': state, 'omitted_projects': omitted}
        return chosen, {'omitted_projects': omitted} if omitted else None

    def remember(self, result, direct, project=None):
        for excerpt in result.get('excerpts', []):
            if project is not None:
                # A reused unchanged unit in a new composition must not accidentally
                # reuse an old session ID and make a fresh batch look like mixed versions.
                internal = excerpt['id']
                identifier = 'e-' + hashlib.sha256(
                    (direct.composition['composition_hash'] + ':' + internal).encode()).hexdigest()[:20]
                with direct._lock:
                    entry = copy.deepcopy(direct._ledger[internal])
                    direct._ledger[identifier] = {**entry, 'id': identifier}
                excerpt['id'] = identifier
                self.content_ids[identifier] = internal
            self.ledger.setdefault(excerpt['id'], (direct, project))
        return result

    @staticmethod
    def metadata(project, sync):
        keys = ('state', 'freshness', 'requested_revision', 'published_revision', 'last_checked_at', 'error')
        return {'state': 'ready', 'project': project,
                'sync': {k: sync[k] for k in keys if k in sync} if sync else None}

    def invoke(self, direct, method, argument, options, project, sync):
        metadata = self.metadata(project, sync)
        budget = options.get('max_chars', 12000)
        reserve = len(json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))) + 32
        if budget - reserve < 512:
            raise SearchError('max_chars is too small for context and sync metadata; increase the budget')
        result = getattr(direct, method)(argument, **(options | {'max_chars': budget - reserve}))
        self.remember(result, direct, project)
        return {**result, **metadata}

    def read_scope(self, requests, composition_id, **selectors):
        origins = []
        for request in requests or []:
            identifier = request if isinstance(request, str) else request.get('id')
            if identifier and identifier not in self.ledger and composition_id is None:
                raise SearchError('Unknown or forged direct evidence ID')
            if identifier and identifier in self.ledger:
                origins.append(self.ledger[identifier])
        if origins and composition_id is None:
            if selectors.get('workspace'):
                raise SearchError('Evidence IDs pin their original scope; omit workspace when reading IDs')
            hashes = {direct.composition['composition_hash'] for direct, _ in origins}
            if len(hashes) != 1:
                raise SearchError('Evidence from different compositions must be read in separate requests')
            if selectors.get('freshness', 'available') != 'available':
                raise SearchError('Evidence IDs pin their original version; use a path with freshness=wait')
            direct, project = origins[0]
            if selectors.get('project') and selectors['project'] != project:
                raise SearchError('Evidence ID belongs to a different project')
            return [(project, direct, None)], None
        return self.scopes(composition_id, **selectors)

    def resolved(self, ids, composition_id=None):
        if composition_id is not None:
            direct = self.explicit_service(composition_id)
            return {'composition_hash': direct.composition['composition_hash'], 'evidence': direct.resolve(ids)}
        evidence = []
        for identifier in ids:
            if identifier not in self.ledger:
                raise SearchError('Unknown or forged direct evidence ID')
            direct, project = self.ledger[identifier]
            item = direct.resolve([identifier])[0]
            evidence.append({**item, 'composition_id': direct.composition_id, 'project': project})
        return {'evidence': evidence}

    def status(self, project=None, workspace=None):
        workspace = workspace or self.workspace
        if workspace and (workspace not in self.config.workspaces or self.workspace and workspace != self.workspace):
            raise ConfigError('Workspace is outside this MCP connection scope')
        result = self.client.request('status')
        if workspace:
            result['projects'] = [p for p in result['projects'] if workspace in p['workspaces']]
        elif self.config.project_config is not None:
            root = str(self.config.project_config.parent)
            result['projects'] = [p for p in result['projects'] if p['path'] == root]
        if project:
            result['projects'] = [p for p in result['projects'] if p['project'] == project]
        return result

    def sync(self, *, wait=True, timeout_ms=30000, project=None, workspace=None):
        workspace = workspace or self.workspace
        if workspace:
            if workspace not in self.config.workspaces or self.workspace and workspace != self.workspace:
                raise ConfigError('Workspace is outside this MCP connection scope')
            candidates = self.client.request('workspace_projects', workspace=workspace)
            if candidates['errors']:
                return {'state': 'error', 'workspace_errors': candidates['errors']}
            candidates = candidates['projects']
            if project:
                candidates = [p for p in candidates if p['project'] == project]
        elif self.config.project_config is not None:
            from .project import load_project
            configured = load_project(self.config.project_config)
            if project and configured.name != project:
                raise ConfigError('Project is outside this MCP connection scope')
            candidates = [{'project': configured.name, 'path': str(self.config.project_config.parent)}]
        else:
            raise ConfigError('Select a project or workspace for synchronization')
        if not candidates:
            raise ConfigError('No matching project to synchronize')
        import time
        deadline = time.monotonic() + timeout_ms / 1000
        values = [{'project': p['project'], **self.client.select(p['path'],
            freshness='wait' if wait else 'available', manual_sync=True,
            timeout_ms=max(0, int((deadline - time.monotonic()) * 1000)))} for p in candidates]
        if len(values) == 1:
            return values[0]
        return {'state': 'ready' if all(v['state'] == 'ready' for v in values) else 'updating', 'projects': values}

    def search(self, query, *, composition_id=None, project=None, workspace=None, projects=None,
               compositions=None, freshness='available', timeout_ms=30000, **options):
        scopes, state = self.scopes(composition_id, project=project, workspace=workspace, projects=projects,
                                    compositions=compositions, freshness=freshness, timeout_ms=timeout_ms)
        if not scopes:
            return state
        responses = []
        for name, direct, sync in scopes:
            if len(scopes) == 1 and composition_id is None and not (workspace or self.workspace or compositions):
                return self.invoke(direct, 'search', query, options, name, sync)
            result = direct.search(query, **options)
            self.remember(result, direct, name)
            if composition_id is None:
                result = {**result, **self.metadata(name, sync)}
            responses.append((name, direct, result))
        if len(responses) == 1 and not (workspace or self.workspace or compositions):
            return responses[0][2]
        return self.combine(responses, options.get('limit', 6), options.get('max_chars', 12000), state)

    def combine(self, responses, limit, budget, state=None):
        # Round-robin ranks, never compare backend similarity scores across models.
        result = {'state': 'ready', 'compositions': [
            {'project': n, 'composition_id': d.composition_id, 'composition_hash': d.composition['composition_hash'],
             } for n, d, r in responses],
            'sync': {n: r.get('sync') for n, _, r in responses},
            'excerpts': [], 'sources': {}, 'truncated': False,
            'omitted_projects': (state or {}).get('omitted_projects', []), 'omitted_count': 0,
            'coverage': {name: {'returned': 0, 'omitted': r.get('omitted_count', 0)} for name, _, r in responses}}
        seen = {}
        rows = []
        for rank in range(max(len(r['excerpts']) for _, _, r in responses)):
            rows.extend((n, r, r['excerpts'][rank]) for n, _, r in responses if rank < len(r['excerpts']))
        for name, response, excerpt in rows:
            content_id = self.content_ids.get(excerpt['id'], excerpt['id'])
            if content_id in seen:
                shared = seen[content_id]
                if name not in shared['projects']:
                    shared['projects'].append(name)
                result['coverage'][name]['returned'] += 1
                continue
            item = {**excerpt, 'project': name}
            item['projects'] = [name]
            if 'next_read' in item:
                item['next_read'] = {**item['next_read'], 'project': name}
            key = name + ':' + excerpt['source_id']
            old_source = result['sources'].get(key)
            result['excerpts'].append(item)
            result['sources'][key] = response['sources'][excerpt['source_id']]
            if len(result['excerpts']) > limit or len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))) > budget - 128:
                result['excerpts'].pop()
                if old_source is None:
                    result['sources'].pop(key)
                result['omitted_count'] += 1
                result['coverage'][name]['omitted'] += 1
            else:
                seen[content_id] = item
                result['coverage'][name]['returned'] += 1
        result['truncated'] = bool(result['omitted_count'] or result['omitted_projects']
                                   or any(r.get('truncated') for _, _, r in responses))
        if len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))) > budget:
            raise SearchError('Workspace metadata exceeds max_chars; narrow projects or increase budget')
        return result

    def read_origins(self, requests, *, max_chars, include_catalog=False):
        groups, versions = {}, {}
        for request in requests:
            identifier = request if isinstance(request, str) else request.get('id')
            if identifier not in self.ledger:
                raise SearchError('Unknown or forged direct evidence ID')
            direct, name = self.ledger[identifier]
            if name in versions and versions[name] != direct.composition_id:
                raise SearchError('Two versions of the same project must be read in separate batches')
            versions[name] = direct.composition_id
            key = (name, direct)
            groups.setdefault(key, []).append(identifier)
        responses = []
        for (name, direct), ids in groups.items():
            response = direct.read_many(ids, max_chars=max_chars, include_catalog=include_catalog)
            self.remember(response, direct, name)
            responses.append((name, direct, response))
        return self.combine(responses, 20, max_chars)
