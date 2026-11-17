"""Shared CLI/MCP calls, with connection scope and session-owned evidence IDs."""
from .config import ConfigError
from .engine_projects import selected_project, workspace_catalog
from .service import ServiceClient


class EngineClient:
    def __init__(self, config, workspace=None):
        self.config, self.workspace = config, workspace
        self.client = ServiceClient(config)
        self.issued = set()
        if workspace is not None and workspace not in config.workspaces:
            raise ConfigError('Unknown workspace: ' + workspace)

    def call(self, operation, **arguments):
        workspace = arguments.get('workspace') or self.workspace
        if self.workspace and workspace != self.workspace:
            raise ConfigError('Workspace is outside this MCP connection scope')
        if workspace:
            arguments['workspace'] = workspace
            settings = self.config.workspaces[workspace]
            if (operation not in {'info', 'plan'} and not settings.get('default_project')
                    and not any(arguments.get(k) for k in ('ids', 'project', 'projects'))):
                for name in settings.get('projects', []):
                    if self.config.projects[name]['path'] == self.config.project_dir:
                        arguments['project'] = name
                        break
        elif not arguments.get('ids'):
            _, project, _ = selected_project(self.config)
            if arguments.get('project') not in (None, project.name):
                raise ConfigError('Project is outside this connection scope; select a workspace')
            arguments['project'] = project.name
        if operation in {'plan', 'info'}:
            if workspace:
                catalog = workspace_catalog(self.config, workspace)
                if catalog['errors']:
                    raise ConfigError(str(catalog['errors']))
                scopes = catalog['projects']
                wanted = arguments.get('projects') or ([arguments['project']] if arguments.get('project') else None)
                if wanted is None and operation == 'plan':
                    default = self.config.workspaces[workspace].get('default_project')
                    wanted = [default] if default else None
                if wanted is not None:
                    if set(wanted) - {s['project'] for s in scopes}:
                        raise ConfigError('Unknown workspace project')
                    scopes = [s for s in scopes if s['project'] in wanted]
            else:
                root, project, item = selected_project(self.config)
                scopes = [{'project': project.name, 'path': str(root), 'description': item.get('description', '')}]
            if operation == 'plan':
                plans = [selected_project(self.config, s['path'])[1].plan(self.config.data_dir, refresh=True) for s in scopes]
                return {'projects': plans, 'ready': all(p['ready'] for p in plans)}
            return {'mode': 'coordinated-live', 'workspace': workspace, 'engines': list(self.config.engines),
                    'projects': [s | {'context': selected_project(self.config, s['path'])[1].context} for s in scopes]}
        ids = arguments.get('ids')
        if ids and any(identifier not in self.issued for identifier in ids):
            raise ConfigError('Unknown evidence ID in this session')
        result = self.client.request('engine', operation=operation, **arguments)
        self.issued.update(e['id'] for e in result.get('excerpts', []))
        return result

    def status(self):
        result = self.client.request('status')
        if self.workspace:
            result['projects'] = [p for p in result['projects'] if self.workspace in p['workspaces']]
        else:
            root, _, _ = selected_project(self.config)
            result['projects'] = [p for p in result['projects'] if p['path'] == str(root)]
        return result

    def close(self):
        self.client.close()
