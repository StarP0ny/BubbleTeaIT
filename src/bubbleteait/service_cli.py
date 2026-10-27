"""Human CLI controls for the shared coordinator."""
from dataclasses import replace
import json
from pathlib import Path
import time

from .config import ConfigError
from .service import ServiceClient, config_payload, select_profile


def add_commands(commands):
    service = commands.add_parser('service', help='Управление сервисом BTI и координатором')
    actions = service.add_subparsers(dest='service_action', required=True)
    for name in ('start', 'stop', 'status'):
        item = actions.add_parser(name)
        item.add_argument('--json', action='store_true')
        if name == 'start':
            item.add_argument('--config', dest='service_config')
            item.add_argument('--embedding-profile')
            item.add_argument('--debounce-ms', type=int)
            item.add_argument('--poll-interval-seconds', type=int)
            item.add_argument('--reconcile-interval-seconds', type=int)
    config = actions.add_parser('config').add_subparsers(dest='config_action', required=True)
    config.add_parser('show').add_argument('--json', action='store_true')
    watch = commands.add_parser('watch', help='Постоянное наблюдение за проектом из bti.toml')
    watch.add_argument('-p', '--path')
    modes = watch.add_mutually_exclusive_group()
    modes.add_argument('--list', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--follow', action='store_true')
    watch.add_argument('--json', action='store_true')
    unwatch = commands.add_parser('unwatch')
    unwatch.add_argument('-p', '--path')
    unwatch.add_argument('--json', action='store_true')
    workspace = commands.add_parser('workspace')
    actions = workspace.add_subparsers(dest='workspace_action', required=True)
    actions.add_parser('list').add_argument('--json', action='store_true')
    status = actions.add_parser('status')
    status.add_argument('name')
    status.add_argument('--json', action='store_true')


def output(value, as_json):
    if as_json:
        print(json.dumps(value, default=str, ensure_ascii=False, indent=2))
        return
    if 'excerpts' in value:
        print('Состояние:', value.get('state', 'ready'))
        for excerpt in value['excerpts']:
            print(f'[{excerpt["id"]}] {excerpt.get("project", "")} {excerpt["path"]}:{excerpt["lines"][0]}-{excerpt["lines"][1]}')
            print(excerpt['text'])
        if value.get('truncated') or value.get('omitted_projects'):
            print('Выдача ограничена; omitted_count:', value.get('omitted_count', 0),
                  'omitted_projects:', value.get('omitted_projects', []))
    elif 'projects' in value:
        print('Сервис:', value.get('state', 'running'))
        for name, path in value.get('workspaces', {}).items():
            print(f'Workspace {name}: {path}')
        print('Workspace\tProject\tPath\tState')
        for p in value['projects']:
            print(f'{",".join(p.get("workspaces", [])) or "—"}\t{p.get("project") or "—"}\t{p.get("path", "")}\t{p["state"]}')
            if p.get('error'):
                print('  Ошибка:', p['error'])
        if value.get('config_error'):
            print('Ошибка конфига:', value['config_error'])
        for name, error in value.get('workspace_errors', {}).items():
            print(f'Workspace {name}: {error}')
        if value.get('saved_registrations'):
            print('Сохранённые регистрации:', value['saved_registrations'])
    else:
        def fields(mapping, indent=''):
            for key, item in mapping.items():
                if isinstance(item, dict):
                    print(f'{indent}{key}:')
                    fields(item, indent + '  ')
                else:
                    print(f'{indent}{key}: {item}')
        fields(value)


def run(args, config):
    config = select_profile(config, getattr(args, 'embedding_profile', None))
    overrides = {key: getattr(args, key) for key in ('debounce_ms', 'poll_interval_seconds', 'reconcile_interval_seconds')
                 if getattr(args, key, None) is not None}
    for key, value in overrides.items():
        if not (0 if key == 'debounce_ms' else 1) <= value <= (60000 if key == 'debounce_ms' else 86400):
            raise ConfigError(f'Invalid {key}')
    if overrides:
        config = replace(config, sync=config.sync | overrides, sync_overrides=overrides)
    client = ServiceClient(config)
    if args.command == 'service':
        action = args.service_action
        if action == 'start':
            client.ensure()
            current = client.request('config')['settings']
            expected = config_payload(config)
            for key in ('data_dir', 'models', 'qwen', 'cocoindex', 'embedding_profile', 'embedding_override',
                        'embedding_profiles', 'sync', 'sync_overrides', 'workspaces', 'service_file'):
                if current.get(key) != expected.get(key):
                    raise ConfigError('Service already uses different settings; stop it before starting with overrides')
            value = client.request('status')
        elif action == 'config':
            value = client.request('config') if client.probe() else {
                'state': 'stopped', 'settings': config_payload(config), 'sources': list(map(str, config.loaded_files))}
        else:
            value = client.request('stop' if action == 'stop' else 'status')
    elif args.command == 'workspace':
        value = client.request('status')
        value.setdefault('workspaces', {name: str(v['path']) for name, v in config.workspaces.items()})
        if args.workspace_action == 'status':
            if args.name not in value['workspaces']:
                raise ConfigError(f'Unknown workspace: {args.name}')
            value['projects'] = [p for p in value['projects'] if args.name in p['workspaces']]
    elif args.command == 'unwatch':
        value = client.request('unwatch', path=str(Path(args.path or args.project_dir).resolve()))
    elif args.list or args.status:
        value = client.request('status')
        if args.status:
            path = str(Path(args.path or args.project_dir).resolve())
            value['projects'] = [p for p in value['projects'] if p['path'] == path]
    else:
        path = Path(args.path or args.project_dir).resolve()
        if args.path and args.project_dir != '.' and path != Path(args.project_dir).resolve():
            raise ConfigError('-p and --project-dir select different directories')
        # Validate before starting a daemon, avoiding side effects on bad input.
        from .project import load_project
        load_project(path / 'bti.toml')
        value = client.request('register', path=str(path))
        if args.follow:
            previous = None
            try:
                while True:
                    current = client.request('status')
                    current['projects'] = [p for p in current['projects'] if p['path'] == str(path)]
                    signature = json.dumps(current, sort_keys=True)
                    if signature != previous:
                        output(current, args.json)
                        previous = signature
                    time.sleep(1)
            except KeyboardInterrupt:
                return 0
    output(value, args.json)
    return 0
