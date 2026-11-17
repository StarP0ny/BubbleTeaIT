"""CLI operations use the same coordinator RPC as the public MCP facade."""
import json
from pathlib import Path
import shutil
import subprocess
import time

from .config import ConfigError
from .engine_client import EngineClient
from .service_cli import output


def doctor(config):
    checks = []
    for name, settings in config.engines.items():
        for key in ('python_executable', 'executable', 'language_server'):
            if key in settings:
                checks.append({'name': f'{name}.{key}', 'ok': shutil.which(settings[key]) is not None})
        if name == 'cocoindex':
            checks.append({'name': 'cocoindex.model_path', 'ok': Path(settings['model_path']).is_dir()})
            if shutil.which(settings['python_executable']):
                code = ('import json,importlib.metadata; import torch; print(json.dumps({"version":'
                        'importlib.metadata.version("cocoindex-code"),"cuda":torch.cuda.is_available()}))')
                probe = subprocess.run([settings['python_executable'], '-c', code], capture_output=True, text=True, timeout=30)
                try:
                    info = json.loads(probe.stdout) if probe.returncode == 0 else {}
                except ValueError:
                    info = {}
                checks.append({'name': 'cocoindex-code.version', 'ok': info.get('version') == '0.2.42', 'version': info.get('version')})
                if settings['device'] == 'cuda':
                    checks.append({'name': 'cocoindex.cuda_runtime', 'ok': info.get('cuda', False)})
    ready = all(c['ok'] for c in checks)
    return {'ok': ready, 'ready': ready, 'mode': 'coordinated-live', 'checks': checks,
            'models_loaded': False}


def run(args, config):
    if args.command == 'doctor':
        value = doctor(config)
        output(value, args.json)
        return 0 if value['ready'] else 2
    client = EngineClient(config, getattr(args, 'workspace', None))
    try:
        if args.command == 'search':
            if args.channel != 'vector':
                raise ConfigError('Native mode uses semantic search; use rg for exact text or a pinned legacy composition')
            value = client.call('search', query=args.query, projects=args.projects, limit=args.limit,
                max_chars=args.max_chars, roles=args.role, source_ids=args.source_id, paths=args.path,
                freshness=args.freshness or 'available')
        elif args.command == 'symbols':
            value = client.call('symbols', project=args.project_name, symbol_operation=args.operation, path=args.path,
                source_id=args.source_id, name=args.name, limit=args.limit, max_chars=args.max_chars)
        elif args.command == 'read':
            request = {'path': args.path, 'start': args.start_line}
            if args.end_line is not None:
                request['end'] = args.end_line
            if args.sha256:
                request['sha256'] = args.sha256
            value = client.call('read', project=args.project_name, requests=[request], max_chars=args.max_chars)
        elif args.command == 'sync':
            deadline = None if args.timeout_ms is None else time.monotonic() + args.timeout_ms / 1000
            while True:
                remaining = 60000 if deadline is None else max(0, min(60000, int((deadline-time.monotonic())*1000)))
                value = client.call('sync', projects=args.projects, freshness='wait', timeout_ms=remaining)
                if value['state'] == 'ready' or any(p['state'] == 'error' for p in value.get('projects', [])):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
        elif args.command in {'plan', 'project-info'}:
            value = client.call('plan' if args.command == 'plan' else 'info')
        elif args.command == 'list-files':
            value = client.call('files', paths=args.path, roles=args.role, source_ids=args.source_id)
            value['total'] = len(value['files'])
            value['files'] = value['files'][args.offset:args.offset+args.limit]
        elif args.command == 'list-sources':
            files = client.call('files')['files']
            sources = {}
            for file in files:
                item = sources.setdefault(file['source_id'], {'source_id': file['source_id'], 'role': file['role'], 'files': 0})
                item['files'] += 1
            value = {'sources': list(sources.values())}
        else:
            raise ConfigError('Unsupported native CLI operation')
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0 if value.get('state', 'ready') == 'ready' and value.get('ready', True) else 2
    finally:
        client.close()
