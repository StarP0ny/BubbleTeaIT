"""Machine-owned engine/workspace settings; no imports of optional engines."""
import math
import re
import sys
import copy
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, _boolean, _path, _string, _strings, _table


def read_engine_settings(raw, filename, env):
    engines = _table(raw.get('engines', {}), 'engines', {'cocoindex', 'serena'})
    if engines and 'cocoindex' not in engines:
        raise ConfigError('engines.cocoindex is required for coordinated live search')
    for name, item in engines.items():
        common = {'python_executable', 'timeout_seconds'}
        fields = ({'model_path', 'model_revision', 'device', 'indexing_params', 'query_params'}
                  if name == 'cocoindex' else {'executable', 'language_server', 'server', 'query_drivers'})
        _table(item, 'engines.' + name, common | fields)
        for key in ('python_executable', 'executable', 'language_server'):
            if key in item:
                value = _string(item[key], f'engines.{name}.{key}')
                item[key] = str(_path(value, filename.parent, env, key)) if '/' in value or '\\' in value else value
        item.setdefault('python_executable', sys.executable)
        timeout = item.setdefault('timeout_seconds', 600 if name == 'cocoindex' else 120)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ConfigError(f'engines.{name}.timeout_seconds must be in (0, 3600]')
        if name == 'cocoindex':
            item['model_path'] = _path(item.get('model_path'), filename.parent, env, 'engines.cocoindex.model_path')
            _string(item.get('model_revision'), 'engines.cocoindex.model_revision')
            if item.setdefault('device', 'cpu') not in {'cpu', 'cuda'}:
                raise ConfigError('engines.cocoindex.device must be cpu or cuda')
            for key in ('indexing_params', 'query_params'):
                params = item.setdefault(key, {})
                _table(params, key, {'prompt_name', 'normalize_embeddings'})
                if 'prompt_name' in params:
                    _string(params['prompt_name'], key + '.prompt_name')
                if 'normalize_embeddings' in params:
                    _boolean(params['normalize_embeddings'], key + '.normalize_embeddings')
        else:
            for key in ('executable', 'language_server'):
                _string(item.get(key), f'engines.serena.{key}')
            if item.setdefault('server', 'clangd') not in {'clangd', 'ccls'}:
                raise ConfigError('engines.serena.server must be clangd or ccls')
            item['query_drivers'] = [str(_path(p, filename.parent, env, 'query_drivers'))
                                     for p in _strings(item.get('query_drivers', []), 'query_drivers')]
    projects = raw.get('projects', {})
    if not isinstance(projects, dict):
        raise ConfigError('projects must be a table')
    for name, item in projects.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', name):
            raise ConfigError('Invalid project name')
        _table(item, 'projects.' + name, {'path', 'project_file', 'include', 'exclude', 'attach',
                                        'respect_gitignore', 'description', 'compile_commands', 'remove_flags', 'add_flags'})
        item['path'] = _path(item.get('path'), filename.parent, env, 'projects.' + name + '.path')
        for key in ('project_file', 'compile_commands'):
            if key in item:
                item[key] = _path(item[key], item['path'], env, key)
        for key in ('include', 'exclude', 'attach', 'remove_flags', 'add_flags'):
            if key in item:
                item[key] = list(_strings(item[key], key))
        if 'respect_gitignore' in item:
            _boolean(item['respect_gitignore'], 'respect_gitignore')
        if 'description' in item:
            _string(item['description'], 'description')
    for name, item in raw.get('workspaces', {}).items():
        if 'projects' in item:
            item['projects'] = list(_strings(item['projects'], 'workspaces.' + name + '.projects'))
            if len(set(item['projects'])) != len(item['projects']):
                raise ConfigError('Workspace projects must be unique')
            if set(item['projects']) - set(projects):
                raise ConfigError('Workspace refers to unknown projects')
        if 'default_project' in item and item['default_project'] not in item.get('projects', []):
            raise ConfigError('Workspace default_project must belong to its explicit projects')


def apply_engine_flags(config, args):
    bindings = {'embedding_model': ('cocoindex', 'model_path'), 'embedding_revision': ('cocoindex', 'model_revision'),
                'embedding_device': ('cocoindex', 'device'), 'embedding_python': ('cocoindex', 'python_executable'),
                'symbol_server': ('serena', 'server'), 'language_server': ('serena', 'language_server')}
    engines = copy.deepcopy(config.engines)
    overrides = copy.deepcopy(config.engine_overrides)
    for flag, (engine, key) in bindings.items():
        value = getattr(args, flag, None)
        if value is None:
            continue
        if engine not in engines:
            raise ConfigError('Configure engines.' + engine + ' before applying flags')
        if key == 'model_path':
            if not getattr(args, 'embedding_revision', None):
                raise ConfigError('--embedding-model requires --embedding-revision')
            value = Path(value).resolve()
        engines[engine][key] = value
        overrides.setdefault(engine, {})[key] = str(value) if isinstance(value, Path) else value
    return replace(config, engines=engines, engine_overrides=overrides)
