"""Read-only Serena sessions with selected-file views and real GCC build commands."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import re

from .config import ConfigError
from .engine_process import EngineProcess
from .registry import _atomic_json, _digest


TOOLS = {'overview': 'get_symbols_overview', 'definition': 'find_symbol', 'references': 'find_referencing_symbols'}


def command_arguments(command):
    if os.name != 'nt':
        return shlex.split(command)
    # CMake emits Windows quoting; posix shlex would eat backslashes in toolchain paths.
    argc = ctypes.c_int()
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argv = shell.CommandLineToArgvW(command, ctypes.byref(argc))
    if not argv:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [argv[i] for i in range(argc.value)]
    finally:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.LocalFree.argtypes = (ctypes.c_void_p,)
        kernel.LocalFree(argv)


class SymbolSession:
    def __init__(self, config, root, project, settings, plan, engine, signature, process_factory=EngineProcess):
        if not plan['ready']:
            raise ConfigError('Source preflight failed before symbol navigation')
        database = settings.get('compile_commands')
        if database is None:
            raise ConfigError('Configure projects.<name>.compile_commands from the selected real build')
        database = Path(database).resolve()
        raw_database = database.read_bytes()
        commands = json.loads(raw_database)
        if not isinstance(commands, list) or not commands:
            raise ConfigError('compile_commands.json must contain real translation-unit commands')
        self.signature = signature
        self.engine, self.root, self.project = engine, root, project
        options = config.engines['serena']
        self.server = options['server']
        self.database_sha = hashlib.sha256(raw_database).hexdigest()
        identity = _digest(json.loads(json.dumps({'root': str(root), 'signature': signature, 'options': options,
                            'database': self.database_sha, 'flags': settings}, default=str)))
        self.directory = config.data_dir / 'engines/serena' / identity
        self.view = self.directory / 'view'
        self.view.mkdir(parents=True, exist_ok=True)
        selected = engine.records(root, plan)
        self.records = {r['path'] if Path(r['absolute_path']).is_relative_to(root) else '__external/' + key: r
                        for key, r in selected.items()}
        self.by_absolute = {r['absolute_path']: key for key, r in self.records.items()}
        for key, record in self.records.items():
            data = Path(record['absolute_path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ConfigError('Source changed while preparing Serena')
            target = self.view / key
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != data:
                target.write_bytes(data)
        roots = sorted({(r['source_root'], str(self.view / Path(r['source_root']).relative_to(root))
                         if Path(r['source_root']).is_relative_to(root)
                         else str(self.view / '__external' / r['source_id'])) for r in self.records.values()},
                       key=lambda pair: len(pair[0]), reverse=True)
        def mapped(path, base):
            absolute = (base / path).resolve()
            for original, destination in roots:
                original = Path(original)
                if absolute.is_relative_to(original):
                    candidate = Path(destination) / absolute.relative_to(original)
                    if candidate.exists():
                        return str(candidate)
                    # Generated/unselected build headers remain at their real build paths.
                    return str(absolute)
            return str(absolute)
        rewritten, covered = [], set()
        for command in commands:
            if not isinstance(command, dict):
                raise ConfigError('Invalid compilation database entry')
            base = (database.parent / command.get('directory', '.')).resolve()
            original = str((base / command['file']).resolve())
            if original not in self.by_absolute:
                continue
            arguments = command.get('arguments') or command_arguments(command['command'])
            if not isinstance(arguments, list) or not arguments or any(not isinstance(a, str) for a in arguments):
                raise ConfigError('Invalid compiler arguments')
            output = [arguments[0]]
            skip = False
            path_next = False
            for argument in arguments[1:]:
                if skip:
                    skip = False
                    continue
                if argument == '-o':
                    skip = True
                    continue
                if path_next:
                    output.append(mapped(argument, base))
                    path_next = False
                    continue
                if any(argument == flag or flag.endswith('*') and argument.startswith(flag[:-1])
                       for flag in settings.get('remove_flags', [])):
                    continue
                if argument.startswith('@'):
                    raise ConfigError('Expand compiler response files in compile_commands.json before navigation')
                if argument in ('-I', '-isystem', '-iquote', '-include', '-imacros'):
                    output.append(argument)
                    path_next = True
                elif argument.startswith('-I') and len(argument) > 2:
                    output.append('-I' + mapped(argument[2:], base))
                elif not argument.startswith('-') and str((base / argument).resolve()) == original:
                    output.append(mapped(argument, base))
                else:
                    # Absolute paths may also occur in header-selection string macros.
                    for original_root, destination in roots:
                        replacement = destination.replace('\\', '/')
                        argument = argument.replace(original_root.replace('\\', '/'), replacement)
                        argument = argument.replace(original_root.replace('/', '\\'), replacement)
                    output.append(argument)
            output += settings.get('add_flags', [])
            rewritten.append({'directory': str(self.view), 'file': str(self.view / self.by_absolute[original]),
                              'arguments': output})
            covered.add(original)
        if not rewritten:
            raise ConfigError('Compilation database has no translation units in the selected sources')
        _atomic_json(self.view / 'compile_commands.json', rewritten)
        cpp = 'cpp' if self.server == 'clangd' else 'cpp_ccls'
        launcher = Path(sys.executable).with_name('bti-language-server.exe' if os.name == 'nt' else 'bti-language-server')
        if not launcher.is_file():
            raise ConfigError('Install the current BTI package to create the bti-language-server launcher')
        executable = shutil.which(options['executable'])
        language_server = shutil.which(options['language_server'])
        if not executable or not language_server:
            raise ConfigError('Configured Serena or language-server executable is unavailable')
        ls_args = []
        if self.server == 'clangd' and options['query_drivers']:
            ls_args.append('--query-driver=' + ','.join(options['query_drivers']))
        _atomic_json(self.view / '.serena/project.yml', {'project_name': 'bti-' + identity[:12],
            'language_servers': [cpp], 'encoding': 'utf-8', 'read_only': True, 'activation_command': None,
            'agent_interface': 'tools', 'ignore_all_files_in_gitignore': False, 'ignored_paths': ['.serena/**'],
            'default_modes': ['planning', 'no-memories'], 'ls_additional_workspace_folders': [],
            'ls_specific_settings': {cpp: {'ls_path': str(launcher), 'compile_commands_dir': str(self.view / '.serena')}}})
        home = self.directory / 'home'
        _atomic_json(home / 'serena_config.yml', {'agent_interface': 'tools', 'projects': [], 'base_modes': [],
            'trusted_project_path_patterns': [str(self.view)], 'web_dashboard': False,
            'web_dashboard_open_on_launch': False, 'gui_log_window': False})
        context = self.directory / 'context.yml'
        _atomic_json(context, {'description': 'Read-only navigation over selected sources and real compiler commands.',
            'prompt': 'Navigate selected source versions. Do not edit files or claim a successful firmware build.',
            'single_project': True, 'fixed_tools': sorted(TOOLS.values())})
        bridge = self.directory / 'bridge.json'
        _atomic_json(bridge, {'executable': executable, 'view': str(self.view), 'directory': str(self.directory),
            'context': str(context), 'home': str(home), 'tools': sorted(TOOLS.values()),
            'language_server': language_server, 'ls_args': ls_args})
        self.process = process_factory(options['python_executable'], Path(__file__).with_name('serena_engine_worker.py'),
                                       bridge, options['timeout_seconds'])
        self.covered = covered
        self.profile = {'compile_context': 'compilation_database', 'compile_db_sha256': self.database_sha,
                        'translation_units': len(covered), 'build_verified': False,
                        'unconfigured_translation_units': sum(r['absolute_path'] not in covered for r in self.records.values()
                            if Path(r['path']).suffix.lower() in {'.c', '.cc', '.cpp', '.cxx'}),
                        'remove_flags': settings.get('remove_flags', []), 'add_flags': settings.get('add_flags', [])}

    def query(self, request):
        operation = request.get('symbol_operation')
        if operation not in TOOLS:
            raise ConfigError('Unsupported symbol operation')
        paths = [k for k, r in self.records.items() if request.get('path') in (r['path'], r['source_path'], r['absolute_path'])
                 and (not request.get('source_id') or r['source_id'] == request['source_id'])]
        if len(paths) != 1:
            raise ConfigError('Symbol path is outside selected sources or ambiguous')
        arguments = {'relative_path': paths[0], 'max_answer_chars': 32000}
        if operation != 'overview':
            name = request.get('name')
            if not isinstance(name, str) or not name.strip() or len(name) > 512:
                raise ConfigError('A symbol name of 1..512 characters is required')
            arguments['name_path_pattern' if operation == 'definition' else 'name_path'] = name
        if operation == 'definition':
            arguments.update(include_body=True, depth=0, max_matches=40)
        elif operation == 'overview':
            arguments['depth'] = 0
        reused = self.process.process is not None and self.process.process.poll() is None
        value = self.process.request('symbols', tool=TOOLS[operation], arguments=arguments)
        entries = []
        if operation == 'overview':
            if not isinstance(value, dict):
                raise ConfigError('Unexpected Serena overview')
            for kind, names in value.items():
                if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
                    raise ConfigError('Unexpected Serena overview symbol')
                entries.extend({'name_path': n, 'kind': kind, 'relative_path': paths[0]} for n in names)
        elif operation == 'references':
            if not isinstance(value, dict):
                raise ConfigError('Unexpected Serena references')
            for path, kinds in value.items():
                if path.replace('\\', '/') not in self.records:
                    raise ConfigError('Serena returned a file outside the selected source whitelist')
                groups = kinds.items() if isinstance(kinds, dict) else [(None, kinds)]
                for kind, symbols in groups:
                    if not isinstance(symbols, list) or any(not isinstance(symbol, dict) for symbol in symbols):
                        raise ConfigError('Unexpected Serena reference symbols')
                    entries.extend({**symbol, 'relative_path': path, '_reference_group': kind} for symbol in symbols)
        else:
            if not isinstance(value, list):
                raise ConfigError('Unexpected Serena definitions')
            entries = value
        mapped = []
        for entry in entries:
            key = entry.get('relative_path', paths[0]).replace('\\', '/')
            if key not in self.records:
                raise ConfigError('Serena returned a file outside the selected source whitelist')
            record = self.records[key]
            source = (self.view / key).read_bytes()
            if hashlib.sha256(source).hexdigest() != record['sha256']:
                raise ConfigError('Serena source view changed')
            result = {'project': self.project.name, 'path': record['path'], 'source_id': record['source_id'],
                      'sha256': record['sha256'], 'name_path': entry['name_path'],
                      'location_kind': 'enclosing_symbol' if operation == 'references' else 'symbol'}
            location = entry.get('body_location')
            if location is not None:
                start, end = location.get('start_line'), location.get('end_line')
                lines = source.decode('utf-8-sig').splitlines(keepends=True)
                if operation == 'references' and entry.get('_reference_group') == 'File':
                    # Serena's file container may include virtual trailing lines.
                    # Return our own verified whole-file context, never claim this is a symbol/call-site span.
                    start, end = 0, len(lines)-1
                    result['location_kind'] = 'file_context'
                    marked = re.findall(r'^\s*>\s*(\d+):(.*)$', entry.get('content_around_reference', ''), re.MULTILINE)
                    verified = [int(number) for number, text in marked
                                if int(number) < len(lines) and lines[int(number)].rstrip('\r\n') == text.rstrip('\r')]
                    if marked and len(verified) == len(marked):
                        start, end = max(0, min(verified)-3), min(len(lines)-1, max(verified)+3)
                        result['location_kind'] = 'reference_context'
                if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(lines):
                    raise ConfigError('Invalid Serena source coordinates')
                body = ''.join(lines[start:end+1])
                if entry.get('body') is not None and entry['body'].replace('\r\n', '\n').strip() not in body.replace('\r\n', '\n').strip():
                    raise ConfigError('Serena body does not match original source lines')
                result.update(lines=[start+1, end+1], text=body,
                    next_read={'path': record['path'], 'source_id': record['source_id'], 'sha256': record['sha256'],
                               'start': start+1, 'end': end+1})
            mapped.append(result)
        limit = request.get('limit', 20)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ConfigError('Invalid symbol limit')
        result = {'state': 'ready', 'project': self.project.name, 'server': self.server, 'operation': operation,
            'session_reused': reused, **self.profile, 'completeness': 'not_established',
            'results': mapped[:limit], 'total': len(mapped), 'truncated': len(mapped) > limit}
        budget = request.get('max_chars', 8000)
        if type(budget) is not int or not 512 <= budget <= 64000:
            raise ConfigError('Invalid symbol max_chars')
        while len(json.dumps(result, ensure_ascii=False)) > budget and result['results']:
            with_text = next((r for r in reversed(result['results']) if 'text' in r), None)
            if with_text is not None:
                del with_text['text']
            else:
                result['results'].pop()
            result['truncated'] = True
        if len(json.dumps(result, ensure_ascii=False)) > budget:
            raise ConfigError('max_chars too small for symbol metadata')
        return result

    def close(self):
        self.process.close()
