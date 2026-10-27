"""Optional standalone Serena probe on exact BTI snapshots; no model calls.

The C/C++ compile context is deliberately unknown. No live compile database
or source checkout is copied, and results are navigation candidates only.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from .qwen_runtime import atomic_json
from .registry import _relative, _safe_file
from .pinned import PinnedSearchService


ALLOWED_TOOLS = frozenset({'get_symbols_overview', 'find_symbol', 'find_referencing_symbols'})


class SerenaWorkspace:
    """Private materialization with an explicit unit/file-to-navigation map."""

    def __init__(self, config, composition, output, *, prefixes=None):
        self.service = PinnedSearchService(config.data_dir, composition)
        self.composition = self.service.pinned
        self.output = Path(output).resolve()
        data = config.data_dir.resolve()
        if self.output == data or not self.output.is_relative_to(data) or self.output.exists():
            raise ValueError('Serena output must be a fresh child of BTI data_dir')
        self.project, self.home = self.output / 'project', self.output / 'home'
        units = [self.service.registry.load_unit(r['unit_id']) for r in self.composition['units']]
        units_by_id = {unit['unit_id']: unit for unit in units}
        prefixes = {} if prefixes is None else dict(prefixes)
        if set(prefixes) - {u['unit_id'] for u in units}:
            raise ValueError('Serena prefixes contain a foreign unit')
        projects = [u for u in units if u['kind'] == 'project']
        app_root = projects[0]['provenance'].get('source_root') if len(projects) == 1 else None
        self.files, seen = {}, set()
        for unit in units:
            uid = unit['unit_id']
            if uid in prefixes:
                prefix = prefixes[uid]
            elif len(projects) == 1 and uid == projects[0]['unit_id']:
                prefix = ''
            elif unit['kind'] == 'documents':
                prefix = 'documents/' + unit['source_id']
            else:
                origin = unit['provenance'].get('source_root')
                if not app_root or not origin or not Path(origin).is_relative_to(Path(app_root)):
                    raise ValueError('An explicit mount prefix is required for this Serena unit')
                prefix = Path(origin).relative_to(Path(app_root)).as_posix()
            if not isinstance(prefix, str):
                raise ValueError('Serena mount prefix must be a relative path')
            if prefix:
                _relative(prefix)
            for entry in unit['files']:
                relative = '/'.join(part for part in (prefix, entry['path']) if part)
                _relative(relative)
                if any(part in {'.git', '.serena', '.clangd', 'compile_commands.json', 'compile_flags.txt'}
                       for part in relative.split('/')):
                    raise ValueError('Navigation configuration cannot come from source snapshots')
                if relative.casefold() in seen:
                    raise ValueError('Serena mounts overlap; source files cannot be overwritten')
                seen.add(relative.casefold())
                self.files[relative] = {'unit_id': uid, 'file_path': entry['path'], 'source_sha256': entry['sha256']}
        self.project.mkdir(parents=True)
        self.home.mkdir()
        for relative, record in self.files.items():
            unit = units_by_id[record['unit_id']]
            _entry, content, _digest = self.service._snapshot_file(unit, record['file_path'])
            destination = self.project / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        atomic_json(self.output / 'mapping.json', {'composition_hash': self.composition['composition_hash'],
            'compile_context': 'unknown', 'files': self.files})
        self.verify()

    def verify(self):
        self.service.load_composition(self.composition['composition_id'])
        for relative, record in self.files.items():
            path = _safe_file(self.project, relative)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['source_sha256']:
                raise ValueError('Private Serena source changed from its pinned snapshot')

    def navigation_path(self, unit_id, file_path):
        _relative(file_path)
        found = [path for path, record in self.files.items()
                 if record['unit_id'] == unit_id and record['file_path'] == file_path]
        if len(found) != 1:
            raise ValueError('Serena request is outside the pinned file whitelist')
        return found[0]

    def source(self, navigation_path, location=None, body=None):
        if not isinstance(navigation_path, str):
            raise ValueError('Serena result has no mapped file path')
        relative = navigation_path.replace('\\', '/')
        _relative(relative)
        record = self.files.get(relative)
        if record is None:
            raise ValueError('Serena returned a file outside the pinned whitelist')
        unit = self.service.registry.load_unit(record['unit_id'])
        _entry, raw, _digest = self.service._snapshot_file(unit, record['file_path'])
        if hashlib.sha256(raw).hexdigest() != record['source_sha256']:
            raise ValueError('Serena mapping no longer matches its source SHA')
        result = dict(record)
        if location is None:
            return result | {'location_status': 'not_provided'}
        if not isinstance(location, dict):
            raise ValueError('Serena body location must be an object')
        start, end = location.get('start_line'), location.get('end_line')
        lines = raw.decode('utf-8').splitlines(keepends=True) or ['']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(lines):
            raise ValueError('Serena returned invalid zero-based source coordinates')
        content = ''.join(lines[start:end + 1])
        if len(content) > 16000:
            raise ValueError('Serena source body exceeds the bounded response size')
        if body is not None and (not isinstance(body, str) or not body.strip()
                or body.replace('\r\n', '\n').strip() not in content.replace('\r\n', '\n').strip()):
            raise ValueError('Serena symbol body does not match original snapshot lines')
        return result | {'start_line': start + 1, 'end_line': end + 1, 'content': content,
                         'location_status': 'verified_snapshot_lines'}

    def map_result(self, tool, value, requested_path):
        if tool == 'get_symbols_overview':
            if not isinstance(value, dict):
                raise ValueError('Unexpected Serena overview format')
            symbols = []
            for kind, names in value.items():
                if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                    raise ValueError('Unsupported Serena overview; refine to a bounded find_symbol call')
                symbols.extend({'name_path': name, 'kind': kind} for name in names)
            return {'source': self.source(requested_path), 'symbols': symbols}
        groups = [(None, value)] if tool == 'find_symbol' else []
        if tool == 'find_referencing_symbols':
            if not isinstance(value, dict):
                raise ValueError('Unexpected Serena references format')
            for path in value:
                self.source(path)
            groups = [(path, entries) for path, kinds in value.items() for entries in
                      (kinds.values() if isinstance(kinds, dict) else [kinds])]
        symbols = []
        for inherited_path, entries in groups:
            if not isinstance(entries, list):
                raise ValueError('Unexpected Serena symbol list')
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get('name_path'), str):
                    raise ValueError('Unexpected Serena symbol record')
                path = entry.get('relative_path', inherited_path)
                mapped = self.source(path, entry.get('body_location'), entry.get('body'))
                symbols.append(mapped | {'name_path': entry['name_path'],
                    'location_kind': 'enclosing_symbol' if inherited_path is not None else 'symbol'})
        return {'symbols': symbols}

    def configure(self, clangd):
        if not Path(clangd).is_file():
            raise ValueError('An existing explicit clangd executable is required')
        project = {'project_name': 'bti-pinned-' + self.composition['composition_hash'][:12],
            'language_servers': ['cpp'], 'encoding': 'utf-8', 'read_only': True,
            'activation_command': None, 'agent_interface': 'tools', 'ignore_all_files_in_gitignore': False,
            'ignored_paths': ['.serena/**'], 'default_modes': ['planning', 'no-memories'],
            'ls_additional_workspace_folders': [],
            'ls_specific_settings': {'cpp': {'ls_path': str(Path(clangd).resolve())}}}
        directory = self.project / '.serena'
        directory.mkdir(exist_ok=True)
        atomic_json(directory / 'project.yml', project)  # JSON is a YAML subset.
        atomic_json(self.home / 'serena_config.yml', {'agent_interface': 'tools', 'projects': [],
            'trusted_project_path_patterns': [str(self.project)], 'base_modes': [],
            'web_dashboard': False, 'web_dashboard_open_on_launch': False, 'gui_log_window': False})
        context = self.output / 'context.yml'
        atomic_json(context, {'description': 'Bounded read-only navigation on exact BTI snapshots; compile context unknown.',
            'prompt': 'Navigate the pinned files. Do not claim active build variants or execute changes.',
            'single_project': True, 'fixed_tools': sorted(ALLOWED_TOOLS)})
        return context


def _decoded(result):
    if getattr(result, 'is_error', False):
        raise ValueError('Serena tool failed; no navigation evidence accepted')
    value = getattr(result, 'structured_content', None)
    if isinstance(value, dict) and set(value) == {'result'}:
        value = value['result']
    if value is None:
        values = [item.text for item in result.content if getattr(item, 'type', None) == 'text']
        if len(values) != 1:
            raise ValueError('Serena must return one structured result')
        value = values[0]
    if isinstance(value, str):
        if len(value) > 32000:
            raise ValueError('Serena response exceeds its bound')
        value = json.loads(value)
    if not isinstance(value, (dict, list)):
        raise ValueError('Serena response must be structured JSON')
    if len(json.dumps(value, ensure_ascii=False)) > 32000:
        raise ValueError('Serena response exceeds its bound')
    return value


@asynccontextmanager
async def _session(parameters):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            yield session


async def probe_symbols(workspace, requests, *, executable, clangd, timeout_seconds=120,
                        session_factory=None, retry_empty_references=0):
    """One private MCP process for a bounded list of file-scoped read requests."""
    from mcp import StdioServerParameters
    if (not isinstance(requests, list) or not 1 <= len(requests) <= 8
            or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600):
        raise ValueError('Serena probe requires 1–8 requests and a bounded timeout')
    if type(retry_empty_references) is not int or not 0 <= retry_empty_references <= 3:
        raise ValueError('Empty reference retries must be an integer from 0 to 3')
    command = shutil.which(str(executable))
    if command is None:
        raise ValueError('An installed Serena executable is required; implicit downloads are disabled')
    calls = []
    for request in requests:
        if (not isinstance(request, dict) or set(request) - {'tool', 'unit_id', 'file_path', 'name'}
                or not {'tool', 'unit_id', 'file_path'} <= request.keys() or request['tool'] not in ALLOWED_TOOLS):
            raise ValueError('Serena probe only allows file-scoped read-only symbol tools')
        path = workspace.navigation_path(request['unit_id'], request['file_path'])
        arguments = {'relative_path': path, 'max_answer_chars': 16000}
        if request['tool'] != 'get_symbols_overview':
            name = request.get('name')
            if not isinstance(name, str) or not name.strip() or len(name) > 512:
                raise ValueError('A bounded symbol name is required')
            arguments['name_path_pattern' if request['tool'] == 'find_symbol' else 'name_path'] = name
        if request['tool'] == 'find_symbol':
            arguments.update(include_body=True, depth=0, max_matches=20)
        elif request['tool'] == 'get_symbols_overview':
            arguments['depth'] = 0
        calls.append((request['tool'], path, arguments))
    workspace.verify()
    context = workspace.configure(clangd)
    parameters = StdioServerParameters(command=command, args=['start-mcp-server', '--project', str(workspace.project),
        '--context', str(context), '--agent-interface', 'tools', '--mode', 'planning', '--mode', 'no-memories',
        '--enable-web-dashboard', 'false', '--open-web-dashboard', 'false', '--enable-gui-log-window', 'false'],
        cwd=str(workspace.output), env=os.environ | {'SERENA_HOME': str(workspace.home), 'PYTHONUTF8': '1',
                                                  'SERENA_USAGE_REPORTING': 'false'})
    results = []
    async with asyncio.timeout(timeout_seconds):
        async with (session_factory or _session)(parameters) as session:
            await session.initialize()
            listed = await session.list_tools()
            if {tool.name for tool in listed.tools} != ALLOWED_TOOLS:
                raise ValueError('Serena tools/list differs from the read-only allowlist')
            for tool, path, arguments in calls:
                retries = retry_empty_references if tool == 'find_referencing_symbols' else 0
                for attempt in range(retries + 1):
                    workspace.verify()
                    value = _decoded(await session.call_tool(tool, arguments))
                    workspace.verify()
                    mapped = {'tool': tool, **workspace.map_result(tool, value, path)}
                    if mapped.get('symbols') or attempt == retries:
                        break
                    await asyncio.sleep(min(1, .25 * 2 ** attempt))
                if tool == 'find_referencing_symbols':
                    mapped['query_attempts'] = attempt + 1
                if len(json.dumps(mapped, ensure_ascii=False)) > 64000:
                    raise ValueError('Mapped Serena source context exceeds its response bound')
                results.append(mapped)
    return {'status': 'completed', 'mode': 'standalone_optional_navigation', 'compile_context': 'unknown',
            'composition_hash': workspace.composition['composition_hash'], 'results': results,
            'tools': sorted(ALLOWED_TOOLS), 'source_verification': 'sha256_pinned_snapshot',
            'limitations': ['No verified compile database or active build context.',
                            'Navigation candidates require original-source reads before evidence.']}
