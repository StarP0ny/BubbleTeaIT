"""Bounded direct C/C++ navigation on private, verified Serena snapshots.

Results are navigation locators, never issued evidence IDs. Each call starts a
cold read-only Serena process; materialized source workspaces are reused.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import threading
import uuid

from .registry import Registry, RegistryError, _relative
from .qwen_runtime import atomic_json
from .serena_workspace import SerenaWorkspace, probe_symbols
from .search import SearchService, _text


class SymbolError(RegistryError):
    """Navigation cannot safely serve the requested pinned source."""


class _RequestRegistry(Registry):
    """Cache verified unit manifests only during one serialized probe."""

    manifests = None

    def load_unit(self, unit_id, *, verify_files=True):
        if verify_files and self.manifests is not None and unit_id in self.manifests:
            return copy.deepcopy(self.manifests[unit_id])
        value = super().load_unit(unit_id, verify_files=verify_files)
        if verify_files and self.manifests is not None:
            self.manifests[unit_id] = copy.deepcopy(value)
        return value


@dataclass
class _Prepared:
    workspace: SerenaWorkspace
    profile: dict
    artifacts: dict
    lock: threading.Lock = field(default_factory=threading.Lock)


_CACHE = {}
_CACHE_LOCK = threading.RLock()
_TOOLS = {'overview': 'get_symbols_overview', 'definition': 'find_symbol',
          'references': 'find_referencing_symbols'}
_CPP = {'.c', '.h', '.cc', '.hh', '.cpp', '.hpp', '.cxx', '.hxx'}
_PROFILE_VERSION = 'synthetic-header-directories-v1'


def _navigation_profile(workspace):
    """Describe every TU using only fixed flags and private snapshot paths.

    This is a parser/indexing aid, not a reconstructed production build. The
    profile deliberately never supplies target, build variant or feature definitions.
    """
    files = sorted(workspace.files)
    headers = [name for name in files if PurePosixPath(name).suffix.casefold() in {'.h', '.hh', '.hpp', '.hxx'}]
    directories = sorted({str(PurePosixPath(name).parent) for name in headers})
    names = {}
    for name in headers:
        basename = PurePosixPath(name).name.casefold()
        names[basename] = names.get(basename, 0) + 1
    commands = []
    for relative in files:
        suffix = PurePosixPath(relative).suffix
        if suffix.casefold() not in {'.c', '.cc', '.cpp', '.cxx'}:
            continue
        own_directory = str(PurePosixPath(relative).parent)
        includes = list(dict.fromkeys([own_directory, '.'] + directories))
        arguments = ['clang', '-fsyntax-only', '-ferror-limit=0',
                     '-std=gnu11' if suffix == '.c' else '-std=gnu++17']
        for directory in includes:
            arguments += ['-I', str(workspace.project / directory)]
        arguments += [str(workspace.project / relative)]
        commands.append({'directory': str(workspace.project),
                         'file': str(workspace.project / relative), 'arguments': arguments})
    if not commands:
        raise SymbolError('Synthetic Serena navigation requires at least one C/C++ translation unit')
    database = workspace.project / 'compile_commands.json'
    atomic_json(database, commands)
    database_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest = {'profile': _PROFILE_VERSION, 'composition_hash': workspace.composition['composition_hash'],
                'compile_context': 'unknown', 'target_definitions': 'not_supplied',
                'translation_units': len(commands), 'header_directories': directories,
                'ambiguous_header_names': sorted(name for name, count in names.items() if count > 1),
                'compile_db_sha256': database_hash}
    artifact = workspace.output / 'navigation-profile.json'
    atomic_json(artifact, manifest)
    manifest_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    profile = {'kind': _PROFILE_VERSION, 'compile_db_sha256': database_hash,
               'manifest_sha256': manifest_hash, 'translation_units': len(commands),
               'header_directories': len(directories),
               'ambiguous_header_names': len(manifest['ambiguous_header_names']),
               'active_build_verified': False}
    return profile, {database: database_hash, artifact: manifest_hash}


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def _expected_group(group):
    """Flatten only anticipated transport/validation failures from anyio."""
    return all(_expected_group(error) if isinstance(error, ExceptionGroup)
               else isinstance(error, (OSError, ValueError, TimeoutError))
               for error in group.exceptions)


class SymbolService:
    """No embedding backend or generative model is constructed by this service."""

    def __init__(self, config, composition_id):
        self.config = config
        self.options = getattr(config, 'symbols', None)
        if self.options is None:
            raise SymbolError('Configure symbols.executable and symbols.clangd_path to enable Serena navigation')
        seconds = self.options.timeout_seconds
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or not 0 < seconds <= 600):
            raise SymbolError('symbols.timeout_seconds must be finite and greater than zero, at most 600')
        if not isinstance(self.options.executable, str) or not self.options.executable.strip():
            raise SymbolError('symbols.executable must identify an installed Serena executable')
        self.service = SearchService(config.data_dir)
        self.composition_id = composition_id
        self.composition = self.service.load_composition(composition_id)
        self.units = {ref['unit_id']: self.service.registry.load_unit(ref['unit_id'])
                      for ref in self.composition['units']}
        projects = [unit for unit in self.units.values() if unit['kind'] == 'project']
        if len(projects) > 1:
            raise SymbolError('Serena requires one product per composition; create a focused composition '
                              'for the requested project and its exact module versions')
        if not projects and len(self.units) != 1:
            raise SymbolError('Without a project root, Serena requires a focused single-source composition')
        self.sources, self.aliases = {}, {}
        for unit in self.units.values():
            alias = 's-' + hashlib.sha256(unit['source_id'].encode()).hexdigest()[:12]
            if alias in self.sources:
                raise SymbolError('Source alias collision')
            self.sources[alias] = unit
            self.aliases[unit['source_id']] = alias
        self._prefixes = None if projects else {next(iter(self.units)): ''}
        self._cache_key = (str(self.service.data_dir), self.composition['composition_hash'], _PROFILE_VERSION)

    def _validate(self):
        current = self.service.load_composition(self.composition_id)
        if current['composition_hash'] != self.composition['composition_hash']:
            raise SymbolError('Composition changed; create a new SymbolService for its current snapshot')

    @staticmethod
    def _private_integrity(prepared):
        workspace = prepared.workspace
        # A stale externally added compile database could make clangd navigate
        # live files even while all mapped source hashes still match.
        for directory in (workspace.output, workspace.project, workspace.home):
            if (directory.resolve() != directory or directory.is_symlink()
                    or (hasattr(directory, 'is_junction') and directory.is_junction())):
                raise SymbolError('Private Serena workspace root changed or became a filesystem link')
        for artifact, expected in prepared.artifacts.items():
            if artifact.is_symlink() or not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise SymbolError('Private Serena navigation artifact changed from its trusted generated hash')
        for path in workspace.project.rglob('*'):
            if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
                raise SymbolError('Private Serena workspace contains an unexpected filesystem link')
            if path.name in {'.clangd', 'compile_commands.json', 'compile_flags.txt'} and path not in prepared.artifacts:
                raise SymbolError('Private Serena workspace contains an unverified compile configuration')
            if path.is_file() and '.serena' not in path.relative_to(workspace.project).parts:
                relative = path.relative_to(workspace.project).as_posix()
                # clangd can create its own cache, which never becomes source.
                if relative not in workspace.files and path not in prepared.artifacts and not relative.startswith('.cache/clangd/'):
                    raise SymbolError('Private Serena workspace contains an unverified extra source')
        workspace.verify()

    def _prepared(self):
        with _CACHE_LOCK:
            entry = _CACHE.get(self._cache_key)
            if entry is not None:
                return entry, True
            output = self.service.data_dir / 'symbols' / self.composition['composition_hash'] / uuid.uuid4().hex
            workspace = SerenaWorkspace(self.config, self.composition, output, prefixes=self._prefixes)
            workspace.service.registry = _RequestRegistry(self.service.data_dir)
            profile, artifacts = _navigation_profile(workspace)
            entry = _Prepared(workspace, profile, artifacts)
            if len(_CACHE) >= 16:
                # Eviction releases only in-memory handles; diagnostic files
                # and old snapshots are never deleted implicitly.
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[self._cache_key] = entry
            return entry, False

    def _invalidate(self, entry=None):
        with _CACHE_LOCK:
            if entry is None or _CACHE.get(self._cache_key) is entry:
                _CACHE.pop(self._cache_key, None)

    def _source(self, source_id, path):
        if not isinstance(source_id, str) or not source_id:
            raise SymbolError('source_id must be a logical source name or a direct s- alias')
        alias = source_id if source_id in self.sources else self.aliases.get(source_id)
        if alias is None:
            raise SymbolError('Unknown source_id in the pinned composition')
        _relative(path)
        unit = self.sources[alias]
        if not any(item['path'] == path for item in unit['files']):
            raise SymbolError('Requested file is outside the pinned source whitelist; use list_files')
        if PurePosixPath(path).suffix.casefold() not in _CPP:
            raise SymbolError('This Serena adapter supports explicit C/C++ source and header files')
        return unit

    def _locator(self, record, requested, operation):
        if not isinstance(record, dict):
            raise SymbolError('Serena returned an invalid symbol record')
        unit = self.units.get(record.get('unit_id'))
        path = record.get('file_path')
        if unit is None:
            raise SymbolError('Serena returned a source outside the pinned composition')
        if operation in {'overview', 'definition'} and (unit['unit_id'], path) != requested:
            raise SymbolError('Serena returned a definition outside the explicitly requested file')
        _, raw, sha = self.service._snapshot_file(unit, path)
        if sha != record.get('source_sha256'):
            raise SymbolError('Serena locator does not match its pinned source SHA')
        lines = _text(raw).splitlines(keepends=True) or ['']
        start, end = record.get('start_line'), record.get('end_line')
        if record.get('location_status') == 'verified_snapshot_lines':
            if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
                raise SymbolError('Serena locator has invalid source coordinates')
        else:
            if operation != 'overview':
                raise SymbolError('Serena did not provide verifiable symbol coordinates')
            start, end = 1, min(120, len(lines))
        alias = self.aliases[unit['source_id']]
        value = {'source_id': alias, 'source': unit['source_id'], 'unit_id': unit['unit_id'],
                 'path': path, 'source_sha256': sha, 'location_status': record['location_status'],
                 'total_lines': len(lines), 'read': {'source_id': alias, 'path': path,
                                                  'unit_id': unit['unit_id'],
                                                  'start': start, 'end': min(end, start + 119)}}
        if operation != 'overview':
            value['lines'] = [start, end]
            if end >= start + 120:
                value['next_read'] = {'source_id': alias, 'path': path, 'start': start + 120,
                                      'unit_id': unit['unit_id'],
                                      'end': min(end, start + 239)}
        return value

    def _response(self, probed, unit, path, operation, limit, max_chars, reused, profile):
        if (not isinstance(probed, dict) or probed.get('status') != 'completed'
                or probed.get('composition_hash') != self.composition['composition_hash']
                or probed.get('compile_context') != 'unknown'
                or not isinstance(probed.get('results'), list) or len(probed['results']) != 1):
            raise SymbolError('Serena response does not match the requested snapshot or compile context')
        mapped = probed['results'][0]
        if mapped.get('tool') != _TOOLS[operation] or not isinstance(mapped.get('symbols'), list):
            raise SymbolError('Serena returned an unexpected navigation result')
        records = mapped['symbols']
        attempts = mapped.get('query_attempts', 1)
        if type(attempts) is not int or not 1 <= attempts <= 4:
            raise SymbolError('Serena returned invalid bounded query-attempt metadata')
        configured = self.composition['configuration']
        result = {'composition_id': self.composition_id, 'composition_hash': self.composition['composition_hash'],
                  'configuration_hash': self.composition['configuration_hash'],
                  'configuration_status': 'declared' if configured else 'unknown',
                  'configuration': copy.deepcopy(configured) if configured else None,
                  'compile_context': 'unknown', 'navigation_only': True,
                  'navigation_profile': copy.deepcopy(profile), 'query_attempts': attempts,
                  'operation': operation, 'workspace_reused': reused, 'process': 'cold',
                  'completeness': 'not_established', 'total': len(records), 'returned': 0,
                  'truncated': bool(records), 'results': []}
        # Validate every returned locator, including entries beyond the output
        # limit. A foreign record must never hide behind response truncation.
        entries = []
        seen_references = set()
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get('name_path'), str):
                raise SymbolError('Serena returned an invalid symbol name')
            source = mapped.get('source') if operation == 'overview' else record
            locator = self._locator(source, (unit['unit_id'], path), operation)
            entry = locator | {'name_path': record['name_path'],
                               'location_kind': ('unlocated_symbol' if operation == 'overview' else
                                                 'enclosing_symbol' if operation == 'references' else 'symbol')}
            if 'kind' in record:
                entry['kind'] = record['kind']
            if operation == 'references':
                key = (entry['unit_id'], entry['path'], tuple(entry['lines']),
                       entry['name_path'], entry['location_kind'])
                if key in seen_references:
                    continue
                seen_references.add(key)
            entries.append(entry)
        if operation == 'references':
            # Serena may repeat an enclosing symbol for several references.
            # This counts backend records, never promises an exact call count.
            result['reported_records'] = len(records)
        result['total'] = len(entries)
        result['truncated'] = bool(entries)
        if _size(result) > max_chars:
            raise SymbolError('Budget is too small for the symbol response and declared configuration')
        for entry in entries[:limit]:
            candidate = copy.deepcopy(result)
            candidate['results'].append(entry)
            candidate['returned'] += 1
            candidate['truncated'] = candidate['returned'] < len(entries)
            if _size(candidate) > max_chars:
                break
            result = candidate
        return result

    async def query(self, operation, source_id, path, name=None, limit=20, max_chars=8000):
        if not isinstance(operation, str) or operation not in _TOOLS:
            raise SymbolError('operation must be overview, definition, or references')
        if type(limit) is not int or not 1 <= limit <= 40:
            raise SymbolError('limit must be an integer from 1 to 40')
        if type(max_chars) is not int or not 512 <= max_chars <= 16000:
            raise SymbolError('max_chars must be an integer from 512 to 16000')
        if operation != 'overview' and (not isinstance(name, str) or not name.strip() or len(name) > 512):
            raise SymbolError('definition and references require a nonempty name of at most 512 characters')
        if operation == 'overview' and name is not None:
            raise SymbolError('overview does not accept a symbol name')
        unit = self._source(source_id, path)
        prepared = None
        invalidate = True
        try:
            self._validate()
            prepared, reused = self._prepared()
            if not prepared.lock.acquire(blocking=False):
                invalidate = False
                raise SymbolError('The pinned Serena workspace is busy; retry after its current query')
            try:
                registry = prepared.workspace.service.registry
                registry.manifests = {}
                self._private_integrity(prepared)
                request = {'tool': _TOOLS[operation], 'unit_id': unit['unit_id'], 'file_path': path}
                if name is not None:
                    request['name'] = name
                probed = await probe_symbols(prepared.workspace, [request],
                    executable=self.options.executable, clangd=self.options.clangd_path,
                    timeout_seconds=self.options.timeout_seconds,
                    retry_empty_references=3 if operation == 'references' else 0)
                registry.manifests = {}
                self._private_integrity(prepared)
                self._validate()
                return self._response(probed, unit, path, operation, limit, max_chars, reused, prepared.profile)
            finally:
                prepared.workspace.service.registry.manifests = None
                prepared.lock.release()
        except asyncio.CancelledError:
            if invalidate:
                self._invalidate(prepared)
            raise
        except ExceptionGroup as exc:
            if invalidate:
                self._invalidate(prepared)
            if not _expected_group(exc):
                raise
            raise SymbolError('Serena MCP process or tool session failed; no navigation locators were accepted') from exc
        except (OSError, ValueError, TimeoutError) as exc:
            if invalidate:
                self._invalidate(prepared)
            if isinstance(exc, SymbolError):
                raise
            if isinstance(exc, TimeoutError):
                raise SymbolError('Serena navigation timed out; no navigation locators were accepted') from exc
            raise SymbolError(f'Serena navigation failed: {exc}') from exc
