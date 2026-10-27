"""Compact, source-verified hybrid retrieval without an analysis model.

Context expansion is conservative: bounded Markdown sections and top-level C
functions, otherwise a nearby line window. It is not a language-server parser.
The response character budget counts the complete compact JSON representation.
Query mentions do not create hidden filters. For exact focus, pass
source_ids=['catalog-source-id'], paths=['src/file.c'] or roles=['docs'].
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import logging
import math
from pathlib import PurePosixPath
import re
import sqlite3
import threading
import time

from .application import ConfiguredBackend, make_backend
from .registry import Registry, RegistryError
from .search import SearchError, SearchService, _filter_values, _path_matches, _text


_FUNCTION_WORDS = frozenset('''a an and are as at be been being by can could did do does for from
    had has have how i if in into is it its may might of on or our should so than that the their
    them then there these they this those to was we were what when where which while who why will
    with would you your а без был была были было бы в во вот все для до его ее ещё если есть же за
    и из или как ко когда кто ли мы на над надо не но о об он она они оно от по под при про с со
    так также там то того только тут ты у уже что чтобы это этой этот эту я'''.split())

_ROLE_TERMS = {
    'code': {'implementation', 'definition', 'source', 'code', 'реализац', 'определени', 'код'},
    'docs': {'requirement', 'specification', 'spec', 'требован', 'спецификац', 'тз'},
    'tests': {'test', 'testing', 'regression', 'тест', 'регресси'},
    'reports': {'report', 'reported', 'reporting', 'отчет', 'отчёт'},
}


def _query_terms(query):
    tokens = re.findall(r'\w+', query)
    quoted = {term.casefold() for term in re.findall(r'`(\w+)`', query)}
    terms, identifiers = set(), set()
    for token in tokens:
        term = token.casefold()
        if term in _FUNCTION_WORDS or len(term) < 2:
            continue
        terms.add(term)
        if not (len(token) == 2 and token.isupper()) and (term in quoted or '_' in token or re.search(r'[a-z][A-Z]', token)
                or len(token) > 2 and token.isupper() and any(c.isalpha() for c in token)
                or any(c.isdigit() for c in token) and any(c.isalpha() for c in token)):
            identifiers.add(term)
    return terms, identifiers


def _term_count(term, folded):
    # Two/three-letter labels must not match inside ordinary prose (AB/this,
    # CD/limits). Underscores delimit labels in BUILD_BB-style identifiers.
    if len(term) <= 3:
        return len(re.findall(r'(?<![^\W_])' + re.escape(term) + r'(?![^\W_])', folded))
    return folded.count(term)


def _matches(terms, text):
    folded = text.casefold()
    return {term for term in terms if _term_count(term, folded)}


def _query_labels(query):
    return {token.casefold() for token in re.findall(r'\b[A-Z][A-Z0-9]{1,3}\b', query)
            if token.casefold() not in _FUNCTION_WORDS}


def _query_intents(query):
    words = re.findall(r'[^\W_]+', query.casefold())
    def matches(term):
        if any('\u0400' <= char <= '\u04ff' for char in term):
            return any(word.startswith(term) for word in words)
        return any(word == term or word == term + 's' for word in words)
    return {role for role, terms in _ROLE_TERMS.items() if any(matches(term) for term in terms)}


def _metadata_labels(labels, source, path):
    value = (source + ' ' + path).casefold()
    return {label for label in labels if re.search(
        r'(?<![^\W_])' + re.escape(label) + r'(?![^\W_])', value)}


def _request(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        # Serialize only this service's requests. Validation is scoped to this
        # call, never cached across requests. Roll back unissued ledger entries
        # when the outer exit check detects a changed source/index manifest.
        with self.service.request_lock:
            outer = self.service._depth == 0
            started, result, succeeded = time.perf_counter(), None, False
            if outer:
                self._timings = {}
                self.service.validation_seconds = 0.0
                self.service.validation_calls = 0
            with self._lock:
                previous = copy.deepcopy(self._ledger)
            try:
                with self.service.transaction():
                    result = method(self, *args, **kwargs)
                succeeded = True
                return result
            except BaseException:
                with self._lock:
                    self._ledger = previous
                raise
            finally:
                if outer:
                    self.last_profile = {'operation': method.__name__, 'succeeded': succeeded,
                        'seconds': time.perf_counter() - started, 'stages': copy.deepcopy(self._timings),
                        'validation_seconds': self.service.validation_seconds,
                        'validation_calls': self.service.validation_calls,
                        'result_chars': _size(result) if result is not None else 0}
                    logging.getLogger(__name__).debug('direct_profile %s', json.dumps(self.last_profile))
    return wrapped


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def _timed(stage):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self._measure(stage):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


def _pack_arguments(max_chars, include_catalog):
    if type(max_chars) is not int or not 512 <= max_chars <= 64000:
        raise SearchError('max_chars must be an integer from 512 to 64000')
    if type(include_catalog) is not bool:
        raise SearchError('include_catalog must be a boolean')


def _role(manifest, path):
    role = manifest['role'].casefold()
    parts = PurePosixPath(path.casefold()).parts
    if 'report' in role:
        return 'reports'
    if any(part in {'test', 'tests', 'testing'} for part in parts[:-1]) or parts[-1].startswith(('test_', 'test.')):
        return 'tests'
    if PurePosixPath(path).suffix.casefold() in {'.md', '.rst', '.txt', '.adoc'}:
        return 'docs'
    return 'code'


def _context(lines, path, start, end):
    """Return bounded inclusive source coordinates; never synthesize text."""
    if end - start >= 120:
        return start, start + 119
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix in {'.md', '.markdown'}:
        headings = [(i + 1, len(match.group(1))) for i, line in enumerate(lines)
                    if (match := re.match(r'^\s{0,3}(#{1,6})\s+', line))]
        # Lexical hits include neighbouring lines and can straddle a heading.
        # Select the section around the hit's centre instead of widening to a file.
        centre = (start + end) // 2
        previous = [(line, level) for line, level in headings if line <= centre]
        if previous:
            first, level = previous[-1]
            last = next((line - 1 for line, depth in headings if line > first and depth <= level), len(lines))
            if centre <= last and last - first < 120:
                return first, last
    elif suffix in {'.c', '.h', '.cc', '.cpp', '.hpp'}:
        # Mask braces in comments/strings before finding top-level function bodies.
        text = ''.join(lines)
        masked = re.sub(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                        lambda m: re.sub(r'[^\r\n]', ' ', m.group()), text)
        depth, opening, boundary = 0, None, 1
        for number, line in enumerate(masked.splitlines(), 1):
            for character in line:
                if character == '{':
                    if depth == 0:
                        header = ''.join(lines[max(0, boundary - 1):number])
                        opening = max(boundary, number - 15) if re.search(r'\)\s*\{', header) else None
                        if opening is not None:
                            while opening < number and not lines[opening - 1].strip():
                                opening += 1
                    depth += 1
                elif character == '}':
                    depth = max(0, depth - 1)
                    if depth == 0:
                        if opening is not None and opening <= start <= end <= number and number - opening < 120:
                            return opening, number
                        boundary, opening = number + 1, None
                elif character == ';' and depth == 0:
                    boundary = number + 1
    first, last = max(1, start - 8), min(len(lines), end + 12)
    if last - first >= 120:
        last = first + 119
    return first, last


class _LazyBackend:
    """Source navigation/reads stay usable without configured embeddings."""
    def __init__(self, config):
        self.config, self.backend, self.registry = config, None, None

    def search(self, *args, **kwargs):
        if self.backend is None:
            self.backend = make_backend(self.config)
            if isinstance(self.backend, ConfiguredBackend):
                self.backend.registry = self.registry
        return self.backend.search(*args, **kwargs)


class _RequestRegistry(Registry):
    """Reuse verified manifests only inside one serialized direct request.

    Entry and exit validation start with empty caches and check every byte.
    Selected source reads still verify their bytes independently.
    """
    manifests = None

    def load_unit(self, unit_id, *, verify_files=True):
        if verify_files and self.manifests is not None and unit_id in self.manifests:
            return copy.deepcopy(self.manifests[unit_id])
        manifest = super().load_unit(unit_id, verify_files=verify_files)
        if verify_files and self.manifests is not None:
            self.manifests[unit_id] = copy.deepcopy(manifest)
        return manifest


class _PinnedSearch(SearchService):
    def __init__(self, data_dir, backend, composition_id):
        super().__init__(data_dir, backend)
        self.registry = _RequestRegistry(self.data_dir)
        if isinstance(backend, (ConfiguredBackend, _LazyBackend)):
            backend.registry = self.registry
        self.request_lock, self._depth = threading.RLock(), 0
        self.pinned = super().load_composition(composition_id)

    def _validate(self):
        started = time.perf_counter()
        try:
            for ref in self.pinned['units']:
                unit = self.registry.load_unit(ref['unit_id'])
                if (unit['source_id'] != ref['source_id'] or unit['recipe_hash'] != ref['recipe_hash']
                        or self.registry.index_status(ref['unit_id'])['status'] != 'ready'):
                    raise SearchError('Pinned direct composition is no longer ready')
        finally:
            self.validation_seconds = getattr(self, 'validation_seconds', 0.0) + time.perf_counter() - started
            self.validation_calls = getattr(self, 'validation_calls', 0) + 1

    @contextmanager
    def transaction(self):
        with self.request_lock:
            outer = self._depth == 0
            try:
                if outer:
                    self.registry.manifests = {}
                    self._validate()
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                    if outer:
                        self.registry.manifests = {}
                        self._validate()
            finally:
                if outer:
                    self.registry.manifests = None

    def load_composition(self, composition_id):
        if composition_id != self.pinned['composition_id']:
            raise SearchError('Direct retrieval is pinned to one composition')
        with self.request_lock:
            if not self._depth:
                self._validate()
        return copy.deepcopy(self.pinned)


class DirectService:
    def __init__(self, config, composition_id):
        from pathlib import Path
        self.project_root = Path(getattr(config, 'project_dir', config.data_dir.parent)).resolve()
        self.service = _PinnedSearch(config.data_dir, _LazyBackend(config), composition_id)
        self.composition = copy.deepcopy(self.service.pinned)
        project_root = self.composition.get('configuration', {}).get('bti', {}).get('project_root')
        if project_root:
            self.project_root = Path(project_root).resolve()
        self.composition_id = composition_id
        self.units = {ref['unit_id']: self.service.registry.load_unit(ref['unit_id']) for ref in self.composition['units']}
        self.sources, self.aliases = {}, {}
        for unit in self.units.values():
            alias = 's-' + hashlib.sha256(unit['source_id'].encode()).hexdigest()[:12]
            if alias in self.sources:
                raise SearchError('Direct source alias collision')
            self.sources[alias] = unit
            self.aliases[unit['source_id']] = alias
        self._ledger, self._lock = {}, threading.RLock()
        self._timings, self.last_profile = {}, None

    @contextmanager
    def _measure(self, stage):
        started = time.perf_counter()
        try:
            yield
        finally:
            value = self._timings.setdefault(stage, {'seconds': 0.0, 'calls': 0})
            value['seconds'] += time.perf_counter() - started
            value['calls'] += 1

    def _filters(self, roles, source_ids):
        roles = _filter_values(roles, 'roles')
        source_ids = _filter_values(source_ids, 'source_ids')
        if source_ids is not None:
            selected = []
            for source in source_ids:
                alias = source if source in self.sources else self.aliases.get(source)
                if alias is None:
                    raise SearchError('Unknown direct source_id')
                selected.append(self.sources[alias]['source_id'])
            source_ids = selected
        allowed_roles = {unit['role'] for unit in self.units.values()} | {'code', 'tests', 'docs', 'reports'}
        if roles is not None and not set(roles) <= allowed_roles:
            raise SearchError('Unknown direct role')
        return roles, source_ids

    @_timed('source_read_expansion')
    def _read(self, unit_id, path, start, end, expand, anchor=None):
        if unit_id not in self.units:
            raise SearchError('Source is outside the pinned direct composition')
        unit = self.units[unit_id]
        _, content, snapshot_hash = self.service._snapshot_file(unit, path)
        lines = _text(content).splitlines(keepends=True) or ['']
        if (type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines)):
            raise SearchError('Direct coordinates exceed source boundaries')
        seed_start, seed_end = anchor or (start, end)
        if not 1 <= seed_start <= seed_end <= len(lines):
            raise SearchError('Direct hit anchor exceeds source boundaries')
        if expand:
            start, end = _context(lines, path, start, end)
            start, end = min(start, seed_start), max(end, seed_end)
            if end - start >= 120:
                start = max(1, (seed_start + seed_end) // 2 - 59)
                end = min(len(lines), start + 119)
        elif end - start >= 120:
            raise SearchError('Direct reads are limited to 120 inclusive lines')
        value = self.service.read_source(unit_id, path, start_line=start, end_line=end,
                                         mode='snapshot', composition_id=self.composition_id)
        if value['source_sha256'] != snapshot_hash or value['content'] != ''.join(lines[start - 1:end]):
            raise SearchError('Direct source changed during context expansion')
        # These internal fields never become evidence. The original hit remains
        # available when expanded context would consume the entire JSON budget.
        value['_source_lines'] = lines
        value['_seed'] = (seed_start, seed_end)
        value['_allow_fit'] = expand
        value['_context_truncated'] = not (start <= seed_start <= seed_end <= end)
        return value

    @_timed('lexical')
    def _lexical(self, query, limit, roles, source_ids, paths):
        """Rank matching lines globally with file rarity and identifier weights.

        This is a small lexical ranker, not a BM25 index. It uses exactly the
        same verified snapshots and SQL GLOB filtering as the public service.
        """
        terms, identifiers = _query_terms(query)
        paths = _filter_values(paths, 'paths')
        documents = []
        frequency = dict.fromkeys(terms, 0)
        connection = sqlite3.connect(':memory:')
        try:
            for unit in self.units.values():
                if (roles is not None and unit['role'] not in roles
                        or source_ids is not None and unit['source_id'] not in source_ids):
                    continue
                for entry in unit['files']:
                    if not _path_matches(connection, entry['path'], paths):
                        continue
                    _, content, _ = self.service._snapshot_file(unit, entry['path'])
                    try:
                        lines = _text(content).splitlines(keepends=True)
                    except SearchError:
                        continue
                    folded = ''.join(lines).casefold()
                    present = _matches(terms, folded)
                    metadata = _matches(terms, unit['source_id'] + ' ' + entry['path'])
                    for term in present:
                        frequency[term] += 1
                    documents.append((unit, entry['path'], lines, present, metadata))
        finally:
            connection.close()
        weights = {term: (4 if term in identifiers else 1) *
                   (1 + math.log((len(documents) + 1) / (frequency[term] + .5))) for term in terms}
        candidates = []
        for unit, path, lines, present, metadata in documents:
            for number, line in enumerate(lines, 1):
                folded = line.casefold()
                matches = _matches(present, folded)
                if not matches:
                    continue
                score = sum(weights[term] * (1 + math.log(min(3, _term_count(term, folded)))) for term in matches)
                # Metadata disambiguates real content hits, never creates a hit
                # on every line just because a product appears in its source ID.
                score *= 1 + min(.5, .15 * len(metadata))
                score /= 1 + max(0, len(line) - 160) / 640
                candidates.append((score, unit, path, number, line))
        candidates.sort(key=lambda row: (-row[0], row[1]['unit_id'], row[2], row[3]))
        return [self.service._decorate(unit, {'file_path': path, 'start_line': number, 'end_line': number,
            'content': line, 'score': score}, rank) for rank, (score, unit, path, number, line)
                in enumerate(candidates[:limit], 1)]

    @_timed('report_routing')
    def _routed_reports(self, query, labels, limit, roles, source_ids, paths):
        """Retrieve small report groups named explicitly by metadata labels.

        The same embedding query is reused; selected filenames are exact
        whitelisted paths. No product-name expansion or topic rules are used.
        """
        if not labels:
            return []
        patterns = _filter_values(paths, 'paths')
        rows = []
        connection = sqlite3.connect(':memory:')
        try:
            for unit in self.units.values():
                if ('report' not in unit['role'].casefold()
                        or roles is not None and unit['role'] not in roles
                        or source_ids is not None and unit['source_id'] not in source_ids):
                    continue
                allowed = {entry['path'] for entry in unit['files'] if
                    _metadata_labels(labels, unit['source_id'], entry['path'])
                    and _path_matches(connection, entry['path'], patterns)}
                if not allowed:
                    continue
                # Escape glob metacharacters to select exact filenames.
                literals = [''.join({'[': '[[]', ']': '[]]', '?': '[?]', '*': '[*]'}.get(c, c)
                                    for c in path) for path in sorted(allowed)]
                found = self.service.backend.search(unit['unit_id'], query, limit=limit, paths=literals)
                if not isinstance(found, list) or len(found) > limit:
                    raise SearchError('Routed report backend returned an invalid result list')
                for rank, row in enumerate(found, 1):
                    value = self.service._decorate(unit, row, rank)
                    if value['file_path'] not in allowed:
                        raise SearchError('Routed report backend violated its exact path filter')
                    rows.append(value)
        finally:
            connection.close()
        return rows

    def effective_role(self, unit_id, path):
        return _role(self.units[unit_id], path)

    def _pack(self, reads, max_chars, *, include_catalog=True, candidate_files=None):
        from .direct_packing import pack_context
        with self._measure('packing'):
            result, pending = pack_context(self, reads, max_chars,
                                           include_catalog=include_catalog, candidate_files=candidate_files)
        self.service.load_composition(self.composition_id)
        with self._lock:
            for identifier, evidence in pending.items():
                if identifier in self._ledger and self._ledger[identifier] != evidence:
                    raise SearchError('Direct evidence identity collision')
            self._ledger.update(pending)
        return result

    @_request
    def list_sources(self, query=None, roles=None, offset=0, limit=20, max_chars=6000):
        from .direct_catalog import list_sources
        return list_sources(self, query=query, roles=roles, offset=offset, limit=limit, max_chars=max_chars)

    @_request
    def list_files(self, query=None, source_ids=None, paths=None, roles=None, offset=0, limit=40, max_chars=8000):
        from .direct_catalog import list_files
        return list_files(self, query=query, source_ids=source_ids, paths=paths, roles=roles,
                          offset=offset, limit=limit, max_chars=max_chars)

    @_request
    def search(self, query, limit=6, max_chars=12000, roles=None, source_ids=None, paths=None, include_catalog=True):
        """Retrieve with soft role intent; explicit filters define the scope.

        A source/path typed inside query remains a retrieval hint. Use the
        catalog source_id with source_ids=[...], paths=['src/file.c'] for exact
        focus; mention of another project never silently changes these filters.
        """
        if type(limit) is not int or not 1 <= limit <= 20:
            raise SearchError('Direct limit must be an integer from 1 to 20')
        _pack_arguments(max_chars, include_catalog)
        roles, source_ids = self._filters(roles, source_ids)
        registry_roles = [role for role in roles or [] if role not in {'code', 'tests', 'docs', 'reports'}]
        # Inferred roles may coexist within a unit, so apply those after source validation.
        pre_roles = registry_roles if roles is not None and len(registry_roles) == len(roles) else None
        candidate_limit = min(100, max(24, limit * 8))
        with self._measure('vector_pipeline'):
            found = self.service.search(self.composition_id, query, channel='vector', limit=candidate_limit,
                                        roles=pre_roles, source_ids=source_ids, paths=paths)
        labels = _query_labels(query)
        intents = _query_intents(query)
        routed = (self._routed_reports(query, labels, min(16, candidate_limit), pre_roles, source_ids, paths)
                  if not intents or 'reports' in intents else [])
        groups = [routed + found['results'], self._lexical(query, candidate_limit, pre_roles, source_ids, paths)]
        if roles is not None:
            registry_roles = set(roles) - {'code', 'tests', 'docs', 'reports'}
            groups = [[row for row in rows if
                       _role(self.units[row['unit_id']], row['file_path']) in roles
                       or row['role'] in registry_roles] for rows in groups]
        terms, identifiers = _query_terms(query)
        report_intent = 'reports' in intents
        has_identifier_hits = any(bool(_matches(identifiers, row['content']))
                                  for rows in groups for row in rows)
        candidates = {}
        for channel, rows in enumerate(groups):
            for rank, row in enumerate(rows, 1):
                category = _role(self.units[row['unit_id']], row['file_path'])
                matches = _matches(terms, row['content'])
                identifier_match = bool(matches & identifiers)
                if has_identifier_hits and not identifier_match and not (report_intent and category == 'reports'):
                    prose = terms - identifiers
                    if not prose or len(matches & prose) < min(2, len(prose)):
                        continue
                key = row['unit_id'], row['file_path'], row['start_line'], row['end_line']
                item = candidates.setdefault(key, {'row': row, 'votes': {}})
                item['row']['_identifier_match'] = item['row'].get('_identifier_match', False) or identifier_match
                item['row']['_labels'] = _metadata_labels(labels, row['source_id'], row['file_path'])
                anchor_key = (not identifier_match, channel != 1, rank, row['end_line'] - row['start_line'])
                if '_anchor_key' not in item['row'] or anchor_key < item['row']['_anchor_key']:
                    item['row']['_anchor_key'] = anchor_key
                    item['row']['_anchor'] = (row['start_line'], row['end_line'])
                item['votes'].setdefault(channel, 1 / (60 + rank))
        # Fuse vector support into exact lexical anchors without replacing
        # several distant anchors with one broad semantic range. A channel
        # contributes only its best rank to each distinct anchor.
        lexical = [item for item in candidates.values() if 1 in item['votes']]
        merged = []
        for _, item in sorted(candidates.items()):
            row = item['row']
            overlaps = [entry for entry in lexical if entry is not item
                and entry['row']['unit_id'] == row['unit_id']
                and entry['row']['file_path'] == row['file_path']
                and row['start_line'] <= entry['row']['end_line']
                and entry['row']['start_line'] <= row['end_line']]
            if 0 in item['votes'] and 1 not in item['votes'] and overlaps:
                for entry in overlaps:
                    entry['votes'][0] = max(entry['votes'].get(0, 0), item['votes'][0])
                continue
            merged.append(item)
        selected, files, categories, label_counts = [], {}, {}, {}
        has_routed_reports = bool(routed)
        while merged and len(selected) < limit:
            def priority(item):
                row = item['row']
                file = row['unit_id'], row['file_path']
                role = _role(self.units[row['unit_id']], row['file_path'])
                score = sum(item['votes'].values()) / (1 + 2 * files.get(file, 0) + .5 * categories.get(role, 0))
                if has_identifier_hits and not row['_identifier_match']:
                    # Diversity must not promote an unrelated file merely to fill a role.
                    score *= .25
                if role in intents:
                    score *= 2
                if has_routed_reports and role == 'reports':
                    score *= 3 if row['_labels'] else .2
                if row['_labels']:
                    score /= 1 + .75 * min(label_counts.get(label, 0) for label in row['_labels'])
                return -score, file, row['start_line'], row['end_line']
            merged.sort(key=priority)
            row = merged.pop(0)['row']
            value = self._read(row['unit_id'], row['file_path'], row['start_line'], row['end_line'], True,
                               anchor=row['_anchor'])
            if any(previous['unit_id'] == value['unit_id'] and previous['file_path'] == value['file_path']
                   and previous['_seed'] == value['_seed'] for previous in selected):
                continue
            selected.append(value)
            file = row['unit_id'], row['file_path']
            role = _role(self.units[row['unit_id']], row['file_path'])
            files[file], categories[role] = files.get(file, 0) + 1, categories.get(role, 0) + 1
            for label in row['_labels']:
                label_counts[label] = label_counts.get(label, 0) + 1
        return self._pack(selected, max_chars, include_catalog=include_catalog,
                          candidate_files=len({(row['row']['unit_id'], row['row']['file_path']) for row in candidates.values()}))

    def path_request(self, request):
        """Resolve a known project/absolute path against the pinned whitelist only."""
        from pathlib import Path
        if not isinstance(request, dict) or 'path' not in request:
            return request
        if not set(request) <= {'source_id', 'path', 'start', 'end', 'unit_id'}:
            raise SearchError('Invalid path request shape')
        raw = request['path']
        if not isinstance(raw, str) or not raw or '..' in raw.replace('\\', '/').split('/'):
            raise SearchError('Path traversal or empty path is not allowed')
        if 'source_id' in request:
            _, sources = self._filters(None, [request['source_id']])
            unit = self.sources[self.aliases[sources[0]]]
            candidates = [(unit, raw)]
        else:
            path = Path(raw)
            target = path if path.is_absolute() else self.project_root / path
            candidates = []
            for unit in self.units.values():
                root = self.composition.get('configuration', {}).get('source_roots', {}).get(unit['source_id'])
                root = root or unit['provenance'].get('source_root')
                if not root:
                    continue
                root = Path(root).resolve()
                try:
                    relative = target.absolute().relative_to(root)
                except ValueError:
                    continue
                # Reject a working junction/symlink escape even though the read uses snapshot bytes.
                if not target.resolve().is_relative_to(root):
                    raise SearchError('Path escapes its declared source root')
                if any(f['path'] == relative.as_posix() for f in unit['files']):
                    candidates.append((unit, relative.as_posix()))
        if len(candidates) != 1:
            raise SearchError('ambiguous_path' if candidates else 'not_found: path is outside the pinned whitelist; check config/ignore or wait for sync')
        unit, path = candidates[0]
        _, content, _ = self.service._snapshot_file(unit, path)
        total = len(_text(content).splitlines()) or 1
        if ('start' in request) != ('end' in request):
            raise SearchError('Provide both start and end, or omit both for the first window')
        return {**request, 'source_id': unit['source_id'], 'path': path,
                'start': request.get('start', 1), 'end': request.get('end', min(total, 120))}

    @_request
    def read_many(self, requests, max_chars=16000, include_catalog=True):
        _pack_arguments(max_chars, include_catalog)
        if not isinstance(requests, list) or not 1 <= len(requests) <= 20:
            raise SearchError('read_many needs 1..20 requests')
        self.service.load_composition(self.composition_id)
        values = []
        for index, request in enumerate(requests):
            try:
                request = self.path_request(request)
                if isinstance(request, str) or isinstance(request, dict) and set(request) == {'id'}:
                    identifier = request if isinstance(request, str) else request['id']
                    evidence = self.resolve([identifier])[0]
                    values.append(self._read(evidence['unit_id'], evidence['file_path'],
                                             evidence['start_line'], evidence['end_line'], True))
                elif (isinstance(request, dict) and {'source_id', 'path', 'start', 'end'} <= set(request)
                      and set(request) <= {'source_id', 'path', 'start', 'end', 'unit_id'}):
                    _, sources = self._filters(None, [request['source_id']])
                    unit = self.sources[self.aliases[sources[0]]]
                    if 'unit_id' in request and request['unit_id'] != unit['unit_id']:
                        raise SearchError('Source version changed: the guarded unit_id is not selected in this composition; refresh navigation')
                    values.append(self._read(unit['unit_id'], request['path'], request['start'], request['end'], False))
                else:
                    raise SearchError('Invalid request shape')
            except (RegistryError, ValueError, TypeError) as exc:
                hint = ''
                if isinstance(request, dict) and set(request) == {'source_id', 'path', 'start', 'end'}:
                    from .direct_catalog import read_error_hint
                    hint = read_error_hint(self, request['source_id'], request['path'], request['start'], request['end'])
                raise SearchError(f'read_many requests[{index}]: {exc}. Expected an issued ID, '
                    '{id: issued_ID}, or {source_id: catalog_ID, path: nonempty_relative_path, '
                    'start: positive_line, end: inclusive_line}; at most 120 lines.'
                    + (f' Navigation: {hint}' if hint else '')) from exc
        return self._pack(values, max_chars, include_catalog=include_catalog)

    @_request
    def resolve(self, ids):
        if not isinstance(ids, list) or len(ids) > 80 or any(not isinstance(identifier, str) or not identifier for identifier in ids):
            raise SearchError('Evidence IDs must be an array of at most 80 nonempty strings')
        with self._lock:
            if any(identifier not in self._ledger for identifier in ids):
                raise SearchError('Unknown or forged direct evidence ID')
            evidence = {identifier: copy.deepcopy(self._ledger[identifier]) for identifier in ids}
        # A citation is published only after rechecking the originally issued
        # range against its pinned snapshot, not merely trusting an old ledger.
        for entry in evidence.values():
            current = self.service.read_source(entry['unit_id'], entry['file_path'],
                start_line=entry['start_line'], end_line=entry['end_line'], mode='snapshot',
                composition_id=self.composition_id)
            if any(current[field] != entry[field] for field in
                   ('unit_id', 'file_path', 'start_line', 'end_line', 'source_sha256')):
                raise SearchError('Direct evidence no longer matches its pinned snapshot')
        return [copy.deepcopy(evidence[identifier]) for identifier in ids]
