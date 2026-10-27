"""Bounded navigation over a DirectService's pinned snapshot whitelist.

DirectService wraps these helpers in its request transaction. Navigation never
loads an encoder or follows paths into the working checkout.
"""
from __future__ import annotations

import copy
import difflib
import json
from pathlib import PurePosixPath
import sqlite3

from .search import SearchError, _filter_values, _path_matches, _text


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def _role(unit, path):
    # Import lazily so DirectService can delegate here without an import cycle.
    from .direct import _role as effective_role
    return effective_role(unit, path)


def _parameters(query, offset, limit, max_chars):
    if query is not None and (not isinstance(query, str) or not query.strip()):
        raise SearchError('query must be a nonempty literal string or null')
    if type(offset) is not int or offset < 0:
        raise SearchError('offset must be a nonnegative integer')
    if type(limit) is not int or not 1 <= limit <= 200:
        raise SearchError('limit must be an integer from 1 to 200')
    if type(max_chars) is not int or not 512 <= max_chars <= 64000:
        raise SearchError('max_chars must be an integer from 512 to 64000')
    return query.casefold() if query is not None else None


def _base(direct, offset, total):
    configuration = direct.composition.get('configuration')
    return {'composition_id': direct.composition_id,
            'composition_hash': direct.composition['composition_hash'],
            'configuration_status': 'declared' if configuration else 'unknown',
            'configuration': copy.deepcopy(configuration) if configuration else None,
            'entries': [], 'offset': offset, 'total': total,
            'next_offset': None, 'truncated': False}


def _page(direct, rows, render, offset, limit, max_chars):
    if offset > len(rows):
        raise SearchError('offset exceeds the filtered result count')
    result = _base(direct, offset, len(rows))
    if _size(result) > max_chars:
        raise SearchError('max_chars is too small for the declared configuration; increase it')
    position = offset
    for row in rows[offset:offset + limit]:
        entry = render(row)
        candidate = {**result, 'entries': result['entries'] + [entry]}
        next_position = position + 1
        candidate['next_offset'] = next_position if next_position < len(rows) else None
        candidate['truncated'] = next_position < len(rows)
        # A budget stop must point at the unreturned row, not skip it. Reserve
        # the cursor before checking so a full last row cannot hide pagination.
        if _size(candidate) > max_chars:
            if not result['entries']:
                raise SearchError('max_chars is too small for the next entry; increase it')
            break
        result = candidate
        position = next_position
    result['next_offset'] = position if position < len(rows) else None
    result['truncated'] = position < len(rows)
    return result


def list_sources(direct, query=None, roles=None, offset=0, limit=20, max_chars=6000):
    """List logical sources; query is a literal substring of source name/role."""
    folded = _parameters(query, offset, limit, max_chars)
    roles, _ = direct._filters(roles, None)
    rows = []
    for unit in sorted(direct.units.values(), key=lambda item: item['source_id']):
        effective = sorted({_role(unit, entry['path']) for entry in unit['files']})
        if roles is not None and not set(roles) & ({unit['role']} | set(effective)):
            continue
        if folded is not None and folded not in (unit['source_id'] + ' ' + unit['role']).casefold():
            continue
        rows.append({'source_id': direct.aliases[unit['source_id']], 'source': unit['source_id'],
                     'unit_id': unit['unit_id'], 'role': unit['role'], 'roles': effective,
                     'file_count': len(unit['files'])})
    return _page(direct, rows, lambda row: row, offset, limit, max_chars)


def list_files(direct, query=None, source_ids=None, paths=None, roles=None,
               offset=0, limit=40, max_chars=8000):
    """List whitelisted files with a ready-to-use, at most 120-line read.

    Query is a case-insensitive literal substring of the POSIX path. ``paths``
    uses the same case-sensitive SQLite GLOB semantics as direct retrieval.
    Roles accept either the source's declared role or the effective file role.
    Only entries attempted for this page have their source bytes read/verified.
    """
    folded = _parameters(query, offset, limit, max_chars)
    roles, source_ids = direct._filters(roles, source_ids)
    paths = _filter_values(paths, 'paths')
    rows = []
    with sqlite3.connect(':memory:') as connection:
        for unit in sorted(direct.units.values(), key=lambda item: item['source_id']):
            if source_ids is not None and unit['source_id'] not in source_ids:
                continue
            for entry in sorted(unit['files'], key=lambda item: item['path']):
                path, role = entry['path'], _role(unit, entry['path'])
                if roles is not None and role not in roles and unit['role'] not in roles:
                    continue
                if folded is not None and folded not in path.casefold():
                    continue
                if not _path_matches(connection, path, paths):
                    continue
                rows.append((unit, path, role))

    def render(row):
        unit, path, role = row
        _, content, _ = direct.service._snapshot_file(unit, path)
        try:
            lines = _text(content).splitlines() or ['']
            readable, total_lines = True, len(lines)
        except SearchError:
            readable, total_lines = False, None
        alias = direct.aliases[unit['source_id']]
        return {'source_id': alias, 'source': unit['source_id'], 'unit_id': unit['unit_id'],
                'path': path, 'role': role, 'text_readable': readable, 'total_lines': total_lines,
                'read': {'source_id': alias, 'path': path, 'start': 1, 'end': min(120, total_lines)}
                        if readable else None}

    return _page(direct, rows, render, offset, limit, max_chars)


def read_error_hint(direct, source_id, path, start=None, end=None, *, max_chars=1600):
    """Return bounded navigation advice without silently executing a new read.

    Call after request validation has failed. Unknown files suggest only real
    paths in this exact source; known files expose valid windows. The caller
    still raises its original error and preserves batch atomicity.
    """
    if type(max_chars) is not int or not 128 <= max_chars <= 64000:
        raise SearchError('Diagnostic max_chars must be an integer from 128 to 64000')
    if not isinstance(source_id, str):
        return ''
    alias = source_id if source_id in direct.sources else direct.aliases.get(source_id)
    if alias is None or not isinstance(path, str) or not path:
        return ''
    unit = direct.sources[alias]
    available = [entry['path'] for entry in unit['files']]
    if path not in available:
        target = PurePosixPath(path).name.casefold()
        ranked = sorted(available, key=lambda name: (
            -difflib.SequenceMatcher(None, target, PurePosixPath(name).name.casefold()).ratio(),
            -difflib.SequenceMatcher(None, path.casefold(), name.casefold()).ratio(), name))
        advice = {'source_id': alias, 'available_paths': [], 'hint': 'Use list_files to discover snapshot paths'}
        for name in ranked[:4]:
            candidate = {**advice, 'available_paths': advice['available_paths'] + [name]}
            if _size(candidate) > max_chars:
                break
            advice = candidate
    else:
        _, content, _ = direct.service._snapshot_file(unit, path)
        try:
            total = len(_text(content).splitlines()) or 1
        except SearchError:
            return 'This file is not UTF-8 text; no source-line read is available'
        advice = {'source_id': alias, 'path': path, 'total_lines': total, 'read_windows': []}
        if type(start) is int and type(end) is int and 1 <= start <= end:
            position, last = min(start, total), min(end, total)
        else:
            position, last = 1, min(total, 120)
        while position <= last and len(advice['read_windows']) < 3:
            window = {'source_id': alias, 'path': path, 'start': position, 'end': min(last, position + 119)}
            candidate = {**advice, 'read_windows': advice['read_windows'] + [window]}
            if _size(candidate) > max_chars:
                break
            advice = candidate
            position = window['end'] + 1
        if position <= last:
            advice['next_start'] = position
    if _size(advice) > max_chars:
        return 'Use list_files to discover valid snapshot paths and read windows'
    return json.dumps(advice, ensure_ascii=False, separators=(',', ':'))
