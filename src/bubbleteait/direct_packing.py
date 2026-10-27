"""Budgeted, extractive packing of already verified source windows.

Admission of compact windows precedes expansion. No model or index is used
here; every published character is a slice of the verified source bytes.
"""
from __future__ import annotations

import hashlib
import json

from .search import SearchError


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def _centred(value, width):
    first, last = value['start_line'], value['end_line']
    width = min(width, last - first + 1)
    seed_first, seed_last = value['_seed']
    centre = (seed_first + seed_last) // 2
    start = max(first, min(centre - (width - 1) // 2, last - width + 1))
    return start, start + width - 1


def _groups(reads, spans):
    """Coalesce overlapping issued windows, keeping every distinct anchor."""
    groups = []
    for index, span in sorted(spans.items(), key=lambda item: (
            reads[item[0]]['unit_id'], reads[item[0]]['file_path'], item[1])):
        value = reads[index]
        group = {'value': value, 'first': span[0], 'last': span[1], 'indices': [index]}
        merged = []
        for previous in groups:
            same = (previous['value']['unit_id'], previous['value']['file_path']) == (value['unit_id'], value['file_path'])
            mergeable = (value.get('_allow_fit') and previous['value'].get('_allow_fit')
                         and max(previous['last'], group['last']) - min(previous['first'], group['first']) < 120)
            if same and mergeable and previous['first'] <= group['last'] + 1 and group['first'] <= previous['last'] + 1:
                if previous['value']['source_sha256'] != value['source_sha256']:
                    raise SearchError('Source changed while packing overlapping windows')
                group['first'] = min(group['first'], previous['first'])
                group['last'] = max(group['last'], previous['last'])
                group['indices'].extend(previous['indices'])
            else:
                merged.append(previous)
        merged.append(group)
        groups = merged
    return sorted(groups, key=lambda group: min(group['indices']))


def pack_context(direct, reads, max_chars, *, include_catalog=True, candidate_files=None):
    if type(max_chars) is not int or not 512 <= max_chars <= 64000:
        raise SearchError('max_chars must be an integer from 512 to 64000')
    if type(include_catalog) is not bool:
        raise SearchError('include_catalog must be a boolean')
    catalog = []
    if include_catalog:
        for logical, alias in sorted(direct.aliases.items()):
            entry = {'source_id': alias, 'source': logical, 'role': direct.sources[alias]['role']}
            if size(catalog + [entry]) <= max_chars // 4:
                catalog.append(entry)
    selected_sources = {value['source_id'] for value in reads}
    selected_roles = {direct.effective_role(value['unit_id'], value['file_path']) for value in reads}

    def render(spans):
        result = {'composition_id': direct.composition_id,
                  'composition_hash': direct.composition['composition_hash'],
                  'sources': {}, 'excerpts': [], 'truncated': len(spans) != len(reads),
                  'catalog': list(catalog),
                  'catalog_complete': include_catalog and len(catalog) == len(direct.sources),
                  'omitted_count': len(reads) - len(spans)}
        pending = {}
        roles, sources = set(), set()
        for group in _groups(reads, spans):
            value, start, end = group['value'], group['first'], group['last']
            alias = direct.aliases[value['source_id']]
            identity = [value['unit_id'], value['file_path'], start, end, value['source_sha256']]
            identifier = 'e-' + hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()[:20]
            truncated = any(reads[index].get('_context_truncated', False)
                            or start > reads[index]['start_line'] or end < reads[index]['end_line']
                            for index in group['indices'])
            excerpt = {'id': identifier, 'source_id': alias, 'source': value['source_id'],
                       'path': value['file_path'], 'lines': [start, end],
                       'text': ''.join(value['_source_lines'][start - 1:end]),
                       'total_lines': value['total_lines'],
                       'role': direct.effective_role(value['unit_id'], value['file_path']),
                       'source_changed': value['source_changed'], 'working_state': value['working_state'],
                       'context_truncated': truncated}
            if truncated:
                excerpt['context_lines'] = [min(reads[i]['start_line'] for i in group['indices']),
                                            max(reads[i]['end_line'] for i in group['indices'])]
            if end < value['total_lines']:
                excerpt['next_read'] = {'source_id': alias, 'path': value['file_path'],
                                        'start': end + 1, 'end': min(value['total_lines'], end + 120)}
            result['sources'][alias] = {'source_id': value['source_id'], 'role': direct.units[value['unit_id']]['role']}
            result['excerpts'].append(excerpt)
            result['truncated'] |= truncated
            roles.add(excerpt['role'])
            sources.add(value['source_id'])
            pending[identifier] = {'id': identifier, 'unit_id': value['unit_id'],
                                   'file_path': value['file_path'], 'start_line': start,
                                   'end_line': end, 'source_sha256': value['source_sha256']}
        # Counts describe this selection, never completeness of the selection.
        # Detailed facets are bounded and can be obtained through navigation.
        if max_chars >= 3000:
            result['coverage'] = {'selected_hits': len(reads), 'covered_hits': len(spans),
                                  'selected_sources': len(selected_sources), 'returned_sources': len(sources),
                                  'selected_roles': sorted(selected_roles), 'returned_roles': sorted(roles)}
            if candidate_files is not None:
                result['coverage']['candidate_files'] = candidate_files
            if result['omitted_count']:
                advice = []
                for index, value in enumerate(reads):
                    if index in spans:
                        continue
                    first, last = (value['_seed'] if value.get('_allow_fit') else
                                   (value['start_line'], value['end_line']))
                    request = {'source_id': direct.aliases[value['source_id']],
                               'path': value['file_path'], 'start': first, 'end': min(last, first + 119)}
                    if size(advice + [request]) <= max_chars // 8:
                        advice.append(request)
                result['omitted_reads'] = advice
                result['omitted_details_complete'] = len(advice) == result['omitted_count']
        return result, pending

    # Even tiny budgets retain a valid envelope; navigation is available if
    # the complete discovery catalog cannot fit alongside it.
    while catalog and size(render({})[0]) > max_chars:
        catalog.pop()
    if size(render({})[0]) > max_chars:
        raise SearchError('Budget is too small for the direct response envelope')

    spans = {}
    # Pass 1: give all candidates a compact initial window before growing any
    # parent. Explicit coordinate requests are indivisible and remain exact.
    for index, value in enumerate(reads):
        if not value.get('_allow_fit'):
            options = [(value['start_line'], value['end_line'])]
        else:
            target = 7
            options = [_centred(value, target)]
            # An admission window should not reserve a large semantic chunk
            # before later hits get any space. Three source lines are the
            # minimum fallback for a multi-line parent with competitors.
            options.append(_centred(value, 3))
            # A narrow fallback is useful for an intrinsically large line or
            # a single semantic hit. Never spend a tail on a bare one-liner
            # from a multi-line parent while other candidates compete.
            if len(reads) == 1:
                options = [_centred(value, width) for width in (7, 5, 3, 1)]
        for span in dict.fromkeys(options):
            proposed = {**spans, index: span}
            if size(render(proposed)[0]) <= max_chars:
                spans = proposed
                break

    # Pass 2: grow in small rounds, so rank does not give the first result
    # ownership of the remaining budget. Merging is based on actual windows,
    # not expanded parents that might cover a different, still unseen anchor.
    changed = True
    while changed:
        changed = False
        for index in list(spans):
            value = reads[index]
            if not value.get('_allow_fit'):
                continue
            first, last = spans[index]
            if (first, last) == (value['start_line'], value['end_line']):
                continue
            alternatives = [(max(value['start_line'], first - 1), min(value['end_line'], last + 1)),
                            (max(value['start_line'], first - 1), last),
                            (first, min(value['end_line'], last + 1))]
            for span in dict.fromkeys(alternatives):
                if span == (first, last):
                    continue
                proposed = {**spans, index: span}
                if size(render(proposed)[0]) <= max_chars:
                    spans = proposed
                    changed = True
                    break
    result, pending = render(spans)
    if size(result) > max_chars:
        raise SearchError('Direct response exceeds its serialization budget')
    return result, pending
