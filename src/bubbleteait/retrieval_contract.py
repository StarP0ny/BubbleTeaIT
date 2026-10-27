"""Deterministic source-expectation checks, separate from retrieval policy.

Candidates are raw, verified retrieval hits before fusion/gating/selection.
Selected rows are the context windows handed to the packer. Packed rows are
only the excerpts actually returned, resolved through the evidence ledger.
No stage is treated as a complete answer or an exhaustive relevance oracle.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time


class ContractError(ValueError):
    """Invalid case, receipt, or frozen-source identity."""


def digest(value):
    return hashlib.sha256(value).hexdigest()


def span_digest(lines):
    """Line-ending independent digest, without a trailing newline."""
    return digest('\n'.join(lines).encode('utf-8'))


def _hash(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))


def _reference(ref, *, substantive=True):
    if not isinstance(ref, dict):
        raise ContractError('Expected source must be an object')
    for key in ('source_id', 'file_path'):
        if not isinstance(ref.get(key), str) or not ref[key]:
            raise ContractError(f'Expected source requires {key}')
    path = PurePosixPath(ref['file_path'])
    if path.is_absolute() or '..' in path.parts or '\\' in ref['file_path'] or ':' in ref['file_path']:
        raise ContractError('Expected file_path must be a source-relative POSIX path')
    if 'unit_id' in ref and not isinstance(ref['unit_id'], str):
        raise ContractError('unit_id must be a string')
    if 'source_sha256' in ref and not _hash(ref['source_sha256']):
        raise ContractError('Invalid expected source SHA-256')
    ranges = ref.get('ranges', [])
    anchors = ref.get('anchors_all', [])
    if not isinstance(ranges, list) or not isinstance(anchors, list):
        raise ContractError('ranges and anchors_all must be arrays')
    if substantive and not ranges and not anchors:
        raise ContractError('Positive source expectations require substantive ranges or anchors')
    for span in ranges:
        if (not isinstance(span, dict) or type(span.get('start_line')) is not int
                or type(span.get('end_line')) is not int
                or not 1 <= span['start_line'] <= span['end_line']
                or not _hash(span.get('sha256'))):
            raise ContractError('Expected range requires inclusive coordinates and SHA-256')
    if any(not isinstance(value, str) or not value.strip() for value in anchors):
        raise ContractError('Anchors must be nonempty substantive strings')


def validate_case(case):
    if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id']:
        raise ContractError('Case requires a nonempty id')
    if not isinstance(case.get('composition_id'), str) or not _hash(case.get('composition_hash')):
        raise ContractError('Case requires a frozen composition id and hash')
    if case.get('purpose') not in {'focused', 'diagnostic', 'fixture'}:
        raise ContractError('Case purpose must be focused, diagnostic or fixture')
    request = case.get('request')
    if not isinstance(request, dict) or set(request) - {'query', 'limit', 'max_chars', 'filters', 'include_catalog'}:
        raise ContractError('Invalid contract search request')
    if not isinstance(request.get('query'), str) or not request['query'].strip():
        raise ContractError('Search query is required')
    if type(request.get('limit')) is not int or not 1 <= request['limit'] <= 20:
        raise ContractError('Search limit must be 1..20')
    if type(request.get('max_chars')) is not int or not 512 <= request['max_chars'] <= 64000:
        raise ContractError('Search max_chars must be 512..64000')
    if 'include_catalog' in request and type(request['include_catalog']) is not bool:
        raise ContractError('include_catalog must be boolean')
    filters = request.get('filters', {})
    if not isinstance(filters, dict) or set(filters) - {'source_ids', 'roles', 'paths'}:
        raise ContractError('Unknown search filters')
    for key, values in filters.items():
        if values is not None and (not isinstance(values, list)
                or any(not isinstance(v, str) or not v for v in values)):
            raise ContractError(f'{key} must be an array of nonempty strings or null')
    stages = case.get('stages')
    if (not isinstance(stages, list) or not stages or len(set(stages)) != len(stages)
            or not set(stages) <= {'candidate', 'packed'}):
        raise ContractError('Case stages must select candidate and/or packed')
    expected = case.get('expected_sources')
    if not isinstance(expected, dict) or set(expected) - {'state', 'all_of', 'none_of', 'unknown_reason'}:
        raise ContractError('Invalid expected_sources')
    if expected.get('state') not in {'supported', 'unknown'}:
        raise ContractError('Expected state must be supported or unknown')
    groups = expected.get('all_of', [])
    if not isinstance(groups, list) or (expected['state'] == 'supported' and not groups):
        raise ContractError('Supported cases require nonempty all_of source groups')
    if expected['state'] == 'unknown' and (groups or not expected.get('unknown_reason')):
        raise ContractError('Unknown cases need a reason and must not assert expected evidence')
    names = set()
    for group in groups:
        if (not isinstance(group, dict) or not isinstance(group.get('id'), str)
                or group['id'] in names or not isinstance(group.get('any_of'), list) or not group['any_of']):
            raise ContractError('Source groups require unique ids and nonempty any_of alternatives')
        names.add(group['id'])
        for ref in group['any_of']:
            _reference(ref)
    negatives = expected.get('none_of', [])
    if not isinstance(negatives, list):
        raise ContractError('none_of must be an array')
    for ref in negatives:
        _reference(ref, substantive=False)
    return case


def load_cases(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or not isinstance(value.get('cases'), list) or not value['cases']:
        raise ContractError('Expected nonempty version-1 case bundle')
    identifiers = set()
    for case in value['cases']:
        validate_case(case)
        if case['id'] in identifiers:
            raise ContractError('Duplicate contract case id')
        identifiers.add(case['id'])
    return value


def _rows(rows):
    if not isinstance(rows, list):
        raise ContractError('Observation stage must be an array')
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('source_id'), str)
                or not isinstance(row.get('file_path'), str) or not _hash(row.get('source_sha256'))
                or type(row.get('start_line')) is not int or type(row.get('end_line')) is not int
                or not 1 <= row['start_line'] <= row['end_line'] or not isinstance(row.get('content'), str)):
            raise ContractError('Observation requires canonical source identity, SHA, coordinates and text')
        # A truncated/partial backend line is not promoted to a full source line.
        # It remains observable and fails range digest checks, rather than being
        # silently replaced by nearby full snapshot text.
        if len(row['content'].splitlines()) != row['end_line'] - row['start_line'] + 1:
            raise ContractError('Observation text and inclusive line coordinates differ')
    return rows


def _match(ref, rows):
    same_file = [r for r in rows if r['source_id'] == ref['source_id'] and r['file_path'] == ref['file_path']]
    matching = [r for r in same_file if ('unit_id' not in ref or r.get('unit_id') == ref['unit_id'])
                and ('source_sha256' not in ref or r['source_sha256'] == ref['source_sha256'])]
    lines = {}
    for row in matching:
        for number, text in enumerate(row['content'].splitlines(), row['start_line']):
            if number in lines and lines[number] != text:
                raise ContractError('Conflicting returned source bytes at identical coordinates')
            lines[number] = text
    missing_ranges = []
    for span in ref.get('ranges', []):
        needed = range(span['start_line'], span['end_line'] + 1)
        if any(n not in lines for n in needed):
            missing_ranges.append({'start_line': span['start_line'], 'end_line': span['end_line'], 'reason': 'not_returned'})
        elif span_digest([lines[n] for n in needed]) != span['sha256']:
            missing_ranges.append({'start_line': span['start_line'], 'end_line': span['end_line'], 'reason': 'text_mismatch'})
    missing_anchors = [a for a in ref.get('anchors_all', []) if not any(a in row['content'] for row in matching)]
    return {'source_id': ref['source_id'], 'file_path': ref['file_path'], 'identity_found': bool(same_file),
            'version_found': bool(matching), 'covered': bool(matching) and not missing_ranges and not missing_anchors,
            'missing_ranges': missing_ranges, 'missing_anchors': missing_anchors}


def evaluate_case(case, observation):
    """Evaluate a preserved observation without search, filesystem reads or LLMs."""
    validate_case(case)
    if observation.get('composition_hash') != case['composition_hash']:
        raise ContractError('Observed composition hash differs from the frozen case')
    if observation.get('request') != case['request']:
        raise ContractError('Observed query/filters/budget differ from the case')
    if observation.get('error'):
        return {'case_id': case['id'], 'purpose': case['purpose'], 'status': 'error',
                'error': observation['error'], 'stages': {}}
    rows = {stage: _rows(observation.get(stage, [])) for stage in ('candidate', 'selected', 'packed')}
    expected, stage_results = case['expected_sources'], {}
    all_groups = expected.get('all_of', [])
    group_coverage = {stage: {} for stage in rows}
    for stage, values in rows.items():
        groups = []
        for group in all_groups:
            alternatives = [_match(ref, values) for ref in group['any_of']]
            covered = any(ref['covered'] for ref in alternatives)
            group_coverage[stage][group['id']] = covered
            item = {'id': group['id'], 'covered': covered, 'alternatives': alternatives}
            if not covered:
                if stage == 'candidate':
                    item['reason'] = 'missing_candidate_source' if not any(a['identity_found'] for a in alternatives) else 'missing_candidate_substance'
                elif group_coverage['selected'].get(group['id']):
                    item['reason'] = 'packing_loss'
                elif group_coverage['candidate'].get(group['id']):
                    item['reason'] = 'selection_loss'
                else:
                    item['reason'] = 'not_available_as_substantive_candidate'
            groups.append(item)
        forbidden = [m for ref in expected.get('none_of', []) if (m := _match(ref, values))['covered']]
        state = ('fail' if forbidden else 'unknown') if expected['state'] == 'unknown' else (
            'pass' if all(g['covered'] for g in groups) and not forbidden else 'fail')
        stage_results[stage] = {'status': state, 'covered_groups': sum(g['covered'] for g in groups),
                                'expected_groups': len(groups), 'groups': groups, 'forbidden': forbidden}
    response_chars = (len(json.dumps(observation['result'], ensure_ascii=False, separators=(',', ':')))
                      if 'result' in observation else None)
    budget = {'max_chars': case['request']['max_chars'], 'response_chars': response_chars,
              'within_budget': response_chars <= case['request']['max_chars'] if response_chars is not None else None}
    requested = [stage_results[stage]['status'] for stage in case['stages']]
    status = 'fail' if 'fail' in requested else ('unknown' if 'unknown' in requested else 'pass')
    if budget['within_budget'] is False:
        status = 'fail'
    return {'case_id': case['id'], 'purpose': case['purpose'], 'status': status,
            'stages': stage_results, 'asserted_stages': case['stages'],
            'budget': budget, 'unknown_reason': expected.get('unknown_reason')}


def verify_expected_sources(case, direct):
    """Verify the oracle against pinned bytes before retrieving anything."""
    validate_case(case)
    if (direct.composition_id != case['composition_id']
            or direct.composition['composition_hash'] != case['composition_hash']):
        raise ContractError('Direct composition differs from frozen contract')
    verified = []
    for group in case['expected_sources'].get('all_of', []):
        for ref in group['any_of']:
            units = [u for u in direct.units.values() if u['source_id'] == ref['source_id']
                     and ('unit_id' not in ref or u['unit_id'] == ref['unit_id'])]
            if len(units) != 1:
                raise ContractError('Expected source is missing or ambiguous in composition')
            unit = units[0]
            _, content, source_hash = direct.service._snapshot_file(unit, ref['file_path'])
            lines = content.decode('utf-8').splitlines()
            row = {'source_id': unit['source_id'], 'unit_id': unit['unit_id'], 'file_path': ref['file_path'],
                   'source_sha256': source_hash, 'start_line': 1, 'end_line': len(lines), 'content': content.decode('utf-8')}
            if not _match(ref, [row])['covered']:
                raise ContractError('Expected substantive source span does not match pinned bytes')
            verified.append({'group_id': group['id'], 'source_id': unit['source_id'], 'unit_id': unit['unit_id'],
                             'file_path': ref['file_path'], 'source_sha256': source_hash})
    return verified


def _record(row, *, channel=None):
    value = {key: copy.deepcopy(row[key]) for key in ('unit_id', 'source_id', 'file_path',
             'start_line', 'end_line', 'source_sha256', 'content')}
    if channel:
        value['channel'] = channel
    return value


def observe_search(direct, case):
    """Instrument one DirectService instance without changing retrieval policy.

    Holds the service request lock while temporary wrappers are installed and
    restores the exact original methods even on failure. The ordinary result
    and all three observed stages are retained for later deterministic replay.
    """
    verified = verify_expected_sources(case, direct)
    request = copy.deepcopy(case['request'])
    observation = {'case_id': case['id'], 'composition_hash': case['composition_hash'], 'request': request,
                   'candidate': [], 'selected': [], 'packed': [], 'expected_sources_verified': verified}
    started = time.perf_counter()
    with direct.service.request_lock:
        originals = {name: getattr(direct, name) for name in ('_lexical', '_routed_reports', '_pack')}
        vector_search = direct.service.search
        attribute_state = [(owner, name, name in vars(owner), vars(owner).get(name))
                           for owner, name in [(direct.service, 'search')]
                           + [(direct, name) for name in originals]]

        def capture(rows, channel):
            for row in rows:
                roles = request.get('filters', {}).get('roles')
                if roles is not None and direct.effective_role(row['unit_id'], row['file_path']) not in roles and row['role'] not in roles:
                    continue
                observation['candidate'].append(_record(row, channel=channel))

        def vector(*args, **kwargs):
            result = vector_search(*args, **kwargs)
            capture(result['results'], 'vector')
            return result

        def lexical(*args, **kwargs):
            rows = originals['_lexical'](*args, **kwargs)
            capture(rows, 'lexical')
            return rows

        def routed(*args, **kwargs):
            rows = originals['_routed_reports'](*args, **kwargs)
            capture(rows, 'routed_reports')
            return rows

        def packed(reads, *args, **kwargs):
            observation['selected'] = [_record(row) for row in reads]
            return originals['_pack'](reads, *args, **kwargs)

        direct.service.search = vector
        direct._lexical, direct._routed_reports, direct._pack = lexical, routed, packed
        try:
            filters = request.get('filters', {})
            result = direct.search(request['query'], limit=request['limit'], max_chars=request['max_chars'],
                                   include_catalog=request.get('include_catalog', True), **filters)
            observation['result'] = copy.deepcopy(result)
            observation['profile'] = copy.deepcopy(direct.last_profile)
            entries = direct.resolve([row['id'] for row in result['excerpts']])
            for excerpt, entry in zip(result['excerpts'], entries):
                row = {**entry, 'source_id': direct.units[entry['unit_id']]['source_id'], 'content': excerpt['text']}
                if (excerpt['source'] != row['source_id'] or excerpt['path'] != row['file_path']
                        or excerpt['lines'] != [row['start_line'], row['end_line']]):
                    raise ContractError('Packed result and issued evidence ledger differ')
                observation['packed'].append(_record(row))
        except Exception as exc:
            observation['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            for owner, name, existed, original in attribute_state:
                if existed:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
    observation['seconds_including_diagnostic_trace'] = time.perf_counter() - started
    return observation


def summarize(results):
    counts = Counter(row['status'] for row in results)
    return {'cases': len(results), 'statuses': dict(counts), 'all_passed': bool(results) and counts['pass'] == len(results),
            'purposes': {purpose: dict(Counter(row['status'] for row in results if row['purpose'] == purpose))
                         for purpose in sorted({row['purpose'] for row in results})}}
