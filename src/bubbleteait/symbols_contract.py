"""Expected-source acceptance for navigation; independent of Serena output."""
from __future__ import annotations

import json
from pathlib import Path
import re


def load_cases(path):
    bundle=json.loads(Path(path).read_text(encoding='utf-8'))
    if bundle.get('schema_version') != 1 or not isinstance(bundle.get('cases'),list) or not bundle['cases']:
        raise ValueError('Expected a nonempty version-1 symbols contract')
    ids=set()
    for case in bundle['cases']:
        if not isinstance(case.get('id'),str) or not case['id'] or case['id'] in ids:
            raise ValueError('Each symbols case needs a unique id')
        ids.add(case['id'])
        if case.get('operation') not in {'definition','references'}:
            raise ValueError('Symbol source acceptance requires a definition or references')
        for key in ('composition_id','source_id','path','name'):
            if not isinstance(case.get(key),str) or not case[key]:
                raise ValueError(f'Missing case {key}')
        if not re.fullmatch('[0-9a-f]{64}',case.get('composition_hash','')):
            raise ValueError('Case must pin the composition hash')
        expected=case.get('expected_sources')
        if not isinstance(expected,list) or not expected:
            raise ValueError('Case must declare expected sources')
        for source in expected:
            if not isinstance(source.get('source'),str) or not isinstance(source.get('path'),str):
                raise ValueError('Expected source needs a logical source and path')
            if not re.fullmatch('[0-9a-f]{64}',source.get('source_sha256','')):
                raise ValueError('Expected source must pin its bytes')
            keys=set(source)&{'lines','covers_lines'}
            if len(keys)!=1:
                raise ValueError('Expected source requires exactly one range criterion')
            span=source[next(iter(keys))]
            if (not isinstance(span,list) or len(span)!=2 or any(type(n)is not int for n in span)
                    or not 1<=span[0]<=span[1]):
                raise ValueError('Expected source ranges must be positive and inclusive')
    return bundle


def _valid_lines(row):
    if not isinstance(row,dict) or row.get('location_status')!='verified_snapshot_lines':
        return False
    span=row.get('lines')
    total=row.get('total_lines')
    return (isinstance(span,list) and len(span)==2 and all(type(n)is int for n in span)
            and type(total)is int and 1<=span[0]<=span[1]<=total)


def matches(expected,row):
    """The shared source/range criterion for scoring and selecting a ready read."""
    if not _valid_lines(row):
        return False
    if any(row.get(key)!=expected[key] for key in ('source','path','source_sha256')):
        return False
    if 'unit_id' in expected and row.get('unit_id')!=expected['unit_id']:
        return False
    span=row['lines']
    if 'lines' in expected:
        return span==expected['lines']
    first,last=expected['covers_lines']
    return span[0]<=first<=last<=span[1]


def ready_read(expected,row):
    """Validate the actual ready-read pointer before issuing a read request.

    This bounded checker does not automatically continue symbols longer than
    120 lines. Such a source needs an explicit continuation implementation;
    its first page alone cannot prove that the complete symbol was read.
    """
    if not matches(expected,row):
        raise ValueError('Ready read locator does not match the expected source/range')
    if not all(isinstance(row.get(key),str) and row[key] for key in ('source_id','unit_id')):
        raise ValueError('Ready read locator lacks a source alias or immutable unit')
    start,end=row['lines']
    if end-start>=120:
        raise ValueError('Symbol exceeds one 120-line read; explicit continuation is required')
    read=row.get('read')
    required={'source_id','path','start','end'}
    if (not isinstance(read,dict) or not required<=set(read) or set(read)-required-{'unit_id'}
            or read['source_id']!=row['source_id'] or read['path']!=row['path']
            or ('unit_id' in read and read['unit_id']!=row['unit_id'])
            or type(read['start'])is not int or type(read['end'])is not int
            or [read['start'],read['end']]!=row['lines']):
        raise ValueError('Ready read pointer differs from the verified symbol coordinates/source')
    return dict(read)


def validate_read_evidence(case,expected,row,read_result,resolved_result):
    """Bind the issued read, returned text and resolved ledger entry together."""
    request=ready_read(expected,row)
    if (not isinstance(read_result,dict)
            or read_result.get('composition_id')!=case['composition_id']
            or read_result.get('composition_hash')!=case['composition_hash']
            or not isinstance(resolved_result,dict)
            or resolved_result.get('composition_hash')!=case['composition_hash']):
        raise ValueError('Read/resolve composition differs from the frozen symbol case')
    excerpts=read_result.get('excerpts')
    evidence=resolved_result.get('evidence')
    if (not isinstance(excerpts,list) or len(excerpts)!=1
            or not isinstance(evidence,list) or len(evidence)!=1):
        raise ValueError('One ready read must return exactly one excerpt and one resolved evidence entry')
    excerpt,entry=excerpts[0],evidence[0]
    if not isinstance(excerpt,dict) or not isinstance(entry,dict):
        raise ValueError('Invalid read/resolve evidence objects')
    alias=row['source_id']
    source_map=read_result.get('sources')
    if (not isinstance(source_map,dict) or not isinstance(source_map.get(alias),dict)
            or source_map[alias].get('source_id')!=expected['source']
            or excerpt.get('source_id')!=alias or excerpt.get('source')!=expected['source']
            or excerpt.get('path')!=expected['path']
            or excerpt.get('lines')!=[request['start'],request['end']]
            or excerpt.get('total_lines')!=row['total_lines']
            or excerpt.get('context_truncated')is not False):
        raise ValueError('Returned excerpt does not match the requested source and complete range')
    text=excerpt.get('text')
    if (not isinstance(text,str) or not text.strip()
            or len(text.splitlines())!=request['end']-request['start']+1):
        raise ValueError('Returned source text does not cover the actual inclusive read range')
    if (not isinstance(excerpt.get('id'),str) or not excerpt['id']
            or entry.get('id')!=excerpt['id']
            or entry.get('unit_id')!=row['unit_id']
            or entry.get('file_path')!=expected['path']
            or entry.get('source_sha256')!=expected['source_sha256']
            or type(entry.get('start_line'))is not int or type(entry.get('end_line'))is not int
            or [entry['start_line'],entry['end_line']]!=excerpt['lines']):
        raise ValueError('Resolved ledger identity/range does not match the returned excerpt')
    # Check the substantive expectation against the coordinates actually read,
    # not just the larger locator originally returned by navigation.
    actual={**row,'source':excerpt['source'],'path':excerpt['path'],
            'lines':excerpt['lines'],'source_sha256':entry['source_sha256']}
    if not matches(expected,actual):
        raise ValueError('Actual read does not cover the expected source range')
    return {'source':expected['source'],'unit_id':entry['unit_id'],'path':entry['file_path'],
            'lines':excerpt['lines'],'source_sha256':entry['source_sha256'],
            'evidence_ids':[entry['id']]}


def evaluate(case,result):
    if not isinstance(result,dict) or result.get('composition_id')!=case['composition_id']:
        raise ValueError('Observed composition id differs from the symbol case')
    if result.get('composition_hash') != case['composition_hash']:
        raise ValueError('Observed composition differs from expected version')
    if result.get('operation') != case['operation'] or result.get('navigation_only') is not True:
        raise ValueError('Expected a navigation result for this operation')
    if not isinstance(result.get('results'),list):
        raise ValueError('Expected an array of navigation results')
    missing=[]
    for expected in case['expected_sources']:
        found=any(matches(expected,row) for row in result['results'])
        if not found:
            missing.append(expected)
    return {'case_id':case['id'],'status':'fail' if missing else 'pass','missing_expected_sources':missing}
