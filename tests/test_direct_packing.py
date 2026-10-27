"""Packing regressions over real snapshots; no model, worker or source mutation."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bubbleteait.direct import DirectService
from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
from bubbleteait.search import SearchError


class Backend:
    def __init__(self):
        self.rows = {}

    def search(self, unit_id, query, *, limit, paths):
        return self.rows.get(unit_id, [])[:limit]


class DirectPackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = Registry(self.root / 'data')
        self.recipe = IndexRecipe('cocoindex-code', 'fixture', 'fixed', 'fixture')
        self.backend = Backend()
        patcher = patch('bubbleteait.direct.make_backend', return_value=self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)

    def unit(self, source, files):
        checkout = self.root / source
        checkout.mkdir()
        for name, text in files.items():
            path = checkout / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode('utf-8'))
        self.registry.register(SourceSpec(source, 'module', checkout))
        unit = self.registry.snapshot(source, self.recipe)
        status = self.registry.data_dir / 'indexes' / unit['unit_id'] / 'status.json'
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps({'schema_version': 1, 'unit_id': unit['unit_id'],
            'recipe_hash': unit['recipe_hash'], 'backend': 'cocoindex-code', 'status': 'ready'}), encoding='utf-8')
        return unit

    def direct(self, *units):
        self.registry.compose('packing', [unit['unit_id'] for unit in units], require_ready=True)
        return DirectService(SimpleNamespace(data_dir=self.registry.data_dir), 'packing')

    def row(self, unit, path, start, end):
        raw = (self.registry.data_dir / 'sources' / unit['unit_id'] / 'files' / path).read_bytes()
        return {'file_path': path, 'start_line': start, 'end_line': end,
                'content': ''.join(raw.decode('utf-8').splitlines(keepends=True)[start - 1:end]), 'score': .5}

    @staticmethod
    def function(number, *, eol='\n'):
        lines = [f'void verifyCircuit{number}(void) {{']
        lines += ['    // unrelated neighbouring setup ' + 'x' * 12] * 51
        setup = len(lines) + 1
        lines += [f'    int localLimit{number} = 7;', f'    int observed{number} = readValue();',
                  f'    if (observed{number} >= localLimit{number}) {{',
                  f'        CHECK(state{number} == expected{number});', '    }']
        anchor = setup + 3
        lines += ['    // unrelated neighbouring cleanup ' + 'y' * 10] * 48
        lines += ['}']
        return eol.join(lines) + eol, setup, anchor

    def assert_exact_citations(self, direct, result, texts):
        ids = []
        for excerpt in result['excerpts']:
            key = (excerpt['source'], excerpt['path'])
            original = texts[key]
            first, last = excerpt['lines']
            self.assertEqual(excerpt['text'], ''.join(original.splitlines(keepends=True)[first - 1:last]))
            ids.append(excerpt['id'])
        resolved = direct.resolve(ids)
        for excerpt, evidence in zip(result['excerpts'], resolved):
            self.assertEqual([evidence['start_line'], evidence['end_line']], excerpt['lines'])
            self.assertEqual(evidence['file_path'], excerpt['path'])
            self.assertEqual(evidence['source_sha256'], hashlib.sha256(texts[excerpt['source'], excerpt['path']].encode('utf-8')).hexdigest())

    def test_budget_is_shared_before_expanding_first_parent(self):
        """Every late condition keeps its local setup rather than a bare assertion."""
        files, rows = {}, []
        for number in range(5):
            text, _setup, anchor = self.function(number, eol='\r\n')
            path = f'tests/check_{number}.c'
            files[path] = text
            rows.append((path, anchor))
        unit = self.unit('circuits', files)
        self.backend.rows[unit['unit_id']] = [self.row(unit, path, anchor - 1, anchor + 1) for path, anchor in rows]
        direct = self.direct(unit)
        texts = {('circuits', path): text for path, text in files.items()}
        for budget in (6000, 9000, 12000):
            with self.subTest(budget=budget):
                result = direct.search('conceptual behaviour', limit=5, max_chars=budget)
                self.assertEqual({e['path'] for e in result['excerpts']}, set(files))
                for excerpt in result['excerpts']:
                    number = int(Path(excerpt['path']).stem.rsplit('_', 1)[1])
                    self.assertIn(f'int localLimit{number} = 7;', excerpt['text'])
                    self.assertIn(f'int observed{number} = readValue();', excerpt['text'])
                    self.assertIn(f'if (observed{number} >= localLimit{number})', excerpt['text'])
                    self.assertIn(f'CHECK(state{number} == expected{number});', excerpt['text'])
                self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), budget)
                self.assertTrue(result['truncated'])
                self.assert_exact_citations(direct, result, texts)

    def test_same_relative_paths_in_two_products_keep_both_substantive_windows(self):
        texts, units = {}, []
        for number, source in enumerate(('product-a', 'product-b')):
            text, _setup, anchor = self.function(number)
            unit = self.unit(source, {'src/decision.c': text})
            units.append(unit)
            texts[source, 'src/decision.c'] = text
            self.backend.rows[unit['unit_id']] = [self.row(unit, 'src/decision.c', anchor - 1, anchor + 1)]
        direct = self.direct(*units)
        result = direct.search('conceptual behaviour', limit=2, max_chars=6000)
        self.assertEqual({e['source'] for e in result['excerpts']}, {'product-a', 'product-b'})
        for number, source in enumerate(('product-a', 'product-b')):
            excerpt = next(e for e in result['excerpts'] if e['source'] == source)
            self.assertIn(f'int localLimit{number} = 7;', excerpt['text'])
            self.assertIn(f'CHECK(state{number} == expected{number});', excerpt['text'])
        self.assert_exact_citations(direct, result, texts)

    def test_overlap_of_parent_context_does_not_erase_a_distinct_anchor(self):
        lines = ['void bothChecks(void) {\n']
        lines += ['    // setup\n'] * 14
        first = len(lines) + 1
        lines += ['    CHECK(firstCircuit == expectedFirst);\n']
        lines += ['    // transition\n'] * 55
        second = len(lines) + 1
        lines += ['    CHECK(secondCircuit == expectedSecond);\n', '}\n']
        text = ''.join(lines)
        unit = self.unit('shared-parent', {'tests/check.c': text})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'tests/check.c', first, first),
                                               self.row(unit, 'tests/check.c', second, second)]
        direct = self.direct(unit)
        result = direct.search('conceptual behaviour', limit=2, max_chars=1300)
        combined = '\n'.join(excerpt['text'] for excerpt in result['excerpts'])
        self.assertIn('CHECK(firstCircuit == expectedFirst);', combined)
        self.assertIn('CHECK(secondCircuit == expectedSecond);', combined)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 1300)
        self.assert_exact_citations(direct, result, {('shared-parent', 'tests/check.c'): text})

    def test_explicit_large_read_is_never_replaced_by_a_smaller_window(self):
        text, _setup, _anchor = self.function(0)
        unit = self.unit('explicit', {'tests/check.c': text})
        direct = self.direct(unit)
        request = {'source_id': 'explicit', 'path': 'tests/check.c', 'start': 1,
                   'end': len(text.splitlines())}
        result = direct.read_many([request], max_chars=1800)
        self.assertEqual(result['excerpts'], [])
        self.assertTrue(result['truncated'])
        self.assertEqual(direct.resolve([]), [])

    def test_indivisible_long_line_is_omitted_and_truncation_is_reported(self):
        text = 'const char *message = "' + 'q' * 3000 + '";\r\n'
        unit = self.unit('long-line', {'message.c': text})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'message.c', 1, 1)]
        direct = self.direct(unit)
        result = direct.search('conceptual behaviour', limit=1, max_chars=512)
        self.assertEqual(result['excerpts'], [])
        self.assertTrue(result['truncated'])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 512)

    def test_published_next_read_is_executable_and_keeps_exact_original_lines(self):
        text, _setup, anchor = self.function(0, eol='\r\n')
        unit = self.unit('continuation', {'tests/check.c': text})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'tests/check.c', anchor - 1, anchor + 1)]
        direct = self.direct(unit)
        first = direct.search('conceptual behaviour', limit=1, max_chars=1800)['excerpts'][0]
        request = first['next_read']
        self.assertEqual(request['start'], first['lines'][1] + 1)
        self.assertLessEqual(request['end'] - request['start'] + 1, 120)
        following = direct.read_many([request], max_chars=12000, include_catalog=False)
        self.assertEqual(len(following['excerpts']), 1)
        self.assertEqual(following['excerpts'][0]['lines'], [request['start'], request['end']])
        self.assert_exact_citations(direct, following, {('continuation', 'tests/check.c'): text})
        self.assertEqual(direct.resolve([first['id']])[0]['start_line'], first['lines'][0])

    def test_oversized_batch_advice_has_executable_windows_and_publishes_no_partial_ids(self):
        text = ''.join(f'int line{number};\r\n' for number in range(1, 301))
        unit = self.unit('windows', {'main.c': text})
        direct = self.direct(unit)
        valid = {'source_id': 'windows', 'path': 'main.c', 'start': 1, 'end': 1}
        original = direct.read_many([valid])['excerpts'][0]
        ledger_before = dict(direct._ledger)
        invalid = {**valid, 'start': 110, 'end': 244}
        with self.assertRaises(SearchError) as error:
            direct.read_many([{**valid, 'start': 2, 'end': 3}, invalid])
        self.assertIn('requests[1]', str(error.exception))
        advice = json.loads(str(error.exception).split(' Navigation: ', 1)[1])
        self.assertEqual(direct._ledger, ledger_before)
        self.assertEqual(advice['total_lines'], 300)
        windows = advice['read_windows']
        self.assertEqual(windows[0]['start'], 110)
        self.assertEqual(windows[-1]['end'], 244)
        for index, window in enumerate(windows):
            self.assertEqual(window['path'], 'main.c')
            self.assertLessEqual(window['end'] - window['start'] + 1, 120)
            if index:
                self.assertEqual(window['start'], windows[index - 1]['end'] + 1)
        recovered = direct.read_many(windows, max_chars=16000, include_catalog=False)
        combined = ''.join(excerpt['text'] for excerpt in recovered['excerpts'])
        self.assertEqual(combined, ''.join(text.splitlines(keepends=True)[109:244]))
        self.assert_exact_citations(direct, recovered, {('windows', 'main.c'): text})
        self.assertEqual(direct.resolve([original['id']])[0]['start_line'], 1)

    def test_missing_configuration_suggests_only_real_pinned_paths(self):
        files = {'trunk/config/configuration.h.in': '#define VARIANT 7\n',
                 'trunk/Src/control.c': 'int control;\n', 'README.md': '# Build\n'}
        unit = self.unit('configuration', files)
        direct = self.direct(unit)
        absent = 'trunk/Src/configuration.h'
        live = self.root / 'configuration' / absent
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text('generated after the snapshot\n', encoding='utf-8')
        with self.assertRaises(SearchError) as error:
            direct.read_many([{'source_id': 'configuration', 'path': absent, 'start': 1, 'end': 1}])
        advice = json.loads(str(error.exception).split(' Navigation: ', 1)[1])
        self.assertTrue(advice['available_paths'])
        self.assertLessEqual(set(advice['available_paths']), set(files))
        self.assertIn('trunk/config/configuration.h.in', advice['available_paths'])
        self.assertNotIn(absent, advice['available_paths'])
        self.assertEqual(direct._ledger, {})
        suggestion = {'source_id': advice['source_id'], 'path': advice['available_paths'][0], 'start': 1, 'end': 1}
        recovered = direct.read_many([suggestion], include_catalog=False)
        self.assertEqual(recovered['excerpts'][0]['text'], files[suggestion['path']])

    def test_catalog_can_be_omitted_without_losing_source_identity_or_citations(self):
        alpha = self.unit('alpha', {'src/common.c': 'void alpha(void) {\n    checkAlpha();\n}\n'})
        beta = self.unit('beta', {'src/common.c': 'void beta(void) {\n    checkBeta();\n}\n'})
        for unit in (alpha, beta):
            self.backend.rows[unit['unit_id']] = [self.row(unit, 'src/common.c', 2, 2)]
        direct = self.direct(alpha, beta)
        full = direct.search('conceptual behaviour', limit=2, max_chars=6000)
        compact = direct.search('conceptual behaviour', limit=2, max_chars=6000, include_catalog=False)
        self.assertTrue(full['catalog_complete'])
        self.assertEqual(compact['catalog'], [])
        self.assertFalse(compact['catalog_complete'])
        self.assertEqual(compact['sources'], full['sources'])
        self.assertEqual(compact['excerpts'], full['excerpts'])
        self.assertEqual(compact['coverage'], full['coverage'])
        self.assertEqual(len(direct.resolve([e['id'] for e in compact['excerpts']])), 2)
        self.assertLess(len(json.dumps(compact, ensure_ascii=False, separators=(',', ':'))),
                        len(json.dumps(full, ensure_ascii=False, separators=(',', ':'))))

    def test_omitted_explicit_read_reports_hit_and_role_coverage_without_issuing_it(self):
        text, _setup, _anchor = self.function(0)
        unit = self.unit('omissions', {'main.c': 'int publicValue;\n', 'tests/check.c': text})
        direct = self.direct(unit)
        small = {'source_id': 'omissions', 'path': 'main.c', 'start': 1, 'end': 1}
        large = {'source_id': 'omissions', 'path': 'tests/check.c', 'start': 1,
                 'end': len(text.splitlines())}
        result = direct.read_many([small, large], max_chars=3000, include_catalog=False)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['omitted_count'], 1)
        self.assertEqual([e['path'] for e in result['excerpts']], ['main.c'])
        self.assertEqual(result['coverage']['selected_hits'], 2)
        self.assertEqual(result['coverage']['covered_hits'], 1)
        self.assertEqual(result['coverage']['selected_sources'], 1)
        self.assertEqual(result['coverage']['returned_sources'], 1)
        self.assertEqual(result['coverage']['selected_roles'], ['code', 'tests'])
        self.assertEqual(result['coverage']['returned_roles'], ['code'])
        self.assertEqual(len(direct._ledger), 1)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 3000)

    def test_broad_semantic_hit_cannot_swallow_two_distinct_lexical_anchors(self):
        lines = ['// filler ' + 'x' * 70 + '\n' for _ in range(70)]
        lines[9] = 'void CheckAlpha(void);\n'
        lines[54] = 'void CheckBeta(void);\n'
        text = ''.join(lines)
        unit = self.unit('broad-semantic', {'src/both.c': text})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'src/both.c', 1, 70)]
        direct = self.direct(unit)
        result = direct.search('CheckAlpha CheckBeta', limit=3, max_chars=2600, include_catalog=False)
        combined = '\n'.join(excerpt['text'] for excerpt in result['excerpts'])
        self.assertIn('void CheckAlpha(void);', combined)
        self.assertIn('void CheckBeta(void);', combined)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 2600)
        self.assert_exact_citations(direct, result, {('broad-semantic', 'src/both.c'): text})

    def test_adjacent_explicit_reads_keep_replayable_ranges_within_120_lines(self):
        text = ''.join(f'int row{number};\r\n' for number in range(1, 301))
        unit = self.unit('adjacent', {'main.c': text})
        direct = self.direct(unit)
        requests = [{'source_id': 'adjacent', 'path': 'main.c', 'start': 1, 'end': 120},
                    {'source_id': 'adjacent', 'path': 'main.c', 'start': 121, 'end': 240}]
        response = direct.read_many(requests, max_chars=16000, include_catalog=False)
        self.assertEqual([e['lines'] for e in response['excerpts']], [[1, 120], [121, 240]])
        self.assert_exact_citations(direct, response, {('adjacent', 'main.c'): text})
        for excerpt in response['excerpts']:
            self.assertLessEqual(excerpt['lines'][1] - excerpt['lines'][0] + 1, 120)
            replay = direct.read_many([excerpt['id']], max_chars=16000, include_catalog=False)
            self.assertEqual(len(replay['excerpts']), 1)
            self.assertEqual(replay['excerpts'][0]['lines'], excerpt['lines'])
            self.assertEqual(replay['excerpts'][0]['text'], excerpt['text'])
            self.assertEqual(replay['excerpts'][0]['id'], excerpt['id'])


if __name__ == '__main__':
    unittest.main()
