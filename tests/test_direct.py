"""Direct retrieval uses real immutable snapshots and a deterministic fake vector backend."""
import hashlib
import json
import fnmatch
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bubbleteait.direct import DirectService
from bubbleteait.registry import IndexRecipe, Registry, RegistryError, SourceSpec
from bubbleteait.search import SearchError


class Backend:
    def __init__(self):
        self.rows, self.calls = {}, []

    def search(self, unit_id, query, *, limit, paths):
        self.calls.append((unit_id, query, paths))
        return self.rows.get(unit_id, [])[:limit]


class DirectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = Registry(self.root / 'data')
        self.recipe = IndexRecipe('cocoindex-code', 'fixture', 'fixed', 'fixture')
        self.backend = Backend()
        self.config = SimpleNamespace(data_dir=self.registry.data_dir)
        patcher = patch('bubbleteait.direct.make_backend', return_value=self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)

    def unit(self, source, files, role='unknown'):
        checkout = self.root / source
        checkout.mkdir(exist_ok=True)
        for name, text in files.items():
            path = checkout / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())
        self.registry.register(SourceSpec(source, 'module', checkout, role=role))
        unit = self.registry.snapshot(source, self.recipe)
        status = self.registry.data_dir / 'indexes' / unit['unit_id'] / 'status.json'
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({'schema_version': 1, 'unit_id': unit['unit_id'], 'recipe_hash': unit['recipe_hash'],
                                     'backend': 'cocoindex-code', 'status': 'ready'}), encoding='utf-8')
        return unit

    def direct(self, *units):
        self.registry.compose('fixture', [unit['unit_id'] for unit in units], require_ready=True)
        return DirectService(self.config, 'fixture')

    def row(self, unit, path, start, end):
        text = (self.registry.data_dir / 'sources' / unit['unit_id'] / 'files' / path).read_bytes().decode()
        return {'file_path': path, 'start_line': start, 'end_line': end,
                'content': ''.join(text.splitlines(keepends=True)[start-1:end]), 'score': .5}

    def test_hybrid_rrf_promotes_consensus_and_deduplicates_overlapping_votes(self):
        unit = self.unit('code', {'semantic.c': 'int conceptual = 1;\n',
                                  'both.c': 'int needle = 2;\n', 'lexical.c': 'int needleOnly = 3;\n'})
        semantic = self.row(unit, 'semantic.c', 1, 1)
        both = self.row(unit, 'both.c', 1, 1)
        self.backend.rows[unit['unit_id']] = [semantic, both, both, both]
        direct = self.direct(unit)
        result = direct.search('needle', limit=3)
        self.assertEqual([hit['path'] for hit in result['excerpts']], ['both.c', 'semantic.c', 'lexical.c'])
        self.assertEqual(len({hit['id'] for hit in result['excerpts']}), 3)
        self.assertNotIn('source_sha256', json.dumps(result))

    def test_excluded_code_identifier_does_not_suppress_requested_docs(self):
        unit = self.unit('mixed', {'README.md': '# Calibration\nA conceptual overview.\n',
                                   'main.c': 'void CalibrateRelay(void) {}\n'})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'README.md', 1, 2)]
        result = self.direct(unit).search('CalibrateRelay', roles=['docs'])
        self.assertEqual([hit['path'] for hit in result['excerpts']], ['README.md'])

    def test_rare_identifier_outranks_common_prose_globally_without_rewriting_vector_query(self):
        common = self.unit('a-common', {'utility.c': '// the and for with в на для управление тест\n'})
        actual = self.unit('z-actual', {'relay.c': 'void CalibrateRelay(void) {}\n'})
        self.backend.rows[common['unit_id']] = [self.row(common, 'utility.c', 1, 1)]
        query = 'CalibrateRelay and the в на для управление регрессионный тест'
        direct = self.direct(common, actual)
        with patch.object(direct.service, 'search', wraps=direct.service.search) as service_search:
            result = direct.search(query, limit=2)
        self.assertEqual(service_search.call_count, 1)
        self.assertEqual(service_search.call_args.kwargs['channel'], 'vector')
        self.assertEqual({call[1] for call in self.backend.calls}, {query})
        self.assertEqual(result['excerpts'][0]['path'], 'relay.c')

    def test_role_diversity_does_not_fill_identifier_results_with_unrelated_documents(self):
        actual = self.unit('code', {'relay.c': 'void CalibrateRelay(void) {}\n'})
        other = self.unit('other', {'README.md': '# General test\nThis is a test guide.\n'}, role='requirements')
        self.backend.rows[other['unit_id']] = [self.row(other, 'README.md', 1, 2)]
        direct = self.direct(actual, other)
        result = direct.search('CalibrateRelay and the тест управление', limit=6)
        self.assertEqual([hit['path'] for hit in result['excerpts']], ['relay.c'])

    def test_short_labels_match_words_and_underscore_segments_not_prose_substrings(self):
        unit = self.unit('products', {'ordinary.c': '// this while limits\n',
                                     'hardware.c': '#define BUILD_BB 1\n'})
        direct = self.direct(unit)
        matches = direct._lexical('AA BB', 20, None, None, None)
        self.assertEqual([hit['file_path'] for hit in matches], ['hardware.c'])

    def test_short_labels_do_not_suppress_semantic_russian_report_and_scope_is_discoverable(self):
        alpha = self.unit('alpha-app-v2', {'src/main.c': '// temperature limits hysteresis implementation tests\n'})
        beta = self.unit('beta-app-v2', {'src/main.c': '// temperature limits hysteresis implementation tests\n'})
        report = self.unit('testing-history', {'old.md': '# Ошибка\nНеверные границы температуры после изменения параметра.\n'}, role='primary_report')
        hidden = self.unit('unretrieved-source', {'other.c': 'int unrelated;\n'})
        for unit in (alpha, beta):
            self.backend.rows[unit['unit_id']] = [self.row(unit, 'src/main.c', 1, 1)]
        self.backend.rows[report['unit_id']] = [self.row(report, 'old.md', 1, 2)]
        direct = self.direct(alpha, beta, report, hidden)
        result = direct.search('AA BB temperature limits hysteresis bug report requirement implementation tests')
        self.assertIn('reports', {hit['role'] for hit in result['excerpts']})
        self.assertEqual({hit['source'] for hit in result['excerpts'] if hit['path'] == 'src/main.c'},
                         {'alpha-app-v2', 'beta-app-v2'})
        self.assertTrue(result['catalog_complete'])
        self.assertEqual({entry['source'] for entry in result['catalog']},
                         {'alpha-app-v2', 'beta-app-v2', 'testing-history', 'unretrieved-source'})
        # A metadata product tag alone must not turn unrelated content lines into hits.
        self.assertEqual(direct._lexical('AA absentSemanticTerm', 20, None, ['alpha-app-v2'], None), [])

    def test_tiny_payload_catalog_remains_bounded_and_explicitly_incomplete(self):
        units = [self.unit('scope-' + str(number), {'a.c': 'int untouched;\n'}) for number in range(6)]
        result = self.direct(*units).search('noMatchingTerm', max_chars=512)
        self.assertEqual(result['excerpts'], [])
        self.assertFalse(result['catalog_complete'])
        self.assertGreater(len(result['catalog']), 0)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 512)

    def test_catalog_reservation_survives_substantive_excerpts_near_total_budget(self):
        unit = self.unit('primary', {'large.txt': 'needle ' + 'x' * 9000 + '\n'})
        others = [self.unit('scope-' + str(number), {'a.c': 'int untouched;\n'}) for number in range(11)]
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'large.txt', 1, 1)]
        result = self.direct(unit, *others).search('needle', max_chars=12000)
        self.assertEqual(len(result['excerpts']), 1)
        self.assertGreater(len(result['excerpts'][0]['text']), 9000)
        self.assertTrue(result['catalog_complete'])
        self.assertEqual(len(result['catalog']), 12)
        self.assertLessEqual(len(json.dumps(result['catalog'], ensure_ascii=False, separators=(',', ':'))), 3000)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 12000)

    def test_budget_fit_preserves_original_hit_and_issues_exact_new_citation(self):
        lines = ['void processFrame(void) {\r\n'] + ['    // ' + 'padding ' * 16 + '\r\n'] * 40
        lines += ['    int targetToken = 17;\r\n']
        anchor = len(lines)
        lines += ['    // ' + 'padding ' * 16 + '\r\n'] * 40 + ['}\r\n']
        text = ''.join(lines)
        unit = self.unit('synthetic-alpha', {'src/frame.c': text})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'src/frame.c', anchor - 3, anchor + 3)]
        direct = self.direct(unit)
        full = direct.search('targetToken', limit=1, max_chars=16000)['excerpts'][0]
        self.assertEqual(full['lines'], [1, len(lines)])
        self.assertFalse(full['context_truncated'])
        result = direct.search('targetToken', limit=1, max_chars=3500)
        hit = result['excerpts'][0]
        self.assertLessEqual(hit['lines'][0], anchor)
        self.assertGreaterEqual(hit['lines'][1], anchor)
        self.assertEqual(hit['text'], ''.join(lines[hit['lines'][0] - 1:hit['lines'][1]]))
        self.assertTrue(hit['context_truncated'])
        self.assertTrue(result['truncated'])
        self.assertNotEqual(hit['id'], full['id'])
        self.assertEqual(hit['source'], 'synthetic-alpha')
        evidence = direct.resolve([hit['id']])[0]
        self.assertEqual((evidence['unit_id'], evidence['file_path']), (unit['unit_id'], 'src/frame.c'))
        self.assertEqual([evidence['start_line'], evidence['end_line']], hit['lines'])
        self.assertEqual(evidence['source_sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 3500)
        # An explicit request is atomic and exact, never silently shortened to its centre.
        requested = {'source_id': 'synthetic-alpha', 'path': 'src/frame.c', 'start': 1, 'end': len(lines)}
        omitted = direct.read_many([requested], max_chars=3500)
        self.assertEqual(omitted['excerpts'], [])
        self.assertTrue(omitted['truncated'])
        explicit = direct.read_many([requested], max_chars=16000)['excerpts'][0]
        self.assertEqual(explicit['lines'], [1, len(lines)])
        self.assertEqual(explicit['text'], text)
        self.assertFalse(explicit['context_truncated'])

    def test_semantic_multiline_seed_shrinks_by_whole_lines_when_seed_itself_is_too_large(self):
        lines = ['void frame(void) {\n'] + ['    // ' + 'x' * 220 + '\n'] * 24 + ['}\n']
        unit = self.unit('synthetic-beta', {'frame.c': ''.join(lines)})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'frame.c', 4, 20)]
        result = self.direct(unit).search('unmatchedtopic', limit=1, max_chars=1800)
        hit = result['excerpts'][0]
        first, last = hit['lines']
        self.assertLess(last - first, 16)
        self.assertTrue(first <= 12 <= last)
        self.assertEqual(hit['text'], ''.join(lines[first - 1:last]))
        self.assertTrue(hit['context_truncated'])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 1800)

    def test_soft_role_intents_prefer_requested_kind_and_no_intent_stays_neutral(self):
        app = self.unit('app-XX', {'src/main.c': '// signal\n', 'tests/test_signal.c': '// signal\n'})
        docs = self.unit('docs-XX', {'requirements.md': '# signal\n'}, role='requirements')
        reports = self.unit('history', {'XX/report.md': '# signal\n'}, role='primary_report')
        self.backend.rows[app['unit_id']] = [self.row(app, path, 1, 1) for path in ('tests/test_signal.c', 'src/main.c')]
        self.backend.rows[docs['unit_id']] = [self.row(docs, 'requirements.md', 1, 1)]
        self.backend.rows[reports['unit_id']] = [self.row(reports, 'XX/report.md', 1, 1)]
        direct = self.direct(app, docs, reports)
        neutral = direct.search('signal', limit=4)
        self.assertEqual([hit['id'] for hit in neutral['excerpts']],
                         [hit['id'] for hit in direct.search('signal discussion', limit=4)['excerpts']])
        self.assertEqual([hit['id'] for hit in neutral['excerpts']],
                         [hit['id'] for hit in direct.search('signal resource inspection', limit=4)['excerpts']])
        for intent, role in (('implementation', 'code'), ('requirements', 'docs'),
                             ('regression tests', 'tests'), ('report', 'reports')):
            with self.subTest(intent=intent):
                self.assertEqual(direct.search('signal ' + intent, limit=4)['excerpts'][0]['role'], role)
        self.backend.calls.clear()
        self.assertEqual(direct.search('XX signal implementation', limit=4)['excerpts'][0]['role'], 'code')
        self.assertEqual(len(self.backend.calls), 3)  # No extra report-routing pass for pure implementation.
        self.assertEqual(direct.search('signal implementation', roles=['docs'])['excerpts'][0]['role'], 'docs')
        filtered = direct.search('signal requirements', source_ids=['history'], roles=['primary_report'])
        self.assertEqual({hit['source'] for hit in filtered['excerpts']}, {'history'})

    def test_explicit_source_filter_wins_over_other_product_mentioned_with_same_path(self):
        alpha = self.unit('Alpha', {'src/main.c': '// signal alpha\n'})
        beta = self.unit('Beta', {'src/main.c': '// signal beta\n'})
        for unit in (alpha, beta):
            self.backend.rows[unit['unit_id']] = [self.row(unit, 'src/main.c', 1, 1)]
        direct = self.direct(alpha, beta)
        result = direct.search('Beta src/main.c signal implementation', source_ids=['Alpha'], paths=['src/main.c'])
        self.assertEqual({hit['source'] for hit in result['excerpts']}, {'Alpha'})
        self.assertEqual([call[0] for call in self.backend.calls], [alpha['unit_id']])
        self.assertEqual(direct.resolve([result['excerpts'][0]['id']])[0]['unit_id'], alpha['unit_id'])

    def test_batch_error_names_bad_request_and_preserves_atomic_ledger(self):
        unit = self.unit('source', {'one.c': 'int value;\n'})
        direct = self.direct(unit)
        valid = {'source_id': 'source', 'path': 'one.c', 'start': 1, 'end': 1}
        with self.assertRaisesRegex(SearchError, r'requests\[1\].*nonempty_relative_path'):
            direct.read_many([valid, {**valid, 'path': ''}])
        self.assertEqual(direct._ledger, {})
        with self.assertRaisesRegex(SearchError, r'requests\[0\].*Expected an issued ID'):
            direct.read_many([{'wrong': 'shape'}])

    def test_explicit_labels_route_small_report_groups_without_hardcoded_products(self):
        unit = self.unit('archive', {'Other/ZZ/note.md': '# Ошибка\nСтарое наблюдение.\n',
                                    'Alpha/XX/note.md': '# Ошибка\nПервое наблюдение.\n',
                                    'Beta/YY/note.md': '# Ошибка\nВторое наблюдение.\n'}, role='primary_report')
        rows = [self.row(unit, path, 1, 2) for path in
                ('Other/ZZ/note.md', 'Alpha/XX/note.md', 'Beta/YY/note.md')]
        calls = []
        def scoped(unit_id, query, *, limit, paths):
            calls.append(paths)
            return [row for row in rows if paths is None or any(fnmatch.fnmatchcase(row['file_path'], pattern)
                                                               for pattern in paths)][:limit]
        self.backend.search = scoped
        direct = self.direct(unit)
        result = direct.search('Alpha Beta XX YY regression report', limit=3)
        self.assertEqual({hit['path'] for hit in result['excerpts'][:2]}, {'Alpha/XX/note.md', 'Beta/YY/note.md'})
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(calls[1]), {'Alpha/XX/note.md', 'Beta/YY/note.md'})
        calls.clear()
        unscoped = direct.search('regression report', limit=3)
        self.assertIn('Other/ZZ/note.md', {hit['path'] for hit in unscoped['excerpts']})
        self.assertEqual(calls, [None])

    def test_one_request_validates_at_entry_and_exit_and_nested_resolve_reuses_scope(self):
        unit = self.unit('code', {'main.c': 'int needle;\n'})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'main.c', 1, 1)]
        direct = self.direct(unit)
        with patch.object(direct.service, '_validate', wraps=direct.service._validate) as validate:
            hit = direct.search('needle')['excerpts'][0]
            self.assertEqual(validate.call_count, 2)
        with patch.object(direct.service, '_validate', wraps=direct.service._validate) as validate:
            direct.read_many([hit['id']])
            self.assertEqual(validate.call_count, 2)

    def test_readiness_mutation_inside_request_rejects_publication_and_no_stale_scope_survives(self):
        unit = self.unit('code', {'main.c': 'int needle;\n'})
        direct = self.direct(unit)
        status = self.registry.data_dir / 'indexes' / unit['unit_id'] / 'status.json'
        ready = status.read_text(encoding='utf-8')
        def change_status(unit_id, query, *, limit, paths):
            value = json.loads(ready)
            value['status'] = 'building'
            status.write_text(json.dumps(value), encoding='utf-8')
            return [self.row(unit, 'main.c', 1, 1)]
        self.backend.search = change_status
        with self.assertRaisesRegex(SearchError, 'no longer ready'):
            direct.search('needle')
        self.assertEqual(direct._ledger, {})
        self.assertEqual(direct.service._depth, 0)
        status.write_text(ready, encoding='utf-8')
        self.backend.search = lambda *args, **kwargs: [self.row(unit, 'main.c', 1, 1)]
        hit = direct.search('needle')['excerpts'][0]
        status.write_text(json.dumps({**json.loads(ready), 'status': 'building'}), encoding='utf-8')
        with self.assertRaisesRegex(SearchError, 'no longer ready'):
            direct.resolve([hit['id']])

    def test_concurrent_requests_share_no_partial_validation_scope(self):
        unit = self.unit('code', {'main.c': 'int needle;\n'})
        direct = self.direct(unit)
        first_inside, second_inside, release, attempting = (threading.Event() for _ in range(4))
        errors = []
        def backend(unit_id, query, *, limit, paths):
            if query == 'first':
                first_inside.set()
                if not release.wait(2):
                    raise RuntimeError('Fixture timed out waiting for release')
            else:
                second_inside.set()
            return [self.row(unit, 'main.c', 1, 1)]
        self.backend.search = backend
        def run(query):
            try:
                if query == 'second':
                    attempting.set()
                direct.search(query)
            except Exception as exc:
                errors.append(exc)
        first, second = threading.Thread(target=run, args=('first',)), threading.Thread(target=run, args=('second',))
        first.start()
        self.assertTrue(first_inside.wait(2))
        second.start()
        self.assertTrue(attempting.wait(2))
        try:
            self.assertFalse(second_inside.wait(.05))
        finally:
            release.set()
            first.join(3)
            second.join(3)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(second_inside.is_set())

    def test_context_expansion_citations_match_exact_snapshot_bytes_and_not_live_files(self):
        text = '// header\r\nint compute(void)\r\n{\r\n    const char *s = "}";\r\n    return 37;\r\n}\r\nint unrelated;\r\n'
        unit = self.unit('code', {'src/main.c': text})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'src/main.c', 5, 5)]
        direct = self.direct(unit)
        (self.root / 'code/src/main.c').write_text('changed working file', encoding='utf-8')
        result = direct.search('compute', limit=1)
        hit = result['excerpts'][0]
        start, end = hit['lines']
        self.assertEqual(end, 6)
        self.assertEqual(hit['text'], ''.join(text.splitlines(keepends=True)[start-1:end]))
        self.assertEqual(hit['total_lines'], 7)
        self.assertTrue(hit['source_changed'])
        self.assertEqual(hit['working_state'], 'available')
        evidence = direct.resolve([hit['id']])[0]
        self.assertEqual(evidence['source_sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual((evidence['start_line'], evidence['end_line']), (start, end))
        self.assertEqual(evidence['file_path'], 'src/main.c')
        # Resolve rechecks the immutable snapshot. Duplicate IDs retain order
        # without duplicating source reads or allowing caller-supplied hashes.
        with patch.object(direct.service, 'read_source', wraps=direct.service.read_source) as read:
            self.assertEqual(direct.resolve([hit['id'], hit['id']]), [evidence, evidence])
            self.assertEqual(read.call_count, 1)

    def test_resolve_rejects_snapshot_tampered_after_a_successful_search(self):
        unit = self.unit('code', {'main.c': 'int needle;\n'})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'main.c', 1, 1)]
        direct = self.direct(unit)
        hit = direct.search('needle')['excerpts'][0]
        self.assertFalse(hit['source_changed'])
        snapshot = self.registry.data_dir / 'sources' / unit['unit_id'] / 'files/main.c'
        snapshot.write_text('int forged;\n', encoding='utf-8')
        with self.assertRaises(RegistryError):
            direct.resolve([hit['id']])

    def test_resolve_bounds_and_missing_working_checkout_are_explicit(self):
        unit = self.unit('code', {'main.c': 'int needle;\n'})
        direct = self.direct(unit)
        (self.root / 'code/main.c').unlink()
        hit = direct.read_many([{'source_id': 'code', 'path': 'main.c', 'start': 1, 'end': 1}])['excerpts'][0]
        self.assertTrue(hit['source_changed'])
        self.assertEqual(hit['working_state'], 'missing')
        with self.assertRaisesRegex(SearchError, '80'):
            direct.resolve([hit['id']] * 81)

    def test_markdown_section_expansion_and_explicit_eof_read(self):
        text = '# Intro\nfirst\n# Details\nneedle\nmore\n# Next\ntail\n'
        unit = self.unit('docs', {'guide.md': text})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'guide.md', 4, 4)]
        direct = self.direct(unit)
        result = direct.search('needle', limit=1)
        self.assertEqual(result['excerpts'][0]['lines'], [3, 5])
        self.assertEqual(result['excerpts'][0]['text'], '# Details\nneedle\nmore\n')
        source = result['excerpts'][0]['source_id']
        last = direct.read_many([{'source_id': source, 'path': 'guide.md', 'start': 7, 'end': 7}])
        self.assertEqual(last['excerpts'][0]['text'], 'tail\n')
        self.assertEqual(last['excerpts'][0]['total_lines'], 7)
        expanded = direct.read_many([result['excerpts'][0]['id']])
        self.assertEqual(expanded['excerpts'][0]['id'], result['excerpts'][0]['id'])

    def test_diversity_covers_code_tests_docs_reports_and_source_filter_is_preretrieval(self):
        code = self.unit('code', {'main.c': 'int needle;\n', 'tests/test_main.c': 'int needleTest;\n',
                                  'docs/guide.md': '# needle guide\ninfo\n'})
        report = self.unit('reports', {'obser.md': '# needle failure\nregression\n'}, role='primary_report')
        self.backend.rows[code['unit_id']] = [self.row(code, path, 1, 1) for path in
                                             ('main.c', 'tests/test_main.c', 'docs/guide.md')]
        self.backend.rows[report['unit_id']] = [self.row(report, 'obser.md', 1, 1)]
        direct = self.direct(code, report)
        result = direct.search('needle', limit=4)
        self.assertEqual({hit['role'] for hit in result['excerpts']}, {'code', 'tests', 'docs', 'reports'})
        self.backend.calls.clear()
        filtered = direct.search('needle', source_ids=['reports'], roles=['primary_report'])
        self.assertEqual([call[0] for call in self.backend.calls], [report['unit_id']])
        self.assertEqual({hit['role'] for hit in filtered['excerpts']}, {'reports'})
        self.backend.calls.clear()
        self.assertEqual(direct.search('needle', source_ids=[])['excerpts'], [])
        self.assertEqual(self.backend.calls, [])
        self.assertEqual({hit['role'] for hit in direct.search('needle', roles=['tests'])['excerpts']}, {'tests'})

    def test_hard_total_payload_budget_skips_unfittable_lines_without_forging_coordinates(self):
        unit = self.unit('long', {'long.md': '# heading\n' + 'needle' * 1000 + '\n'})
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'long.md', 2, 2)]
        result = self.direct(unit).search('needle', max_chars=512)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 512)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['excerpts'], [])

    def test_explicit_read_line_limit_and_character_budget_do_not_publish_skipped_ids(self):
        unit = self.unit('lines', {'many.txt': ''.join(f'line {number}\n' for number in range(1, 122))})
        direct = self.direct(unit)
        request = {'source_id': 'lines', 'path': 'many.txt', 'start': 1, 'end': 120}
        full = direct.read_many([request])
        identifier = full['excerpts'][0]['id']
        self.assertEqual(full['excerpts'][0]['lines'], [1, 120])
        empty_service = DirectService(self.config, 'fixture')
        bounded = empty_service.read_many([request], max_chars=512)
        self.assertTrue(bounded['truncated'])
        self.assertEqual(bounded['excerpts'], [])
        with self.assertRaisesRegex(SearchError, 'forged'):
            empty_service.resolve([identifier])
        with self.assertRaisesRegex(SearchError, '120'):
            direct.read_many([{**request, 'end': 121}])

    def test_forged_ids_unknown_scope_traversal_and_invalid_coordinates_fail(self):
        unit = self.unit('code', {'main.c': 'int needle;\n'})
        direct = self.direct(unit)
        for request in ('e-forged', {'id': 'e-forged'},
                        {'source_id': 'foreign', 'path': 'main.c', 'start': 1, 'end': 1},
                        {'source_id': 'code', 'path': '../secret', 'start': 1, 'end': 1},
                        {'source_id': 'code', 'path': 'main.c', 'start': 1, 'end': 2},
                        {'source_id': 'code', 'path': 'main.c', 'start': 0, 'end': 1},
                        {'id': 'e-forged', 'source_sha256': 'f'*64}):
            with self.subTest(request=request), self.assertRaises(RegistryError):
                direct.read_many([request])
        with self.assertRaises(SearchError):
            direct.search('needle', source_ids=['foreign'])
        with self.assertRaises(SearchError):
            direct.search('needle', roles=['imaginary'])
        with self.assertRaises(SearchError):
            direct.resolve(['e-forged'])

    def test_republished_composition_does_not_switch_pinned_sources_and_tamper_fails(self):
        old = self.unit('code', {'main.c': 'int oldNeedle;\n'})
        direct = self.direct(old)
        old_hash = direct.composition['composition_hash']
        new = self.unit('code', {'main.c': 'int newNeedle;\n'})
        self.registry.compose('fixture', [new['unit_id']], require_ready=True)
        result = direct.read_many([{'source_id': 'code', 'path': 'main.c', 'start': 1, 'end': 1}])
        self.assertEqual(result['composition_hash'], old_hash)
        self.assertEqual(result['excerpts'][0]['text'], 'int oldNeedle;\n')
        snapshot = self.registry.data_dir / 'sources' / old['unit_id'] / 'files/main.c'
        snapshot.write_text('corrupted', encoding='utf-8')
        with self.assertRaises(RegistryError):
            direct.read_many([result['excerpts'][0]['id']])

    def test_backend_forged_content_and_path_filter_violation_fail_closed(self):
        unit = self.unit('code', {'a.c': 'int needle;\n', 'b.c': 'int other;\n'})
        direct = self.direct(unit)
        self.backend.rows[unit['unit_id']] = [{**self.row(unit, 'a.c', 1, 1), 'content': 'invented text'}]
        with self.assertRaisesRegex(SearchError, 'does not match'):
            direct.search('needle')
        self.backend.rows[unit['unit_id']] = [self.row(unit, 'b.c', 1, 1)]
        with self.assertRaisesRegex(SearchError, 'path filter'):
            direct.search('needle', paths=['a.c'])


if __name__ == '__main__':
    unittest.main()
