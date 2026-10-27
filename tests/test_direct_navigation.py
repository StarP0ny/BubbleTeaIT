"""Navigation stays bounded and references only exact pinned snapshot files."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bubbleteait.direct import DirectService
from bubbleteait.direct_catalog import list_files, list_sources, read_error_hint
from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
from bubbleteait.search import SearchError


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


class Backend:
    def search(self, unit_id, query, *, limit, paths):
        raise AssertionError('Navigation must not invoke embeddings')


class NavigationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = Registry(self.root / 'data')
        self.recipe = IndexRecipe('cocoindex-code', 'fixture', 'fixed', 'fixture')
        self.config = SimpleNamespace(data_dir=self.registry.data_dir)
        patcher = patch('bubbleteait.direct.make_backend', return_value=Backend())
        patcher.start()
        self.addCleanup(patcher.stop)

    def unit(self, source, files, role='unknown'):
        checkout = self.root / source
        checkout.mkdir()
        for name, value in files.items():
            path = checkout / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value.encode('utf-8') if isinstance(value, str) else value)
        self.registry.register(SourceSpec(source, 'module', checkout, role=role))
        manifest = self.registry.snapshot(source, self.recipe)
        status = self.registry.data_dir / 'indexes' / manifest['unit_id'] / 'status.json'
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps({'schema_version': 1, 'unit_id': manifest['unit_id'],
            'recipe_hash': manifest['recipe_hash'], 'backend': 'cocoindex-code', 'status': 'ready'}), encoding='utf-8')
        return manifest

    def direct(self, *units, configuration=None):
        self.registry.compose('fixture', [u['unit_id'] for u in units],
                              configuration=configuration, require_ready=True)
        return DirectService(self.config, 'fixture')

    def test_same_paths_keep_distinct_sources_and_return_executable_windows(self):
        alpha = self.unit('alpha', {'src/control.c': 'int alpha;\r\n' * 150})
        beta = self.unit('beta', {'src/control.c': 'int beta;\n'})
        direct = self.direct(alpha, beta)
        response = list_files(direct, query='CONTROL.C')
        self.assertEqual(response['total'], 2)
        self.assertEqual([r['source'] for r in response['entries']], ['alpha', 'beta'])
        self.assertEqual([r['total_lines'] for r in response['entries']], [150, 1])
        self.assertNotEqual(*[r['source_id'] for r in response['entries']])
        request = response['entries'][0]['read']
        self.assertEqual(request, {'source_id': direct.aliases['alpha'], 'path': 'src/control.c', 'start': 1, 'end': 120})
        excerpt = direct.read_many([request])['excerpts'][0]
        self.assertEqual(excerpt['lines'], [1, 120])
        self.assertEqual(excerpt['text'], 'int alpha;\r\n' * 120)

    def test_source_aliases_roles_and_glob_filters_have_one_scope(self):
        alpha = self.unit('alpha', {'src/main.c': 'int alpha;', 'tests/test.c': 'int test;',
                              'docs/guide.md': '# Guide\n'}, role='application')
        beta = self.unit('beta', {'src/main.c': 'int beta;'}, role='application')
        direct = self.direct(alpha, beta)
        self.assertEqual([r['source'] for r in list_sources(direct, roles=['tests'])['entries']], ['alpha'])
        response = list_files(direct, source_ids=[direct.aliases['alpha']], paths=['tests/*.c'], roles=['tests'])
        self.assertEqual([r['path'] for r in response['entries']], ['tests/test.c'])
        self.assertEqual(response['entries'][0]['role'], 'tests')
        self.assertEqual(len(list_files(direct, source_ids=['alpha'], roles=['application'])['entries']), 3)
        self.assertEqual(list_files(direct, paths=['TESTS/*.c'])['total'], 0)

    def test_configuration_is_declared_not_inferred_or_changed(self):
        unit = self.unit('app', {'src/main.c': '#define BUILD B\n'})
        direct = self.direct(unit)
        self.assertEqual(list_sources(direct)['configuration_status'], 'unknown')
        self.assertIsNone(list_sources(direct)['configuration'])
        direct = self.direct(unit, configuration={'BUILD': 'B', 'flags': ['LOCAL']})
        response = list_files(direct)
        self.assertEqual(response['configuration_status'], 'declared')
        self.assertEqual(response['configuration'], {'BUILD': 'B', 'flags': ['LOCAL']})
        response['configuration']['flags'].append('OTHER')
        self.assertEqual(direct.composition['configuration']['flags'], ['LOCAL'])

    def test_budget_and_limit_pagination_never_skip_unreturned_entry(self):
        unit = self.unit('source', {f'src/file-{i:02}.c': f'int value{i};\n' for i in range(12)})
        direct = self.direct(unit)
        with patch.object(direct.service, '_snapshot_file', wraps=direct.service._snapshot_file) as read:
            first = list_files(direct, limit=2)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(first['next_offset'], 2)
        self.assertTrue(first['truncated'])
        offset, paths = 0, []
        while True:
            response = list_files(direct, offset=offset, limit=9, max_chars=1400)
            self.assertLessEqual(size(response), 1400)
            paths.extend(row['path'] for row in response['entries'])
            cursor = response['next_offset']
            if cursor is None:
                break
            self.assertGreater(cursor, offset)
            offset = cursor
        self.assertEqual(paths, [f'src/file-{i:02}.c' for i in range(12)])
        self.assertEqual(list_files(direct, offset=12)['entries'], [])
        self.assertFalse(list_files(direct, offset=12)['truncated'])

    def test_source_catalog_paginates_and_filters_literal_query(self):
        units = [self.unit(f'app-{i}', {'main.c': 'int a;'}) for i in range(6)]
        direct = self.direct(*units)
        first = list_sources(direct, limit=2)
        self.assertEqual([r['source'] for r in first['entries']], ['app-0', 'app-1'])
        self.assertEqual(first['next_offset'], 2)
        self.assertEqual(list_sources(direct, query='APP-4')['entries'][0]['source'], 'app-4')
        self.assertEqual(list_sources(direct, query='app-?')['total'], 0)
        self.assertEqual(list_sources(direct, roles=[])['total'], 0)
        self.assertEqual(list_files(direct, source_ids=[])['total'], 0)

    def test_invalid_parameters_and_unknown_sources_fail_before_reading(self):
        direct = self.direct(self.unit('app', {'main.c': 'int a;'}))
        for arguments in ({'offset': True}, {'offset': -1}, {'offset': 3}, {'limit': 0},
                          {'limit': True}, {'limit': 201}, {'max_chars': 511},
                          {'max_chars': 64001}, {'query': ''}, {'query': []}):
            with self.subTest(arguments=arguments), self.assertRaises(SearchError):
                list_files(direct, **arguments)
        for arguments in ({'source_ids': ['absent']}, {'roles': ['absent']},
                          {'paths': ['']}, {'source_ids': 'app'}):
            with self.subTest(arguments=arguments), self.assertRaises(SearchError):
                list_files(direct, **arguments)
        smallest = list_files(direct, max_chars=512)
        self.assertLessEqual(size(smallest), 512)
        self.assertEqual(len(smallest['entries']), smallest['total'])

    def test_large_configuration_cannot_escape_response_budget(self):
        unit = self.unit('app', {'main.c': 'int a;'})
        direct = self.direct(unit, configuration={'description': 'x' * 1000})
        with self.assertRaisesRegex(SearchError, 'configuration'):
            list_sources(direct, max_chars=512)

    def test_empty_and_binary_files_have_explicit_readability(self):
        unit = self.unit('app', {'empty.c': '', 'binary.bin': b'\xff\xfe', 'one.c': 'int a;'})
        direct = self.direct(unit)
        rows = {r['path']: r for r in list_files(direct)['entries']}
        self.assertEqual(rows['empty.c']['total_lines'], 1)
        self.assertEqual(rows['empty.c']['read']['end'], 1)
        self.assertFalse(rows['binary.bin']['text_readable'])
        self.assertIsNone(rows['binary.bin']['read'])

    def test_hints_use_only_actual_source_paths_and_bounded_read_windows(self):
        unit = self.unit('app', {'trunk/config/configuration.h.in': '#define X 1\n',
                                 'trunk/Src/control.c': '// row\r\n' * 500})
        direct = self.direct(unit)
        hint = json.loads(read_error_hint(direct, 'app', 'trunk/Src/configuration.h'))
        self.assertIn('trunk/config/configuration.h.in', hint['available_paths'])
        self.assertNotIn('trunk/Src/configuration.h', hint['available_paths'])
        windows = json.loads(read_error_hint(direct, 'app', 'trunk/Src/control.c', 110, 244))
        self.assertEqual([(r['start'], r['end']) for r in windows['read_windows']], [(110, 229), (230, 244)])
        for request in windows['read_windows']:
            direct.read_many([request])
        longer = json.loads(read_error_hint(direct, 'app', 'trunk/Src/control.c', 1, 500))
        self.assertEqual(longer['next_start'], 361)
        self.assertLessEqual(len(read_error_hint(direct, 'app', 'missing.c')), 1600)
        self.assertEqual(read_error_hint(direct, 'unknown', 'missing.c'), '')

    def test_navigation_detects_snapshot_corruption_without_following_working_checkout(self):
        unit = self.unit('app', {'src/main.c': 'int original;\n'})
        direct = self.direct(unit)
        (self.root / 'app/src/main.c').write_text('int working_change;\n')
        row = list_files(direct)['entries'][0]
        self.assertEqual(row['total_lines'], 1)
        path = self.registry.data_dir / 'sources' / unit['unit_id'] / 'files/src/main.c'
        path.write_text('int corrupt;\n')
        with self.assertRaisesRegex(SearchError, 'hash'):
            list_files(direct)


if __name__ == '__main__':
    unittest.main()
