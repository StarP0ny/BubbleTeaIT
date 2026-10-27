"""Navigation CLI discovers exact snapshot files without an ML runtime."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.cli import main
from bubbleteait.registry import IndexRecipe, Registry, SourceSpec


class DirectNavigationCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root/'.local').mkdir()
        (self.root/'.local/config.toml').write_text(
            '[paths]\ndata_dir="../data"\n[backends.cocoindex]\n'
            'python_executable="missing-navigation-ml-runtime"\n', encoding='utf-8')
        registry = Registry(self.root/'data')
        recipe = IndexRecipe('cocoindex-code', 'fixture', 'fixed', 'fixture')
        units = []
        for source in ('alpha', 'beta'):
            checkout = self.root/source
            (checkout/'src').mkdir(parents=True)
            (checkout/'src/control.c').write_text('int value = 37;\n' * 150, encoding='utf-8')
            registry.register(SourceSpec(source, 'module', checkout, role='code'))
            manifest = registry.snapshot(source, recipe)
            status = registry.data_dir/'indexes'/manifest['unit_id']/'status.json'
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({'schema_version': 1, 'unit_id': manifest['unit_id'],
                'recipe_hash': manifest['recipe_hash'], 'backend': 'cocoindex-code', 'status': 'ready'}), encoding='utf-8')
            units.append(manifest['unit_id'])
        registry.compose('products', units, require_ready=True, configuration={'product': 'declared-example'})
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, command, *args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(['--project-dir', str(self.root), command, '--project', 'products', *args, '--json'])
        return code, json.loads(output.getvalue())

    def test_alias_pagination_filters_and_read_windows_without_ml(self):
        with patch('bubbleteait.application.ConfiguredBackend._for_unit', side_effect=AssertionError('Navigation touched ML')):
            code, first = self.run_cli('list-sources', '--limit', '1')
            self.assertEqual(code, 0)
            self.assertEqual(first['configuration_status'], 'declared')
            self.assertEqual(first['configuration'], {'product': 'declared-example'})
            self.assertEqual(first['next_offset'], 1)
            code, second = self.run_cli('list-sources', '--offset', str(first['next_offset']), '--limit', '1')
            self.assertEqual(code, 0)
            self.assertNotEqual(first['entries'][0]['source_id'], second['entries'][0]['source_id'])
            self.assertIsNone(second['next_offset'])
            alias = second['entries'][0]['source_id']
            code, files = self.run_cli('list-files', '--source-id', alias, '--query', 'CONTROL', '--path', 'src/*.c')
            self.assertEqual(code, 0)
            self.assertEqual(files['total'], 1)
            entry = files['entries'][0]
            self.assertEqual(entry['source_id'], alias)
            self.assertEqual(entry['path'], 'src/control.c')
            self.assertEqual(entry['total_lines'], 150)
            self.assertLessEqual(entry['read']['end'] - entry['read']['start'] + 1, 120)

    def test_invalid_bounds_and_foreign_scope_return_structured_errors(self):
        for command, args in (('list-sources', ('--offset', '-1')),
                              ('list-files', ('--limit', '201')),
                              ('list-files', ('--max-chars', '511')),
                              ('list-files', ('--source-id', 'foreign'))):
            with self.subTest(command=command, args=args):
                code, result = self.run_cli(command, *args)
                self.assertEqual(code, 2)
                self.assertFalse(result['ok'])
                self.assertEqual(result['checks'][0]['status'], 'error')


if __name__ == '__main__':
    unittest.main()
