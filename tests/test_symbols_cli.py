"""Symbols CLI awaits the navigation service without cloud or embedding setup."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bubbleteait.cli import main
from bubbleteait.symbols import SymbolError


class SymbolsCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root/'.local').mkdir()
        (self.root/'.local/config.toml').write_text('[paths]\ndata_dir="../data"\n', encoding='utf-8')
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def run_cli(self, *arguments):
        argv = ['--project-dir', str(self.root), 'symbols', '--project','alpha',
                '--source-id','alpha-app', '--path','src/control.c', *arguments, '--json']
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(argv)
        return code, json.loads(output.getvalue())

    def service(self):
        service = MagicMock()
        service.query = AsyncMock(return_value={'navigation_only':True, 'compile_context':'unknown',
            'results':[{'source':'alpha-app', 'path':'src/control.c', 'name_path':'heatingAllowed',
                        'read':{'source_id':'s-alpha-fixture', 'path':'src/control.c', 'start':1, 'end':3}}]})
        return service

    def test_default_definition_awaits_the_service_and_preserves_expected_source_path(self):
        service = self.service()
        with patch('bubbleteait.symbols.SymbolService', return_value=service) as constructor:
            code, result = self.run_cli('--name','heatingAllowed')
        self.assertEqual(code, 0)
        config, composition = constructor.call_args.args
        self.assertEqual(composition, 'alpha')
        self.assertIsNone(config.llm)
        self.assertIsNone(config.cocoindex)
        self.assertIsNone(config.qwen)
        service.query.assert_awaited_once_with('definition','alpha-app','src/control.c',
            name='heatingAllowed', limit=20, max_chars=8000)
        self.assertEqual((result['results'][0]['source'], result['results'][0]['path']),
                         ('alpha-app', 'src/control.c'))
        self.assertEqual(result['results'][0]['read']['start'], 1)
        self.assertTrue(result['navigation_only'])
        self.assertEqual(result['compile_context'], 'unknown')

    def test_overview_and_references_forward_explicit_options(self):
        for operation, extra, expected_name in (
                ('overview', [], None), ('references', ['--name','heatingAllowed'], 'heatingAllowed')):
            with self.subTest(operation=operation):
                service = self.service()
                with patch('bubbleteait.symbols.SymbolService', return_value=service):
                    code, result = self.run_cli('--operation',operation, '--limit','3', '--max-chars','1400', *extra)
                self.assertEqual(code, 0)
                self.assertEqual(result['results'][0]['path'], 'src/control.c')
                service.query.assert_awaited_once_with(operation,'alpha-app','src/control.c',
                    name=expected_name, limit=3, max_chars=1400)

    def test_bad_parser_arguments_cannot_construct_a_navigation_service(self):
        with patch('bubbleteait.symbols.SymbolService', side_effect=AssertionError('Bad CLI args reached sources')) as constructor:
            for arguments in (('--operation','edit'), ('--limit','three'), ('--max-chars','false')):
                with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    self.run_cli(*arguments)
                self.assertEqual(raised.exception.code, 2)
            constructor.assert_not_called()

    def test_async_domain_failure_is_a_structured_cli_error(self):
        service = self.service()
        service.query.side_effect = SymbolError('Unknown pinned path; suggested_paths=[src/control.c]')
        with patch('bubbleteait.symbols.SymbolService', return_value=service):
            code, result = self.run_cli('--name','heatingAllowed')
        self.assertEqual(code, 2)
        self.assertFalse(result['ok'])
        self.assertEqual(result['checks'][0]['status'], 'error')
        self.assertIn('suggested_paths=[src/control.c]', json.dumps(result))
        service.query.assert_awaited_once()

    def test_missing_config_reports_the_required_symbols_settings_without_inference(self):
        with patch('bubbleteait.application.make_backend', side_effect=AssertionError('Symbols loaded embeddings')) as backend, \
             patch('bubbleteait.serena_workspace.probe_symbols', new_callable=AsyncMock) as probe:
            code, result = self.run_cli('--name','heatingAllowed')
        self.assertEqual(code, 2)
        self.assertFalse(result['ok'])
        self.assertIn('symbols.executable', json.dumps(result))
        self.assertIn('symbols.clangd_path', json.dumps(result))
        backend.assert_not_called()
        probe.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
