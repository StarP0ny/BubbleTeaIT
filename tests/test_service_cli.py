import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bubbleteait.cli import main


class ServiceCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'bti.toml').write_text('schema_version=1\n[component]\nname="app"\n[source]\ninclude=["*.c"]\n')
        (self.root / 'main.c').write_text('int main;\n')
        self.client = MagicMock()
        self.client.request.return_value = {'state': 'pending', 'project': 'app', 'path': str(self.root)}
        mock = patch('bubbleteait.service_cli.ServiceClient', return_value=self.client)
        mock.start()
        self.addCleanup(mock.stop)
        env = patch.dict(os.environ, {'PATH': os.environ['PATH'], 'BTI_DATA_DIR': str(self.root / 'data')}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def run_cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()) as result, contextlib.redirect_stderr(io.StringIO()):
            code = main(args)
        return code, result.getvalue()

    def test_watch_path_is_exact_and_json_is_only_output_format(self):
        code, output = self.run_cli(['watch', '-p', str(self.root), '--json'])
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output)['project'], 'app')
        self.client.request.assert_called_once_with('register', path=str(self.root))
        self.client.reset_mock()
        code, output = self.run_cli(['watch', '-p', str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn('state: pending', output)
        self.client.request.assert_called_once_with('register', path=str(self.root))

    def test_missing_local_config_does_not_watch_parent(self):
        child = self.root / 'child'
        child.mkdir()
        code, output = self.run_cli(['--project-dir', str(child), 'watch'])
        self.assertEqual(code, 2, output)
        self.client.request.assert_not_called()

    def test_watch_without_path_uses_current_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            code, output = self.run_cli(['watch'])
        finally:
            os.chdir(previous)
        self.assertEqual(code, 0, output)
        self.client.request.assert_called_once_with('register', path=str(self.root))

    def test_list_does_not_register_a_project_or_start_a_model(self):
        self.client.request.return_value = {'state': 'stopped', 'projects': []}
        code, output = self.run_cli(['--project-dir', str(self.root), 'watch', '--list', '--json'])
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output)['state'], 'stopped')
        self.client.request.assert_called_once_with('status')

    def test_mismatched_path_options_are_rejected(self):
        code, output = self.run_cli(['--project-dir', str(self.root / 'other'), 'watch', '-p', str(self.root)])
        self.assertEqual(code, 2, output)
        self.client.request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
