import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.cli import main


class SourceCLITests(unittest.TestCase):
    def test_ignore_flags_and_repeatable_attachments_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            (root / "source/artifacts").mkdir(parents=True)
            (root / "source/local").mkdir()
            (root / ".local").mkdir()
            (root / ".local/config.toml").write_text('[paths]\ndata_dir="../data"\n', encoding="utf-8")

            def run(*args):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    code = main(["--project-dir", str(root), "source", "add", "app", "--kind", "project",
                                 "--path", str(root / "source"), *args, "--json"])
                return code, json.loads(output.getvalue())

            code, default = run()
            self.assertEqual(code, 0)
            self.assertTrue(default["respect_gitignore"])
            self.assertEqual(default["attach_paths"], [])
            code, attached = run("--respect-gitignore", "--attach-path", "artifacts", "--attach-path", "local")
            self.assertEqual(code, 0)
            self.assertTrue(attached["respect_gitignore"])
            self.assertEqual(attached["attach_paths"], ["artifacts", "local"])
            code, disabled = run("--no-respect-gitignore")
            self.assertEqual(code, 0)
            self.assertFalse(disabled["respect_gitignore"])

    def test_register_snapshot_status_and_draft_without_embedding_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            (root / 'source').mkdir()
            (root / 'source/main.c').write_bytes(b'int value = 7;\r\n')
            (root / '.local').mkdir()
            (root / '.local/config.toml').write_text('[paths]\ndata_dir="../data"\n', encoding='utf-8')
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps(dict(backend='fixture', embedding_model='test',
                embedding_revision='revision-1', chunker='fixture')), encoding='utf-8')

            def run(*args):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    code = main(['--project-dir', str(root), *args, '--json'])
                return code, json.loads(output.getvalue())

            code, source = run('source', 'add', 'app', '--kind', 'project',
                               '--path', str(root / 'source'), '--include', '**/*.c', '--no-respect-gitignore')
            self.assertEqual(code, 0, source)
            code, unit = run('index', 'app', '--snapshot-only', '--recipe', str(recipe))
            self.assertEqual(code, 0, unit)
            self.assertEqual([file['path'] for file in unit['files']], ['main.c'])
            code, status = run('status', '--unit', unit['unit_id'])
            self.assertEqual(status['status'], 'missing')
            code, rejected = run('compose', 'release', '--unit', unit['unit_id'])
            self.assertNotEqual(code, 0)
            self.assertFalse(rejected['ok'])
            code, draft = run('compose', 'release', '--unit', unit['unit_id'], '--draft')
            self.assertEqual(code, 0)
            self.assertFalse(draft['ready'])


if __name__ == '__main__':
    unittest.main()
