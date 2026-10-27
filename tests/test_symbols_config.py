"""Optional navigation configuration is explicit and independent of models."""
from pathlib import Path
import tempfile
import unittest

from bubbleteait.config import ConfigError, load_config


class SymbolsConfigTests(unittest.TestCase):
    def test_absent_symbols_does_not_enable_any_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(temporary, env={})
            self.assertIsNone(config.symbols)
            self.assertIsNone(config.qwen)

    def test_paths_belong_to_defining_config_and_layers_preserve_other_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'configs').mkdir(); (root/'.local').mkdir()
            (root/'configs/bti.toml').write_text('[symbols]\nexecutable="../runtime/serena"\n'
                'clangd_path="../llvm/clangd"\ntimeout_seconds=60\n',encoding='utf-8')
            (root/'.local/config.toml').write_text('[symbols]\ntimeout_seconds=75\n',encoding='utf-8')
            config = load_config(root,env={})
            self.assertEqual(config.symbols.executable,str(root/'runtime/serena'))
            self.assertEqual(config.symbols.clangd_path,root/'llvm/clangd')
            self.assertEqual(config.symbols.timeout_seconds,75)
            self.assertIsNone(config.cocoindex)

    def test_bad_options_missing_paths_and_wrong_timeouts_are_rejected(self):
        bodies = ('[symbols]\nexecutable="serena"',
                  '[symbols]\nclangd_path="clangd"',
                  '[symbols]\nexecutable="serena"\nclangd_path="clangd"\nunknown=true',
                  *('[symbols]\nexecutable="serena"\nclangd_path="clangd"\ntimeout_seconds='+value
                    for value in ('true','0','601','"120"','nan','inf')))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'.local').mkdir()
            for body in bodies:
                with self.subTest(body=body),self.assertRaises(ConfigError):
                    (root/'.local/config.toml').write_text(body,encoding='utf-8')
                    load_config(root,env={})
