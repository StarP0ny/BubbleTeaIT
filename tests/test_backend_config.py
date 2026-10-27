from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

from bubbleteait.config import CocoIndexConfig, Config, ConfigError, ModelConfig, load_config
from bubbleteait.doctor import run_doctor


class BackendConfigTests(unittest.TestCase):
    def test_backend_python_path_belongs_to_defining_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'configs').mkdir()
            (root / 'configs/bti.toml').write_text(
                '[backends.cocoindex]\npython_executable="../env/python"\nmodel_profile="e5"\n', encoding='utf-8')
            config = load_config(root, env={})
            self.assertEqual(config.cocoindex.python_executable, str(root / 'env/python'))
            self.assertEqual(config.cocoindex.model_profile, 'e5')

    def test_backend_options_and_timeout_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.local').mkdir()
            config = root / '.local/config.toml'
            for body in ('[backends]\nunknown={}',
                         '[backends.cocoindex]\npython_executable="python"\ntimeout_seconds=0',
                         '[backends.cocoindex]\npython_executable="python"\ntimeout_seconds=true'):
                with self.subTest(body=body):
                    config.write_text(body, encoding='utf-8')
                    with self.assertRaises(ConfigError):
                        load_config(root, env={})

    def test_doctor_fails_for_missing_python_unknown_or_disabled_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ModelConfig("e5", root, enabled=True)
            config = Config(root, root, root, models=(model,), cocoindex=CocoIndexConfig(sys.executable, "e5"))
            cases = (
                (replace(config, cocoindex=CocoIndexConfig("bti-absent-worker-python", "e5")), "executable is missing"),
                (replace(config, cocoindex=CocoIndexConfig(sys.executable, "unknown")), "Unknown model profile"),
                (replace(config, models=(replace(model, enabled=False),)), "profile is disabled"),
            )
            for candidate, expected in cases:
                with self.subTest(expected=expected):
                    report = run_doctor(candidate, env={})
                    check = next(check for check in report.checks if check.name == "backend:cocoindex")
                    self.assertEqual(check.status, "error")
                    self.assertIn(expected, check.detail)
                    self.assertFalse(report.ok)

    def test_selected_optional_model_is_required_by_backend_and_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            model_path.mkdir()
            (model_path / "config.json").write_bytes(b"{}")
            model = ModelConfig("e5", model_path, required=False, files=("config.json", "model.safetensors"))
            config = Config(root, root, root, models=(model,), cocoindex=CocoIndexConfig(sys.executable, "e5"))
            missing = run_doctor(config, env={})
            statuses = {check.name: check.status for check in missing.checks}
            self.assertEqual(statuses["model:e5"], "warning")
            self.assertEqual(statuses["backend:cocoindex"], "error")
            self.assertFalse(missing.ok)
            (model_path / "model.safetensors").write_bytes(b"fixture")
            healthy = run_doctor(config, env={})
            self.assertTrue(healthy.ok)
            check = next(check for check in healthy.checks if check.name == "backend:cocoindex")
            self.assertEqual(check.status, "ok")
            self.assertIn("model loading not tested", check.detail)


if __name__ == '__main__':
    unittest.main()
