"""Configuration/CLI integration stays model-free, including mixed old units."""
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.application import make_backend, qwen_model
from bubbleteait.cli import _recipe, main
from bubbleteait.config import (CocoIndexConfig, Config, ConfigError, ModelConfig,
                               QwenConfig, load_config)
from bubbleteait.doctor import run_doctor


REPOSITORY = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


class QwenConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_path = self.root / "models/qwen"
        self.model_path.mkdir(parents=True)
        self.manifest = self.model_path / "bti-model-manifest.json"
        self.write_manifest()
        self.config = Config(self.root, self.root / "data", self.root / "models",
                             models=(ModelConfig("qwen-0.6b", self.model_path),),
                             qwen=QwenConfig(sys.executable))

    def write_manifest(self, **updates):
        self.manifest.write_text(json.dumps(dict(status="complete", repository=REPOSITORY,
                                                 revision=REVISION) | updates), encoding="utf-8")

    def test_legacy_configs_have_no_implicit_qwen_and_positional_fields_stay_compatible(self):
        old = Config(self.root, self.root, self.root, (), None, (), (), CocoIndexConfig("python"))
        self.assertIsNone(old.qwen)
        self.assertEqual(old.cocoindex.model_profile, "e5-small")
        self.assertIsNone(load_config(self.root, env={}).qwen)

    def test_qwen_paths_defaults_and_both_backends_are_preserved(self):
        (self.root / "configs").mkdir()
        (self.root / ".local").mkdir()
        (self.root / "configs/bti.toml").write_text(
            '[backends.qwen]\npython_executable="../env/python"\n'
            '[backends.cocoindex]\npython_executable="python"\n', encoding="utf-8")
        (self.root / ".local/config.toml").write_text(
            '[backends.qwen]\ntimeout_seconds=90\n[models."qwen-0.6b"]\npath="../models/qwen"\n',
            encoding="utf-8")
        config = load_config(self.root, env={})
        self.assertEqual(config.qwen.python_executable, str(self.root / "env/python"))
        self.assertEqual(config.qwen.model_profile, "qwen-0.6b")
        self.assertEqual(config.qwen.timeout_seconds, 90)
        self.assertEqual(config.models[0].path, self.model_path)
        self.assertEqual(config.cocoindex.python_executable, "python")

    def test_rejects_unknown_qwen_settings_and_bad_types(self):
        (self.root / "configs").mkdir()
        path = self.root / "configs/bti.toml"
        for options in ('device="cuda"', 'timeout_seconds=0', 'timeout_seconds=true',
                        'timeout_seconds=3601', 'model_profile=false', 'python_executable=""'):
            with self.subTest(options=options):
                prefix = '' if options.startswith('python_executable') else 'python_executable="python"\n'
                path.write_text('[backends.qwen]\n' + prefix + options, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(self.root, env={})

    def test_default_recipe_is_fixed_and_requires_correct_complete_local_identity(self):
        from bubbleteait.backends.qwen import QWEN_OPTIONS
        recipe = _recipe(self.config)
        self.assertEqual(recipe.backend, "qwen-dense")
        self.assertEqual(recipe.chunker, "bti-lines-v1")
        self.assertEqual(recipe.embedding_model, REPOSITORY)
        self.assertEqual(recipe.embedding_revision, REVISION)
        self.assertEqual(recipe.options, QWEN_OPTIONS)
        for changes in ({"repository": "Qwen/Qwen3-Embedding-4B"},
                        {"revision": "wrong"}, {"status": "downloading"}):
            with self.subTest(changes=changes):
                self.write_manifest(**changes)
                with self.assertRaises(ConfigError):
                    _recipe(self.config)

    def test_qwen_paths_disabled_profiles_and_unreadable_manifests_fail_before_runtime(self):
        candidates = (
            replace(self.config, models=()),
            replace(self.config, models=(replace(self.config.models[0], enabled=False),)),
            replace(self.config, models_dir=self.root / "elsewhere"),
            replace(self.config, qwen=None),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ConfigError):
                    qwen_model(candidate)
        self.manifest.write_text("not-json", encoding="utf-8")
        with self.assertRaises(ConfigError):
            qwen_model(self.config)

    def test_explicit_recipe_is_preserved_but_qwen_options_must_match(self):
        from dataclasses import asdict
        recipe = _recipe(self.config)
        filename = self.root / "recipe.json"
        filename.write_text(json.dumps(asdict(recipe)), encoding="utf-8")
        self.assertEqual(_recipe(self.config, filename), recipe)
        data = asdict(recipe)
        data["options"] = {}
        filename.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            _recipe(self.config, filename)
        legacy = dict(backend="fixture", embedding_model="test", embedding_revision="v1", chunker="fixture")
        filename.write_text(json.dumps(legacy), encoding="utf-8")
        self.assertEqual(_recipe(self.config, filename).backend, "fixture")

    def test_router_uses_unit_recipe_and_does_not_eagerly_touch_unused_qwen(self):
        from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
        registry = Registry(self.config.data_dir)
        source = self.root / "source"
        source.mkdir()
        (source / "control.c").write_text("int value = 1;\n", encoding="utf-8")
        registry.register(SourceSpec("app", "project", source, respect_gitignore=False))
        old = registry.snapshot("app", IndexRecipe("cocoindex-code", "intfloat/multilingual-e5-small",
                                                   "legacy", "cocoindex-code-default@0.2.42"))
        new = registry.snapshot("app", _recipe(self.config))
        e5 = ModelConfig("e5-small", self.root / "models/e5")
        config = replace(self.config, cocoindex=CocoIndexConfig(sys.executable),
                         models=(*self.config.models, e5))
        with patch("bubbleteait.backends.cocoindex.CocoIndexBackend") as coco, \
             patch("bubbleteait.backends.qwen.QwenBackend") as qwen:
            coco.return_value.search.return_value = [{"backend": "coco"}]
            qwen.return_value.search.return_value = [{"backend": "qwen"}]
            backend = make_backend(config)
            self.write_manifest(revision="bad-but-unused")
            self.assertEqual(backend.search(old["unit_id"], "query"), [{"backend": "coco"}])
            qwen.assert_not_called()
            self.write_manifest()
            self.assertEqual(backend.search(new["unit_id"], "query", limit=3, paths=["*.c"]),
                             [{"backend": "qwen"}])
            backend.index(old["unit_id"])
            backend.index(new["unit_id"])
            self.assertEqual(coco.call_count, 1)
            self.assertEqual(qwen.call_count, 1)
            qwen.return_value.search.assert_called_once_with(new["unit_id"], "query", limit=3, paths=["*.c"])

    def test_router_rejects_unconfigured_or_unknown_unit_backends(self):
        from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
        registry = Registry(self.config.data_dir)
        source = self.root / "source"
        source.mkdir()
        (source / "a.txt").write_text("fixture", encoding="utf-8")
        registry.register(SourceSpec("notes", "documents", source, respect_gitignore=False))
        for name in ("cocoindex-code", "unknown"):
            unit = registry.snapshot("notes", IndexRecipe(name, "fixture", "v1", "fixture"))
            with self.subTest(name=name), self.assertRaises(ConfigError):
                make_backend(self.config).index(unit["unit_id"])

    def test_worker_cli_routes_control_without_running_real_worker(self):
        with patch.dict(os.environ, {}, clear=True):
            (self.root / ".local").mkdir()
            (self.root / ".local/config.toml").write_text(
                '[paths]\ndata_dir="../data"\nmodels_dir="../models"\n[models."qwen-0.6b"]\npath="../models/qwen"\n'
                '[backends.qwen]\npython_executable=' + json.dumps(Path(sys.executable).as_posix()) + '\n',
                encoding="utf-8")
            for command, action in (("status", "health"), ("load", "load"),
                                    ("unload", "unload"), ("stop", "shutdown")):
                with self.subTest(command=command), \
                     patch("bubbleteait.qwen_runtime.QwenRuntimeClient") as client, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    client.return_value.request.return_value = {"ok": True, "worker_running": False}
                    code = main(["--project-dir", str(self.root), "worker", command, "--json"])
                    self.assertEqual(code, 0, output.getvalue())
                    self.assertTrue(json.loads(output.getvalue())["ok"])
                    client.return_value.request.assert_called_once_with(action)
                    self.assertEqual(client.call_args.args[1], str(Path(sys.executable)))

    def test_cli_snapshot_uses_default_qwen_recipe_without_starting_worker(self):
        with patch.dict(os.environ, {}, clear=True):
            (self.root / ".local").mkdir()
            (self.root / ".local/config.toml").write_text(
                '[paths]\ndata_dir="../data"\nmodels_dir="../models"\n'
                '[models."qwen-0.6b"]\npath="../models/qwen"\n'
                '[backends.qwen]\npython_executable=' + json.dumps(Path(sys.executable).as_posix()) + '\n',
                encoding="utf-8")
            source = self.root / "source"
            source.mkdir()
            (source / "control.c").write_text("int value = 7;\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--project-dir", str(self.root), "source", "add", "app",
                    "--kind", "project", "--path", str(source), "--no-respect-gitignore", "--json"]), 0)
            with patch("bubbleteait.qwen_runtime.QwenRuntimeClient") as client, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                code = main(["--project-dir", str(self.root), "index", "app", "--snapshot-only", "--json"])
                self.assertEqual(code, 0, output.getvalue())
                snapshot = json.loads(output.getvalue())
                self.assertEqual(snapshot["recipe"]["backend"], "qwen-dense")
                self.assertEqual(snapshot["recipe"]["embedding_model"], REPOSITORY)
                client.assert_not_called()

    def test_worker_cli_missing_configuration_reports_json_error(self):
        with patch.dict(os.environ, {"BTI_DATA_DIR": str(self.root / "data")}, clear=True), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(["--project-dir", str(self.root), "worker", "status", "--json"])
            self.assertEqual(code, 2)
            self.assertFalse(json.loads(output.getvalue())["ok"])
            self.assertIn("backends.qwen", output.getvalue())

    def test_doctor_validates_fixed_identity_without_starting_client(self):
        with patch("bubbleteait.qwen_runtime.QwenRuntimeClient") as client:
            healthy = run_doctor(self.config, env={})
            self.assertEqual(next(c.status for c in healthy.checks if c.name == "backend:qwen"), "ok")
            self.write_manifest(revision="unknown")
            bad = run_doctor(self.config, env={})
            self.assertEqual(next(c.status for c in bad.checks if c.name == "backend:qwen"), "error")
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
