import os
from pathlib import Path
import tempfile
import unittest

from bubbleteait.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_each_layer_resolves_paths_relative_to_its_own_file(self):
        self.write("configs/bti.toml", '[paths]\ndata_dir="../shared-data"\nmodels_dir="../shared-models"\n[models.e5]\npath="../weights/e5"\nrequired=true\n[llm]\nbase_url="http://localhost:9931/v1"\nmodel="original"\n')
        self.write(".local/config.toml", '[paths]\nmodels_dir="../local-models"\n[models.e5]\nrequired=false\n[llm]\nmodel="override"\n')
        config = load_config(self.root, env={})
        self.assertEqual(config.data_dir, self.root / "shared-data")
        self.assertEqual(config.models_dir, self.root / "local-models")
        self.assertEqual(config.models[0].path, self.root / "weights/e5")
        self.assertFalse(config.models[0].required)
        self.assertEqual(config.llm.model, "override")
        self.assertEqual(len(config.loaded_files), 2)

    def test_environment_overrides_paths_and_default_model_locations(self):
        self.write("configs/bti.toml", '[paths]\nmodels_dir="../ignored"\n[models.e5]\nfiles=["config.json"]\n')
        config = load_config(self.root, env={"BTI_DATA_DIR": "runtime", "BTI_MODELS_DIR": "weights"})
        self.assertEqual(config.data_dir, self.root / "runtime")
        self.assertEqual(config.models_dir, self.root / "weights")
        self.assertEqual(config.models[0].path, self.root / "weights/e5")

    def test_path_environment_expansion_and_undefined_variable(self):
        self.write("configs/bti.toml", '[paths]\ndata_dir="${STORAGE}/data"\nmodels_dir="%STORAGE%/models"\n')
        config = load_config(self.root, env={"STORAGE": str(self.root)})
        self.assertEqual(config.data_dir, self.root / "data")
        self.assertEqual(config.models_dir, self.root / "models")
        with self.assertRaisesRegex(ConfigError, "Undefined environment"):
            load_config(self.root, env={})

    def test_explicit_missing_file_is_an_error_implicit_files_are_optional(self):
        self.assertEqual(load_config(self.root, env={}).loaded_files, ())
        with self.assertRaisesRegex(ConfigError, "Cannot load"):
            load_config(self.root, config_path="missing.toml", env={})

    def test_rejects_typos_wrong_types_and_credentials_in_urls(self):
        invalid = (
            '[paths]\nmodel_dir="wrong-key"',
            '[models.e5]\nrequired="false"',
            '[models.e5]\nfiles=["../escape.bin"]',
            '[models.e5]\nfiles="weights.bin"',
            '[llm]\nbase_url="http://secret:password@localhost/v1"',
            '[llm]\nbase_url="http://localhost/v1"\ntimeout_seconds=-1',
            '[llm]\nbase_url="http://localhost:99999/v1"',
            '[mcp.tool]\ntransport="stdio"',
            '[mcp.tool]\ntransport="unknown"\ncommand="python"',
        )
        for content in invalid:
            with self.subTest(content=content):
                self.write("configs/bti.toml", content)
                with self.assertRaises(ConfigError):
                    load_config(self.root, env={})

    def test_mcp_commands_paths_and_disabled_placeholders(self):
        self.write("configs/bti.toml", '[mcp.tool]\ncommand="../bin/tool"\ncwd=".."\nargs=["serve"]\n[mcp.later]\nenabled=false\n')
        config = load_config(self.root, env={})
        self.assertEqual(config.mcp[0].command, str(self.root / "bin/tool"))
        self.assertEqual(config.mcp[0].cwd, self.root)
        self.assertEqual(config.mcp[0].args, ("serve",))
        self.assertFalse(config.mcp[1].enabled)


if __name__ == "__main__":
    unittest.main()
