from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.cli import main
from bubbleteait.config import Config, LLMConfig, MCPConfig, ModelConfig
from bubbleteait.doctor import run_doctor


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.enterContext(patch.dict(os.environ, {key: value for key, value in os.environ.items() if key not in {"BTI_DATA_DIR", "BTI_MODELS_DIR"}}, clear=True))
        self.config = Config(self.root, self.root / "data", self.root / "models")

    def test_offline_check_does_not_create_directories_or_contact_services(self):
        config = replace(self.config, llm=LLMConfig("http://localhost:9931/v1"))
        with patch("bubbleteait.doctor.urlopen") as request:
            report = run_doctor(config, env={})
        request.assert_not_called()
        self.assertTrue(report.ok)
        self.assertFalse(config.data_dir.exists())
        self.assertFalse(config.models_dir.exists())
        self.assertEqual(next(check.status for check in report.checks if check.name == "llm"), "skipped")

    def test_required_missing_weights_fails_then_recovers_with_declared_files(self):
        path = self.root / "weights"
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")
        model = ModelConfig("test", path, required=True, files=("config.json", "model.safetensors"))
        config = replace(self.config, models=(model,))
        self.assertFalse(run_doctor(config, env={}).ok)
        (path / "model.safetensors").write_bytes(b"fixture")
        report = run_doctor(config, env={})
        self.assertTrue(report.ok)
        detail = next(check.detail for check in report.checks if check.name == "model:test")
        self.assertIn("runtime loading not tested", detail)

    def test_optional_model_missing_warns_disabled_model_is_skipped(self):
        missing = ModelConfig("optional", self.root / "missing")
        disabled = ModelConfig("disabled", self.root / "missing", required=True, enabled=False)
        report = run_doctor(replace(self.config, models=(missing, disabled)), env={})
        self.assertTrue(report.ok)
        statuses = {check.name: check.status for check in report.checks}
        self.assertEqual(statuses["model:optional"], "warning")
        self.assertEqual(statuses["model:disabled"], "skipped")

    def test_path_that_is_a_file_is_an_error(self):
        path = self.root / "data-file"
        path.write_text("not a directory", encoding="utf-8")
        self.assertFalse(run_doctor(replace(self.config, data_dir=path), env={}).ok)

    def test_missing_api_key_never_leaks_or_attempts_a_request(self):
        config = replace(self.config, llm=LLMConfig("http://localhost/v1", api_key_env="REQUIRED_KEY"))
        with patch("bubbleteait.doctor.urlopen") as request:
            report = run_doctor(config, probe_endpoints=True, env={})
        request.assert_not_called()
        self.assertFalse(report.ok)

    def test_opt_in_probe_checks_models_list_without_generation(self):
        config = replace(self.config, llm=LLMConfig("http://localhost/v1", "test-model", "KEY", 2))
        response = StringIO('{"data":[{"id":"test-model"}]}')
        with patch("bubbleteait.doctor.urlopen", return_value=response) as request:
            report = run_doctor(config, probe_endpoints=True, env={"KEY": "never-print-this"})
        self.assertTrue(report.ok)
        self.assertEqual(request.call_args.args[0].full_url, "http://localhost/v1/models")
        self.assertEqual(request.call_args.kwargs["timeout"], 2)
        self.assertNotIn("never-print-this", json.dumps(report.to_dict()))

    def test_probe_rejects_unlisted_model_and_invalid_payload(self):
        config = replace(self.config, llm=LLMConfig("http://localhost/v1", "wanted"))
        for payload in ('{"data":[{"id":"other"}]}', '{"unexpected":true}', 'broken-json'):
            with self.subTest(payload=payload):
                with patch("bubbleteait.doctor.urlopen", return_value=StringIO(payload)):
                    self.assertFalse(run_doctor(config, probe_endpoints=True, env={}).ok)

    def test_mcp_checks_executable_cwd_environment_without_launch(self):
        servers = (
            MCPConfig("ready", command=sys.executable, required=True),
            MCPConfig("absent", command="bti-definitely-not-installed", required=False),
            MCPConfig("bad-cwd", command=sys.executable, cwd=self.root / "absent", required=True),
            MCPConfig("remote", transport="http", url="http://localhost/mcp", env_names=("TOKEN",)),
        )
        report = run_doctor(replace(self.config, mcp=servers), env={})
        statuses = {check.name: check.status for check in report.checks}
        self.assertEqual(statuses["mcp:ready"], "ok")
        self.assertEqual(statuses["mcp:absent"], "warning")
        self.assertEqual(statuses["mcp:bad-cwd"], "error")
        self.assertEqual(statuses["mcp:remote"], "warning")

    def test_cli_json_errors_and_output_are_machine_readable(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main(["--project-dir", str(self.root), "--config", "missing.toml", "doctor", "--json"])
        self.assertEqual(status, 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        output = StringIO()
        with redirect_stdout(output):
            status = main(["--project-dir", str(self.root), "doctor", "--json"])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())["ok"])

    def init_config(self):
        path = self.root / "config.toml"
        path.write_text('[paths]\ndata_dir="data"\nmodels_dir="weights"\n', encoding="utf-8")
        return ["--project-dir", str(self.root), "--config", str(path), "init", "--json"]

    def test_init_is_idempotent_and_preserves_existing_data(self):
        args = self.init_config()
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args), 0)
        first = json.loads(output.getvalue())
        self.assertTrue(first["ok"])
        self.assertEqual(len(first["directories"]), 8)
        self.assertTrue(all(directory["created"] for directory in first["directories"]))
        evidence = self.root / "data/sources/preserved.txt"
        evidence.write_text("existing source", encoding="utf-8")
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args), 0)
        second = json.loads(output.getvalue())
        self.assertFalse(any(directory["created"] for directory in second["directories"]))
        self.assertEqual(evidence.read_text(encoding="utf-8"), "existing source")

    def test_init_detects_file_collision_before_creating_layout(self):
        args = self.init_config()
        (self.root / "data").mkdir()
        obstacle = self.root / "data/indexes"
        obstacle.write_bytes(b"preserve this file")
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args), 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        self.assertFalse((self.root / "data/sources").exists())
        self.assertEqual(obstacle.read_bytes(), b"preserve this file")

    def test_init_fails_if_parent_of_data_directory_is_a_file(self):
        path = self.root / "config.toml"
        path.write_text('[paths]\ndata_dir="obstacle/nested"\n', encoding="utf-8")
        (self.root / "obstacle").write_bytes(b"existing")
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--project-dir", str(self.root), "--config", str(path), "init", "--json"]), 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])


if __name__ == "__main__":
    unittest.main()
