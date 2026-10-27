import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.backends.cocoindex import BackendError, CocoIndexBackend, _unit_lock
from bubbleteait.backends.base import RetrievalBackend
from bubbleteait.registry import IndexRecipe, Registry, RegistryError, SourceSpec


class CocoIndexAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.text = "// Heating remains blocked after a thermal trip.\nint heatingAllowed(int trip) { return !trip; }\n"
        (self.source / "control.c").write_text(self.text, encoding="utf-8")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "weights.bin").write_bytes(b"test-only-weights")
        model_manifest = {"repository": "test/embedding", "revision": "a" * 40, "status": "complete",
                          "files": [{"path": "weights.bin", "sha256": hashlib.sha256(b"test-only-weights").hexdigest()}]}
        (self.model / "bti-model-manifest.json").write_text(json.dumps(model_manifest), encoding="utf-8")
        self.registry = Registry(self.root / "data")
        self.registry.register(SourceSpec("fixture", "module", self.source, include=("**/*.c",)))
        recipe = IndexRecipe("cocoindex-code", "test/embedding", "a" * 40, "cocoindex-code-default@0.2.42")
        self.manifest = self.registry.snapshot("fixture", recipe)
        self.unit_id = self.manifest["unit_id"]
        self.backend = CocoIndexBackend(self.registry.data_dir, sys.executable, self.model, models_dir=self.root)
        self.index_root = self.registry.data_dir / "indexes" / self.unit_id

    @staticmethod
    def completed(payload):
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "diagnostic output")

    def index(self):
        with patch("bubbleteait.backends.cocoindex.subprocess.run", return_value=self.completed({"ok": True, "index": {"total_files": 1}})):
            return self.backend.index(self.unit_id)

    def row(self, **values):
        return {"file_path": "control.c", "language": "c", "start_line": 1, "end_line": 2,
                "content": self.text, "score": 0.9, **values}

    def search(self, row):
        with patch("bubbleteait.backends.cocoindex.subprocess.run", return_value=self.completed({"ok": True, "results": [row]})):
            return self.backend.search(self.unit_id, "thermal trip")

    def test_index_uses_only_whitelisted_snapshot_and_marks_ready(self):
        snapshot = self.registry.data_dir / "sources" / self.unit_id
        (snapshot / "files" / "unlisted.md").write_text("never index this", encoding="utf-8")
        original = snapshot.joinpath("files/control.c").read_bytes()
        status = self.index()
        self.assertEqual(status["status"], "ready")
        self.assertEqual(self.registry.index_status(self.unit_id)["recipe_hash"], self.manifest["recipe_hash"])
        self.assertTrue((self.index_root / "workspace/control.c").is_file())
        self.assertFalse((self.index_root / "workspace/unlisted.md").exists())
        self.assertFalse((snapshot / "files/.cocoindex_code").exists())
        self.assertEqual(snapshot.joinpath("files/control.c").read_bytes(), original)

    def test_search_returns_provenance_and_verified_coordinates(self):
        self.index()
        result = self.search(self.row())[0]
        self.assertEqual(result["unit_id"], self.unit_id)
        self.assertEqual(result["source_sha256"], self.manifest["files"][0]["sha256"])
        self.assertEqual(result["source_id"], "fixture")

    def test_empty_and_whitespace_files_remain_in_snapshot_without_chunks(self):
        (self.source / "empty.h").write_bytes(b"")
        whitespace = b" \t\r\n\n   "
        (self.source / "whitespace.md").write_bytes(whitespace)
        self.registry.register(SourceSpec("fixture", "module", self.source, include=("**/*",)))
        self.manifest = self.registry.snapshot("fixture", IndexRecipe("cocoindex-code", "test/embedding", "a" * 40, "cocoindex-code-default@0.2.42"))
        self.unit_id = self.manifest["unit_id"]
        self.index_root = self.registry.data_dir / "indexes" / self.unit_id
        status = self.index()
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["whitelist_files"], 3)
        self.assertEqual(status["total_files"], 1)
        self.assertEqual(status["empty_count"], 2)
        self.assertEqual(status["zero_chunk_files"], ["empty.h", "whitespace.md"])
        snapshot = self.registry.data_dir / "sources" / self.unit_id / "files"
        self.assertEqual((snapshot / "empty.h").read_bytes(), b"")
        self.assertEqual((snapshot / "whitespace.md").read_bytes(), whitespace)
        self.assertEqual((self.index_root / "workspace/whitespace.md").read_bytes(), whitespace)
        self.assertEqual(self.search(self.row())[0]["file_path"], "control.c")

    def test_invalid_result_path_or_coordinates_are_rejected(self):
        self.index()
        for row in [self.row(file_path="../control.c"), self.row(start_line=0), self.row(end_line=200),
                    self.row(content="hallucinated text"), self.row(score=float("inf"))]:
            with self.subTest(row=row), self.assertRaises(BackendError):
                self.search(row)

    def test_wrong_model_revision_never_launches_worker(self):
        path = self.model / "bti-model-manifest.json"
        value = json.loads(path.read_text())
        value["revision"] = "b" * 40
        path.write_text(json.dumps(value))
        with patch("bubbleteait.backends.cocoindex.subprocess.run") as worker, self.assertRaises(BackendError):
            self.backend.index(self.unit_id)
        worker.assert_not_called()

    def test_changed_model_bytes_are_rejected(self):
        (self.model / "weights.bin").write_bytes(b"unexpected")
        with self.assertRaises(BackendError):
            self.backend.index(self.unit_id)

    def test_changed_snapshot_does_not_reach_worker(self):
        snapshot = self.registry.data_dir / "sources" / self.unit_id / "files/control.c"
        snapshot.write_text("changed")
        with patch("bubbleteait.backends.cocoindex.subprocess.run") as worker, self.assertRaises(RegistryError):
            self.backend.index(self.unit_id)
        worker.assert_not_called()

    def test_partial_index_does_not_become_ready(self):
        with patch("bubbleteait.backends.cocoindex.subprocess.run", return_value=self.completed({"ok": True, "index": {"total_files": 0}})), self.assertRaises(BackendError):
            self.backend.index(self.unit_id)
        self.assertEqual(self.registry.index_status(self.unit_id)["status"], "error")

    def test_worker_failure_records_error_and_search_is_blocked(self):
        failed = subprocess.CompletedProcess([], 1, json.dumps({"ok": False, "error": "model failed"}), "details")
        with patch("bubbleteait.backends.cocoindex.subprocess.run", return_value=failed), self.assertRaisesRegex(BackendError, "model failed"):
            self.backend.index(self.unit_id)
        self.assertEqual(self.registry.index_status(self.unit_id)["status"], "error")
        with self.assertRaises(BackendError):
            self.backend.search(self.unit_id, "anything")

    def test_failed_timeout_cleanup_requires_recovery(self):
        with patch("bubbleteait.backends.cocoindex.subprocess.run", side_effect=subprocess.TimeoutExpired([], 1)), self.assertRaisesRegex(BackendError, "timed out"):
            self.backend.index(self.unit_id)
        self.assertTrue((self.index_root / "runtime/recovery-required.json").exists())
        with self.assertRaisesRegex(BackendError, "recovery"):
            self.backend.index(self.unit_id)

    def test_writer_lock_prevents_overlapping_operations_and_releases(self):
        with _unit_lock(self.index_root / "writer.lock"):
            with self.assertRaisesRegex(BackendError, "busy"):
                self.backend.index(self.unit_id)
        self.assertEqual(self.index()["status"], "ready")

    def test_path_command_is_resolved_and_capabilities_are_explicit(self):
        with patch("bubbleteait.backends.cocoindex.shutil.which", return_value=sys.executable):
            backend = CocoIndexBackend(self.registry.data_dir, "python", self.model)
        self.assertEqual(backend.python_executable, Path(sys.executable).resolve())
        self.assertIsInstance(backend, RetrievalBackend)
        self.assertFalse(backend.capabilities.generates_answers)

    def test_model_outside_central_directory_is_rejected(self):
        with self.assertRaisesRegex(BackendError, "models_dir"):
            CocoIndexBackend(self.registry.data_dir, sys.executable, self.model, models_dir=self.root / "elsewhere")

    def test_malformed_model_manifest_fails_with_backend_error(self):
        path = self.model / "bti-model-manifest.json"
        for manifest in [[], 123, {"status": "complete", "repository": "test/embedding", "revision": "a"*40, "files": [None]},
                         {"status": "complete", "repository": "test/embedding", "revision": "a"*40, "files": [{"path": "weights.bin"}]}]:
            with self.subTest(manifest=manifest):
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(BackendError):
                    self.backend.index(self.unit_id)


if __name__ == "__main__":
    unittest.main()
