from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bubbleteait.registry import IndexRecipe, Registry, RegistryError, SourceSpec
from bubbleteait.search import SearchError, SearchService


class FakeBackend:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, unit_id, query, *, limit=10, paths=None):
        self.calls.append({"unit_id": unit_id, "query": query, "limit": limit, "paths": paths})
        return self.rows.get(unit_id, [])[:limit]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = Registry(self.root / "data")
        self.recipe = IndexRecipe("cocoindex-code", "test/e5", "fixed-revision", "fixture")
        self.service = SearchService(self.registry.data_dir)

    def unit(self, source_id="sensor", *, role="unknown", content=b"int sensorError = 7;\n", filename="sensor.c", recipe=None):
        checkout = self.root / source_id
        checkout.mkdir(exist_ok=True)
        path = checkout / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.registry.register(SourceSpec(source_id, "module", checkout, role=role))
        manifest = self.registry.snapshot(source_id, recipe or self.recipe)
        self.status(manifest)
        return manifest

    def status(self, manifest, state="ready"):
        path = self.registry.data_dir / "indexes" / manifest["unit_id"] / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 1, "unit_id": manifest["unit_id"], "recipe_hash": manifest["recipe_hash"], "backend": manifest["recipe"]["backend"], "status": state}), encoding="utf-8")

    def compose(self, *manifests, name="product", configuration=None):
        return self.registry.compose(name, [manifest["unit_id"] for manifest in manifests], configuration=configuration, require_ready=True)

    def row(self, content="int sensorError = 7;\n", *, filename="sensor.c", score=0.5):
        return {"file_path": filename, "start_line": 1, "end_line": 1, "content": content, "score": score}

    def test_draft_stale_readiness_and_tampered_fingerprints_fail(self):
        manifest = self.unit()
        self.registry.compose("product", [manifest["unit_id"]])
        with self.assertRaisesRegex(SearchError, "draft"):
            self.service.load_composition("product")
        original = self.compose(manifest)
        self.status(manifest, "building")
        with self.assertRaisesRegex(SearchError, "no longer ready"):
            self.service.load_composition("product")
        self.status(manifest)
        path = self.registry.data_dir / "metadata/compositions/product.json"
        for key, wrong in (("configuration_hash", "0" * 64), ("composition_hash", "0" * 64)):
            with self.subTest(key=key):
                path.write_text(json.dumps({**original, key: wrong}), encoding="utf-8")
                with self.assertRaisesRegex(SearchError, "fingerprint"):
                    self.service.load_composition("product")
        path.write_text(json.dumps(original), encoding="utf-8")
        original["units"][0]["recipe_hash"] = "0" * 64
        path.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(SearchError, "recipe"):
            self.service.load_composition("product")

    def test_switching_composition_selects_only_the_new_module_version(self):
        first = self.unit(content=b"int oldSensorApi = 1;\n")
        self.compose(first)
        second = self.unit(content=b"int newSensorApi = 2;\n")
        self.compose(second)
        backend = FakeBackend({first["unit_id"]: [self.row("int oldSensorApi = 1;\n")], second["unit_id"]: [self.row("int newSensorApi = 2;\n")]})
        result = SearchService(self.registry.data_dir, backend).search("product", "sensor")
        self.assertEqual([call["unit_id"] for call in backend.calls], [second["unit_id"]])
        self.assertEqual(result["results"][0]["unit_id"], second["unit_id"])
        self.assertIn("newSensorApi", result["results"][0]["content"])

    def test_role_and_source_filters_apply_before_backend_retrieval(self):
        code = self.unit("sensor", role="unknown")
        report = self.unit("report", role="primary_report", content=b"sensorError noted\n", filename="report.md")
        self.compose(code, report)
        backend = FakeBackend({report["unit_id"]: [self.row("sensorError noted\n", filename="report.md")]})
        service = SearchService(self.registry.data_dir, backend)
        result = service.search("product", "sensorError", roles=["primary_report"], source_ids=["report"], paths=["*.md"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["unit_id"], report["unit_id"])
        self.assertEqual(backend.calls[0]["paths"], ["*.md"])
        self.assertEqual(result["results"][0]["role"], "primary_report")
        self.assertEqual(result["results"][0]["provenance"], report["provenance"])
        empty = service.search("product", "sensorError", source_ids=[])
        self.assertEqual(empty["results"], [])
        self.assertEqual(len(backend.calls), 1)

    def test_vector_scores_are_preserved_but_never_compared_across_units(self):
        first = self.unit("first")
        second = self.unit("second", recipe=replace(self.recipe, embedding_model="test/other"))
        self.compose(first, second)
        backend = FakeBackend({
            first["unit_id"]: [self.row(score=0.01), self.row(score=0.02)],
            second["unit_id"]: [self.row(score=9999), self.row(score=8888)],
        })
        result = SearchService(self.registry.data_dir, backend).search("product", "sensorError", limit=4)
        self.assertEqual([row["source_id"] for row in result["results"]], ["first", "second", "first", "second"])
        self.assertEqual([row["unit_rank"] for row in result["results"]], [1, 1, 2, 2])
        self.assertEqual([row["score"] for row in result["results"]], [0.01, 9999, 0.02, 8888])
        self.assertFalse(result["scores_comparable_across_units"])

    def test_path_filter_violation_is_an_error_rather_than_postfilter(self):
        manifest = self.unit(filename="src/sensor.c")
        # A second whitelisted file makes it possible to return a valid but
        # incorrectly filtered row; the result must not silently disappear.
        path = self.root / "sensor" / "private.c"
        path.write_text("privateSymbol\n", encoding="utf-8")
        manifest = self.registry.snapshot("sensor", self.recipe)
        self.status(manifest)
        self.compose(manifest)
        backend = FakeBackend({manifest["unit_id"]: [self.row("privateSymbol\n", filename="private.c")]})
        with self.assertRaisesRegex(SearchError, "path filter"):
            SearchService(self.registry.data_dir, backend).search("product", "symbol", paths=["src/*"])
        self.assertEqual(backend.calls[0]["paths"], ["src/*"])

    def test_text_search_applies_sqlite_glob_before_line_ranking(self):
        manifest = self.unit(filename="src/sensor.c", content=b"// header\nint exactSymbol = 1;\n// tail\n")
        (self.root / "sensor/private.c").write_text("exactSymbol exactSymbol\n", encoding="utf-8")
        manifest = self.registry.snapshot("sensor", self.recipe)
        self.status(manifest)
        self.compose(manifest)
        result = self.service.search("product", "exactSymbol", channel="text", limit=1, paths=["src/*"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["file_path"], "src/sensor.c")
        self.assertIn("exactSymbol", result["results"][0]["content"])
        self.assertEqual(result["results"][0]["backend"], "snapshot-text")
        # SQL GLOB '*' includes directory separators, as CocoIndex does.
        self.assertEqual(len(self.service.search("product", "exactSymbol", channel="text", paths=["*.c"])["results"]), 2)

    def test_source_read_checks_whitelist_coordinates_and_current_working_hash(self):
        content = b"line one\r\nint sensorError = 7;\r\nline three\n"
        manifest = self.unit(content=content)
        result = self.service.read_source(manifest["unit_id"], "sensor.c", start_line=2, end_line=2)
        self.assertEqual(result["content"], "int sensorError = 7;\r\n")
        self.assertEqual(result["source_sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(result["used_sha256"], result["source_sha256"])
        self.assertFalse(result["source_changed"])
        (self.root / "sensor/sensor.c").write_bytes(b"changed source\n")
        snapshot = self.service.read_source(manifest["unit_id"], "sensor.c")
        self.assertTrue(snapshot["source_changed"])
        self.assertEqual(snapshot["content"], content.decode())
        working = self.service.read_source(manifest["unit_id"], "sensor.c", mode="working")
        self.assertEqual(working["content"], "changed source\n")
        self.assertNotEqual(working["source_sha256"], working["used_sha256"])
        (self.root / "sensor/private.txt").write_text("must not read", encoding="utf-8")
        for relative in ("private.txt", "../private.txt", "/absolute", "C:/escape"):
            with self.subTest(relative=relative):
                with self.assertRaises(RegistryError):
                    self.service.read_source(manifest["unit_id"], relative, mode="working")
        with self.assertRaisesRegex(SearchError, "coordinates"):
            self.service.read_source(manifest["unit_id"], "sensor.c", start_line=999)

    def test_working_root_uses_composition_mapping_not_latest_registration(self):
        old = self.unit(content=b"old source\n")
        old_checkout = self.root / "sensor"
        newer_checkout = self.root / "newer-checkout"
        newer_checkout.mkdir()
        (newer_checkout / "sensor.c").write_bytes(b"new source\n")
        self.registry.register(SourceSpec("sensor", "module", newer_checkout))
        new = self.registry.snapshot("sensor", self.recipe)
        self.status(new)
        self.compose(old, name="old-product", configuration={"source_roots": {"sensor": str(old_checkout)}})
        self.compose(new, name="new-product", configuration={"source_roots": {"sensor": str(newer_checkout)}})
        old_read = self.service.read_source(old["unit_id"], "sensor.c", mode="working", composition_id="old-product")
        new_read = self.service.read_source(new["unit_id"], "sensor.c", mode="working", composition_id="new-product")
        self.assertEqual(old_read["content"], "old source\n")
        self.assertEqual(new_read["content"], "new source\n")
        self.assertFalse(old_read["source_changed"])
        # No composition also uses that unit's provenance rather than registry.
        self.assertEqual(self.service.read_source(old["unit_id"], "sensor.c", mode="working")["content"], "old source\n")
        with self.assertRaisesRegex(SearchError, "does not belong"):
            self.service.read_source(new["unit_id"], "sensor.c", composition_id="old-product")

    def test_snapshot_read_survives_missing_working_checkout(self):
        manifest = self.unit()
        (self.root / "sensor/sensor.c").unlink()
        result = self.service.read_source(manifest["unit_id"], "sensor.c")
        self.assertEqual(result["working_state"], "missing")
        self.assertTrue(result["source_changed"])
        self.assertIsNone(result["working_sha256"])
        with self.assertRaisesRegex(SearchError, "missing"):
            self.service.read_source(manifest["unit_id"], "sensor.c", mode="working")

    def test_working_symlink_outside_root_is_blocked_snapshot_is_available(self):
        manifest = self.unit()
        outside = self.root / "outside.c"
        outside.write_text("outside secret", encoding="utf-8")
        path = self.root / "sensor/sensor.c"
        path.unlink()
        try:
            path.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable")
        result = self.service.read_source(manifest["unit_id"], "sensor.c")
        self.assertEqual(result["working_state"], "blocked")
        self.assertNotIn("outside secret", result["content"])
        with self.assertRaises(RegistryError):
            self.service.read_source(manifest["unit_id"], "sensor.c", mode="working")

    def test_tampered_snapshot_and_non_whitelisted_vector_rows_fail(self):
        manifest = self.unit()
        self.compose(manifest)
        backend = FakeBackend({manifest["unit_id"]: [self.row(filename="not-whitelisted.c")]})
        with self.assertRaisesRegex(SearchError, "whitelist"):
            SearchService(self.registry.data_dir, backend).search("product", "sensor")
        snapshot = self.registry.data_dir / "sources" / manifest["unit_id"] / "files/sensor.c"
        snapshot.write_bytes(b"corrupt snapshot")
        with self.assertRaises(RegistryError):
            self.service.read_source(manifest["unit_id"], "sensor.c")

    def test_invalid_scores_and_changed_composition_during_backend_call_fail(self):
        manifest = self.unit()
        self.compose(manifest)
        backend = FakeBackend({manifest["unit_id"]: [self.row(score=float("nan"))]})
        with self.assertRaisesRegex(SearchError, "finite"):
            SearchService(self.registry.data_dir, backend).search("product", "sensor")
        backend.rows[manifest["unit_id"]] = [self.row()]
        original = backend.search

        def change_composition(*args, **kwargs):
            self.compose(manifest, configuration={"variant": "different"})
            return original(*args, **kwargs)

        backend.search = change_composition
        with self.assertRaisesRegex(SearchError, "changed during retrieval"):
            SearchService(self.registry.data_dir, backend).search("product", "sensor")

    def test_backend_content_must_be_exact_slice_not_substring_of_range(self):
        manifest = self.unit(content=b"one\ntwo\nthree\n")
        self.compose(manifest)
        backend = FakeBackend({manifest["unit_id"]: [{
            "file_path": "sensor.c", "start_line": 1, "end_line": 3,
            "content": "one\n", "score": 0.5}]})
        with self.assertRaisesRegex(SearchError, "match source coordinates"):
            SearchService(self.registry.data_dir, backend).search("product", "one")

    def test_composition_source_root_must_match_registered_source(self):
        manifest = self.unit()
        self.compose(manifest, configuration={"source_roots": {"sensor": str(self.root / "other")}})
        with self.assertRaisesRegex(SearchError, "source_roots"):
            self.service.load_composition("product")

    def test_empty_file_read_has_one_empty_editor_line(self):
        manifest = self.unit(content=b"")
        result = self.service.read_source(manifest["unit_id"], "sensor.c")
        self.assertEqual(result["content"], "")
        self.assertEqual((result["start_line"], result["end_line"], result["total_lines"]), (1, 1, 1))
        self.assertFalse(result["source_changed"])


if __name__ == "__main__":
    unittest.main()
