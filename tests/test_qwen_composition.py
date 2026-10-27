"""Global exact cosine ranking is confined to homogeneous Qwen recipes."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
from bubbleteait.search import SearchError, SearchService


class FakeBackend:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, unit_id, query, *, limit=10, paths=None):
        self.calls.append((unit_id, limit, paths))
        return self.rows[unit_id][:limit]


class QwenCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = Registry(self.root / "data")
        self.recipe = IndexRecipe("qwen-dense", "Qwen/Qwen3-Embedding-0.6B",
                                  "fixed-revision", "bti-lines-v1")

    def unit(self, name, recipe=None):
        root = self.root / name
        root.mkdir()
        (root / "sensor.c").write_text("int sensor = 1;\n", encoding="utf-8")
        self.registry.register(SourceSpec(name, "module", root, respect_gitignore=False))
        manifest = self.registry.snapshot(name, recipe or self.recipe)
        status = self.registry.data_dir / "indexes" / manifest["unit_id"] / "status.json"
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps({"schema_version": 1, "unit_id": manifest["unit_id"],
            "recipe_hash": manifest["recipe_hash"], "backend": manifest["recipe"]["backend"],
            "status": "ready"}), encoding="utf-8")
        return manifest

    def search(self, manifests, scores, *, limit=1, kinds=None, missing_score=False, channel="vector"):
        self.registry.compose("product", [unit["unit_id"] for unit in manifests], require_ready=True)
        rows = {}
        for index, (unit, score) in enumerate(zip(manifests, scores)):
            row = {"file_path": "sensor.c", "start_line": 1, "end_line": 1,
                   "content": "int sensor = 1;\n", "score": score,
                   "score_kind": kinds[index] if kinds else "cosine"}
            if missing_score and index == 0:
                del row["score"]
            rows[unit["unit_id"]] = [row]
        backend = FakeBackend(rows)
        return SearchService(self.registry.data_dir, backend).search("product", "sensor", limit=limit,
                                                                     channel=channel), backend

    def test_best_unit_last_survives_global_limit_one(self):
        manifests = [self.unit(f"module-{index}") for index in range(8)]
        result, backend = self.search(manifests, [0.1] * 7 + [0.95])
        self.assertEqual(result["ranking"], "global_cosine_same_recipe")
        self.assertTrue(result["scores_comparable_across_units"])
        self.assertEqual(result["results"][0]["unit_id"], manifests[-1]["unit_id"])
        self.assertEqual(result["results"][0]["score"], 0.95)
        self.assertEqual(result["results"][0]["rank"], 1)
        self.assertEqual(result["results"][0]["unit_rank"], 1)
        self.assertEqual(len(backend.calls), 8)

    def test_different_recipe_hash_never_compares_even_with_same_model(self):
        first = self.unit("first")
        other = replace(self.recipe, options={"query_prompt": "different instruction"})
        second = self.unit("second", other)
        result, _ = self.search([first, second], [0.1, 0.95])
        self.assertEqual(result["results"][0]["unit_id"], first["unit_id"])
        self.assertEqual(result["ranking"], "per_unit_rank_interleave")
        self.assertFalse(result["scores_comparable_across_units"])

    def test_legacy_backend_remains_interleaved_even_with_same_recipe_and_cosine(self):
        recipe = replace(self.recipe, backend="cocoindex-code")
        first, second = self.unit("first", recipe), self.unit("second", recipe)
        result, _ = self.search([first, second], [0.1, 0.95])
        self.assertEqual(result["results"][0]["unit_id"], first["unit_id"])
        self.assertFalse(result["scores_comparable_across_units"])

    def test_missing_score_or_non_cosine_falls_back_to_legacy_order(self):
        first, second = self.unit("first"), self.unit("second")
        for options in ({"missing_score": True}, {"kinds": ["cosine", "distance"]}):
            with self.subTest(options=options):
                result, _ = self.search([first, second], [0.1, 0.95], **options)
                self.assertEqual(result["results"][0]["unit_id"], first["unit_id"])
                self.assertFalse(result["scores_comparable_across_units"])

    def test_equal_cosines_have_stable_unit_order_and_global_ranks(self):
        first, second = self.unit("first"), self.unit("second")
        result, _ = self.search([first, second], [0.8, 0.8], limit=2)
        self.assertEqual([row["unit_id"] for row in result["results"]],
                         sorted([first["unit_id"], second["unit_id"]]))
        self.assertEqual([row["rank"] for row in result["results"]], [1, 2])

    def test_nonfinite_cosine_remains_an_error(self):
        first, second = self.unit("first"), self.unit("second")
        with self.assertRaisesRegex(SearchError, "finite"):
            self.search([first, second], [0.1, float("nan")])

    def test_text_channel_does_not_claim_global_cosine(self):
        first, second = self.unit("first"), self.unit("second")
        result, backend = self.search([first, second], [0.1, 0.95], channel="text")
        self.assertEqual(result["results"][0]["unit_id"], first["unit_id"])
        self.assertFalse(result["scores_comparable_across_units"])
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
