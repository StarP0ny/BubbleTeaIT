from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.registry import IndexRecipe, Registry, RegistryError, SnapshotChangedError, SourceSpec


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.registry = Registry(self.root / "data")
        self.recipe = IndexRecipe("cocoindex-code", "test/e5", "pinned-revision", "tree-sitter-v1")

    def write(self, relative, content=b"fixture", root=None):
        path = (root or self.checkout) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def register(self, **kwargs):
        return self.registry.register(SourceSpec("app", "project", self.checkout, **kwargs))

    def snapshot(self):
        return self.registry.snapshot("app", self.recipe)

    def status(self, manifest, **overrides):
        path = self.registry.data_dir / "indexes" / manifest["unit_id"] / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"schema_version": 1, "unit_id": manifest["unit_id"], "recipe_hash": manifest["recipe_hash"], "backend": "cocoindex-code", "status": "ready"}
        value.update(overrides)
        path.write_text(json.dumps(value), encoding="utf-8")

    def git(self, *args, root=None):
        return subprocess.run(["git", "-C", str(root or self.checkout), *args], capture_output=True, check=True)

    def test_c_headers_raw_bytes_and_explicitly_attached_ignored_local_are_selected(self):
        content = b"int test(void);\r\n// non-UTF8: \xff\x80\r\n"
        self.write("trunk/Src/main.c", b"int test(void) { return 7; }\n")
        self.write("trunk/Src/main.h", content)
        self.write("docs/local/obser.1.4.md", "Температура".encode())
        self.write(".gitignore", b"docs/local/\n")
        self.write("trunk/build/firmware.hex")
        self.write("vendor/large/source.c")
        self.write(".venv/bin/file.py")
        self.write(".git/config")
        self.register(include=("**/*.c", "**/*.h", "docs/local/*.md"), attach_paths=("docs/local",))
        manifest = self.snapshot()
        self.assertEqual([entry["path"] for entry in manifest["files"]], ["docs/local/obser.1.4.md", "trunk/Src/main.c", "trunk/Src/main.h"])
        header = next(entry for entry in manifest["files"] if entry["path"].endswith("main.h"))
        self.assertEqual(header["sha256"], hashlib.sha256(content).hexdigest())
        copied = self.registry.data_dir / "sources" / manifest["unit_id"] / "files/trunk/Src/main.h"
        self.assertEqual(copied.read_bytes(), content)
        self.assertFalse(self.registry.index_status(manifest["unit_id"])["status"] == "ready")

    def test_ids_are_stable_and_excluded_files_do_not_affect_identity(self):
        self.write("main.c", b"int value = 1;\n")
        excluded = self.write("scratch/output.txt", b"first")
        self.register(include=("**/*.c",), exclude=("scratch/**",))
        first = self.snapshot()
        excluded.write_bytes(b"second")
        self.write("build/additional.bin", b"new build output")
        self.assertEqual(self.snapshot()["unit_id"], first["unit_id"])
        self.assertEqual(self.registry.load_unit(first["unit_id"]), first)
        self.assertEqual(len([path for path in (self.registry.data_dir / "sources").iterdir() if path.name.startswith("u-")]), 1)

    def test_dirty_changes_recipe_changes_and_source_identity_change_unit(self):
        source = self.write("main.c", b"old")
        self.register()
        first = self.snapshot()
        source.write_bytes(b"new")
        second = self.snapshot()
        self.assertNotEqual(first["unit_id"], second["unit_id"])
        revised = self.registry.snapshot("app", replace(self.recipe, embedding_revision="different-revision"))
        self.assertNotEqual(second["unit_id"], revised["unit_id"])
        self.assertNotEqual(second["recipe_hash"], revised["recipe_hash"])
        self.registry.register(SourceSpec("same-bytes-other-source", "project", self.checkout))
        separate = self.registry.snapshot("same-bytes-other-source", self.recipe)
        self.assertNotEqual(second["unit_id"], separate["unit_id"])

    def test_relocated_source_reuses_same_logical_artifact(self):
        self.write("main.c", b"portable source")
        self.register()
        original = self.snapshot()
        other = self.root / "relocated"
        shutil.copytree(self.checkout, other)
        self.registry.register(SourceSpec("app", "project", other))
        moved = self.snapshot()
        self.assertEqual(original["unit_id"], moved["unit_id"])
        self.assertEqual(self.registry.source("app").root, other)

    @unittest.skipUnless(shutil.which("git"), "Git is optional")
    def test_git_head_is_provenance_content_hash_still_changes_for_ignored_local(self):
        self.git("init", "-q")
        self.git("config", "user.name", "Registry Test")
        self.git("config", "user.email", "registry-test@example.invalid")
        self.write("main.c", b"tracked")
        self.write(".gitignore", b"docs/local/\n")
        ignored = self.write("docs/local/report.md", b"first")
        self.git("add", "main.c", ".gitignore")
        self.git("commit", "-qm", "fixture")
        self.register(include=("**/*.c", "docs/local/*.md"), attach_paths=("docs/local",))
        clean = self.snapshot()
        self.assertTrue(clean["provenance"]["git"]["available"])
        self.assertFalse(clean["provenance"]["git"]["dirty"])
        ignored.write_bytes(b"second")
        changed = self.snapshot()
        self.assertEqual(changed["provenance"]["git"]["head"], clean["provenance"]["git"]["head"])
        self.assertFalse(changed["provenance"]["git"]["dirty"])
        self.assertNotEqual(changed["content_hash"], clean["content_hash"])
        self.write("main.c", b"modified tracked")
        dirty = self.snapshot()
        self.assertTrue(dirty["provenance"]["git"]["dirty"])

    def test_documents_and_module_without_git_are_valid(self):
        self.write("report.md", b"document")
        self.registry.register(SourceSpec("reports", "documents", self.checkout, role="primary_report"))
        report = self.registry.snapshot("reports", self.recipe)
        self.assertEqual(report["role"], "primary_report")
        self.registry.register(SourceSpec("module", "module", self.checkout))
        module = self.registry.snapshot("module", self.recipe)
        self.assertEqual(module["kind"], "module")
        self.assertFalse(module["provenance"]["git"]["available"])

    @unittest.skipUnless(shutil.which("git"), "Git is optional")
    def test_git_submodule_revision_is_recorded_without_changing_checkout(self):
        module = self.root / "module-upstream"
        module.mkdir()
        for root in (self.checkout, module):
            self.git("init", "-q", root=root)
            self.git("config", "user.name", "Registry Test", root=root)
            self.git("config", "user.email", "registry-test@example.invalid", root=root)
        self.write("sensor.c", b"sensor module", root=module)
        self.git("add", "sensor.c", root=module)
        self.git("commit", "-qm", "module fixture", root=module)
        revision = self.git("rev-parse", "HEAD", root=module).stdout.decode().strip()
        self.git("-c", "protocol.file.allow=always", "submodule", "add", "-q", str(module), "trunk/modules/sensor")
        self.git("commit", "-qam", "parent fixture")
        before = self.git("status", "--porcelain=v1").stdout
        self.register(include=("**/*.c",))
        manifest = self.snapshot()
        submodules = manifest["provenance"]["git"]["submodules"]
        self.assertTrue(any(revision in line and "trunk/modules/sensor" in line for line in submodules))
        self.assertEqual(self.git("status", "--porcelain=v1").stdout, before)

    def test_empty_selection_and_unknown_source_do_not_publish_units(self):
        self.write("report.md")
        self.register(include=("**/*.c",))
        with self.assertRaisesRegex(RegistryError, "No files selected"):
            self.snapshot()
        with self.assertRaisesRegex(RegistryError, "Unknown source_id"):
            self.registry.snapshot("absent", self.recipe)
        self.assertFalse((self.registry.data_dir / "sources").exists())

    def test_source_and_composition_ids_and_globs_cannot_escape_metadata(self):
        self.write("main.c")
        for source_id in ("../escape", "a/b", "C:\\escape", "", ".", ".."):
            with self.subTest(source_id=source_id):
                with self.assertRaises(RegistryError):
                    self.registry.register(SourceSpec(source_id, "project", self.checkout))
        for pattern in ("../*.c", "/absolute/*", "C:/project/*", "a\\*.c"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(RegistryError):
                    self.register(include=(pattern,))
        with self.assertRaises(RegistryError):
            self.registry.compose("../escape", ["u-" + "0" * 64])
        with self.assertRaises(RegistryError):
            self.registry.load_unit("../../escape")

    def test_symlink_outside_source_is_rejected_without_reading_target(self):
        outside = self.write("secret.txt", b"outside", root=self.root)
        link = self.checkout / "linked.c"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation not permitted on this host")
        self.register(include=("**/*.c",))
        original = Path.read_bytes
        def safe_read(path):
            if path == outside or path == link:
                raise AssertionError("must not read symlink target")
            return original(path)
        with patch.object(Path, "read_bytes", safe_read):
            with self.assertRaisesRegex(RegistryError, "symlink"):
                self.snapshot()

    def test_tampered_snapshot_bytes_are_rejected(self):
        self.write("main.c", b"original")
        self.register()
        manifest = self.snapshot()
        copied = self.registry.data_dir / "sources" / manifest["unit_id"] / "files/main.c"
        copied.write_bytes(b"tampered")
        with self.assertRaisesRegex(RegistryError, "bytes do not match"):
            self.registry.load_unit(manifest["unit_id"])
        with self.assertRaises(RegistryError):
            self.snapshot()

    def test_tampered_manifest_traversal_is_rejected_even_without_byte_verification(self):
        self.write("main.c")
        self.register()
        manifest = self.snapshot()
        path = self.registry.data_dir / "sources" / manifest["unit_id"] / "manifest.json"
        manifest["files"][0]["path"] = "../outside.c"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RegistryError, "traversal"):
            self.registry.load_unit(manifest["unit_id"], verify_files=False)

    def test_copy_time_source_change_never_publishes_ready_snapshot(self):
        source = self.write("main.c", b"first")
        self.register()
        from bubbleteait import registry as implementation
        original = implementation._read_file
        calls = 0

        def mutate_between_inventory_and_copy(path):
            nonlocal calls
            calls += 1
            if calls == 2:
                source.write_bytes(b"changed")
            return original(path)

        with patch("bubbleteait.registry._read_file", side_effect=mutate_between_inventory_and_copy):
            with self.assertRaises(SnapshotChangedError):
                self.snapshot()
        snapshots = list((self.registry.data_dir / "sources").iterdir())
        self.assertFalse(any(path.name.startswith("u-") for path in snapshots))
        self.assertFalse(any((path / "manifest.json").exists() for path in snapshots))

    def test_file_added_during_capture_is_detected(self):
        self.write("main.c", b"first")
        self.register()
        original = self.registry._inventory
        calls = 0

        def add_after_copy(spec):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.write("new.c", b"added during capture")
            return original(spec)

        with patch.object(self.registry, "_inventory", side_effect=add_after_copy):
            with self.assertRaises(SnapshotChangedError):
                self.snapshot()
        self.assertFalse(any(path.name.startswith("u-") for path in (self.registry.data_dir / "sources").iterdir()))

    def test_own_data_directory_is_never_captured(self):
        self.write("main.c")
        self.registry = Registry(self.checkout / "runtime")
        self.register()
        manifest = self.snapshot()
        self.assertEqual([entry["path"] for entry in manifest["files"]], ["main.c"])

    def test_composition_is_draft_until_matching_index_is_ready(self):
        self.write("main.c")
        self.register()
        manifest = self.snapshot()
        unit_id = manifest["unit_id"]
        draft = self.registry.compose("release", [unit_id], configuration={"variant": "variant-a"})
        self.assertFalse(draft["ready"])
        with self.assertRaisesRegex(RegistryError, "not ready"):
            self.registry.compose("release", [unit_id], require_ready=True)
        self.status(manifest, recipe_hash="0" * 64)
        with self.assertRaisesRegex(RegistryError, "does not match"):
            self.registry.compose("release", [unit_id], require_ready=True)
        self.status(manifest)
        self.assertFalse(self.registry.compose("draft", [unit_id])["ready"])
        release = self.registry.compose("release", [unit_id], configuration={"variant": "variant-a"}, require_ready=True)
        self.assertTrue(release["ready"])
        other = self.registry.compose("other-composition", [unit_id], configuration={"variant": "variant-b"}, require_ready=True)
        self.assertEqual(release["units"][0]["unit_id"], other["units"][0]["unit_id"])
        self.assertNotEqual(release["configuration_hash"], other["configuration_hash"])

    def test_composition_replaces_one_module_version_without_mutating_old_unit(self):
        source = self.write("sensor.c", b"version one")
        self.register()
        first = self.snapshot()
        self.status(first)
        self.registry.compose("product", [first["unit_id"]], require_ready=True)
        source.write_bytes(b"version two")
        second = self.snapshot()
        self.status(second)
        composed = self.registry.compose("product", [second["unit_id"]], require_ready=True)
        self.assertEqual(composed["units"][0]["unit_id"], second["unit_id"])
        self.assertEqual(self.registry.load_unit(first["unit_id"])["files"][0]["sha256"], hashlib.sha256(b"version one").hexdigest())
        with self.assertRaisesRegex(RegistryError, "one exact version"):
            self.registry.compose("mixed", [first["unit_id"], second["unit_id"]])


if __name__ == "__main__":
    unittest.main()
