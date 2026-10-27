from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.registry import IndexRecipe, Registry, RegistryError, SnapshotChangedError, SourceSpec


@unittest.skipUnless(shutil.which("git"), "Ignore semantics require Git")
class GitIgnoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.registry = Registry(self.root / "data")
        self.spec = SourceSpec("fixture", "project", self.checkout, include=("**/*.md", "**/*.c"))
        self.recipe = IndexRecipe("fixture", "fixture", "1", "fixture")

    def write(self, relative, text="fixture", root=None):
        path = (root or self.checkout) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def git(self, *args, root=None):
        return subprocess.run(["git", "-C", str(root or self.checkout), *args], capture_output=True, check=True)

    def capture(self, **changes):
        self.registry.register(replace(self.spec, **changes))
        return self.registry.snapshot("fixture", self.recipe)

    def paths(self, **changes):
        return [entry["path"] for entry in self.capture(**changes)["files"]]

    def test_default_ignores_local_even_with_matching_include_and_without_git(self):
        self.write(".gitignore", "local/\n")
        self.write("local/obser.md")
        self.write("main.c")
        self.assertEqual(self.paths(), ["main.c"])
        self.assertFalse((self.checkout / ".git").exists())

    def test_nested_negation_and_escaped_spaces_use_git_semantics(self):
        self.write(".gitignore", "*.tmp.md\nspace\\ name.md\n")
        self.write("docs/.gitignore", "*.md\n!keep.md\n")
        for name in ("main.c", "discard.tmp.md", "space name.md", "docs/skip.md", "docs/keep.md"):
            self.write(name)
        self.assertEqual(self.paths(), ["docs/keep.md", "main.c"])

    def test_attachment_overrides_ignored_ancestor_for_only_selected_child(self):
        self.write(".gitignore", "local/\n")
        self.write("main.c")
        self.write("local/junk.md")
        self.write("local/artifacts/keep.md")
        self.write("local/artifacts/cache.tmp")
        self.assertEqual(self.paths(attach_paths=("local/artifacts",)), ["local/artifacts/keep.md", "main.c"])

    def test_explicit_file_attachment_and_absolute_attachment(self):
        self.write(".gitignore", "artifacts/\n")
        keep = self.write("artifacts/keep.md")
        self.write("artifacts/skip.md")
        self.assertEqual(self.paths(attach_paths=(str(keep),)), ["artifacts/keep.md"])

    def test_attachment_overrides_automatic_defaults_but_not_explicit_excludes(self):
        self.write("main.c")
        self.write(".local/keep.md")
        self.write(".local/excluded.md")
        self.write(".local/build/also-keep.md")
        self.assertEqual(self.paths(attach_paths=(".local",), exclude=("**/excluded.md",)),
                         [".local/build/also-keep.md", ".local/keep.md", "main.c"])
        self.assertEqual(self.paths(attach_paths=(".local/keep.md",), exclude=(".local/**",)), ["main.c"])

    def test_disabling_ignore_keeps_automatic_defaults_and_include_filter(self):
        self.write(".gitignore", "local/\n")
        self.write("local/report.md")
        self.write("local/blob.bin")
        self.write(".local/private.md")
        self.assertEqual(self.paths(respect_gitignore=False), ["local/report.md"])

    def test_root_inside_ignored_parent_repo_requires_explicit_attachment(self):
        self.git("init", "-q")
        self.write(".gitignore", "artifacts/\n")
        self.write("artifacts/keep.md")
        self.spec = replace(self.spec, root=self.checkout / "artifacts")
        with self.assertRaisesRegex(RegistryError, "No files selected"):
            self.capture()
        self.assertEqual(self.paths(attach_paths=(".",)), ["keep.md"])

    def test_nested_repo_applies_its_own_ignores(self):
        self.git("init", "-q")
        nested = self.checkout / "nested"
        nested.mkdir()
        self.git("init", "-q", root=nested)
        self.write("nested/.gitignore", "skip.md\n")
        self.write("nested/skip.md")
        self.write("nested/keep.md")
        self.assertEqual(self.paths(), ["nested/keep.md"])

    def test_subdirectory_root_uses_worktree_relative_anchored_ignores(self):
        self.git("init", "-q")
        self.write(".gitignore", "/docs/hidden.md\n")
        self.write("docs/hidden.md")
        self.write("docs/visible.md")
        self.spec = replace(self.spec, root=self.checkout / "docs")
        self.assertEqual(self.paths(), ["visible.md"])

    def test_info_exclude_and_relative_configured_excludes_are_respected(self):
        self.git("init", "-q")
        self.write(".git/info/exclude", "/docs/info-hidden.md\n")
        self.write("custom-ignore", "/docs/config-hidden.md\n")
        self.git("config", "core.excludesFile", "custom-ignore")
        for name in ("info-hidden.md", "config-hidden.md", "visible.md"):
            self.write("docs/" + name)
        self.spec = replace(self.spec, root=self.checkout / "docs")
        self.assertEqual(self.paths(), ["visible.md"])

    def test_submodule_applies_its_own_ignore_rules_without_mutation(self):
        upstream = self.root / "upstream"
        upstream.mkdir()
        for root in (self.checkout, upstream):
            self.git("init", "-q", root=root)
            self.git("config", "user.name", "BTI Fixture", root=root)
            self.git("config", "user.email", "fixture@example.invalid", root=root)
        self.write("keep.md", root=upstream)
        self.write(".gitignore", "skip.md\n", root=upstream)
        self.git("add", ".", root=upstream)
        self.git("commit", "-qm", "fixture", root=upstream)
        self.git("-c", "protocol.file.allow=always", "submodule", "add", "-q", str(upstream), "modules/component")
        self.write("modules/component/skip.md")
        before = self.git("status", "--porcelain=v1").stdout
        self.assertEqual(self.paths(), ["modules/component/keep.md"])
        self.assertEqual(self.git("status", "--porcelain=v1").stdout, before)

    def test_git_metadata_and_own_data_never_attached(self):
        self.git("init", "-q")
        self.write("main.c")
        self.registry = Registry(self.checkout / "runtime")
        self.write("runtime/secret.md")
        self.assertEqual(self.paths(attach_paths=(".",)), ["main.c"])
        with self.assertRaisesRegex(RegistryError, "Git metadata"):
            self.capture(attach_paths=(".git",))

    def test_own_data_root_and_descendant_cannot_be_registered_or_snapshotted(self):
        self.registry.data_dir.mkdir()
        (self.registry.data_dir / "sources").mkdir()
        for root in (self.registry.data_dir, self.registry.data_dir / "sources"):
            self.write("private.md", root=root)
            spec = replace(self.spec, root=root, attach_paths=(".",))
            with self.subTest(root=root):
                with self.assertRaisesRegex(RegistryError, "registry data directory"):
                    self.registry.register(spec)
                # A prior/manual registration must not bypass the snapshot gate.
                self.registry.sources_file.parent.mkdir(exist_ok=True)
                self.registry.sources_file.write_text(json.dumps({"schema_version": 1, "sources": {
                    "fixture": {"source_id": "fixture", "kind": "project", "root": str(root),
                                "include": ["**/*.md"], "attach_paths": ["."]}}}), encoding="utf-8")
                with self.assertRaisesRegex(RegistryError, "registry data directory"):
                    self.registry.snapshot("fixture", self.recipe)

    def test_attachment_paths_outside_source_are_rejected(self):
        self.write("main.c")
        outside = self.write("other.md", root=self.root)
        for path in (str(outside), "../other.md", "missing"):
            with self.subTest(path=path), self.assertRaises(RegistryError):
                self.capture(attach_paths=(path,))

    def test_ignore_policy_changes_selection_and_unit_identity(self):
        self.write("main.c")
        self.write("local/report.md")
        self.write(".gitignore", "local/\n")
        first = self.capture()
        second = self.capture(attach_paths=("local",))
        self.assertNotEqual(first["unit_id"], second["unit_id"])
        self.assertEqual(self.registry.load_unit(first["unit_id"]), first)

    def test_old_registration_gets_safe_defaults_without_mutating_snapshot(self):
        self.write("main.c")
        self.registry.register(self.spec)
        sources = json.loads(self.registry.sources_file.read_text())
        for field in ("respect_gitignore", "attach_paths"):
            del sources["sources"]["fixture"][field]
        self.registry.sources_file.write_text(json.dumps(sources), encoding="utf-8")
        source = self.registry.source("fixture")
        self.assertTrue(source.respect_gitignore)
        self.assertEqual(source.attach_paths, ())

    def test_ignored_file_changes_do_not_change_snapshot(self):
        self.write("main.c")
        ignored = self.write("local/skip.md")
        self.write(".gitignore", "local/\n")
        first = self.capture()
        ignored.write_text("changed", encoding="utf-8")
        self.assertEqual(first["unit_id"], self.capture()["unit_id"])

    def test_rule_change_during_copy_detected_even_if_selected_files_stay_same(self):
        self.write("main.c")
        rules = self.write(".gitignore", "local/\n")
        self.registry.register(self.spec)
        original = self.registry._inventory
        calls = 0

        def inventory(spec):
            nonlocal calls
            calls += 1
            if calls == 2:
                rules.write_text("local/\n# changed during capture\n", encoding="utf-8")
            return original(spec)

        with patch.object(self.registry, "_inventory", side_effect=inventory):
            with self.assertRaises(SnapshotChangedError):
                self.registry.snapshot("fixture", self.recipe)
        self.assertFalse(any(path.name.startswith("u-") for path in (self.registry.data_dir / "sources").iterdir()))

    def test_missing_git_is_explicit_error_unless_policy_disabled(self):
        self.write("main.c")
        self.registry.register(self.spec)
        with patch("bubbleteait.registry.shutil.which", return_value=None):
            with self.assertRaisesRegex(RegistryError, "requires Git"):
                self.registry.snapshot("fixture", self.recipe)
            self.assertEqual(self.paths(respect_gitignore=False), ["main.c"])


if __name__ == "__main__":
    unittest.main()
