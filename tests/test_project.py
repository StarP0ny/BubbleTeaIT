import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.cli import main
from bubbleteait.config import Config, ConfigError, load_config
from bubbleteait.project import discover_project, load_project, project_info, sync_project
from bubbleteait.registry import IndexRecipe, Registry, _atomic_json
from bubbleteait.search import SearchService


class FixtureBackend:
    def __init__(self, root):
        self.registry = Registry(root)
        self.calls = []
        self.fail = False

    def index(self, unit_id):
        if self.fail:
            raise RuntimeError('fixture indexing failure')
        self.calls.append(unit_id)
        unit = self.registry.load_unit(unit_id)
        status = {'schema_version': 1, 'unit_id': unit_id, 'recipe_hash': unit['recipe_hash'],
                  'backend': 'fixture', 'status': 'ready'}
        _atomic_json(self.registry.data_dir / 'indexes' / unit_id / 'status.json', status)
        return status


class ProjectTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.write('bti.toml', '''schema_version=1
[component]
name="product"
kind="project"
[source]
include=["**/*.c", "**/*.h"]
[context]
variant="declared-only"
[dependencies.sdk]
path="vendor/sdk"
variant="v2"
''')
        self.write('main.c', 'int application = 1;\n')
        self.write('vendor/sdk/bti.toml', '''schema_version=1
[component]
name="sdk"
kind="sdk"
[source]
include=["**/*.c"]
[variants.v2]
include=["v2/**", "common/**"]
''')
        self.write('vendor/sdk/v2/driver.c', 'int sdk_v2 = 1;\n')
        self.write('vendor/sdk/v3/driver.c', 'int sdk_v3 = 1;\n')
        self.write('vendor/sdk/common/helper.c', 'int common_helper;\n')
        self.config = Config(self.root, self.root/'data', self.root/'models', project_config=self.root/'bti.toml')
        self.recipe = IndexRecipe('fixture', 'none', '1', 'lines')
        self.backend = FixtureBackend(self.config.data_dir)

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return target

    def test_tree_paths_variants_and_parent_do_not_index_dependency_twice(self):
        project = load_project(self.root/'bti.toml')
        plan = project.plan(self.config.data_dir)
        self.assertTrue(plan['ready'])
        selected = {s['source_id']: [f['path'] for f in s['files']] for s in plan['sources']}
        self.assertEqual(selected, {'product.source': ['main.c'],
                                    'sdk.source': ['common/helper.c', 'v2/driver.c']})
        self.assertEqual(project.sources[1].root, self.root/'vendor/sdk')
        self.assertEqual(project.sources[1].role, 'sdk')
        self.assertEqual(plan['components'][1]['selection']['variant'], 'v2')
        self.assertEqual(plan['components'][1]['parent'], 'product')
        self.assertFalse((self.config.data_dir).exists())

    def test_parent_include_is_intersection_not_replacement(self):
        path = self.root/'bti.toml'
        path.write_text(path.read_text() + 'include=["v2/**", "**/*.h"]\n', encoding='utf-8')
        self.write('vendor/sdk/v2/hidden.h', 'int hidden;\n')
        selected = load_project(path).plan(self.config.data_dir)['sources'][1]
        self.assertEqual([f['path'] for f in selected['files']], ['v2/driver.c'])

    def test_reused_dependency_deduplicates_but_conflicting_selection_fails(self):
        path = self.root/'bti.toml'
        original = path.read_text()
        path.write_text(original + '[dependencies.same]\npath="vendor/sdk"\nvariant="v2"\n')
        self.assertEqual(len(load_project(path).sources), 2)
        path.write_text(original + '[dependencies.same]\npath="vendor/sdk"\n')
        with self.assertRaisesRegex(ConfigError, 'Conflicting selections'):
            load_project(path)

    def test_cycle_missing_dependency_unknown_variant_and_schema_errors(self):
        path = self.root/'bti.toml'
        original = path.read_text()
        for old, new, message in (
                ('variant="v2"', 'variant="wrong"', 'Unknown variant'),
                ('path="vendor/sdk"', 'path="absent"', 'Cannot load'),
                ('schema_version=1', 'schema_version=true', 'schema_version'),
                ('kind="project"', 'kind="wrong"', 'component.kind'),
                ('variant="v2"', 'variant="v2"\ntypo=true', 'unknown options')):
            path.write_text(original.replace(old, new))
            with self.subTest(message=message), self.assertRaisesRegex(ConfigError, message):
                load_project(path)
        path.write_text(original)
        sdk = self.root/'vendor/sdk/bti.toml'
        sdk.write_text(sdk.read_text() + '[dependencies.parent]\npath="../.."\n')
        with self.assertRaisesRegex(ConfigError, 'cycle'):
            load_project(path)

    def test_plan_reports_ignore_rules_attachments_and_encoding_before_index(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, capture_output=True)
        self.write('.gitignore', 'ignored/\n')
        self.write('ignored/hide.c', 'int hidden;\n')
        self.write('main.c', 'int application;\n')
        broken = self.root/'legacy.c'
        broken.write_bytes(b'// \xff\n')
        plan = load_project(self.root/'bti.toml').plan(self.config.data_dir)
        app = plan['sources'][0]
        self.assertFalse(plan['ready'])
        self.assertEqual(app['encoding_errors'][0]['path'], 'legacy.c')
        self.assertIn({'path': 'ignored/', 'reason': 'gitignore'}, app['skipped'])
        self.assertTrue(any('.gitignore' in key for key in app['ignore_rules']))
        broken.unlink()
        path = self.root/'bti.toml'
        path.write_text(path.read_text().replace('[source]', '[source]\nattach=["ignored"]'))
        selected = load_project(path).plan(self.config.data_dir)['sources'][0]['files']
        self.assertIn('ignored/hide.c', [f['path'] for f in selected])

    def test_discovery_stops_at_git_boundary_and_loads_user_runtime_layer(self):
        folder = self.root/'src/deep'
        folder.mkdir(parents=True)
        self.assertEqual(discover_project(folder), self.root/'bti.toml')
        self.write('user/BubbleTeaIT/config.toml', '[paths]\ndata_dir="runtime"\n')
        config = load_config(folder, env={'LOCALAPPDATA': str(self.root/'user')})
        self.assertEqual(config.project_config, self.root/'bti.toml')
        self.assertEqual(config.data_dir, self.root/'user/BubbleTeaIT/runtime')
        (folder/'.git').mkdir()
        self.assertIsNone(discover_project(folder))
        explicit = load_config(folder, project_file='../../bti.toml', env={})
        self.assertEqual(explicit.project_config, self.root/'bti.toml')

    def test_sync_reuses_units_pins_history_and_preserves_alias_on_failure(self):
        first = sync_project(load_project(self.root/'bti.toml'), self.config, self.recipe, self.backend)
        self.assertEqual(len(self.backend.calls), 2)
        repeated = sync_project(load_project(self.root/'bti.toml'), self.config, self.recipe, self.backend)
        self.assertEqual(first, repeated)
        self.assertEqual(len(self.backend.calls), 2)
        self.write('main.c', 'int application = 2;\n')
        self.backend.fail = True
        with self.assertRaisesRegex(RuntimeError, 'fixture indexing failure'):
            sync_project(load_project(self.root/'bti.toml'), self.config, self.recipe, self.backend)
        service = SearchService(self.config.data_dir)
        self.assertEqual(service.load_composition('product')['composition_hash'], first['composition_hash'])
        self.backend.fail = False
        second = sync_project(load_project(self.root/'bti.toml'), self.config, self.recipe, self.backend)
        self.assertNotEqual(first['pinned_composition_id'], second['pinned_composition_id'])
        self.assertEqual(first['units'][1], second['units'][1])
        self.assertEqual(service.load_composition(first['pinned_composition_id'])['composition_hash'], first['composition_hash'])
        info = project_info(self.config)
        self.assertEqual(info['composition_id'], second['pinned_composition_id'])
        self.assertFalse(info['config_changed'])
        path = self.root/'bti.toml'
        path.write_text(path.read_text() + '\n# new configuration\n')
        self.assertTrue(project_info(self.config)['config_changed'])

    def test_plan_cli_needs_no_embedding_config_and_implicit_search_uses_pinned_id(self):
        with patch.dict(os.environ, {'BTI_DATA_DIR': str(self.config.data_dir), 'PATH': os.environ['PATH']}, clear=True), contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(['--project-dir', str(self.root), 'plan', '--json'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['file_count'], 3)
        sync_project(load_project(self.root/'bti.toml'), self.config, self.recipe, self.backend)
        with patch.dict(os.environ, {'BTI_DATA_DIR': str(self.config.data_dir), 'PATH': os.environ['PATH']}, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(['--project-dir', str(self.root/'vendor/sdk/v2'), '--project-file', str(self.root/'bti.toml'),
                         'list-sources', '--json'])
        self.assertEqual(code, 0, output.getvalue())
        self.assertEqual(len(json.loads(output.getvalue())['entries']), 2)

    def test_sync_rejects_changed_plan_config_and_concurrent_writer(self):
        project = load_project(self.root/'bti.toml')
        first = sync_project(project, self.config, self.recipe, self.backend)
        plan = project.plan(self.config.data_dir)
        self.write('main.c', 'int changed_after_plan;\n')
        with self.assertRaisesRegex(ConfigError, 'Source changed after preflight'):
            sync_project(project, self.config, self.recipe, self.backend, plan=plan)
        self.assertEqual(SearchService(self.config.data_dir).load_composition('product')['composition_hash'],
                         first['composition_hash'])
        # A config edit during a backend operation must not publish a new alias.
        original_index = self.backend.index
        def change_config(unit):
            result = original_index(unit)
            path = self.root/'bti.toml'
            path.write_text(path.read_text() + '\n# edited while indexing\n')
            return result
        with patch.object(self.backend, 'index', side_effect=change_config):
            with self.assertRaisesRegex(ConfigError, 'configuration changed during sync'):
                sync_project(project, self.config, self.recipe, self.backend)
        self.assertEqual(SearchService(self.config.data_dir).load_composition('product')['composition_hash'],
                         first['composition_hash'])
        from bubbleteait.qwen_runtime import file_lock, QwenRuntimeError
        with file_lock(self.config.data_dir/'metadata/project-sync.lock'):
            with self.assertRaisesRegex(QwenRuntimeError, 'busy'):
                sync_project(load_project(self.root/'bti.toml'), self.config, self.recipe, self.backend)


if __name__ == '__main__':
    unittest.main()
