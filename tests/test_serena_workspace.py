"""Read-only Serena workspace/transport tests, without running Serena or LLMs."""
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from bubbleteait.config import Config
from bubbleteait.qwen_runtime import atomic_json
from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
from bubbleteait.serena_workspace import ALLOWED_TOOLS, SerenaWorkspace, probe_symbols


class FakeSession:
    def __init__(self, reply, extra_tools=()):
        self.reply, self.extra_tools = reply, extra_tools
        self.calls, self.initialized = [], False

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name=n) for n in sorted(ALLOWED_TOOLS | set(self.extra_tools))])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        value = self.reply(name, arguments) if callable(self.reply) else self.reply
        return SimpleNamespace(is_error=False, structured_content={'result': json.dumps(value)})


class SerenaAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config(self.root, self.root / 'data', self.root / 'models')
        self.registry = Registry(self.config.data_dir)
        self.app_root = self.root / 'source'
        (self.app_root / 'trunk/Src').mkdir(parents=True)
        (self.app_root / 'trunk/Src/control.c').write_text(
            'int heatingAllowed(int trip) {\n    return trip == 0;\n}\n', encoding='utf-8')
        module_root = self.app_root / 'trunk/modules/control-module'
        module_root.mkdir(parents=True)
        (module_root / 'control.c').write_text('int sensorReady(void) {\n    return 1;\n}\n', encoding='utf-8')
        self.units = []
        for name, kind, root, include in (
                ('app', 'project', self.app_root, ('trunk/Src/*.c',)),
                ('control-module', 'module', module_root, ('*.c',))):
            self.registry.register(SourceSpec(name, kind, root, include=include, respect_gitignore=False))
            unit = self.registry.snapshot(name, IndexRecipe('text-fixture', 'none', 'none', 'lines'))
            atomic_json(self.config.data_dir / 'indexes' / unit['unit_id'] / 'status.json',
                {'schema_version': 1, 'unit_id': unit['unit_id'], 'recipe_hash': unit['recipe_hash'],
                 'backend': 'text-fixture', 'status': 'ready'})
            self.units.append(unit)
        self.composition = self.registry.compose('project', [u['unit_id'] for u in self.units], require_ready=True)
        self.workspace = SerenaWorkspace(self.config, self.composition, self.config.data_dir / 'serena/one')
        self.originals = {str(p): p.read_bytes() for p in self.app_root.rglob('*.c')}

    def assert_sources_unchanged(self):
        for filename, content in self.originals.items():
            self.assertEqual(Path(filename).read_bytes(), content)
        self.assertFalse((self.app_root / '.serena').exists())

    def request(self, tool='find_symbol'):
        return {'tool': tool, 'unit_id': self.units[0]['unit_id'],
                'file_path': 'trunk/Src/control.c', 'name': 'heatingAllowed'}

    def symbol(self, **changes):
        return {'name_path': 'heatingAllowed', 'relative_path': 'trunk/Src/control.c',
                'body_location': {'start_line': 0, 'end_line': 2},
                'body': 'int heatingAllowed(int trip) {\n    return trip == 0;\n}'} | changes

    async def probe(self, session, requests=None):
        @asynccontextmanager
        async def factory(parameters):
            self.parameters = parameters
            yield session
        return await probe_symbols(self.workspace, requests or [self.request()], executable=sys.executable,
            clangd=sys.executable, session_factory=factory)

    async def test_mounts_are_exact_snapshot_copies_with_origin_mapping(self):
        self.assertEqual(set(self.workspace.files), {'trunk/Src/control.c',
                                                     'trunk/modules/control-module/control.c'})
        for relative, record in self.workspace.files.items():
            path = self.workspace.project / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['source_sha256'])
        self.assertEqual(self.workspace.navigation_path(self.units[1]['unit_id'], 'control.c'),
                         'trunk/modules/control-module/control.c')
        with self.assertRaises(ValueError):
            SerenaWorkspace(self.config, self.composition, self.config.data_dir / 'serena/overlap',
                prefixes={self.units[1]['unit_id']: 'trunk/Src'})
        self.assert_sources_unchanged()

    async def test_symbol_zero_based_lines_map_to_bti_one_based_proof(self):
        session = FakeSession([self.symbol()])
        result = await self.probe(session)
        self.assertTrue(session.initialized)
        symbol = result['results'][0]['symbols'][0]
        self.assertEqual(symbol['unit_id'], self.units[0]['unit_id'])
        self.assertEqual((symbol['start_line'], symbol['end_line']), (1, 3))
        self.assertEqual(symbol['source_sha256'], self.units[0]['files'][0]['sha256'])
        self.assertEqual(result['compile_context'], 'unknown')
        self.assertEqual(result['mode'], 'standalone_optional_navigation')
        self.assertEqual(session.calls[0][1]['relative_path'], 'trunk/Src/control.c')
        self.assertIn('--mode', self.parameters.args)
        project_config = json.loads((self.workspace.project / '.serena/project.yml').read_text(encoding='utf-8'))
        self.assertTrue(project_config['read_only'])
        self.assertIsNone(project_config['activation_command'])
        self.assertEqual(project_config['ls_additional_workspace_folders'], [])
        self.assertEqual(self.parameters.env['SERENA_HOME'], str(self.workspace.home))
        self.assertEqual(self.parameters.env['SERENA_USAGE_REPORTING'], 'false')
        self.assert_sources_unchanged()

    async def test_read_only_tools_list_is_required_before_any_symbol_calls(self):
        session = FakeSession([self.symbol()], extra_tools=('write_memory',))
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            await self.probe(session)
        self.assertEqual(session.calls, [])

    async def test_foreign_requests_and_write_tools_are_rejected_without_startup(self):
        for changes in ({'tool': 'replace_symbol_body'}, {'unit_id': 'u-' + 'a' * 64},
                        {'file_path': '../private.c'}, {'file_path': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                await self.probe(FakeSession([]), [self.request() | changes])

    async def test_unmapped_output_files_forged_bodies_and_locations_are_rejected(self):
        for changes in ({'relative_path': 'outside.c'}, {'relative_path': '../outside.c'},
                        {'body': 'invented body'}, {'body_location': {'start_line': 0, 'end_line': 99}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                await self.probe(FakeSession([self.symbol(**changes)]))
        self.assert_sources_unchanged()

    async def test_private_copy_tampering_and_original_snapshot_tampering_fail(self):
        path = self.workspace.project / 'trunk/Src/control.c'
        original = path.read_bytes()
        path.write_bytes(b'modified private copy')
        with self.assertRaisesRegex(ValueError, 'Private Serena source changed'):
            await self.probe(FakeSession([]))
        path.write_bytes(original)
        snapshot = self.config.data_dir / 'sources' / self.units[0]['unit_id'] / 'files/trunk/Src/control.c'
        snapshot.write_bytes(b'modified immutable source')
        with self.assertRaises(ValueError):
            await self.probe(FakeSession([]))
        self.assert_sources_unchanged()

    async def test_overview_and_references_use_verified_source_mapping(self):
        overview = await self.probe(FakeSession({'Function': ['heatingAllowed']}),
                                    [self.request('get_symbols_overview')])
        source = overview['results'][0]['source']
        self.assertEqual(source['location_status'], 'not_provided')
        self.assertNotIn('start_line', source)
        reference = self.symbol()
        reference.pop('relative_path')
        reference.pop('body')
        result = await self.probe(FakeSession({'trunk\\Src\\control.c': {'Function': [reference]}}),
                                  [self.request('find_referencing_symbols')])
        self.assertEqual(result['results'][0]['symbols'][0]['location_kind'], 'enclosing_symbol')
        with self.assertRaisesRegex(ValueError, 'outside the pinned whitelist'):
            await self.probe(FakeSession({'outside.c': {}}), [self.request('find_referencing_symbols')])

    async def test_output_must_be_fresh_inside_data_root_and_pinned_refs_valid(self):
        for output in (self.root / 'outside', self.workspace.output, self.config.data_dir):
            with self.subTest(output=output), self.assertRaises(ValueError):
                SerenaWorkspace(self.config, self.composition, output)
        changed = dict(self.composition, composition_hash='0' * 64)
        with self.assertRaises(ValueError):
            SerenaWorkspace(self.config, changed, self.config.data_dir / 'serena/tampered')

    async def test_empty_reference_readiness_retry_keeps_same_session_and_expected_source(self):
        from unittest.mock import AsyncMock, patch
        count = 0
        reference = self.symbol()
        reference.pop('relative_path')
        reference.pop('body')
        def reply(tool, arguments):
            nonlocal count
            count += 1
            return {} if count == 1 else {'trunk/Src/control.c': {'Function': [reference]}}
        session = FakeSession(reply)
        @asynccontextmanager
        async def factory(parameters):
            yield session
        with patch('bubbleteait.serena_workspace.asyncio.sleep', new_callable=AsyncMock) as wait:
            result = await probe_symbols(self.workspace, [self.request('find_referencing_symbols')],
                executable=sys.executable, clangd=sys.executable, session_factory=factory,
                retry_empty_references=3)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(wait.await_count, 1)
        mapped = result['results'][0]
        self.assertEqual(mapped['query_attempts'], 2)
        symbol = mapped['symbols'][0]
        self.assertEqual((symbol['unit_id'], symbol['file_path'], symbol['start_line'], symbol['end_line']),
                         (self.units[0]['unit_id'], 'trunk/Src/control.c', 1, 3))

    async def test_reference_readiness_is_bounded_and_never_turns_empty_into_evidence(self):
        from unittest.mock import AsyncMock, patch
        session = FakeSession({})
        @asynccontextmanager
        async def factory(parameters):
            yield session
        with patch('bubbleteait.serena_workspace.asyncio.sleep', new_callable=AsyncMock) as wait:
            result = await probe_symbols(self.workspace, [self.request('find_referencing_symbols')],
                executable=sys.executable, clangd=sys.executable, session_factory=factory,
                retry_empty_references=3)
        self.assertEqual(len(session.calls), 4)
        self.assertEqual(wait.await_count, 3)
        self.assertEqual(result['results'][0]['query_attempts'], 4)
        self.assertEqual(result['results'][0]['symbols'], [])


if __name__ == '__main__':
    unittest.main()
