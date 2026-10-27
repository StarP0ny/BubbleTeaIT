"""Public MCP symbols contract: scoped locators, async errors, no inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bubbleteait.config import Config
from bubbleteait.mcpserver import build_server
from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
from bubbleteait.symbols import SymbolError

HAS_MCP = importlib.util.find_spec('mcp') is not None
if HAS_MCP:
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError


@unittest.skipUnless(HAS_MCP, 'Install the mcp extra')
class SymbolsMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_guarded_read_rejects_stale_unit_after_composition_publication(self):
        request={'source_id':'alpha-app','path':'src/control.c','start':1,'end':3,'unit_id':self.unit['unit_id']}
        first=await self.server.call_tool('read_sources',{'composition_id':'alpha','requests':[request]})
        self.assertFalse(first.is_error)
        self.assertEqual(first.structured_content['excerpts'][0]['lines'],[1,3])
        (self.source/'src/control.c').write_text('int heatingAllowed(int trip) {\n    return trip == 1;\n}\n',encoding='utf-8')
        self.publish()
        with self.assertRaisesRegex(ToolError,'Source version changed'):
            await self.server.call_tool('read_sources',{'composition_id':'alpha','requests':[request]})
        current=await self.server.call_tool('read_sources',{'composition_id':'alpha','requests':[
            {**request,'unit_id':self.unit['unit_id']}]})
        self.assertIn('trip == 1',current.structured_content['excerpts'][0]['text'])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/'checkout'
        (self.source/'src').mkdir(parents=True)
        (self.source/'src/control.c').write_text(
            'int heatingAllowed(int trip) {\n    return trip == 0;\n}\n', encoding='utf-8')
        self.config = Config(self.root, self.root/'data', self.root/'models')
        self.registry = Registry(self.config.data_dir)
        self.registry.register(SourceSpec('alpha-app', 'project', self.source))
        self.recipe = IndexRecipe('text-fixture', 'none', 'none', 'lines')
        self.publish()
        self.server = build_server(self.config)

    def publish(self):
        self.unit = self.registry.snapshot('alpha-app', self.recipe)
        status = self.config.data_dir/'indexes'/self.unit['unit_id']/'status.json'
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({'schema_version':1, 'unit_id':self.unit['unit_id'],
            'recipe_hash':self.unit['recipe_hash'], 'backend':'text-fixture', 'status':'ready'}), encoding='utf-8')
        self.composition = self.registry.compose('alpha', [self.unit['unit_id']], require_ready=True)

    def arguments(self, **changes):
        return {'composition_id':'alpha', 'operation':'definition', 'source_id':'alpha-app',
                'path':'src/control.c', 'name':'heatingAllowed', **changes}

    def response(self):
        return {'composition_id':'alpha', 'composition_hash':self.composition['composition_hash'],
            'compile_context':'unknown', 'navigation_only':True, 'results':[
                {'source_id':'s-alpha-fixture', 'source':'alpha-app', 'unit_id':self.unit['unit_id'],
                 'path':'src/control.c', 'source_sha256':self.unit['files'][0]['sha256'],
                 'name_path':'heatingAllowed', 'lines':[1,3],
                 'read':{'source_id':'s-alpha-fixture', 'path':'src/control.c', 'start':1, 'end':3}}]}

    async def test_schema_is_read_only_and_bad_arguments_never_construct_the_service(self):
        tools = {tool.name:tool for tool in await self.server.list_tools()}
        tool = tools['code_symbols']
        self.assertTrue(tool.annotations.read_only_hint)
        self.assertFalse(tool.annotations.destructive_hint)
        properties = tool.input_schema['properties']
        self.assertEqual(set(properties['operation']['enum']), {'overview','definition','references'})
        self.assertEqual(properties['limit']['minimum'], 1)
        self.assertEqual(properties['limit']['maximum'], 40)
        self.assertEqual(properties['limit']['default'], 20)
        self.assertEqual(properties['max_chars']['minimum'], 512)
        self.assertEqual(properties['max_chars']['maximum'], 16000)
        self.assertEqual(properties['max_chars']['default'], 8000)
        self.assertEqual(set(tool.input_schema['required']), {'operation', 'path'})
        invalid = ({'operation':'edit'}, {'limit':True}, {'limit':'3'}, {'limit':0},
                   {'limit':41}, {'max_chars':True}, {'max_chars':'8000'},
                   {'max_chars':511}, {'max_chars':16001}, {'path':17})
        with patch('bubbleteait.symbols.SymbolService', side_effect=AssertionError('Invalid wire reached sources')) as service:
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ToolError):
                    await self.server.call_tool('code_symbols', self.arguments(**changes))
            missing = self.arguments()
            del missing['path']
            with self.assertRaises(ToolError):
                await self.server.call_tool('code_symbols', missing)
            service.assert_not_called()

    async def test_async_query_forwards_scope_and_returns_the_expected_source_locator(self):
        direct = MagicMock()
        direct.composition = self.composition
        direct.query = AsyncMock(return_value=self.response())
        with patch('bubbleteait.symbols.SymbolService', return_value=direct) as service:
            result = await self.server.call_tool('code_symbols', self.arguments(limit=3, max_chars=1400))
        service.assert_called_once_with(self.config, 'alpha')
        direct.query.assert_awaited_once_with('definition', 'alpha-app', 'src/control.c',
            name='heatingAllowed', limit=3, max_chars=1400)
        locator = result.structured_content['results'][0]
        self.assertEqual((locator['source'], locator['unit_id'], locator['path']),
                         ('alpha-app', self.unit['unit_id'], 'src/control.c'))
        self.assertEqual(locator['read'], {'source_id':'s-alpha-fixture', 'path':'src/control.c', 'start':1, 'end':3})
        self.assertNotIn('id', locator)
        self.assertTrue(result.structured_content['navigation_only'])
        self.assertEqual(result.structured_content['compile_context'], 'unknown')

    async def test_cached_service_refreshes_after_publishing_a_different_snapshot(self):
        created = []
        def create(config, composition_id):
            service = MagicMock()
            self.assertEqual(composition_id, 'alpha')
            service.composition = dict(self.composition)
            service.query = AsyncMock(return_value=self.response())
            created.append(service)
            return service
        with patch('bubbleteait.symbols.SymbolService', side_effect=create):
            await self.server.call_tool('code_symbols', self.arguments())
            await self.server.call_tool('code_symbols', self.arguments(operation='overview', name=None))
            self.assertEqual(len(created), 1)
            old_unit = self.unit['unit_id']
            (self.source/'src/control.c').write_text('int heatingAllowed(void) { return 1; }\n', encoding='utf-8')
            self.publish()
            result = await self.server.call_tool('code_symbols', self.arguments())
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].query.await_count, 2)
        self.assertNotEqual(self.unit['unit_id'], old_unit)
        self.assertEqual(result.structured_content['results'][0]['unit_id'], self.unit['unit_id'])

    async def test_async_domain_diagnostic_reaches_sdk_but_unexpected_errors_remain_masked(self):
        service = MagicMock()
        service.composition = self.composition
        service.query = AsyncMock(side_effect=SymbolError('Unknown pinned path; suggested_paths=[src/control.c]'))
        with patch('bubbleteait.symbols.SymbolService', return_value=service):
            with self.assertRaises(ToolError) as raised:
                await self.server.call_tool('code_symbols', self.arguments())
            self.assertNotIsInstance(raised.exception, UnexpectedToolError)
            self.assertIn('suggested_paths=[src/control.c]', str(raised.exception))
            service.query.side_effect = RuntimeError('private programming failure')
            with self.assertRaises(UnexpectedToolError) as unexpected:
                await self.server.call_tool('code_symbols', self.arguments())
            self.assertNotIn('private programming failure', str(unexpected.exception))

    async def test_missing_symbols_config_is_actionable_without_cloud_or_embeddings(self):
        self.assertIsNone(self.config.llm)
        self.assertIsNone(self.config.symbols)
        with patch('bubbleteait.application.make_backend', side_effect=AssertionError('Symbols loaded embeddings')) as backend, \
             patch('bubbleteait.serena_workspace.probe_symbols', new_callable=AsyncMock) as probe:
            with self.assertRaises(ToolError) as raised:
                await self.server.call_tool('code_symbols', self.arguments())
            self.assertNotIsInstance(raised.exception, UnexpectedToolError)
            self.assertIn('symbols.executable', str(raised.exception))
            self.assertIn('symbols.clangd_path', str(raised.exception))
            backend.assert_not_called()
            probe.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
