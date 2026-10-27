"""Public compact MCP methods share a version-scoped evidence ledger."""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from bubbleteait.config import Config
from bubbleteait.mcpserver import build_server
from bubbleteait.registry import Registry, SourceSpec, IndexRecipe

HAS_MCP = importlib.util.find_spec('mcp') is not None
if HAS_MCP:
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError


class CaptureServer:
    def __init__(self,*args,**kwargs):self.tools={}
    def tool(self,**kwargs):
        def register(function):self.tools[function.__name__]=function;return function
        return register


class EmptyBackend:
    def search(self,*args,**kwargs):return []


@unittest.skipUnless(HAS_MCP, 'Install the mcp extra')
class DirectMCPTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'source';self.source.mkdir()
        (self.source/'control.c').write_text('int threshold = 37;\n',encoding='utf-8')
        self.config=Config(self.root,self.root/'data',self.root/'models')
        self.registry=Registry(self.config.data_dir)
        self.registry.register(SourceSpec('app','project',self.source,role='code'))
        self.recipe=IndexRecipe('fixture','none','none','lines')
        self.publish()
        module=ModuleType('mcp.server.mcpserver');module.MCPServer=CaptureServer
        with patch.dict(sys.modules,{module.__name__:module}):self.server=build_server(self.config)
        patcher=patch('bubbleteait.direct.make_backend',return_value=EmptyBackend());patcher.start();self.addCleanup(patcher.stop)

    def publish(self):
        unit=self.registry.snapshot('app',self.recipe)
        path=self.config.data_dir/'indexes'/unit['unit_id']/'status.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'schema_version':1,'unit_id':unit['unit_id'],'recipe_hash':unit['recipe_hash'],'backend':'fixture','status':'ready'}),encoding='utf-8')
        self.registry.compose('project',[unit['unit_id']],require_ready=True)

    def test_search_batch_and_resolution_use_same_public_ledger(self):
        result=self.server.tools['context_search']('project','threshold')
        identifier=result['excerpts'][0]['id']
        self.assertEqual(result['excerpts'][0]['text'],(self.source/'control.c').read_bytes().decode('utf-8'))
        batch=self.server.tools['read_sources']('project',[identifier])
        self.assertEqual(batch['excerpts'][0]['id'],identifier)
        canonical=self.server.tools['resolve_sources']('project',[identifier])['evidence'][0]
        self.assertEqual(canonical['file_path'],'control.c')
        self.assertEqual(len(canonical['source_sha256']),64)

    def test_mixed_issued_id_and_path_batch_requires_explicit_composition(self):
        identifier = self.server.tools['context_search']('project', 'threshold')['excerpts'][0]['id']
        with self.assertRaises(ToolError):
            self.server.tools['read_sources'](None, [identifier, {'path': 'control.c'}])

    def test_composition_switch_does_not_resolve_old_evidence_in_new_scope(self):
        identifier=self.server.tools['context_search']('project','threshold')['excerpts'][0]['id']
        (self.source/'control.c').write_text('int threshold = 42;\n',encoding='utf-8');self.publish()
        with self.assertRaises(ToolError):self.server.tools['resolve_sources']('project',[identifier])
        self.assertIn('42',self.server.tools['context_search']('project','threshold')['excerpts'][0]['text'])

    def test_navigation_returns_snapshot_read_requests_without_embedding_calls(self):
        with patch.object(EmptyBackend, 'search', side_effect=AssertionError('Navigation loaded embeddings')):
            sources = self.server.tools['list_sources']('project')
            self.assertEqual(sources['configuration_status'], 'unknown')
            self.assertIsNone(sources['configuration'])
            alias = sources['entries'][0]['source_id']
            files = self.server.tools['list_files']('project', query='CONTROL', source_ids=[alias])
            entry = files['entries'][0]
            self.assertEqual(entry['path'], 'control.c')
            read = self.server.tools['read_sources']('project', [entry['read']], include_catalog=False)
        self.assertIn('threshold = 37', read['excerpts'][0]['text'])
        self.assertEqual(read['catalog'], [])
        identifier = read['excerpts'][0]['id']
        self.assertEqual(self.server.tools['resolve_sources']('project', [identifier])['evidence'][0]['file_path'],
                         entry['path'])

    def test_navigation_scope_and_composition_refresh_do_not_leak_working_content(self):
        before = self.server.tools['list_files']('project')
        (self.source/'control.c').write_text('int threshold = 42;\nint added;\n', encoding='utf-8')
        unchanged = self.server.tools['list_files']('project')
        self.assertEqual(unchanged['composition_hash'], before['composition_hash'])
        self.assertEqual(unchanged['entries'][0]['total_lines'], 1)
        with self.assertRaises(ToolError):
            self.server.tools['list_files']('project', source_ids=['outside'])
        self.publish()
        refreshed = self.server.tools['list_files']('project')
        self.assertNotEqual(refreshed['composition_hash'], before['composition_hash'])
        self.assertEqual(refreshed['entries'][0]['total_lines'], 2)


@unittest.skipUnless(HAS_MCP, 'Install the mcp extra')
class DirectNavigationWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_domain_errors_preserve_read_hints_and_snapshot_scope_through_sdk(self):
        from unittest.mock import MagicMock
        from bubbleteait.config import ConfigError
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            source.mkdir()
            (source/'control.c').write_text('int value;\n' * 130, encoding='utf-8')
            registry = Registry(root/'data')
            registry.register(SourceSpec('source', 'module', source))
            unit = registry.snapshot('source', IndexRecipe('fixture', 'none', 'none', 'lines'))
            status = root/'data/indexes'/unit['unit_id']/'status.json'
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({'schema_version':1, 'unit_id':unit['unit_id'],
                'recipe_hash':unit['recipe_hash'], 'backend':'fixture', 'status':'ready'}), encoding='utf-8')
            composition = registry.compose('project', [unit['unit_id']], require_ready=True)
            server = build_server(Config(root, root/'data', root/'models'))
            backend = patch('bubbleteait.direct.make_backend', return_value=EmptyBackend())
            backend.start()
            self.addCleanup(backend.stop)
            # A checkout-only path must never become a snapshot suggestion.
            (source/'control-local-only.c').write_text('unindexed;\n', encoding='utf-8')
            common = {'composition_id':'project'}
            first = await server.call_tool('read_sources', {**common,
                'requests':[{'source_id':'source', 'path':'control.c', 'start':1, 'end':1}]})
            identifier = first.structured_content['excerpts'][0]['id']
            bad_requests = [
                ({'source_id':'source', 'path':'control.c', 'start':1, 'end':125}, 'read_windows'),
                ({'source_id':'source', 'path':'conrol.c', 'start':1, 'end':1}, 'control.c')]
            for request, diagnostic in bad_requests:
                with self.subTest(request=request), self.assertRaises(ToolError) as raised:
                    await server.call_tool('read_sources', {**common, 'requests':[request]})
                self.assertNotIsInstance(raised.exception, UnexpectedToolError)
                self.assertIn(diagnostic, str(raised.exception))
                self.assertNotIn('control-local-only.c', str(raised.exception))
            with self.assertRaises(ToolError) as forged:
                await server.call_tool('resolve_sources', {**common, 'ids':['e-forged']})
            self.assertNotIsInstance(forged.exception, UnexpectedToolError)
            self.assertIn('Unknown or forged', str(forged.exception))
            resolved = await server.call_tool('resolve_sources', {**common, 'ids':[identifier]})
            self.assertEqual(resolved.structured_content['evidence'][0]['file_path'], 'control.c')

            # Expected configuration errors stay useful, while programmer errors
            # retain the SDK's masking instead of exposing arbitrary exception text.
            clean_server = build_server(Config(root, root/'data', root/'models'))
            with patch('bubbleteait.direct.DirectService', side_effect=ConfigError('Fixture backend is not configured')):
                with self.assertRaises(ToolError) as configuration:
                    await clean_server.call_tool('list_sources', common)
                self.assertNotIsInstance(configuration.exception, UnexpectedToolError)
                self.assertIn('Fixture backend is not configured', str(configuration.exception))
            direct = MagicMock()
            direct.composition = composition
            direct.list_files.side_effect = RuntimeError('private programmer diagnostic')
            with patch('bubbleteait.direct.DirectService', return_value=direct):
                with self.assertRaises(UnexpectedToolError) as unexpected:
                    await clean_server.call_tool('list_files', common)
                self.assertNotIn('private programmer diagnostic', str(unexpected.exception))

    async def test_navigation_schema_bounds_reject_bad_values_before_service_construction(self):
        from mcp.server.mcpserver.exceptions import ToolError
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = build_server(Config(root, root/'data', root/'models'))
            listed = {tool.name: tool for tool in await server.list_tools()}
            self.assertEqual(set(listed), {'project_info', 'search', 'read_source', 'index_status', 'context_search', 'code_symbols',
                'read_sources', 'resolve_sources', 'list_sources', 'list_files',
                'sync_project', 'sync_status', 'watch_status'})
            self.assertFalse(listed['sync_project'].annotations.read_only_hint)
            for name in set(listed) - {'sync_project'}:
                self.assertTrue(listed[name].annotations.read_only_hint)
                self.assertFalse(listed[name].annotations.destructive_hint)
            for name in ('list_sources', 'list_files'):
                properties = listed[name].input_schema['properties']
                self.assertEqual(properties['offset']['minimum'], 0)
                self.assertEqual(properties['limit']['maximum'], 200)
                self.assertEqual(properties['max_chars']['maximum'], 64000)
            invalid = [('list_files', {'offset': True}), ('list_sources', {'offset': -1}),
                       ('list_files', {'limit': 201}), ('list_sources', {'limit': 0}),
                       ('list_files', {'max_chars': 511}), ('list_sources', {'max_chars': 64001}),
                       ('context_search', {'query': 'value', 'include_catalog': 'false'}),
                       ('context_search', {'query': 'value', 'limit': True}),
                       ('context_search', {'query': 'value', 'limit': '4'}),
                       ('context_search', {'query': 'value', 'max_chars': '6000'}),
                       ('read_sources', {'requests': ['issued'], 'max_chars': '6000'}),
                       ('read_sources', {'requests': ['issued'], 'max_chars': True}),
                       ('read_sources', {'requests': [], 'include_catalog': 1}),
                       ('read_sources', {'requests': [{'source_id':'s', 'path':'a.c', 'start':True, 'end':1}]}),
                       ('read_sources', {'requests': [{'id':'issued', 'path':'guessed.c'}]}),
                       ('read_sources', {'requests': [{'source_id':'s', 'path':'a.c', 'start':0, 'end':1}]}),
                       ('read_sources', {'requests': ['issued'] * 21})]
            with patch('bubbleteait.direct.DirectService', side_effect=AssertionError('Bad wire arguments reached sources')) as direct:
                for name, values in invalid:
                    with self.subTest(name=name, values=values), self.assertRaises(ToolError):
                        await server.call_tool(name, {'composition_id': 'project', **values})
                direct.assert_not_called()

    async def test_read_wire_union_normalizes_existing_shapes_and_leaves_range_hints_to_service(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = Registry(root/'data')
            source = root/'source'
            source.mkdir()
            (source/'a.c').write_text('int value;\n', encoding='utf-8')
            registry.register(SourceSpec('source', 'module', source))
            unit = registry.snapshot('source', IndexRecipe('fixture', 'none', 'none', 'lines'))
            status = root/'data/indexes'/unit['unit_id']/'status.json'
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({'schema_version':1, 'unit_id':unit['unit_id'],
                'recipe_hash':unit['recipe_hash'], 'backend':'fixture', 'status':'ready'}), encoding='utf-8')
            composition = registry.compose('project', [unit['unit_id']], require_ready=True)
            server = build_server(Config(root, root/'data', root/'models'))
            direct = MagicMock()
            direct.composition = composition
            direct.read_many.return_value = {'excerpts': [], 'truncated': True}
            request = {'source_id': 'source', 'path': 'a.c', 'start': 110, 'end': 244}
            with patch('bubbleteait.direct.DirectService', return_value=direct):
                result = await server.call_tool('read_sources', {'composition_id':'project',
                    'requests':['issued', {'id':'issued'}, request], 'include_catalog':False})
            self.assertFalse(result.is_error)
            direct.read_many.assert_called_once_with(['issued', {'id':'issued'}, request],
                max_chars=16000, include_catalog=False)


if __name__=='__main__':unittest.main()
