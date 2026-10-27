"""Real private source copies and fake Serena transport; no model or LSP calls."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from bubbleteait.config import Config, SymbolsConfig
from bubbleteait.direct import DirectService
from bubbleteait.qwen_runtime import atomic_json
from bubbleteait.registry import IndexRecipe, Registry, SourceSpec
from bubbleteait.symbols import SymbolError, SymbolService
from bubbleteait.search import SearchError
import bubbleteait.symbols as symbols_module
from test_serena_workspace import FakeSession


class SymbolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = Config(self.root, self.root / 'data', self.root / 'models',
                             symbols=SymbolsConfig(sys.executable, Path(sys.executable), 2))
        self.registry = Registry(self.config.data_dir)
        self.text = 'int heatingAllowed(int trip) {\r\n    return trip == 0;\r\n}\r\n'
        self.unit = self.make_unit('app', 'project', {
            'src/control.c': self.text,
            'src/caller.c': 'int caller(void) {\n    return heatingAllowed(0);\n}\n'})
        self.registry.compose('symbols', [self.unit['unit_id']], require_ready=True)
        self.service = SymbolService(self.config, 'symbols')
        self.reply = self.definition()
        self.session = FakeSession(lambda name, arguments: self.reply)
        self.parameters = []

        @asynccontextmanager
        async def factory(parameters):
            self.parameters.append(parameters)
            yield self.session

        self.transport = patch('bubbleteait.serena_workspace._session', factory)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        backend = patch('bubbleteait.application.make_backend', side_effect=AssertionError('No embedding backend'))
        backend.start()
        self.addCleanup(backend.stop)

    def make_unit(self, name, kind, files):
        source = self.root / name
        for relative, text in files.items():
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(text.encode('utf-8'))
        self.registry.register(SourceSpec(name, kind, source, respect_gitignore=False))
        unit = self.registry.snapshot(name, IndexRecipe('fixture', 'none', 'none', 'lines'))
        atomic_json(self.config.data_dir / 'indexes' / unit['unit_id'] / 'status.json',
                    {'schema_version': 1, 'unit_id': unit['unit_id'], 'recipe_hash': unit['recipe_hash'],
                     'backend': 'fixture', 'status': 'ready'})
        return unit

    def definition(self, **changes):
        return [{'name_path': 'heatingAllowed', 'relative_path': 'src/control.c',
                 'body_location': {'start_line': 0, 'end_line': 2},
                 'body': self.text.strip()} | changes]

    async def query(self, operation='definition', source='app', path='src/control.c', **kwargs):
        name = None if operation == 'overview' else 'heatingAllowed'
        return await self.service.query(operation, source, path, name=name, **kwargs)

    async def test_definition_is_snapshot_locator_and_never_evidence_id(self):
        result = await self.query()
        self.assertTrue(result['navigation_only'])
        self.assertEqual(result['compile_context'], 'unknown')
        self.assertEqual(result['configuration_status'], 'unknown')
        entry = result['results'][0]
        alias = 's-' + hashlib.sha256(b'app').hexdigest()[:12]
        self.assertEqual(entry['source_id'], alias)
        self.assertEqual(entry['path'], 'src/control.c')
        self.assertEqual(entry['lines'], [1, 3])
        self.assertEqual(entry['source_sha256'], hashlib.sha256(self.text.encode()).hexdigest())
        self.assertEqual(entry['read'], {'source_id': alias, 'path': 'src/control.c',
                                        'unit_id': self.unit['unit_id'], 'start': 1, 'end': 3})
        self.assertNotIn('id', entry)
        self.assertNotIn('content', entry)
        self.assertIsNone(self.config.qwen)
        self.assertIsNone(self.config.cocoindex)
        self.assertEqual(self.parameters[0].env['SERENA_USAGE_REPORTING'], 'false')
        self.assertFalse((self.root / 'app/.serena').exists())

    async def test_same_hash_reuses_private_workspace_across_service_instances_and_aliases(self):
        first = await self.query()
        self.assertFalse(first['workspace_reused'])
        original_workspace = symbols_module._CACHE[self.service._cache_key].workspace
        self.service = SymbolService(self.config, 'symbols')
        second = await self.query(source=first['results'][0]['source_id'])
        self.assertTrue(second['workspace_reused'])
        self.assertIs(symbols_module._CACHE[self.service._cache_key].workspace, original_workspace)
        self.assertEqual(first['results'], second['results'])
        self.assertIsNone(original_workspace.service.registry.manifests)

    async def test_declared_configuration_never_claims_verified_compile_context(self):
        configuration = {'BUILD': 'A', 'build': {'variant': 3}}
        self.registry.compose('configured', [self.unit['unit_id']], configuration=configuration, require_ready=True)
        self.service = SymbolService(self.config, 'configured')
        result = await self.query()
        self.assertEqual(result['configuration'], configuration)
        self.assertEqual(result['configuration_status'], 'declared')
        self.assertEqual(result['compile_context'], 'unknown')

    async def test_overview_has_no_fabricated_locations_and_reference_is_enclosing_symbol(self):
        self.reply = {'Function': ['heatingAllowed']}
        overview = await self.query('overview')
        entry = overview['results'][0]
        self.assertEqual(entry['location_status'], 'not_provided')
        self.assertEqual(entry['location_kind'], 'unlocated_symbol')
        self.assertNotIn('lines', entry)
        self.assertEqual(entry['read']['start'], 1)
        self.reply = {'src/caller.c': {'Function': [{'name_path': 'caller',
                       'body_location': {'start_line': 0, 'end_line': 2}}]}}
        references = await self.query('references')
        self.assertEqual(references['results'][0]['path'], 'src/caller.c')
        self.assertEqual(references['results'][0]['location_kind'], 'enclosing_symbol')
        self.assertEqual(references['completeness'], 'not_established')

    async def test_large_symbol_has_executable_bounded_source_windows(self):
        text = ''.join(f'int line{number};\n' for number in range(200))
        unit = self.make_unit('single-module', 'module', {'long.c': text})
        self.registry.compose('long', [unit['unit_id']], require_ready=True)
        self.service = SymbolService(self.config, 'long')
        self.reply = [{'name_path': 'largeSymbol', 'relative_path': 'long.c',
                       'body_location': {'start_line': 0, 'end_line': 199}}]
        result = await self.query(source='single-module', path='long.c')
        entry = result['results'][0]
        self.assertEqual(entry['lines'], [1, 200])
        self.assertEqual((entry['read']['start'], entry['read']['end']), (1, 120))
        self.assertEqual((entry['next_read']['start'], entry['next_read']['end']), (121, 200))
        self.assertEqual(entry['next_read']['unit_id'], unit['unit_id'])

    async def test_foreign_file_coordinates_and_forged_bodies_fail_closed(self):
        for changes in ({'relative_path': 'outside.c'},
                        {'relative_path': 'src/caller.c', 'body': None},
                        {'body_location': {'start_line': 0, 'end_line': 99}},
                        {'body': 'int fabricated;'}):
            self.reply = self.definition(**changes)
            with self.subTest(changes=changes), self.assertRaises(SymbolError):
                await self.query()

    async def test_private_copy_or_injected_compile_configuration_is_not_reused(self):
        for relative, content in (('src/control.c', b'corrupt private source'),
                                  ('compile_commands.json', b'[]')):
            await self.query()
            prepared = symbols_module._CACHE[self.service._cache_key]
            (prepared.workspace.project / relative).write_bytes(content)
            calls = len(self.session.calls)
            with self.subTest(relative=relative), self.assertRaises(SymbolError):
                await self.query()
            self.assertEqual(len(self.session.calls), calls)
            self.assertNotIn(self.service._cache_key, symbols_module._CACHE)

    async def test_unselected_snapshot_mutation_during_probe_and_next_request_fail(self):
        snapshot = self.config.data_dir / 'sources' / self.unit['unit_id'] / 'files/src/caller.c'
        original = snapshot.read_bytes()

        def mutate(name, arguments):
            snapshot.write_bytes(b'corrupt snapshot')
            return self.definition()

        self.session.reply = mutate
        with self.assertRaises(SymbolError):
            await self.query()
        self.assertNotIn(self.service._cache_key, symbols_module._CACHE)
        self.session.reply = lambda name, arguments: self.definition()
        with self.assertRaises(SymbolError):
            await self.query()
        snapshot.write_bytes(original)
        self.assertEqual((await self.query())['returned'], 1)

    async def test_changed_composition_and_multiple_products_require_focused_service(self):
        self.registry.compose('symbols', [self.unit['unit_id']], configuration={'BUILD': 'new'}, require_ready=True)
        with self.assertRaisesRegex(SymbolError, 'Composition changed'):
            await self.query()
        foreign = self.make_unit('other-product', 'project', {'other.c': 'int other;\n'})
        self.registry.compose('mixed', [self.unit['unit_id'], foreign['unit_id']], require_ready=True)
        with self.assertRaisesRegex(SymbolError, 'focused composition'):
            SymbolService(self.config, 'mixed')

    async def test_limits_budget_and_invalid_requests_are_bounded_before_startup(self):
        invalid = ({'operation': []}, {'limit': True}, {'limit': 0}, {'max_chars': 511},
                   {'source': 'foreign'}, {'path': '../private.c'}, {'path': 'missing.c'})
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                await self.query(**change)
        self.assertEqual(self.session.calls, [])
        self.reply = {'Function': ['symbol' + str(number) for number in range(30)]}
        result = await self.query('overview', limit=2, max_chars=1800)
        self.assertEqual(result['total'], 30)
        self.assertLessEqual(result['returned'], 2)
        self.assertTrue(result['truncated'])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 1800)

    async def test_unconfigured_and_timeout_have_actionable_errors_without_models(self):
        with self.assertRaisesRegex(SymbolError, 'Configure symbols'):
            SymbolService(replace(self.config, symbols=None), 'symbols')
        self.service.options = replace(self.config.symbols, timeout_seconds=.001)

        async def slow(name, arguments):
            await asyncio.sleep(.05)
            raise AssertionError('Should have timed out')

        self.session.call_tool = slow
        with self.assertRaisesRegex(SymbolError, 'timed out'):
            await self.query()
        self.assertNotIn(self.service._cache_key, symbols_module._CACHE)

    async def test_mcp_task_group_transport_failure_is_an_actionable_symbol_error(self):
        async def fail(name, arguments):
            raise ExceptionGroup('outer anyio context', [ExceptionGroup('inner MCP context',
                [OSError('closed transport'), ValueError('Invalid source coordinates')])])

        self.session.call_tool = fail
        with self.assertRaisesRegex(SymbolError, 'MCP process or tool session failed'):
            await self.query()
        self.assertNotIn(self.service._cache_key, symbols_module._CACHE)

    async def test_task_groups_do_not_mask_programmer_errors_or_cancellation(self):
        original = ExceptionGroup('mixed errors', [ValueError('Expected input failure'),
                                                   AssertionError('Programming defect')])

        async def fail(name, arguments):
            raise original

        self.session.call_tool = fail
        with self.assertRaises(ExceptionGroup) as error:
            await self.query()
        self.assertIs(error.exception, original)
        self.assertNotIn(self.service._cache_key, symbols_module._CACHE)

        async def cancel(name, arguments):
            raise asyncio.CancelledError()

        self.session.call_tool = cancel
        with self.assertRaises(asyncio.CancelledError):
            await self.query()
        self.assertNotIn(self.service._cache_key, symbols_module._CACHE)

    async def test_synthetic_database_enumerates_every_private_translation_unit_with_fixed_flags(self):
        result = await self.query()
        prepared = symbols_module._CACHE[self.service._cache_key]
        workspace = prepared.workspace
        commands = json.loads((workspace.project / 'compile_commands.json').read_text())
        self.assertEqual({Path(command['file']).relative_to(workspace.project).as_posix() for command in commands},
                         {'src/control.c', 'src/caller.c'})
        for command in commands:
            self.assertEqual(Path(command['directory']), workspace.project)
            arguments = command['arguments']
            self.assertEqual(arguments[:4], ['clang', '-fsyntax-only', '-ferror-limit=0', '-std=gnu11'])
            self.assertTrue(Path(arguments[-1]).is_relative_to(workspace.project))
            self.assertFalse(any(argument.startswith(('-D', '--target', '-include')) for argument in arguments))
            for index, argument in enumerate(arguments):
                if argument == '-I':
                    self.assertTrue(Path(arguments[index + 1]).is_relative_to(workspace.project))
        profile = result['navigation_profile']
        self.assertEqual(profile['translation_units'], 2)
        self.assertFalse(profile['active_build_verified'])
        self.assertEqual(profile['compile_db_sha256'], hashlib.sha256(
            (workspace.project / 'compile_commands.json').read_bytes()).hexdigest())
        self.assertEqual(self.service._cache_key[-1], profile['kind'])

    async def test_duplicate_header_names_are_deterministic_and_explicitly_ambiguous(self):
        unit = self.make_unit('header-profile', 'project', {'src/control.c': self.text,
            'src/extra.cpp': 'int extra() { return 1; }\n',
            'a/shared.h': '#define A 1\n', 'b/shared.h': '#define B 1\n'})
        self.registry.compose('headers', [unit['unit_id']], require_ready=True)
        self.service = SymbolService(self.config, 'headers')
        result = await self.query(source='header-profile')
        profile = result['navigation_profile']
        self.assertEqual(profile['header_directories'], 2)
        self.assertEqual(profile['ambiguous_header_names'], 1)
        prepared = symbols_module._CACHE[self.service._cache_key]
        commands = json.loads((prepared.workspace.project / 'compile_commands.json').read_text())
        cpp = next(command for command in commands if command['file'].endswith('extra.cpp'))
        self.assertIn('-std=gnu++17', cpp['arguments'])
        included = [Path(cpp['arguments'][index + 1]).relative_to(prepared.workspace.project).as_posix()
                    for index, argument in enumerate(cpp['arguments']) if argument == '-I']
        self.assertEqual(included, ['src', '.', 'a', 'b'])
        self.assertEqual(result['compile_context'], 'unknown')

    async def test_generated_profile_manifest_mutation_is_rejected_before_mcp_start(self):
        await self.query()
        prepared = symbols_module._CACHE[self.service._cache_key]
        artifact = prepared.workspace.output / 'navigation-profile.json'
        value = json.loads(artifact.read_text())
        value['compile_context'] = 'forged-active-build'
        artifact.write_text(json.dumps(value))
        calls = len(self.session.calls)
        with self.assertRaisesRegex(SymbolError, 'trusted generated hash'):
            await self.query()
        self.assertEqual(len(self.session.calls), calls)
        self.assertNotIn(self.service._cache_key, symbols_module._CACHE)

    async def test_nested_compile_database_cannot_override_the_trusted_synthetic_database(self):
        await self.query()
        workspace = symbols_module._CACHE[self.service._cache_key].workspace
        (workspace.project / 'src/compile_commands.json').write_text('[]')
        calls = len(self.session.calls)
        with self.assertRaisesRegex(SymbolError, 'unverified compile configuration'):
            await self.query()
        self.assertEqual(len(self.session.calls), calls)

    async def test_empty_reference_retry_finds_a_cross_file_caller_in_the_same_session(self):
        reference = {'src/caller.c': {'Function': [{'name_path': 'caller',
                     'body_location': {'start_line': 0, 'end_line': 2}}]}}
        self.session.reply = lambda name, arguments: {} if len(self.session.calls) < 3 else reference
        with patch('bubbleteait.serena_workspace.asyncio.sleep', new=AsyncMock()):
            result = await self.query('references')
        self.assertEqual(len(self.parameters), 1)
        self.assertEqual(result['query_attempts'], 3)
        self.assertEqual(len(self.session.calls), 3)
        self.assertEqual(result['results'][0]['path'], 'src/caller.c')
        self.assertEqual(result['results'][0]['location_kind'], 'enclosing_symbol')

    async def test_exhausted_reference_retry_is_bounded_and_does_not_claim_absence(self):
        self.reply = {}
        with patch('bubbleteait.serena_workspace.asyncio.sleep', new=AsyncMock()):
            result = await self.query('references')
        self.assertEqual(len(self.session.calls), 4)
        self.assertEqual(result['query_attempts'], 4)
        self.assertEqual(result['returned'], 0)
        self.assertEqual(result['completeness'], 'not_established')
        self.assertEqual(result['compile_context'], 'unknown')

    async def test_repeated_reference_records_keep_distinct_verified_callers_before_budgeting(self):
        caller = ('int caller(void) {\n    return heatingAllowed(0);\n}\n'
                  'int otherCaller(void) {\n    return heatingAllowed(1);\n}\n')
        unit = self.make_unit('callers', 'project', {'src/control.c': self.text, 'src/caller.c': caller})
        self.registry.compose('dedup', [unit['unit_id']], require_ready=True)
        self.service = SymbolService(self.config, 'dedup')
        first = {'name_path': 'caller', 'body_location': {'start_line': 0, 'end_line': 2}}
        second = {'name_path': 'otherCaller', 'body_location': {'start_line': 3, 'end_line': 5}}
        self.reply = {'src/caller.c': {'Function': [first, first, first, second]}}
        result = await self.query('references', source='callers')
        self.assertEqual(result['reported_records'], 4)
        self.assertEqual((result['total'], result['returned'], result['truncated']), (2, 2, False))
        self.assertEqual([entry['name_path'] for entry in result['results']], ['caller', 'otherCaller'])
        self.assertEqual([entry['lines'] for entry in result['results']], [[1, 3], [4, 6]])
        self.assertEqual({entry['source_sha256'] for entry in result['results']},
                         {hashlib.sha256(caller.encode()).hexdigest()})
        limited = await self.query('references', source='callers', limit=1, max_chars=2000)
        self.assertEqual((limited['total'], limited['returned'], limited['truncated']), (2, 1, True))
        self.assertEqual(limited['reported_records'], 4)
        self.assertLessEqual(len(json.dumps(limited, ensure_ascii=False, separators=(',', ':'))), 2000)

    async def test_ready_read_refuses_republished_source_version_without_any_model(self):
        locator = (await self.query())['results'][0]
        ready = locator['read']
        with patch('bubbleteait.direct.make_backend', return_value=None):
            direct = DirectService(self.config, 'symbols')
            before = direct.read_many([ready], include_catalog=False)
            self.assertEqual(before['excerpts'][0]['text'], self.text)
            changed = self.make_unit('app', 'project', {'src/control.c':
                'int heatingAllowed(int trip) {\r\n    return trip != 0;\r\n}\r\n'})
            self.assertNotEqual(changed['unit_id'], ready['unit_id'])
            self.registry.compose('symbols', [changed['unit_id']], require_ready=True)
            current = DirectService(self.config, 'symbols')
            with self.assertRaises(SearchError) as error:
                current.read_many([ready], include_catalog=False)
            self.assertIn('unit', str(error.exception).casefold())
            self.assertEqual(current._ledger, {})


if __name__ == '__main__':
    unittest.main()
