"""Coordinator regressions: real byte snapshots, isolated worktrees, no models."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from bubbleteait.config import Config, ConfigError, load_config
from bubbleteait.direct import DirectService
from bubbleteait.direct_session import DirectSession
from bubbleteait.project import load_project, sync_project
from bubbleteait.registry import IndexRecipe, Registry, _atomic_json
from bubbleteait.search import SearchError, SearchService
from bubbleteait.service import Coordinator, ServiceClient, config_payload, config_from_payload, select_profile


class Backend:
    def __init__(self, data_dir):
        self.registry = Registry(data_dir)
        self.calls = []
        self.fail = False
        self.after_index = None

    def index(self, unit):
        self.calls.append(unit)
        if self.fail:
            raise RuntimeError('fixture backend unavailable')
        manifest = self.registry.load_unit(unit)
        _atomic_json(self.registry.data_dir / 'indexes' / unit / 'status.json', {
            'schema_version': 1, 'unit_id': unit, 'recipe_hash': manifest['recipe_hash'],
            'backend': 'fixture', 'status': 'ready'})
        if self.after_index:
            self.after_index()

    def search(self, *args, **kwargs):
        return []


class ServiceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.project = self.root / 'workspace/app'
        self.write_project(self.project, 'app')
        self.config = Config(self.project, self.root / 'data', self.root / 'models',
                             project_config=self.project / 'bti.toml', sync={'debounce_ms': 0})
        self.recipe = IndexRecipe('fixture', 'none', 'v1', 'lines')
        self.backend = Backend(self.config.data_dir)
        self.coordinator = Coordinator(self.config, recipe_factory=lambda c: self.recipe,
                                       backend_factory=lambda c: self.backend)

    def write_project(self, root, name):
        root.mkdir(parents=True)
        (root / 'bti.toml').write_text(f'''schema_version=1
[component]
name="{name}"
kind="project"
[source]
include=["*.c"]
''', encoding='utf-8')
        (root / 'main.c').write_text('int threshold = 37;\n', encoding='utf-8')

    def register(self):
        self.coordinator.register(self.project, manual=True)
        self.coordinator.check(str(self.project), force=True)
        return self.coordinator.entries[str(self.project)]

    def start(self):
        self.coordinator.start()
        def close():
            self.coordinator.stop.set()
            self.coordinator.thread.join(10)
            self.assertFalse(self.coordinator.thread.is_alive())
        self.addCleanup(close)

    def test_reconcile_reuses_units_and_detects_add_edit_delete_without_event(self):
        entry = self.register()
        first = entry['composition_id']
        original_hash = SearchService(self.config.data_dir).load_composition(first)['composition_hash']
        count = len(self.backend.calls)
        self.coordinator.check(str(self.project), force=True)
        self.assertEqual(count, len(self.backend.calls))
        (self.project / 'added.c').write_text('int added;\n')
        self.coordinator.check(str(self.project), force=True)
        self.assertNotEqual(entry['composition_id'], first)
        (self.project / 'main.c').write_text('int threshold = 42;\n')
        self.coordinator.check(str(self.project), force=True)
        (self.project / 'added.c').unlink()
        self.coordinator.check(str(self.project), force=True)
        unit = SearchService(self.config.data_dir).load_composition('app')['units'][0]['unit_id']
        self.assertEqual([f['path'] for f in Registry(self.config.data_dir).load_unit(unit)['files']], ['main.c'])
        self.assertEqual(SearchService(self.config.data_dir).load_composition(first)['composition_hash'],
                         original_hash)

    def test_backend_failure_and_mid_build_edit_do_not_publish_partial_composition(self):
        first = self.register()['composition_id']
        (self.project / 'main.c').write_text('int changed;\n')
        self.backend.fail = True
        self.coordinator.check(str(self.project), force=True)
        self.assertEqual(self.coordinator.entries[str(self.project)]['state'], 'error')
        self.assertEqual(self.coordinator.entries[str(self.project)]['composition_id'], first)
        self.backend.fail = False
        self.backend.after_index = lambda: (self.project / 'main.c').write_text('int changed_again;\n')
        self.coordinator.check(str(self.project), force=True)
        self.assertIn('Source changed during sync', self.coordinator.entries[str(self.project)]['error'])
        self.backend.after_index = None
        self.coordinator.check(str(self.project), force=True)
        self.assertEqual(self.coordinator.entries[str(self.project)]['state'], 'ready')

    def test_registration_is_exact_persistent_and_conflicting_names_are_rejected(self):
        self.register()
        child = self.project / 'child'
        child.mkdir()
        with self.assertRaises(ConfigError):
            self.coordinator.register(child, manual=True)
        other = self.root / 'other'
        self.write_project(other, 'app')
        with self.assertRaisesRegex(ConfigError, 'Duplicate project name'):
            self.coordinator.register(other, manual=True)
        restored = Coordinator(self.config)
        self.assertTrue(restored.entries[str(self.project)]['manual'])
        self.coordinator.unwatch(self.project)
        self.assertFalse(Coordinator(self.config).entries)

    def test_wait_barrier_recovers_after_config_edit_and_old_ids_stay_pinned(self):
        self.start()
        first = self.coordinator.select(self.project, lease='client', freshness='wait', timeout_ms=5000)
        self.assertEqual(first['state'], 'ready', first)
        direct = DirectService(self.config, first['composition_id'])
        read = direct.read_many([{'path': 'main.c'}], include_catalog=False)
        identifier = read['excerpts'][0]['id']
        (self.project / 'main.c').write_text('int threshold = 42;\n')
        second = self.coordinator.select(self.project, lease='client', freshness='wait', timeout_ms=5000)
        self.assertEqual(second['state'], 'ready', second)
        self.assertNotEqual(first['composition_id'], second['composition_id'])
        self.assertIn('37', direct.read_many([identifier])['excerpts'][0]['text'])
        path = self.project / 'bti.toml'
        path.write_text(path.read_text() + '\n[sources.docs]\npath="docs"\nkind="documents"\nrole="docs"\ninclude=["*.md"]\n')
        (self.project / 'docs').mkdir()
        (self.project / 'docs/acceptance.md').write_text('Threshold acceptance criteria.\n')
        ready = self.coordinator.select(self.project, lease='client', freshness='wait', timeout_ms=5000)
        self.assertEqual(ready['state'], 'ready', ready)
        result = DirectService(self.config, ready['composition_id']).read_many([{'path': 'docs/acceptance.md'}])
        self.assertIn('criteria', result['excerpts'][0]['text'])

    def test_concurrent_manual_waiters_are_not_lost_when_auto_sync_is_disabled(self):
        self.coordinator.config = replace(self.config, sync={'enabled': False, 'debounce_ms': 0})
        entered, release = threading.Event(), threading.Event()
        self.backend.after_index = lambda: (entered.set(), release.wait(5))
        self.start()
        results = []
        def request(lease):
            results.append(self.coordinator.select(self.project, lease=lease, freshness='wait',
                                                    timeout_ms=5000, manual_sync=True))
        first = threading.Thread(target=request, args=('first',))
        second = threading.Thread(target=request, args=('second',))
        try:
            first.start()
            self.assertTrue(entered.wait(3))
            second.start()
            deadline = time.monotonic() + 3
            while self.coordinator.entries[str(self.project)]['requested_revision'] < 2 and time.monotonic() < deadline:
                time.sleep(.01)
        finally:
            release.set()
            first.join(6)
            second.join(6)
        self.assertEqual([r['state'] for r in results], ['ready', 'ready'], results)
        self.assertEqual(len(self.backend.calls), 1)

    def test_workspace_discovers_projects_and_unwatch_does_not_remove_membership(self):
        self.coordinator.config = replace(self.config, workspaces={'work': {
            'path': self.project.parent, 'discover': ['*/bti.toml'], 'exclude': ['excluded/**'], 'watch': True}})
        self.write_project(self.project.parent / 'second', 'second')
        self.write_project(self.project.parent / 'excluded', 'excluded')
        self.coordinator.refresh_workspaces()
        names = {p['project'] for p in self.coordinator.status()['projects']}
        self.assertEqual(names, {'app', 'second'})
        self.coordinator.register(self.project, manual=True)
        self.assertTrue(self.coordinator.unwatch(self.project)['watching'])
        self.write_project(self.project.parent / 'third', 'third')
        self.coordinator.refresh_workspaces()
        self.assertIn('third', {p['project'] for p in self.coordinator.status()['projects']})

    def test_path_reads_boundaries_and_explicit_range_validation(self):
        entry = self.register()
        direct = DirectService(self.config, entry['composition_id'])
        for path in ('main.c', str(self.project / 'main.c')):
            self.assertIn('37', direct.read_many([{'path': path}])['excerpts'][0]['text'])
        for request in ({'path': '../outside.c'}, {'path': str(self.root / 'absent.c')},
                        {'path': 'main.c', 'start': 1}, {'path': 'main.c', 'start': 1, 'end': 2}):
            with self.assertRaises(SearchError):
                direct.read_many([request])

    def test_session_ids_survive_new_composition_and_cache_eviction(self):
        entry = self.register()
        first = entry['composition_id']
        cache = {}
        session = DirectSession(self.config, lambda cid: cache.setdefault(cid, DirectService(self.config, cid)))
        direct = DirectService(self.config, first)
        result = session.remember(direct.read_many([{'path': 'main.c'}]), direct, 'app')
        identifier = result['excerpts'][0]['id']
        (self.project / 'main.c').write_text('int threshold = 42;\n')
        self.coordinator.check(str(self.project), force=True)
        cache.clear()
        selected, _ = session.read_scope([identifier], None)
        self.assertEqual(selected[0][1].composition_id, first)
        self.assertEqual(session.resolved([identifier])['evidence'][0]['composition_id'], first)
        with self.assertRaises(SearchError):
            session.resolved(['forged'])

    def test_unchanged_units_receive_new_session_ids_when_composition_changes(self):
        first = self.register()['composition_id']
        session = DirectSession(self.config, lambda cid: DirectService(self.config, cid))
        before = DirectService(self.config, first)
        old_id = session.remember(before.read_many([{'path': 'main.c'}]), before, 'app')['excerpts'][0]['id']
        manifest = self.project / 'bti.toml'
        manifest.write_text(manifest.read_text() + '\n# composition-only edit\n')
        self.coordinator.check(str(self.project), force=True)
        after = DirectService(self.config, self.coordinator.entries[str(self.project)]['composition_id'])
        new_id = session.remember(after.read_many([{'path': 'main.c'}]), after, 'app')['excerpts'][0]['id']
        self.assertNotEqual(old_id, new_id)
        self.assertEqual(session.read_scope([new_id], None)[0][0][1].composition_id, after.composition_id)
        with self.assertRaises(SearchError):
            session.read_scope([old_id, new_id], None)

    def test_service_config_priority_supported_profiles_and_roundtrip(self):
        settings = self.root / 'config-service.toml'
        settings.write_text('''[service]
data_dir="storage"
[paths]
models_dir="models"
[sync]
debounce_ms=100
[embedding]
default_profile="q06"
[embedding.profiles.q06]
backend="qwen-dense"
model_path="models/qwen"
[workspaces.work]
path="workspace"
''')
        config = load_config(self.root, config_path=settings, env={})
        self.assertEqual(config.data_dir, self.root / 'storage')
        self.assertEqual(config.qwen.model_profile, 'q06')
        self.assertEqual(config, config_from_payload(config_payload(config)))
        settings.write_text(settings.read_text() + '\n[embedding.profiles.alternative]\nbackend="qwen-dense"\nmodel_path="models/alternative"\n')
        alternative = select_profile(load_config(self.root, config_path=settings, env={}), 'alternative')
        self.assertEqual(alternative.qwen.model_profile, 'alternative')
        self.assertEqual(alternative.embedding_profile, 'alternative')
        overridden = load_config(self.root, config_path=settings, env={'BTI_DATA_DIR': str(self.root / 'override')})
        self.assertEqual(overridden.data_dir, self.root / 'override')
        settings.write_text(settings.read_text().replace('backend="qwen-dense"', 'backend="made-up"'))
        with self.assertRaisesRegex(ConfigError, 'unsupported backend'):
            load_config(self.root, config_path=settings, env={})

    def test_reload_keeps_last_valid_settings_applies_model_choice_and_preserves_flag_overrides(self):
        settings = self.root / 'config-service.toml'
        body = '''[service]
data_dir="data"
[sync]
debounce_ms=100
[embedding]
default_profile="q06"
[embedding.profiles.q06]
backend="qwen-dense"
model_path="models/qwen"
[embedding.profiles.e5]
backend="cocoindex-code"
model_path="models/e5"
'''
        settings.write_text(body)
        config = load_config(self.root, config_path=settings, env={})
        config = replace(config, sync_overrides={'debounce_ms': 0}, sync={'debounce_ms': 0})
        coordinator = Coordinator(config)
        with patch.dict(os.environ, {'PATH': os.environ['PATH']}, clear=True):
            coordinator.refresh_settings()
            self.assertEqual(coordinator.config.sync['debounce_ms'], 0)
            settings.write_text(body.replace('default_profile="q06"', 'default_profile="e5"'))
            coordinator.refresh_settings()
            self.assertEqual(coordinator.config.embedding_profile, 'e5')
            settings.write_text('invalid=[\n')
            coordinator.refresh_settings()
            self.assertIsNotNone(coordinator.config_error)
            self.assertEqual(coordinator.config.embedding_profile, 'e5')
            settings.write_text(body)
            coordinator.refresh_settings()
            self.assertIsNone(coordinator.config_error)
            self.assertEqual(coordinator.config.embedding_profile, 'q06')

    def local_client(self):
        coordinator = self.coordinator
        class LocalClient:
            def ensure(self):
                pass
            def request(self, action, **options):
                if action == 'workspace_projects':
                    return coordinator.workspace_projects(options['workspace'])
                return coordinator.status()
            def select(self, path, **options):
                return coordinator.select(path, lease='local-mcp', **options)
        return LocalClient()

    def test_workspace_search_pins_all_versions_and_batches_ids_across_projects(self):
        self.write_project(self.project.parent / 'second', 'second')
        config = replace(self.config, workspaces={'work': {
            'path': self.project.parent, 'discover': ['*/bti.toml'], 'exclude': [], 'watch': True}})
        self.coordinator.config = config
        self.coordinator.refresh_workspaces()
        self.start()
        session = DirectSession(config, lambda cid: DirectService(config, cid), workspace='work')
        session.client = self.local_client()
        with patch('bubbleteait.direct.make_backend', return_value=self.backend):
            result = session.search('threshold', freshness='wait', max_chars=12000)
            self.assertEqual({r['project'] for r in result['excerpts']}, {'app', 'second'})
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 12000)
            pinned = result['compositions']
            ids = [row['id'] for row in result['excerpts']]
            read = session.read_origins(ids, max_chars=16000)
            self.assertEqual({r['project'] for r in read['excerpts']}, {'app', 'second'})
            (self.project.parent / 'second/main.c').write_text('int threshold = 42;\n')
            current = session.search('threshold', freshness='wait', max_chars=12000)
            self.assertTrue(any('42' in r['text'] for r in current['excerpts']))
            previous = session.search('threshold', compositions=pinned, max_chars=12000)
            self.assertTrue(all('37' in r['text'] for r in previous['excerpts']))
            narrow = session.search('threshold', projects=['app'], max_chars=2000)
            self.assertTrue(all(r['project'] == 'app' for r in narrow['excerpts']))
            with self.assertRaises(ConfigError):
                session.search('threshold', projects=['foreign'], max_chars=12000)

    @unittest.skipUnless(importlib.util.find_spec('mcp'), 'Install MCP extra')
    def test_public_mcp_first_search_path_read_and_versioned_resolution(self):
        import asyncio
        from bubbleteait.mcpserver import build_server
        self.start()
        with patch('bubbleteait.direct_session.ServiceClient', return_value=self.local_client()), \
             patch('bubbleteait.direct.make_backend', return_value=self.backend):
            server = build_server(self.config)
            async def scenario():
                tools = {t.name: t for t in await server.list_tools()}
                self.assertNotIn('composition_id', tools['context_search'].input_schema.get('required', []))
                self.assertFalse(tools['context_search'].annotations.read_only_hint)
                searched = await server.call_tool('context_search', {'query': 'threshold', 'freshness': 'wait'})
                self.assertFalse(searched.is_error)
                first = searched.structured_content
                identifier = first['excerpts'][0]['id']
                read = await server.call_tool('read_sources', {'requests': [{'path': 'main.c'}]})
                self.assertIn('37', read.structured_content['excerpts'][0]['text'])
                (self.project / 'main.c').write_text('int threshold = 42;\n')
                updated = await server.call_tool('context_search', {'query': 'threshold', 'freshness': 'wait'})
                self.assertIn('42', updated.structured_content['excerpts'][0]['text'])
                previous = await server.call_tool('read_sources', {'requests': [identifier]})
                self.assertIn('37', previous.structured_content['excerpts'][0]['text'])
                resolved = await server.call_tool('resolve_sources', {'ids': [identifier]})
                self.assertEqual(resolved.structured_content['evidence'][0]['composition_id'], first['composition_id'])
            asyncio.run(scenario())


    def test_manual_sync_without_wait_reports_updating_and_then_waits(self):
        self.register()
        self.start()
        (self.project / 'main.c').write_text('int threshold = 42;\n', encoding='utf-8')
        queued = self.coordinator.select(self.project, lease='client', freshness='available', manual_sync=True)
        self.assertEqual(queued['state'], 'updating', queued)
        ready = self.coordinator.select(self.project, lease='client', freshness='wait', timeout_ms=5000)
        self.assertEqual(ready['state'], 'ready', ready)

    def test_missing_index_is_rebuilt_before_reporting_ready_again(self):
        entry = self.register()
        unit = SearchService(self.config.data_dir).load_composition(entry['composition_id'])['units'][0]['unit_id']
        status = self.config.data_dir / 'indexes' / unit / 'status.json'
        status.unlink()
        self.assertFalse(self.coordinator._composition_ready(entry['composition_id']))
        calls = len(self.backend.calls)
        self.coordinator.check(str(self.project), force=True)
        self.assertEqual(entry['state'], 'ready')
        self.assertEqual(len(self.backend.calls), calls + 1)
        self.assertTrue(self.coordinator._composition_ready(entry['composition_id']))

    def test_heartbeat_reports_unknown_lease(self):
        self.assertFalse(self.coordinator.heartbeat('missing'))
        self.coordinator.register(self.project, lease='known')
        self.assertTrue(self.coordinator.heartbeat('known'))

    def test_broken_service_state_is_quarantined_and_treated_as_stopped(self):
        client = ServiceClient(self.config)
        client.directory.mkdir(parents=True)
        client.state_path.write_text('{not json', encoding='utf-8')
        self.assertIsNone(client.state())
        self.assertFalse(client.state_path.exists())
        self.assertTrue(list(client.directory.glob('state.invalid-*.json')))


class ServiceProcessTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('mcp'), 'Install MCP extra')
    def test_real_stdio_project_path_read_without_discovery_calls_or_embedding_runtime(self):
        import asyncio
        import sys
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / 'app'
            project.mkdir()
            (project / 'bti.toml').write_text('schema_version=1\n[component]\nname="app"\n[source]\ninclude=["*.c"]\n')
            (project / 'main.c').write_text('int threshold = 37;\n')
            settings = root / 'config-service.toml'
            settings.write_text('[service]\ndata_dir="data"\n[sync]\nenabled=false\n')
            config = load_config(project, config_path=settings, env={})
            sync_project(load_project(project / 'bti.toml'), config,
                         IndexRecipe('fixture', 'none', 'fixed', 'lines'), Backend(config.data_dir))
            client = ServiceClient(config)
            environment = os.environ | {'BTI_DATA_DIR': str(config.data_dir), 'PYTHONUTF8': '1'}
            params = StdioServerParameters(command=sys.executable,
                args=['-m', 'bubbleteait', '--project-dir', str(project), '--config', str(settings), 'mcp'],
                cwd=str(project), env=environment)
            async def scenario():
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool('read_sources', {'requests': [{'path': 'main.c'}]})
                        self.assertFalse(result.is_error)
                        body = result.structured_content
                        self.assertIn('37', body['excerpts'][0]['text'])
                        canonical = await session.call_tool('resolve_sources', {'ids': [body['excerpts'][0]['id']]})
                        self.assertEqual(canonical.structured_content['evidence'][0]['file_path'], 'main.c')
                        status = await session.call_tool('watch_status', {})
                        self.assertEqual(status.structured_content['projects'][0]['project'], 'app')
            try:
                asyncio.run(scenario())
            finally:
                client.request('stop')

    def test_real_daemon_status_registration_authentication_and_restart(self):
        import urllib.error
        import urllib.request
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / 'app'
            project.mkdir()
            (project / 'bti.toml').write_text('schema_version=1\n[component]\nname="app"\n[source]\ninclude=["*.c"]\n')
            (project / 'main.c').write_text('int main;\n')
            config = Config(project, root / 'data', root / 'models', sync={'enabled': False})
            client = ServiceClient(config)
            try:
                self.assertEqual(client.request('status')['state'], 'stopped')
                client.ensure()
                client.request('register', path=str(project))
                self.assertTrue(client.request('status')['projects'][0]['manual'])
                state = client.state()
                request = urllib.request.Request(f'http://127.0.0.1:{state["port"]}/rpc', data=b'{}')
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    urllib.request.urlopen(request)
                self.assertEqual(rejected.exception.code, 403)
                client.request('stop')
                deadline = time.monotonic() + 10
                while client.probe() and time.monotonic() < deadline:
                    time.sleep(.05)
                client.ensure()
                self.assertTrue(client.request('status')['projects'][0]['manual'])
                self.assertIsNone(client.request('status')['projects'][0]['composition_id'])
            finally:
                client.request('stop')
                deadline = time.monotonic() + 10
                while client.probe() and time.monotonic() < deadline:
                    time.sleep(.05)
                if client.spawned is not None:
                    client.spawned.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
