"""Coordinator contracts independent of optional ML/LSP installations."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import concurrent.futures
import threading
from unittest.mock import Mock, patch
from argparse import Namespace

from bubbleteait.config import Config, ConfigError, load_config
from bubbleteait.engine_coordinator import EngineCoordinator
from bubbleteait.engine_index import CocoEngine
from bubbleteait.service import config_payload, config_from_payload


class NativeProcess:
    def __init__(self, *args):
        self.process = None
        self.calls = []
        self.fail = False

    def request(self, action, **args):
        self.process = Mock()
        self.process.poll.return_value = None
        self.calls.append((action, args))
        if self.fail:
            raise ConfigError('fixture native engine failure')
        root = Path(args['project'])
        if action == 'index':
            directory = root / '.cocoindex_code'
            directory.mkdir(exist_ok=True)
            (directory / 'target_sqlite.db').touch()
            (directory / 'cocoindex.db').mkdir(exist_ok=True)
            files = [p for p in root.rglob('*') if p.is_file() and '.cocoindex_code' not in p.parts]
            return {'success': True, 'index': {'total_files': sum(p.stat().st_size > 0 for p in files)},
                    'files': [p.relative_to(root).as_posix() for p in files]}
        results = []
        for path in args['paths'][:args['limit']]:
            text = (root / path).read_text(encoding='utf-8')
            results.append({'file_path': path, 'content': text, 'start_line': 1,
                            'end_line': len(text.splitlines()), 'score': .9})
        return {'success': True, 'results': results}

    def close(self):
        self.process = None


class EngineFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'models/model').mkdir(parents=True)
        projects = {}
        for name in ('controller', 'reference'):
            path = self.root / name
            path.mkdir()
            (path / 'main.c').write_text('int threshold = 37;\n', encoding='utf-8')
            projects[name] = {'path': path, 'include': ['**/*.c'], 'description': name}
        self.config = Config(self.root / 'controller', self.root / 'data', self.root / 'models',
            engines={'cocoindex': {'python_executable': 'python', 'model_path': self.root / 'models/model',
                'model_revision': 'fixture-v1', 'device': 'cpu', 'timeout_seconds': 10,
                'indexing_params': {}, 'query_params': {}}}, projects=projects,
            workspaces={'work': {'path': self.root, 'projects': list(projects), 'default_project': 'controller', 'watch': True}},
            sync={'debounce_ms': 0, 'poll_interval_seconds': 1, 'enabled': True})
        self.coordinator = EngineCoordinator(self.config, engine_factory=lambda c: CocoEngine(c, NativeProcess))
        self.addCleanup(self.coordinator.close)

    def index(self, name='controller'):
        path = self.config.projects[name]['path']
        self.coordinator.register(path, manual=True)
        self.coordinator.check(str(path), force=True)
        self.assertEqual(self.coordinator.entries[str(path)]['state'], 'ready')
        return path

    def call(self, operation, **args):
        return self.coordinator.engine_call({'lease': 'fixture', 'operation': operation, **args})


class EngineTests(EngineFixture, unittest.TestCase):
    def test_empty_file_is_selected_and_has_no_native_chunks(self):
        (self.config.project_dir / 'empty.c').touch()
        root = self.index()
        manifest = self.coordinator.engine.manifest(root)
        self.assertEqual(manifest['empty_files'], ['controller.source/empty.c'])
        self.assertEqual(len(manifest['files']), 2)
        self.assertEqual(manifest['index']['total_files'], 1)

    def test_workspace_without_default_uses_the_client_project(self):
        from bubbleteait.engine_client import EngineClient
        self.config.workspaces['work'].pop('default_project')
        config = replace(self.config, project_dir=self.config.projects['reference']['path'])
        with patch('bubbleteait.engine_client.ServiceClient') as service:
            service.return_value.request.return_value = {}
            client = EngineClient(config, 'work')
            client.call('search', query='threshold')
            self.assertEqual(service.return_value.request.call_args.kwargs['project'], 'reference')
            client.call('search', query='threshold', projects=['controller','reference'])
            self.assertNotIn('project', service.return_value.request.call_args.kwargs)

    def test_search_follow_up_recovers_function_from_original_bytes(self):
        source = self.root / 'controller/main.c'
        source.write_text('int before;\nint check(\n    int v)\n{\n    /* } */\n    return v;\n}\nint after;\n', encoding='utf8')
        self.index()
        hit = {'file_path':'controller.source/main.c', 'start_line':6, 'end_line':6, 'content':'return v;', 'score':1}
        with patch.object(self.coordinator.engine, 'search', return_value={'results':[hit]}):
            excerpt = self.call('search', query='check')['excerpts'][0]
        self.assertEqual(excerpt['context_kind'], 'function')
        follow = excerpt['next_read']
        read = follow['requests'][0]
        self.assertEqual((read['start'],read['end']), (2,7))
        source.write_text('int changed;\n', encoding='utf8')
        request = dict(project=follow['project'], ids=[read['anchor_id']], anchors=[read])
        old = self.call('read', **request)['excerpts'][0]
        self.assertEqual(old['generation'],excerpt['generation'])
        self.assertTrue(old['source_changed'])
        self.assertEqual(old['lines'],[2,7])
        self.assertEqual(old['text'].splitlines()[-2:], ['    return v;', '}'])
        with self.assertRaisesRegex(ConfigError,'locator'):
            self.call('read', **(request | {'project':'reference'}))

    def test_long_function_follow_up_pages_keep_exact_line_boundaries(self):
        source = self.config.project_dir / 'main.c'
        body = 'int f(void) {\n' + '    /* ' + 'x' * 33000 + ' */\n'
        body += '    /* ' + 'y' * 33000 + ' */\n    return 0;\n}\n'
        source.write_bytes(body.encode())
        self.index()
        hit = {'file_path':'controller.source/main.c','start_line':4,'end_line':4,'content':'return 0;', 'score':1}
        with patch.object(self.coordinator.engine, 'search', return_value={'results':[hit]}):
            follow = self.call('search', query='f')['excerpts'][0]['next_read']
        pages = []
        while follow:
            read = follow['requests'][0]
            result = self.call('read', project=follow['project'], ids=[read['anchor_id']],
                               anchors=[read], max_chars=follow['max_chars'])
            page, = result['excerpts']
            pages.append(page['text'])
            follow = page.get('next_read')
        self.assertEqual(len(pages), 2)
        self.assertEqual(''.join(pages), body)

    def test_waiting_model_query_does_not_block_known_path_read(self):
        self.index()
        started, release = threading.Event(), threading.Event()
        native = self.coordinator.search_scope
        def gated(*args):
            started.set()
            release.wait(10)
            return native(*args)
        self.coordinator.search_scope = gated
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            search = pool.submit(self.call, 'search', query='threshold')
            try:
                self.assertTrue(started.wait(3))
                read = pool.submit(self.call, 'read', requests=[{'path':'main.c'}])
                self.assertEqual(read.result(timeout=3)['state'], 'ready')
            finally:
                release.set()
            self.assertEqual(search.result(timeout=3)['state'], 'ready')

    def test_explicit_legacy_config_is_not_ignored_when_machine_engine_config_exists(self):
        user = self.root/'user/BubbleTeaIT'
        user.mkdir(parents=True)
        (user/'config-service.toml').write_text('[service]\ndata_dir="machine-data"\n')
        explicit = self.root/'legacy.toml'
        explicit.write_text('[paths]\ndata_dir="explicit-data"\n')
        config = load_config(self.root, config_path=explicit, env={'LOCALAPPDATA':str(user.parent)})
        self.assertEqual(config.data_dir, self.root/'explicit-data')

    def test_plan_and_info_do_not_start_service_or_models(self):
        from bubbleteait.engine_client import EngineClient
        with patch('bubbleteait.engine_client.ServiceClient') as service:
            client = EngineClient(self.config, 'work')
            self.assertTrue(client.call('plan')['ready'])
            self.assertEqual(len(client.call('info')['projects']), 2)
            service.return_value.request.assert_not_called()

    def test_flags_require_model_revision_and_keep_overrides_separate(self):
        from bubbleteait.engine_config import apply_engine_flags
        with self.assertRaisesRegex(ConfigError, 'requires'):
            apply_engine_flags(self.config, Namespace(embedding_model=str(self.root/'models/model')))
        changed = apply_engine_flags(self.config, Namespace(embedding_model=str(self.root/'models/model'),
            embedding_revision='new-revision', embedding_device='cuda'))
        self.assertEqual(changed.engines['cocoindex']['model_revision'], 'new-revision')
        self.assertEqual(self.config.engines['cocoindex']['model_revision'], 'fixture-v1')
        self.assertEqual(changed.engine_overrides['cocoindex']['device'], 'cuda')

    def test_one_project_search_failure_keeps_other_results_with_explicit_error(self):
        self.index()
        self.index('reference')
        native = self.coordinator.engine.process.request
        def fail_reference(action, **arguments):
            if action == 'search' and any(p.startswith('reference.') for p in arguments.get('paths', [])):
                raise ConfigError('reference unavailable')
            return native(action, **arguments)
        self.coordinator.engine.process.request = fail_reference
        result = self.call('search', query='threshold', workspace='work', projects=['controller', 'reference'])
        self.assertEqual(result['state'], 'partial')
        self.assertEqual([e['project'] for e in result['excerpts']], ['controller'])
        self.assertEqual(result['omitted_projects'][0]['project'], 'reference')

    def test_removed_native_database_is_not_reported_ready(self):
        root = self.index()
        directory = self.coordinator.engine.project_directory(root)
        (directory/'view/.cocoindex_code/target_sqlite.db').unlink()
        self.assertFalse(self.coordinator.engine.available(root))
        self.coordinator.check(str(root), force=True)
        self.assertTrue(self.coordinator.engine.available(root))

    def test_manifest_tampering_is_detected_before_returning_sources(self):
        root = self.index()
        path = self.coordinator.engine.project_directory(root)/'current.json'
        manifest = json.loads(path.read_text())
        next(iter(manifest['files'].values()))['absolute_path'] = str(self.root/'private.txt')
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ConfigError, 'manifest'):
            self.coordinator.engine.manifest(root)

    def test_unavailable_registered_sibling_is_reported_as_partial_failure(self):
        self.index()
        (self.root/'reference/main.c').unlink()
        (self.root/'reference').rmdir()
        result = self.call('search', query='threshold', workspace='work', projects=['controller', 'reference'])
        self.assertEqual(result['state'], 'partial')
        self.assertEqual(result['excerpts'][0]['project'], 'controller')
        self.assertEqual(result['omitted_projects'][0]['project'], 'reference')
    def test_known_path_read_does_not_start_embedding_engine(self):
        result = self.call('read', requests=[{'path': 'main.c'}])
        self.assertEqual(result['excerpts'][0]['text'], (self.config.project_dir / 'main.c').read_bytes().decode())
        self.assertEqual(self.coordinator.engine.process.calls, [])
        self.assertFalse(self.coordinator.status()['engines']['cocoindex']['bridge_running'])

    def test_default_project_and_explicit_cross_project_search_share_runtime(self):
        self.index()
        self.index('reference')
        default = self.call('search', query='threshold', workspace='work')
        self.assertEqual([e['project'] for e in default['excerpts']], ['controller'])
        cross = self.call('search', query='threshold', workspace='work', projects=['controller', 'reference'])
        self.assertEqual([e['project'] for e in cross['excerpts']], ['controller', 'reference'])
        self.assertEqual(len({args['project'] for action, args in self.coordinator.engine.process.calls if action == 'index'}), 2)
        self.assertEqual(cross['state'], 'ready')

    def test_incremental_stable_view_add_edit_delete_and_old_evidence(self):
        root = self.index()
        old = self.call('search', query='threshold')['excerpts'][0]
        old_view = self.coordinator.engine.project_directory(root) / 'view'
        (root / 'main.c').write_text('int threshold = 42;\n')
        (root / 'added.c').write_text('int added;\n')
        self.coordinator.check(str(root), force=True)
        new = self.call('search', query='threshold')
        self.assertTrue(any('42' in e['text'] for e in new['excerpts']))
        self.assertTrue(any(e['path'] == 'added.c' for e in new['excerpts']))
        (root / 'added.c').unlink()
        self.coordinator.check(str(root), force=True)
        self.assertFalse((old_view / 'controller.source/added.c').exists())
        pinned = self.call('read', ids=[old['id']])['excerpts'][0]
        self.assertIn('37', pinned['text'])
        self.assertTrue(pinned['source_changed'])

    def test_failed_native_update_does_not_publish_or_serve_partial_index(self):
        root = self.index()
        old = self.coordinator.engine.manifest(root)
        (root / 'extra.c').write_text('int extra;\n')
        self.coordinator.engine.process.fail = True
        self.coordinator.check(str(root), force=True)
        self.assertEqual(self.coordinator.engine.manifest(root), old)
        result = self.call('search', query='threshold')
        self.assertEqual(result['state'], 'error')
        self.assertFalse(result['excerpts'])
        # A file added during a failed staging pass must not leak into the next index.
        (root / 'extra.c').unlink()
        self.coordinator.engine.process.fail = False
        self.coordinator.check(str(root), force=True)
        self.assertEqual(self.coordinator.entries[str(root)]['state'], 'ready')
        view = self.coordinator.engine.project_directory(root) / 'view'
        self.assertFalse((view / 'controller.source/extra.c').exists())

    def test_ignore_attach_exclude_and_traversal(self):
        root = self.config.projects['controller']['path']
        (root / '.gitignore').write_text('*.c\n')
        (root / 'excluded.c').write_text('int forbidden;\n')
        item = self.config.projects['controller']
        item.update(attach=['.'], exclude=['excluded.c'])
        result = self.call('files')
        self.assertEqual([r['path'] for r in result['files']], ['main.c'])
        for path in ('../reference/main.c', 'excluded.c'):
            with self.assertRaises(ConfigError):
                self.call('read', requests=[{'path': path}])

    def test_stale_path_guard_and_unknown_id_are_rejected(self):
        old = self.call('read', requests=[{'path': 'main.c'}])['excerpts'][0]
        (self.config.project_dir / 'main.c').write_text('int changed;\n')
        with self.assertRaisesRegex(ConfigError, 'Stale'):
            self.call('read', requests=[{'path': 'main.c', 'sha256': old['sha256']}])
        with self.assertRaisesRegex(ConfigError, 'Unknown'):
            self.call('read', ids=['e-forged'])

    def test_unchanged_reconciliation_does_not_invoke_native_index(self):
        root = self.index()
        count = len(self.coordinator.engine.process.calls)
        self.coordinator.check(str(root), force=True)
        self.assertEqual(len(self.coordinator.engine.process.calls), count)

    def test_settings_round_trip_and_model_recipe_isolation(self):
        restored = config_from_payload(config_payload(self.config))
        self.assertEqual(restored.engines, self.config.engines)
        self.assertEqual(restored.projects, self.config.projects)
        settings = dict(self.config.engines['cocoindex'], query_params={'prompt_name': 'query'})
        alternative = CocoEngine(replace(self.config, engines={'cocoindex': settings}), NativeProcess)
        self.addCleanup(alternative.close)
        self.assertNotEqual(self.coordinator.engine.fingerprint, alternative.fingerprint)

    def test_machine_config_explicit_projects_and_selectable_language_server(self):
        config_path = self.root / 'config-service.toml'
        config_path.write_text('''[service]
data_dir='data'
[paths]
models_dir='models'
[engines.cocoindex]
model_path='models/model'
model_revision='fixture'
device='cpu'
[engines.serena]
executable='serena'
language_server='ccls'
server='ccls'
[projects.controller]
path='controller'
include=['**/*.c']
[workspaces.work]
path='.'
projects=['controller']
default_project='controller'
''', encoding='utf-8')
        loaded = load_config(self.root, config_path=config_path, env={})
        self.assertEqual(loaded.engines['serena']['server'], 'ccls')
        self.assertEqual(loaded.projects['controller']['path'], self.root / 'controller')
        config_path.write_text(config_path.read_text().replace("server='ccls'", "server='gcc'"))
        with self.assertRaisesRegex(ConfigError, 'clangd or ccls'):
            load_config(self.root, config_path=config_path, env={})


if __name__ == '__main__':
    unittest.main()
