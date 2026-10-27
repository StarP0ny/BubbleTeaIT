"""Snapshot, chunk-budget, publication and exact-ranking tests without model loads."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.backends.qwen import BackendError, QwenBackend, QwenIndexStore, default_recipe, validate_recipe
from bubbleteait.qwen_encoder import encoder_recipe
from bubbleteait.registry import Registry, SourceSpec, _digest
from bubbleteait.search import SearchService

try:
    import numpy as np
except ImportError:
    np = None


class QwenImportTests(unittest.TestCase):
    def test_control_process_import_has_no_numpy_or_torch(self):
        code = "import sys; import bubbleteait.backends.qwen; assert not {'torch','numpy','sentence_transformers'} & set(sys.modules)"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_proxy_only_forwards_index_and_search(self):
        with patch('bubbleteait.qwen_runtime.QwenRuntimeClient') as client:
            backend = QwenBackend('data', sys.executable, 'model', models_dir='models')
            client.return_value.request.return_value = {'ok': True, 'index': {'status': 'ready'}}
            self.assertEqual(backend.index('unit'), {'status': 'ready'})
            client.return_value.request.assert_called_with('index', unit_id='unit')
            client.return_value.request.return_value = {'ok': True, 'results': [{'file_path': 'a.c'}]}
            self.assertEqual(backend.search('unit', 'cold', paths=['a.c']), [{'file_path': 'a.c'}])
            client.return_value.request.assert_called_with('search', unit_id='unit', query='cold', limit=10, paths=['a.c'])

    def test_recipe_is_pinned_and_options_are_independent(self):
        recipe = default_recipe()
        self.assertEqual(recipe.embedding_model, 'Qwen/Qwen3-Embedding-0.6B')
        self.assertEqual(recipe.options['encoder']['dimensions'], 1024)
        validate_recipe(recipe)
        recipe.options['encoder']['dimensions'] = 2560
        with self.assertRaises(BackendError):
            validate_recipe(recipe)
        validate_recipe(default_recipe())
        wrong_type = default_recipe()
        wrong_type.options['encoder']['use_cache'] = 0
        with self.assertRaises(BackendError):
            validate_recipe(wrong_type)


class FakeEncoder:
    def __init__(self):
        self.model_manifest_sha256 = hashlib.sha256(b'fake pinned model').hexdigest()
        self.document_calls, self.query_calls, self.load_count = [], [], 0
        self.last_encode_stats = {}
        self.failure, self.truncated = False, False

    def recipe(self):
        return encoder_recipe()

    def load(self):
        self.load_count = 1

    def count_tokens(self, texts):
        # Deliberately stricter than ordinary tokenization, so tests exercise
        # budget shrinking and demonstrate that metadata also consumes budget.
        return [len(text) for text in texts]

    def values(self, texts):
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if 'cold' in text.lower() else 1] = 1
        return vectors

    def encode_documents(self, texts):
        self.document_calls.append(texts)
        if self.failure:
            raise RuntimeError('encoder failed')
        self.last_encode_stats = {'n': len(texts), 'truncated': int(self.truncated)}
        return self.values(texts)

    def encode_queries(self, texts):
        self.query_calls.append(texts)
        return self.values(texts)

    def health(self):
        return {'loaded': bool(self.load_count)}


@unittest.skipIf(np is None, 'NumPy worker environment required; control import tests remain active')
class QwenStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'a.c').write_bytes(b'void hot(void) {\r\n    heat();\r\n}\r\n')
        (self.source / 'b.c').write_text('void cold(void) {\n    freeze();\n}\n', encoding='utf-8')
        self.registry = Registry(self.root / 'data')
        self.registry.register(SourceSpec('fixture', 'module', self.source))
        self.unit = self.registry.snapshot('fixture', default_recipe())
        self.unit_id = self.unit['unit_id']
        self.encoder = FakeEncoder()
        self.store = QwenIndexStore(self.registry.data_dir, self.encoder)
        self.index_root = self.registry.data_dir / 'indexes' / self.unit_id

    def refresh(self):
        self.unit = self.registry.snapshot('fixture', default_recipe())
        self.unit_id = self.unit['unit_id']
        self.index_root = self.registry.data_dir / 'indexes' / self.unit_id

    def test_index_ready_reuse_and_restart_do_not_encode_or_overwrite(self):
        first = self.store.index(self.unit_id)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.index_root / 'artifact').iterdir()}
        status_before = (self.index_root / 'status.json').read_bytes()
        again = QwenIndexStore(self.registry.data_dir, self.encoder).index(self.unit_id)
        self.assertEqual(first['status'], 'ready')
        self.assertTrue(again['reused'])
        self.assertEqual(len(self.encoder.document_calls), 1)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.index_root / 'artifact').iterdir()})
        self.assertEqual(status_before, (self.index_root / 'status.json').read_bytes())
        self.assertEqual(self.registry.index_status(self.unit_id)['status'], 'ready')

    def test_crash_after_artifact_publication_recovers_ready_without_reencoding(self):
        self.store.index(self.unit_id)
        status_path = self.index_root / 'status.json'
        ready = json.loads(status_path.read_text(encoding='utf-8'))
        original_artifact = {p.name: p.read_bytes() for p in (self.index_root / 'artifact').iterdir()}
        for previous in ('building', 'error', 'missing'):
            if previous == 'missing':
                status_path.unlink()
            else:
                interrupted = {key: ready[key] for key in ('schema_version', 'unit_id', 'recipe_hash', 'backend', 'model_manifest_sha256')}
                interrupted['status'] = previous
                status_path.write_text(json.dumps(interrupted), encoding='utf-8')
            with self.assertRaisesRegex(BackendError, 'not ready'):
                self.store.search(self.unit_id, 'cold')
            recovered = QwenIndexStore(self.registry.data_dir, self.encoder).index(self.unit_id)
            self.assertEqual(recovered['status'], 'ready')
            self.assertTrue(recovered['recovered'])
            self.assertEqual(len(self.encoder.document_calls), 1)
            self.assertEqual(original_artifact, {p.name: p.read_bytes() for p in (self.index_root / 'artifact').iterdir()})
            self.assertTrue(self.store.search(self.unit_id, 'cold'))

    def test_recovery_does_not_accept_corrupt_or_resealed_stale_artifacts(self):
        self.store.index(self.unit_id)
        status_path = self.index_root / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        status['status'] = 'building'
        status_path.write_text(json.dumps(status), encoding='utf-8')
        path = self.index_root / 'artifact/manifest.json'
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaisesRegex(BackendError, 'checksum'):
            self.store.index(self.unit_id)
        path.write_bytes(original)
        vectors = self.index_root / 'artifact/vectors.npy'
        vectors.write_bytes(vectors.read_bytes() + b'tamper')
        # Even absent status cannot turn an invalid checksum into ready.
        status_path.unlink()
        with self.assertRaisesRegex(BackendError, 'checksum'):
            self.store.index(self.unit_id)
        self.assertEqual(len(self.encoder.document_calls), 1)

    def test_filter_is_applied_before_topk_and_results_have_exact_source_proof(self):
        self.store.index(self.unit_id)
        self.assertEqual(self.store.search(self.unit_id, 'hot', limit=1)[0]['file_path'], 'a.c')
        found = self.store.search(self.unit_id, 'hot', limit=1, paths=['b.c'])
        self.assertEqual(found[0]['file_path'], 'b.c')
        self.assertEqual(found[0]['score'], 0)
        calls = len(self.encoder.query_calls)
        self.assertEqual(self.store.search(self.unit_id, 'hot', paths=[]), [])
        self.assertEqual(len(self.encoder.query_calls), calls)
        for row in self.store.search(self.unit_id, 'cold', limit=10):
            original = (self.registry.data_dir / 'sources' / self.unit_id / 'files' / row['file_path']).read_bytes()
            lines = original.decode().splitlines(keepends=True)
            self.assertEqual(row['content'], ''.join(lines[row['start_line'] - 1:row['end_line']]))
            self.assertEqual(row['source_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(row['recipe_hash'], self.unit['recipe_hash'])
        composition = self.registry.compose('fixture-project', [self.unit_id], require_ready=True)
        result = SearchService(self.registry.data_dir, self.store).search('fixture-project', 'hot')
        self.assertEqual(result['state'], 'ready')

    def test_token_budget_reduces_lines_without_gaps_or_silent_truncation(self):
        text = ''.join(f'int row{i:02d} = {i}; // ' + 'x' * 55 + '\n' for i in range(80))
        (self.source / 'many.c').write_text(text, encoding='utf-8')
        self.refresh()
        status = self.store.index(self.unit_id)
        self.assertGreater(status['chunks_reduced_to_budget'], 0)
        self.assertEqual(status['document_truncated'], 0)
        chunks = [json.loads(x) for x in (self.index_root / 'artifact/chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        many = [c for c in chunks if c['file_path'] == 'many.c']
        covered = set()
        for chunk in many:
            self.assertLessEqual(chunk['token_length'], 1024)
            self.assertLessEqual(len(chunk['content']), 1800)
            covered.update(range(chunk['start_line'], chunk['end_line'] + 1))
        self.assertEqual(covered, set(range(1, 81)))

    def test_long_line_and_unexpected_truncation_never_publish_partial_index(self):
        (self.source / 'long.c').write_text('x' * 2000, encoding='utf-8')
        self.refresh()
        with self.assertRaisesRegex(BackendError, 'long.c:1'):
            self.store.index(self.unit_id)
        self.assertFalse((self.index_root / 'artifact').exists())
        self.assertEqual(self.registry.index_status(self.unit_id)['status'], 'error')
        self.assertFalse(list(self.index_root.glob('.qwen-stage-*')))
        self.encoder.truncated = True
        (self.source / 'long.c').write_text('int small;', encoding='utf-8')
        self.refresh()
        with self.assertRaisesRegex(BackendError, 'unexpectedly truncated'):
            self.store.index(self.unit_id)
        self.assertFalse((self.index_root / 'artifact').exists())

    def test_encoder_failure_is_atomic_and_retry_uses_unchanged_snapshot(self):
        self.encoder.failure = True
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            self.store.index(self.unit_id)
        self.assertFalse((self.index_root / 'artifact').exists())
        self.assertEqual(self.registry.index_status(self.unit_id)['status'], 'error')
        self.encoder.failure = False
        (self.source / 'a.c').write_text('working checkout changed', encoding='utf-8')
        self.assertEqual(self.store.index(self.unit_id)['status'], 'ready')
        row = self.store.search(self.unit_id, 'hot', paths=['a.c'])[0]
        self.assertIn('heat()', row['content'])

    def test_artifact_tamper_source_drift_and_stale_model_fail(self):
        self.store.index(self.unit_id)
        vectors = self.index_root / 'artifact/vectors.npy'
        original = vectors.read_bytes()
        vectors.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(BackendError, 'checksum'):
            self.store.search(self.unit_id, 'cold')
        vectors.write_bytes(original)
        self.encoder.model_manifest_sha256 = 'b' * 64
        with self.assertRaisesRegex(BackendError, 'stale'):
            self.store.search(self.unit_id, 'cold')
        self.encoder.model_manifest_sha256 = hashlib.sha256(b'fake pinned model').hexdigest()
        snapshot = self.registry.data_dir / 'sources' / self.unit_id / 'files/a.c'
        snapshot.write_text('snapshot was modified', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.store.search(self.unit_id, 'cold')

    def test_stale_artifact_recipe_is_rejected_even_with_resealed_file_checksum(self):
        self.store.index(self.unit_id)
        path = self.index_root / 'artifact/manifest.json'
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata['recipe']['options']['encoder']['max_length'] = 512
        path.write_text(json.dumps(metadata), encoding='utf-8')
        status_path = self.index_root / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        status['artifact_manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        status_path.write_text(json.dumps(status), encoding='utf-8')
        with self.assertRaisesRegex(BackendError, 'stale'):
            self.store.search(self.unit_id, 'cold')

    def test_snapshot_changed_during_encoding_never_becomes_ready(self):
        original_encode = self.encoder.encode_documents
        def changed(texts):
            vectors = original_encode(texts)
            snapshot = self.registry.data_dir / 'sources' / self.unit_id / 'files/a.c'
            snapshot.write_text('modified during encoding', encoding='utf-8')
            return vectors
        self.encoder.encode_documents = changed
        with self.assertRaises(ValueError):
            self.store.index(self.unit_id)
        self.assertFalse((self.index_root / 'artifact').exists())
        # Read the status directly: registry validation correctly refuses the
        # now-corrupt snapshot before even inspecting backend readiness.
        self.assertEqual(json.loads((self.index_root / 'status.json').read_text())['status'], 'error')

    def test_invalid_queries_paths_profiles_and_vector_values_are_rejected(self):
        for kwargs in ({'limit': True}, {'limit': 0}, {'paths': '../escape'}, {'paths': ['../escape']}, {'paths': ['C:/x']}):
            with self.assertRaises(ValueError):
                self.store.search(self.unit_id, 'cold', **kwargs)
        self.encoder.recipe = lambda: {'different': True}
        with self.assertRaisesRegex(BackendError, 'profile'):
            self.store.index(self.unit_id)
        with self.assertRaises(BackendError):
            self.store._matrix(np.full((1, 1024), np.nan, dtype=np.float32), 1)
        with self.assertRaises(BackendError):
            self.store._matrix(np.zeros((1, 1024), dtype=np.float32), 1)


if __name__ == '__main__':
    unittest.main()
