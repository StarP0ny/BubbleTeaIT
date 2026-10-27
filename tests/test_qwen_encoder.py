"""Manifest rejection and fake-model normalization/prompt/cache checks."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from bubbleteait.qwen_encoder import QwenEncoder, QWEN_REPOSITORY, QWEN_REVISION, EXPECTED_VERSIONS, encoder_recipe

try:
    import numpy as np
except ImportError:
    np = None


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.encoder = QwenEncoder(self.root / 'qwen', self.root)

    def test_profile_is_fixed_and_returns_independent_recipe(self):
        recipe = encoder_recipe()
        self.assertEqual((recipe['repository'], recipe['revision'], recipe['dimensions'], recipe['max_length']),
                         (QWEN_REPOSITORY, QWEN_REVISION, 1024, 1024))
        recipe['dimensions'] = 2560
        self.assertEqual(encoder_recipe()['dimensions'], 1024)
        with self.assertRaises(ValueError):
            QwenEncoder(self.root.parent / 'outside', self.root)

    def test_bad_identity_and_weights_are_rejected_before_ml_import(self):
        self.encoder.model_path.mkdir()
        manifest_path = self.encoder.model_path / 'bti-model-manifest.json'
        manifest = {'status': 'complete', 'repository': QWEN_REPOSITORY, 'revision': 'wrong', 'files': []}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'identity/revision'):
            self.encoder._verify_model()
        manifest.update(revision=QWEN_REVISION, files=[{'path': 'model.safetensors', 'sha256': 'wrong'}])
        (self.encoder.model_path / 'model.safetensors').write_bytes(b'not real weights')
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.encoder._verify_model()
        manifest['files'][0]['path'] = '../outside'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'file path'):
            self.encoder._verify_model()

    def test_runtime_version_guard_refuses_incompatible_or_missing_packages_before_weights(self):
        recipe, identity = self.encoder.recipe(), self.encoder.profile_hash
        with patch('bubbleteait.qwen_encoder.metadata.version', side_effect=EXPECTED_VERSIONS.__getitem__):
            self.encoder._verify_versions()
        self.assertEqual(self.encoder.health()['runtime_versions'], EXPECTED_VERSIONS)
        for package in EXPECTED_VERSIONS:
            observed = {**EXPECTED_VERSIONS, package: 'incompatible'}
            with self.subTest(package=package), patch('bubbleteait.qwen_encoder.metadata.version', side_effect=observed.__getitem__), \
                    patch.object(self.encoder, '_verify_model') as weights:
                with self.assertRaisesRegex(ValueError, 'Unsupported Qwen runtime versions.*' + package):
                    self.encoder.load()
                weights.assert_not_called()
                self.assertFalse(self.encoder.health()['resident'])
                self.assertEqual(self.encoder.health()['runtime_versions'], observed)
        from importlib.metadata import PackageNotFoundError
        with patch('bubbleteait.qwen_encoder.metadata.version', side_effect=PackageNotFoundError('missing')), \
                patch.object(self.encoder, '_verify_model') as weights:
            with self.assertRaisesRegex(ValueError, 'got None'):
                self.encoder.load()
            weights.assert_not_called()
        self.assertEqual(self.encoder.recipe(), recipe)
        self.assertEqual(self.encoder.profile_hash, identity)
        self.assertEqual(self.encoder.load_count, 0)

    @unittest.skipIf(np is None, 'Numeric tests use indexing environment')
    def test_native_query_document_routes_float32_normalization_and_shared_query_cache(self):
        calls = []
        class Model:
            prompts = {'query': 'query-prefix:'}
            def tokenizer(self, texts, **options):
                return {'length': [len(text) for text in texts]}
            def encode_query(self, texts, **options):
                calls.append(('query', texts, options))
                return np.ones((1, 1024), dtype=np.float64)
            def encode_document(self, texts, **options):
                calls.append(('document', texts, options))
                return np.ones((1, 1024), dtype=np.float64) * 2
        self.encoder.model, self.encoder._np = Model(), np
        self.encoder._torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None,
            memory_allocated=lambda: 0, memory_reserved=lambda: 0))
        documents = self.encoder.encode_documents(['source'])
        queries = self.encoder.encode_queries(['same'])
        queries[0, 0] = -99
        cached = self.encoder.encode_queries(['same'])
        self.assertEqual(documents.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(documents, axis=1), 1)
        self.assertGreater(cached[0, 0], 0)
        self.assertEqual([row[0] for row in calls], ['document', 'query'])
        self.assertEqual(calls[0][2]['prompt'], '')
        self.assertEqual(calls[1][2]['prompt_name'], 'query')
        self.assertFalse(calls[0][2]['normalize_embeddings'])
        self.assertEqual(self.encoder.last_encode_stats['cache_hits'], 1)
        self.assertEqual(self.encoder.count_tokens(['abc']), [3])
        with self.assertRaisesRegex(ValueError, '1024-token'):
            self.encoder.encode_queries(['x' * 1025])


if __name__ == '__main__':
    unittest.main()
