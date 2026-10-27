"""Pinned offline Qwen encoder. Importing this module never imports ML packages."""
from __future__ import annotations

from collections import OrderedDict
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
import time

QWEN_REPOSITORY = 'Qwen/Qwen3-Embedding-0.6B'
QWEN_REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
# Matches the saved raw Qwen 0.6B benchmark recipe. Refuse different encoder
# implementations rather than silently reusing their persisted index vectors.
EXPECTED_VERSIONS = {'torch': '2.11.0+cu128', 'transformers': '5.20.2',
    'sentence-transformers': '6.2.0', 'numpy': '2.5.4'}
QWEN_PROFILE = {'repository': QWEN_REPOSITORY, 'revision': QWEN_REVISION,
    'dimensions': 1024, 'max_length': 1024, 'dtype': 'bfloat16', 'device': 'cuda',
    'pooling': 'last_token', 'normalization': 'float32_L2', 'query_prompt_name': 'query',
    'document_prompt': '', 'batch_size': 1, 'use_cache': False}


def encoder_recipe():
    return dict(QWEN_PROFILE)


def profile_hash():
    return hashlib.sha256(json.dumps(encoder_recipe(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class QwenEncoder:
    def __init__(self, model_path, models_dir):
        self.model_path, self.models_dir = Path(model_path).resolve(), Path(models_dir).resolve()
        if not self.model_path.is_relative_to(self.models_dir):
            raise ValueError('Qwen model must be contained in models_dir')
        self.model = self._torch = self._np = None
        self.load_count, self.error = 0, None
        self.model_manifest_sha256 = None
        self.last_encode_stats = {}
        self._queries = OrderedDict()
        self._hooks = []
        self._loading = False
        self.runtime_versions = {}

    def recipe(self):
        return encoder_recipe()

    @property
    def profile_hash(self):
        return profile_hash()

    def _verify_model(self):
        path = self.model_path / 'bti-model-manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (manifest.get('status'), manifest.get('repository'), manifest.get('revision')) != ('complete', QWEN_REPOSITORY, QWEN_REVISION):
            raise ValueError('Qwen model manifest identity/revision differs')
        if manifest.get('local_modifications'):
            raise ValueError('Qwen model must contain unmodified upstream files')
        entries = manifest.get('files')
        if not isinstance(entries, list) or not entries:
            raise ValueError('Qwen model manifest has no files')
        seen = set()
        for entry in entries:
            relative = Path(entry['path'])
            target = self.model_path / relative
            if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen
                    or target.is_symlink() or not target.resolve().is_relative_to(self.model_path)):
                raise ValueError('Invalid Qwen manifest file path')
            seen.add(relative.as_posix())
            with target.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != entry['sha256']:
                raise ValueError('Qwen model file hash mismatch: ' + relative.as_posix())
        required = {'model.safetensors', 'modules.json', 'config.json', '1_Pooling/config.json',
                    'config_sentence_transformers.json', 'tokenizer.json', 'tokenizer_config.json'}
        if not required <= seen:
            raise ValueError('Qwen manifest omits required encoder files')
        for target in self.model_path.rglob('*'):
            if target.suffix in ('.safetensors', '.bin') and target.relative_to(self.model_path).as_posix() not in seen:
                raise ValueError('Unmanifested Qwen weights')
        pooling = json.loads((self.model_path / '1_Pooling/config.json').read_text(encoding='utf-8'))
        if not pooling.get('pooling_mode_lasttoken') or any(value for key, value in pooling.items()
                if key.startswith('pooling_mode_') and key != 'pooling_mode_lasttoken'):
            raise ValueError('Qwen native pooling must be last-token only')
        self.model_manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def _verify_versions(self):
        observed = {}
        for package in EXPECTED_VERSIONS:
            try:
                observed[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                observed[package] = None
        self.runtime_versions = observed
        mismatches = [f'{package}: expected {expected}, got {observed[package]}'
            for package, expected in EXPECTED_VERSIONS.items() if observed[package] != expected]
        if mismatches:
            raise ValueError('Unsupported Qwen runtime versions: ' + '; '.join(mismatches))

    def load(self):
        if self.model is not None:
            return self.health()
        try:
            self._loading = True
            self._verify_versions()
            cache, temporary = self.models_dir / 'cache' / 'huggingface', self.models_dir / 'tmp'
            cache.mkdir(parents=True, exist_ok=True)
            temporary.mkdir(parents=True, exist_ok=True)
            os.environ.update(HF_HOME=str(cache), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_DATASETS_OFFLINE='1', TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary))
            tempfile.tempdir = str(temporary)
            self._verify_model()
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
            self._np, self._torch = np, torch
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise ValueError('Qwen requires CUDA with BF16 support')
            torch.set_num_threads(3)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.model = SentenceTransformer(str(self.model_path), device='cuda', local_files_only=True,
                trust_remote_code=False, model_kwargs={'dtype': torch.bfloat16, 'low_cpu_mem_usage': True})
            self.model.eval()
            self.model.max_seq_length = 1024
            for _, module in self.model.named_modules():
                config = getattr(module, 'config', None)
                if config is not None and hasattr(config, 'use_cache'):
                    config.use_cache = False
            def normalize_hook(module, args):
                features = args[0]
                key = module.module_input_name
                if key in features:
                    features[key] = features[key].float()
                return (features, *args[1:])
            for module in self.model:
                if type(module).__name__ == 'Normalize':
                    self._hooks.append(module.register_forward_pre_hook(normalize_hook))
            if not self._hooks or 'query' not in self.model.prompts or self.model.get_sentence_embedding_dimension() != 1024:
                raise ValueError('Qwen native Normalize/query prompt/dimension differs')
            self._queries.clear()
            self.load_count += 1
            self.error = None
            torch.cuda.synchronize()
        except Exception as exc:
            self.unload()
            self.error = str(exc)
            raise
        finally:
            self._loading = False
        return self.health()

    def unload(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self.model = None
        self._queries.clear()
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_initialized():
            self._torch.cuda.empty_cache()
        return self.health()

    def health(self):
        memory = {}
        if self._torch is not None and self.model is not None:
            memory = {'cuda_allocated_bytes': self._torch.cuda.memory_allocated(),
                      'cuda_reserved_bytes': self._torch.cuda.memory_reserved()}
        return {'resident': self.model is not None, 'ready': self.model is not None and not self._loading,
            'load_count': self.load_count, 'model': QWEN_REPOSITORY, 'revision': QWEN_REVISION,
            'device': 'cuda', 'profile_hash': self.profile_hash, 'model_manifest_sha256': self.model_manifest_sha256,
            'memory': memory, 'error': self.error, 'query_cache_entries': len(self._queries),
            'runtime_versions': dict(self.runtime_versions)}

    def _encode(self, texts, query):
        if not isinstance(texts, (list, tuple)) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Encoder texts must be nonempty strings')
        self.load()
        np = self._np
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        started = time.perf_counter()
        options = {'prompt_name': 'query'} if query else {'prompt': ''}
        prefix = self.model.prompts['query'] if query else ''
        lengths = [int(n) for n in self.model.tokenizer([prefix + text for text in texts], padding=False,
                                                       truncation=False, return_length=True)['length']]
        if query and any(length > 1024 for length in lengths):
            raise ValueError('Query exceeds the fixed 1024-token encoder cap')
        values, hits = [], 0
        method = self.model.encode_query if query else self.model.encode_document
        for text in texts:
            if query and text in self._queries:
                vector = self._queries.pop(text)
                self._queries[text] = vector
                hits += 1
            else:
                encoded = method([text], batch_size=1, show_progress_bar=False, convert_to_numpy=True,
                                 normalize_embeddings=False, **options)
                vector = np.asarray(encoded[0], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
                    raise ValueError('Invalid Qwen embedding output')
                vector = vector / norm
                if query:
                    self._queries[text] = vector.copy()
                    if len(self._queries) > 128:
                        self._queries.popitem(last=False)
            values.append(vector.copy())
        self._torch.cuda.synchronize()
        self.last_encode_stats = {'n': len(texts), 'token_lengths': lengths, 'truncated': sum(n > 1024 for n in lengths),
            'encode_seconds': time.perf_counter() - started, 'cache_hits': hits}
        return np.stack(values).astype(np.float32, copy=False)

    def encode_documents(self, texts):
        return self._encode(texts, False)

    def count_tokens(self, texts):
        self.load()
        return [int(n) for n in self.model.tokenizer(texts, padding=False, truncation=False, return_length=True)['length']]

    def encode_queries(self, texts):
        return self._encode(texts, True)
