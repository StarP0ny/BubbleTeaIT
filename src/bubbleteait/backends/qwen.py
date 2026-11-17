"""Pinned Qwen retrieval over immutable snapshots; ML imports are worker-only."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import uuid

from .base import BackendCapabilities
from .cocoindex import BackendError, _json_write, _sha256, _unit_lock
from ..qwen_encoder import QWEN_REPOSITORY, QWEN_REVISION, encoder_recipe
from ..registry import IndexRecipe, Registry, _digest, _relative, _safe_file


MODEL_REPOSITORY = QWEN_REPOSITORY
MODEL_REVISION = QWEN_REVISION
_CHUNK_BUDGET = {'max_lines': 32, 'overlap_lines': 4, 'max_chars': 1800,
                 'max_tokens': 1024, 'long_line': 'error'}
# Markdown is split by H2 sections; C/C++ is split by top-level function in the
# current buildable recipe. Both share the same window/char/token budget.
V1_OPTIONS = {'encoder': encoder_recipe(),
              'chunking': {'strategy': 'markdown-h2-sections-v1', **_CHUNK_BUDGET},
              'metadata': 'path-heading-function-v1'}
V2_OPTIONS = {'encoder': encoder_recipe(),
              'chunking': {'strategy': 'markdown-h2-function-v2', **_CHUNK_BUDGET},
              'metadata': 'path-heading-function-v1'}
LEGACY_OPTIONS = {'encoder': encoder_recipe(),
                  'chunking': dict(_CHUNK_BUDGET),
                  'metadata': 'path-heading-function-v1'}
# Buildable default; config/tests also read this as the current profile.
QWEN_OPTIONS = V2_OPTIONS


def default_recipe():
    return IndexRecipe('qwen-dense', MODEL_REPOSITORY, MODEL_REVISION, 'bti-sections-v2',
                       options=json.loads(json.dumps(V2_OPTIONS)))


def sections_v1_recipe():
    return IndexRecipe('qwen-dense', MODEL_REPOSITORY, MODEL_REVISION, 'bti-sections-v1',
                       options=json.loads(json.dumps(V1_OPTIONS)))


def legacy_recipe():
    return IndexRecipe('qwen-dense', MODEL_REPOSITORY, MODEL_REVISION, 'bti-lines-v1',
                       options=json.loads(json.dumps(LEGACY_OPTIONS)))


default_qwen_recipe = default_recipe


def _recipe_kind(recipe):
    recipe = asdict(recipe) if isinstance(recipe, IndexRecipe) else recipe
    if isinstance(recipe, dict) and _digest(recipe) == _digest(asdict(default_recipe())):
        return 'sections-v2'
    if isinstance(recipe, dict) and _digest(recipe) == _digest(asdict(sections_v1_recipe())):
        return 'sections-v1'
    if isinstance(recipe, dict) and _digest(recipe) == _digest(asdict(legacy_recipe())):
        return 'lines'
    raise BackendError('Qwen requires a pinned 0.6B CUDA BF16 cap1024 recipe')


def validate_recipe(recipe):
    if _recipe_kind(recipe) != 'sections-v2':
        raise BackendError('Qwen requires the pinned 0.6B CUDA BF16 cap1024 bti-sections-v2 recipe')
    return asdict(recipe) if isinstance(recipe, IndexRecipe) else recipe


def _query(query, limit, paths):
    if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 100:
        raise BackendError('Search requires a query and integer limit from 1 to 100')
    if paths is not None:
        if not isinstance(paths, list):
            raise BackendError('paths must be a list of relative filters')
        for path in paths:
            _relative(path)


class QwenBackend:
    capabilities = BackendCapabilities(name='qwen-dense')

    def __init__(self, data_dir, python_executable, model_path, *, models_dir=None, timeout_seconds=600):
        from ..qwen_runtime import QwenRuntimeClient
        self.client = QwenRuntimeClient(data_dir, python_executable, model_path,
            models_dir=models_dir, timeout_seconds=timeout_seconds)

    def index(self, unit_id):
        return self.client.request('index', unit_id=unit_id)['index']

    def search(self, unit_id, query, *, limit=10, paths=None):
        _query(query, limit, paths)
        return self.client.request('search', unit_id=unit_id, query=query, limit=limit, paths=paths)['results']


# Single-line C/C++ function definitions: `{` on the signature line, or a bare
# signature whose opening brace sits on a following line. Shared by metadata
# and the function-aware chunker.
_FUNC_BRACE = re.compile(r'^\s*(?:[A-Za-z_]\w*[\s*]+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{\s*$')
_FUNC_HEADER = re.compile(r'^\s*(?:[A-Za-z_]\w*[\s*]+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$')
_FUNC_WORDS = frozenset(('if', 'for', 'while', 'switch'))


def _metadata(path, lines, start):
    prefix = 'Источник: ' + path
    if Path(path).suffix.lower() in ('.md', '.markdown'):
        fenced, heading = False, None
        for line in lines[:start]:
            if re.match(r'^\s*(```|~~~)', line):
                fenced = not fenced
            found = re.match(r'^#{1,6}\s+(.+?)\s*#*\s*$', line) if not fenced else None
            if found:
                heading = found.group(1)
        if heading:
            prefix += '\nРаздел: ' + heading
    elif Path(path).suffix.lower() in ('.c', '.h', '.cpp', '.hpp', '.cc'):
        found = _FUNC_BRACE.match(lines[start - 1]) or _FUNC_HEADER.match(lines[start - 1])
        if found and found.group(1) not in _FUNC_WORDS:
            prefix += '\nФункция: ' + found.group(1)
    return prefix


class QwenIndexStore:
    """Worker-local exact dense store; never reads the mutable working checkout."""

    def __init__(self, data_dir, encoder):
        self.data_dir, self.encoder = Path(data_dir).resolve(), encoder
        self.registry = Registry(self.data_dir)

    def _context(self, unit_id):
        manifest = self.registry.load_unit(unit_id, verify_files=True)
        _recipe_kind(manifest['recipe'])
        if self.encoder.recipe() != encoder_recipe():
            raise BackendError('Encoder profile differs from pinned recipe')
        root = self.data_dir / 'indexes' / unit_id
        if root.is_symlink() or not root.resolve().is_relative_to(self.data_dir):
            raise BackendError('Index directory escapes data directory')
        root.mkdir(parents=True, exist_ok=True)
        return manifest, root

    def _load_encoder(self):
        self.encoder.load()
        checksum = getattr(self.encoder, 'model_manifest_sha256', None)
        if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise BackendError('Encoder has no verified model manifest identity')
        return checksum

    def _source(self, manifest, entry):
        root = self.data_dir / 'sources' / manifest['unit_id'] / 'files'
        path = _safe_file(root, entry['path'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise BackendError('Snapshot source hash mismatch: ' + entry['path'])
        try:
            return raw.decode('utf-8').splitlines(keepends=True)
        except UnicodeDecodeError as error:
            raise BackendError('Qwen text recipe requires UTF-8: ' + entry['path']) from error

    @staticmethod
    def _markdown_spans(lines):
        """Split Markdown at H2 boundaries without separating a lone title.

        H3 and deeper headings stay inside their H2 section. Files without H2
        use the ordinary line windows, while an H1 title followed immediately
        by the first H2 stays with that section.
        """
        headings, fenced = [], False
        for number, line in enumerate(lines, 1):
            if re.match(r'^\s*(```|~~~)', line):
                fenced = not fenced
                continue
            if fenced:
                continue
            match = re.match(r'^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$', line)
            if match:
                headings.append((number, len(match.group(1))))
        starts = [number for number, level in headings if level == 2]
        if not starts:
            return [(1, len(lines))]
        nonblank = [line for line in lines[:starts[0] - 1] if line.strip()]
        title_only = (len(nonblank) == 1 and re.match(r'^\s{0,3}#\s+\S', nonblank[0]) is not None)
        starts = [1] + [number for number in starts if number > 1 and not (title_only and number == starts[0])]
        spans = []
        for index, start in enumerate(starts):
            end = starts[index + 1] - 1 if index + 1 < len(starts) else len(lines)
            if end >= start and any(line.strip() for line in lines[start - 1:end]):
                spans.append((start, end))
        return spans

    @staticmethod
    def _c_function_spans(lines, max_lines=32):
        """Split C/C++ at top-level functions and long-switch case labels.

        Detects single-line signatures whose `{` is on the signature line or on
        a following blank/comment line. Each span runs to the next top-level
        function; leading includes/macros/globals stay a preamble span. Inside a
        function, a switch whose body is longer than ``max_lines`` is split at
        its ``case``/``default`` labels so a window does not cross a case
        branch. Functions larger than the window are still sub-split inside
        their own boundaries. This is a heuristic: a top-level brace inside a
        char/string literal can shift a boundary, but coverage is preserved
        because spans always partition the file.
        """
        starts, depth = [], 0
        for number, line in enumerate(lines, 1):
            name = None
            if depth == 0:
                match = _FUNC_BRACE.match(line)
                if match:
                    name = match.group(1)
                else:
                    header = _FUNC_HEADER.match(line)
                    if header:
                        for look in range(number, min(len(lines), number + 3)):
                            candidate = lines[look].strip()
                            if candidate.startswith('{'):
                                name = header.group(1)
                                break
                            if candidate and not candidate.startswith(('//', '/*', '*')):
                                break
            if name:
                starts.append(number)
            depth += line.count('{') - line.count('}')
            if depth < 0:
                depth = 0
        if not starts:
            return [(1, len(lines))]
        base = []
        if starts[0] > 1:
            base.append((1, starts[0] - 1, False))
        for index, start in enumerate(starts):
            end = starts[index + 1] - 1 if index + 1 < len(starts) else len(lines)
            if end >= start:
                base.append((start, end, True))
        spans = []
        for span_start, span_end, is_function in base:
            if not is_function:
                spans.append((span_start, span_end))
                continue
            cases = QwenIndexStore._long_switch_case_lines(lines, span_start, span_end, max_lines)
            previous = span_start
            for case_line in sorted(cases):
                if previous <= case_line - 1:
                    spans.append((previous, case_line - 1))
                previous = case_line
            if previous <= span_end:
                spans.append((previous, span_end))
        return spans

    @staticmethod
    def _long_switch_case_lines(lines, start, end, max_lines):
        """Case-label lines of switches whose body exceeds ``max_lines``.

        Tracks brace depth and a stack of switch-body depths. Only direct case
        labels of the innermost switch are recorded; when a switch closes, its
        labels are kept only if the switch body was long enough to need them.
        """
        case_labels, depth = set(), 0
        switches = []  # [body_depth, open_line, ...case lines]
        for number in range(start, end + 1):
            line = lines[number - 1]
            stripped = line.strip()
            if switches and depth == switches[-1][0] and not stripped.startswith(('//', '/*', '*')):
                if re.match(r'^\s*case\s+', line) or re.match(r'^\s*default\s*:', line):
                    switches[-1].append(number)
            if re.search(r'\bswitch\b', line) and '(' in line and not stripped.startswith(('//', '/*', '*', '#')):
                switches.append([depth + 1, number])
            depth += line.count('{') - line.count('}')
            if depth < 0:
                depth = 0
            while switches and switches[-1][0] > depth:
                switch = switches.pop()
                if number - switch[1] > max_lines:
                    case_labels.update(switch[2:])
        return case_labels

    def _chunks(self, manifest):
        chunks, empty, reductions = [], [], 0
        options = QWEN_OPTIONS['chunking']
        for entry in manifest['files']:
            lines = self._source(manifest, entry)
            if not ''.join(lines).strip():
                empty.append(entry['path'])
                continue
            suffix = Path(entry['path']).suffix.lower()
            if suffix in ('.md', '.markdown'):
                spans = self._markdown_spans(lines)
            elif suffix in ('.c', '.h', '.cpp', '.hpp', '.cc'):
                spans = self._c_function_spans(lines, options['max_lines'])
            else:
                spans = [(1, len(lines))]
            for span_start, span_end in spans:
                start = span_start
                while start <= span_end:
                    end = min(span_end, start + options['max_lines'] - 1)
                    prefix = _metadata(entry['path'], lines, start)
                    initial_end = end
                    while True:
                        content = ''.join(lines[start - 1:end])
                        text = prefix + '\n\n' + content
                        tokens = self.encoder.count_tokens([text])
                        if (not isinstance(tokens, list) or len(tokens) != 1
                                or type(tokens[0]) is not int or tokens[0] < 0):
                            raise BackendError('Encoder returned invalid token lengths')
                        if len(content) <= options['max_chars'] and tokens[0] <= options['max_tokens']:
                            break
                        if end == start:
                            raise BackendError(f'Unsupported long line {entry["path"]}:{start}; '
                                f'{tokens[0]} tokens with metadata / {len(content)} source chars')
                        end -= 1
                    reductions += end != initial_end
                    if content.strip():
                        value = {'file_path': entry['path'], 'start_line': start, 'end_line': end,
                                 'source_sha256': entry['sha256'], 'content': content,
                                 'embedding_text': text, 'token_length': tokens[0]}
                        value['chunk_id'] = _digest({k: value[k] for k in ('file_path', 'start_line', 'end_line', 'source_sha256', 'content')})
                        chunks.append(value)
                    if end == span_end:
                        break
                    start = max(start + 1, end - options['overlap_lines'] + 1)
        return chunks, {'empty_count': len(empty), 'zero_chunk_files': empty,
                        'chunks_reduced_to_budget': reductions, 'document_truncated': 0,
                        'max_document_tokens': max((c['token_length'] for c in chunks), default=0)}

    @staticmethod
    def _matrix(value, count):
        import numpy as np
        matrix = np.asarray(value)
        if (matrix.dtype != np.float32 or matrix.shape != (count, 1024) or not np.isfinite(matrix).all()
                or (count and not np.allclose(np.linalg.norm(matrix, axis=1), 1, rtol=0, atol=2e-5))):
            raise BackendError('Encoder/index vectors must be finite float32 normalized [N,1024]')
        return matrix

    def _read(self, manifest, root, model_hash, *, require_ready=True):
        import numpy as np
        artifact = root / 'artifact'
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(root):
            raise BackendError('Index artifact escapes unit directory')
        try:
            status = self.registry.index_status(manifest['unit_id'])
            metadata = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
            state = status.get('status')
            if (require_ready and state != 'ready') or state not in ('ready', 'building', 'error', 'missing'):
                raise BackendError('Qwen index is not ready')
            artifact_hash = status.get('artifact_manifest_sha256')
            if ((state == 'ready' or artifact_hash is not None)
                    and artifact_hash != _sha256(artifact / 'manifest.json')):
                raise BackendError('Index artifact manifest checksum mismatch')
            status_model = status.get('model_manifest_sha256')
            if ((state == 'ready' or status_model is not None) and status_model != model_hash):
                raise BackendError('Qwen index status is stale or belongs to different model files')
            kind = _recipe_kind(manifest['recipe'])
            code_hash = metadata.get('backend_code_sha256')
            code_ok = (code_hash == _sha256(Path(__file__)) if kind == 'sections-v2'
                       else isinstance(code_hash, str) and re.fullmatch('[0-9a-f]{64}', code_hash) is not None)
            if (metadata.get('unit_id') != manifest['unit_id']
                    or metadata.get('recipe_hash') != manifest['recipe_hash']
                    or _digest(metadata.get('recipe')) != _digest(manifest['recipe'])
                    or metadata.get('source_files') != manifest['files']
                    or metadata.get('model_manifest_sha256') != model_hash
                    or not code_ok):
                raise BackendError('Qwen index is not ready or has a stale source/model/recipe')
            if set(metadata['files']) != {'chunks.jsonl', 'vectors.npy'}:
                raise BackendError('Index artifact file list differs')
            for name, expected in metadata['files'].items():
                path = _safe_file(artifact, name)
                if _sha256(path) != expected:
                    raise BackendError('Index artifact checksum mismatch: ' + name)
            chunks = [json.loads(line) for line in (artifact / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
            if len(chunks) != metadata['chunks'] or len({c['chunk_id'] for c in chunks}) != len(chunks):
                raise BackendError('Index chunk identities/count differ')
            entries = {entry['path']: entry for entry in manifest['files']}
            sources = {}
            for chunk in chunks:
                entry = entries.get(chunk['file_path'])
                if entry is None or chunk['source_sha256'] != entry['sha256']:
                    raise BackendError('Index chunk outside source whitelist')
                if entry['path'] not in sources:
                    sources[entry['path']] = self._source(manifest, entry)
                lines = sources[entry['path']]
                start, end = chunk['start_line'], chunk['end_line']
                # The current buildable recipe must reproduce its embedding
                # text from source + metadata; frozen read-only recipes keep
                # the embedding text they were published with.
                expected_text = (_metadata(entry['path'], lines, start) + '\n\n' + chunk['content']
                                 if kind == 'sections-v2' else chunk['embedding_text'])
                if (type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines)
                        or chunk['content'] != ''.join(lines[start - 1:end])
                        or chunk['embedding_text'] != expected_text
                        or type(chunk['token_length']) is not int or not 0 <= chunk['token_length'] <= 1024):
                    raise BackendError('Index chunk source coordinates/text differ')
                identity = _digest({k: chunk[k] for k in ('file_path', 'start_line', 'end_line', 'source_sha256', 'content')})
                if chunk['chunk_id'] != identity:
                    raise BackendError('Index chunk identity differs from source proof')
            vectors = self._matrix(np.load(artifact / 'vectors.npy', allow_pickle=False), len(chunks))
            return status, chunks, vectors
        except (OSError, ValueError, KeyError, TypeError) as error:
            if isinstance(error, BackendError):
                raise
            raise BackendError('Cannot validate Qwen index artifact: ' + str(error)) from error

    def index(self, unit_id):
        import numpy as np
        manifest, root = self._context(unit_id)
        kind = _recipe_kind(manifest['recipe'])
        with _unit_lock(root / 'writer.lock'):
            model_hash = self._load_encoder()
            artifact = root / 'artifact'
            if artifact.exists():
                if artifact.is_symlink() or not artifact.resolve().is_relative_to(root):
                    raise BackendError('Index artifact escapes unit directory')
                try:
                    artifact_meta = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
                except (OSError, ValueError) as error:
                    raise BackendError('Cannot read Qwen index artifact manifest') from error
                # A change in the chunker implementation invalidates existing
                # vectors for the buildable recipe; rebuild under the new code.
                # Read-only recipes keep their frozen artifact regardless.
                stale_code = (kind == 'sections-v2'
                              and artifact_meta.get('backend_code_sha256') != _sha256(Path(__file__)))
                if stale_code:
                    shutil.rmtree(artifact)
                else:
                    status, chunks, _ = self._read(manifest, root, model_hash, require_ready=False)
                    if status.get('status') != 'ready':
                        # Publication is an atomic directory rename followed by a
                        # separate registry status update. A crash between them
                        # leaves a complete artifact; recover only after the same
                        # source/recipe/model/vector validation used by search.
                        metadata = artifact_meta
                        status = {'schema_version': 1, 'unit_id': unit_id,
                            'recipe_hash': manifest['recipe_hash'], 'backend': 'qwen-dense',
                            'model_manifest_sha256': model_hash, 'status': 'ready',
                            'chunks': len(chunks), 'whitelist_files': len(manifest['files']),
                            **{key: metadata['stats'][key] for key in ('empty_count', 'zero_chunk_files',
                                'chunks_reduced_to_budget', 'document_truncated', 'max_document_tokens')},
                            'encode_stats': metadata['encode_stats'],
                            'artifact_manifest_sha256': _sha256(artifact / 'manifest.json'),
                            'updated_at': datetime.now(timezone.utc).isoformat(), 'recovered': True}
                        _json_write(root / 'status.json', status)
                    return {**status, 'reused': True}
            if kind != 'sections-v2':
                raise BackendError('Legacy Qwen recipe is read-only; reindex this source with bti-sections-v2')
            status = {'schema_version': 1, 'unit_id': unit_id, 'recipe_hash': manifest['recipe_hash'],
                      'backend': 'qwen-dense', 'model_manifest_sha256': model_hash}
            _json_write(root / 'status.json', {**status, 'status': 'building'})
            stage = root / ('.qwen-stage-' + uuid.uuid4().hex)
            stage.mkdir()
            try:
                chunks, stats = self._chunks(manifest)
                vectors = self._matrix(self.encoder.encode_documents([c['embedding_text'] for c in chunks]), len(chunks)) if chunks else np.empty((0, 1024), dtype=np.float32)
                encode_stats = dict(getattr(self.encoder, 'last_encode_stats', {})) if chunks else {}
                if encode_stats.get('truncated', 0):
                    raise BackendError('Encoder unexpectedly truncated document input; index not published')
                (stage / 'chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks), encoding='utf-8')
                with (stage / 'vectors.npy').open('wb') as stream:
                    np.save(stream, vectors, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                metadata = {**status, 'recipe': manifest['recipe'], 'source_files': manifest['files'],
                            'backend_code_sha256': _sha256(Path(__file__)), 'chunks': len(chunks),
                            'stats': stats, 'encode_stats': encode_stats,
                            'files': {name: _sha256(stage / name) for name in ('chunks.jsonl', 'vectors.npy')}}
                _json_write(stage / 'manifest.json', metadata)
                self.registry.load_unit(unit_id, verify_files=True)
                os.replace(stage, root / 'artifact')
                status.update(status='ready', chunks=len(chunks), whitelist_files=len(manifest['files']),
                    **stats, encode_stats=encode_stats,
                    artifact_manifest_sha256=_sha256(root / 'artifact/manifest.json'),
                    updated_at=datetime.now(timezone.utc).isoformat())
                _json_write(root / 'status.json', status)
                return status
            except Exception as error:
                _json_write(root / 'status.json', {**status, 'status': 'error', 'error': str(error)})
                raise
            finally:
                if stage.exists():
                    resolved = stage.resolve()
                    if resolved.is_relative_to(root.resolve()) and resolved.name.startswith('.qwen-stage-'):
                        shutil.rmtree(resolved)

    def search(self, unit_id, query, *, limit=10, paths=None):
        _query(query, limit, paths)
        manifest, root = self._context(unit_id)
        with _unit_lock(root / 'writer.lock'):
            model_hash = self._load_encoder()
            _, chunks, vectors = self._read(manifest, root, model_hash)
            with sqlite3.connect(':memory:') as connection:
                eligible = [i for i, chunk in enumerate(chunks) if paths is None or any(
                    connection.execute('SELECT ? GLOB ?', (chunk['file_path'], pattern)).fetchone()[0] for pattern in paths)]
            if not eligible:
                return []
            query_vector = self._matrix(self.encoder.encode_queries([query]), 1)[0]
            scores = vectors[eligible] @ query_vector
            ranked = sorted(zip(eligible, scores), key=lambda pair: (-float(pair[1]), chunks[pair[0]]['file_path'], chunks[pair[0]]['start_line']))[:limit]
            self.registry.load_unit(unit_id, verify_files=True)
            return [{k: chunks[i][k] for k in ('file_path', 'start_line', 'end_line', 'content', 'source_sha256')}
                    | {'score': float(score), 'recipe_hash': manifest['recipe_hash'], 'backend': 'qwen-dense',
                       'score_kind': 'cosine', 'unit_id': unit_id, 'source_id': manifest['source_id']}
                    for i, score in ranked]
