"""Stable selected-file views and content-addressed reads around stock indexes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .config import ConfigError
from .engine_process import EngineProcess
from .registry import _atomic_json, _digest, _relative


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.bti-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class CocoEngine:
    def __init__(self, config, process_factory=EngineProcess):
        self.config = config
        self.settings = config.engines['cocoindex']
        model = Path(self.settings['model_path']).resolve()
        if not model.is_dir() or not model.is_relative_to(config.models_dir.resolve()):
            raise ConfigError('CocoIndex model must be a local directory inside models_dir')
        identity = {**self.settings, 'model_path': str(model), 'adapter': 'cocoindex-code@0.2.42/bti-live-v1'}
        # In-place model metadata changes create another namespace, not mixed vectors.
        for filename in ('config.json', 'bti-model-manifest.json', 'modules.json'):
            path = model / filename
            if path.is_file():
                identity[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fingerprint = _digest(identity)
        self.directory = config.data_dir / 'engines/cocoindex' / self.fingerprint
        self.runtime = self.directory / 'runtime'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.worker_settings = self.directory / 'bridge.json'
        _atomic_json(self.worker_settings, {'runtime': str(self.runtime)})
        _atomic_json(self.runtime / 'global_settings.yml', {'embedding': {
            'provider': 'sentence-transformers', 'model': str(model), 'device': self.settings['device'],
            'indexing_params': self.settings['indexing_params'], 'query_params': self.settings['query_params']},
            'daemon': {'idle_timeout_minutes': 60}})
        self.process = process_factory(self.settings['python_executable'],
            Path(__file__).with_name('coco_engine_worker.py'), self.worker_settings, self.settings['timeout_seconds'])

    def project_directory(self, root):
        return self.directory / 'projects' / _digest(str(root))

    def manifest(self, root):
        manifest = read_json(self.project_directory(root) / 'current.json')
        if manifest is None:
            return None
        if (manifest.get('root') != str(root) or manifest.get('engine') != self.fingerprint
                or manifest.get('generation') != 'g-' + _digest({'files': manifest.get('files'),
                    'config': manifest.get('config_hash'), 'engine': self.fingerprint})):
            raise ConfigError('Native source manifest does not match its generation/root/engine')
        for key in manifest['files']:
            _relative(key)
        return manifest

    def blob(self, sha):
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ConfigError('Invalid source hash')
        return self.config.data_dir / 'engines/blobs' / sha[:2] / sha

    def text(self, record):
        raw = self.blob(record['sha256']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ConfigError('Stored source bytes changed')
        return raw.decode('utf-8-sig')

    def records(self, root, plan):
        records = {}
        for source in plan['sources']:
            spec = source['selection']
            for entry in source['files']:
                absolute = (Path(spec['root']) / entry['path']).absolute()
                if absolute.resolve() != absolute or not absolute.is_relative_to(Path(spec['root']).resolve()):
                    raise ConfigError('Selected source path changed or escaped its root')
                relative = absolute.relative_to(root).as_posix() if absolute.is_relative_to(root) else None
                key = spec['source_id'] + '/' + entry['path']
                records[key] = {'source_id': spec['source_id'], 'source_path': entry['path'],
                    'path': relative or key, 'absolute_path': str(absolute), 'sha256': entry['sha256'],
                    'role': spec['role'], 'source_root': str(spec['root'])}
        return records

    def synchronize(self, root, project, plan, *, verify=lambda: None):
        if not plan['ready']:
            raise ConfigError('Source preflight failed; inspect bti plan')
        records = self.records(root, plan)
        generation = 'g-' + _digest({'files': records, 'config': project.config_hash, 'engine': self.fingerprint})
        directory = self.project_directory(root)
        view = directory / 'view'
        if view.is_symlink() or view.is_junction():
            raise ConfigError('Engine view cannot be a filesystem link')
        view.mkdir(parents=True, exist_ok=True)
        previous = self.manifest(root)
        if previous and previous['generation'] == generation and self.available(root):
            return previous
        # Until the native transaction completes, never advertise this view as ready.
        _atomic_json(directory / 'pending.json', {'generation': generation})
        inventory = set(read_json(directory / 'staged.json') or []) | set((previous or {}).get('files', {}))
        _atomic_json(directory / 'staged.json', sorted(inventory | set(records)))
        _atomic_json(view / '.cocoindex_code/settings.yml', {'include_patterns': ['**/*'],
            'exclude_patterns': ['.cocoindex_code/**', '.serena/**', '.clangd', 'compile_commands.json']})
        for key, record in records.items():
            destination = view / key
            if destination.resolve() != destination.absolute():
                raise ConfigError('Engine view contains a filesystem link')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = Path(record['absolute_path'])
            if source.resolve() != source or source.is_symlink():
                raise ConfigError('Selected source became a filesystem link')
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != record['sha256']:
                raise ConfigError('Source changed during staging; retry synchronization')
            blob = self.blob(record['sha256'])
            if not blob.exists():
                atomic_bytes(blob, content)
            if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != record['sha256']:
                atomic_bytes(destination, content)
        # Only remove files named by our own previous manifest; engine databases stay owned by the engine.
        for key in inventory - set(records):
            _relative(key)
            target = view / key
            if target.resolve() != target.absolute():
                raise ConfigError('Engine view deletion crosses a filesystem link')
            target.unlink(missing_ok=True)
        indexed = self.process.request('index', project=str(view))
        if set(indexed.get('files', [])) != set(records):
            raise ConfigError('Native file selection differs from the exact BTI source selection')
        native_files = indexed.get('index', {}).get('total_files')
        # CocoIndex counts distinct paths with chunks. Empty package markers still
        # belong to the selected file inventory but cannot produce a search hit.
        empty_sha = hashlib.sha256(b'').hexdigest()
        empty_paths = [key for key, record in records.items() if record['sha256'] == empty_sha]
        expected = len(records) - len(empty_paths)
        if native_files is not None and native_files != expected:
            raise ConfigError(f'Native file coverage differs: selected={len(records)}, empty={len(empty_paths)}, indexed={native_files}')
        verify()
        manifest = {'generation': generation, 'engine': self.fingerprint, 'project': project.name,
                    'root': str(root), 'config_hash': project.config_hash, 'files': records, 'empty_files': empty_paths,
                    'index': indexed.get('index', {})}
        _atomic_json(directory / 'generations' / (generation + '.json'), manifest)
        _atomic_json(directory / 'current.json', manifest)
        _atomic_json(directory / 'staged.json', sorted(records))
        (directory / 'pending.json').unlink(missing_ok=True)
        return manifest

    def available(self, root):
        directory = self.project_directory(root)
        return (self.manifest(root) is not None and not (directory / 'pending.json').exists()
                and all((directory / 'view/.cocoindex_code' / name).exists()
                        for name in ('cocoindex.db', 'target_sqlite.db')))

    def search(self, root, query, limit, paths):
        if not self.available(root):
            raise ConfigError('Native index needs reconciliation before search')
        return self.process.request('search', project=str(self.project_directory(root) / 'view'),
                                    query=query, limit=limit, paths=paths)

    def close(self):
        self.process.close()
        # Also recover the private daemon if its bridge died or timed out.
        if (self.runtime / 'daemon.pid').exists():
            with self.worker_settings.with_suffix('.log').open('ab') as log:
                subprocess.run([self.settings['python_executable'], str(Path(__file__).with_name('coco_engine_worker.py')),
                    '--config', str(self.worker_settings), '--stop'], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
