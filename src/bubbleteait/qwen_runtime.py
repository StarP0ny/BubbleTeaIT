"""Standard-library client for one authenticated local Qwen worker per data root."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from .qwen_encoder import profile_hash


class QwenRuntimeError(ValueError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise QwenRuntimeError('Private worker RPC redirects are forbidden')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.qwen-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def file_lock(path, timeout=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise QwenRuntimeError('busy: local worker lock is held') from exc
                time.sleep(.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def runtime_identity(data_dir, executable, model_path, models_dir):
    return {'data_dir': os.path.normcase(str(Path(data_dir).resolve())),
        'python_executable': os.path.normcase(str(Path(executable).resolve())),
        'model_path': os.path.normcase(str(Path(model_path).resolve())),
        'models_dir': os.path.normcase(str(Path(models_dir).resolve())), 'profile_hash': profile_hash()}


class QwenRuntimeClient:
    def __init__(self, data_dir, python_executable, model_path, *, models_dir, timeout_seconds=600):
        executable = shutil.which(str(python_executable))
        if not executable:
            raise QwenRuntimeError('Qwen Python executable was not found')
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (float, int)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise QwenRuntimeError('timeout_seconds must be positive and finite')
        self.data_dir, self.model_path, self.models_dir = map(lambda p: Path(p).resolve(), (data_dir, model_path, models_dir))
        if not self.model_path.is_relative_to(self.models_dir):
            raise QwenRuntimeError('Qwen model must be inside models_dir')
        self.python_executable, self.timeout_seconds = Path(executable).resolve(), timeout_seconds
        self.identity = runtime_identity(self.data_dir, self.python_executable, self.model_path, self.models_dir)
        self.runtime_dir = self.data_dir / 'runtime' / 'qwen'
        self.state_path = self.runtime_dir / 'state.json'

    def _state(self):
        try:
            state = json.loads(self.state_path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        if state.get('identity') != self.identity:
            raise QwenRuntimeError('Worker identity/profile differs; use its original configuration or explicitly stop it')
        if type(state.get('port')) is not int or not 1 <= state['port'] <= 65535 or not isinstance(state.get('token'), str) or len(state['token']) != 64:
            raise QwenRuntimeError('Invalid worker state')
        return state

    def _call(self, state, action, payload):
        request = urllib.request.Request('http://127.0.0.1:%d/rpc' % state['port'],
            data=json.dumps({'action': action, 'identity': self.identity, **payload}, allow_nan=False).encode(),
            headers={'Authorization': 'Bearer ' + state['token'], 'Content-Type': 'application/json'}, method='POST')
        # Never use environment proxy settings for the private loopback endpoint.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=min(5, self.timeout_seconds) if action == 'health' else self.timeout_seconds) as response:
                value = json.loads(response.read(4 * 1024 * 1024 + 1))
        except urllib.error.HTTPError as exc:
            raise QwenRuntimeError('Worker RPC rejected request: HTTP%d' % exc.code) from exc
        if not isinstance(value, dict) or value.get('identity') != self.identity:
            raise QwenRuntimeError('Worker handshake identity/profile differs')
        if value.get('ok') is not True:
            raise QwenRuntimeError(value.get('error', 'Worker request failed'))
        return value

    def _probe(self, state):
        if state is None:
            return None
        try:
            return self._call(state, 'health', {})
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return None

    def _start(self):
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self.runtime_dir / 'startup.lock', min(self.timeout_seconds, 30)):
            state = self._state()
            if self._probe(state):
                return state
            # A timed-out but live worker is never replaced or stopped.
            with file_lock(self.runtime_dir / 'worker.lock'):
                config = {'identity': self.identity, 'token': secrets.token_hex(32)}
                launch = self.runtime_dir / 'launch.json'
                atomic_json(launch, config)
                environment = os.environ | {'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
                source = str(Path(__file__).resolve().parents[1])
                environment['PYTHONPATH'] = source + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
                process = self._spawn(launch, environment)
            deadline = time.monotonic() + min(self.timeout_seconds, 30)
            while time.monotonic() < deadline:
                state = self._state()
                if state and state['token'] == config['token'] and self._probe(state):
                    return state
                if process is not None and process.poll() is not None:
                    raise QwenRuntimeError('Worker startup failed; inspect runtime/qwen/worker.log')
                time.sleep(.05)
            raise QwenRuntimeError('Worker startup timed out; inspect runtime/qwen/worker.log')

    def _spawn(self, launch, environment):
        if os.name == 'nt':
            from .qwen_windows import spawn_worker
            launcher = Path(__file__).resolve().parents[2] / 'tools' / 'qwen_worker.py'
            try:
                spawn_worker(self.python_executable, launcher, launch, self.data_dir)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise QwenRuntimeError(str(exc)) from exc
            return None  # Readiness/identity timeout bounds the WMI-owned process.
        with (self.runtime_dir / 'worker.log').open('ab') as log:
            return subprocess.Popen([str(self.python_executable), '-m', 'bubbleteait.qwen_worker', '--config', str(launch)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=self.data_dir, env=environment, start_new_session=True)

    def request(self, action, **payload):
        if action not in ('health', 'load', 'unload', 'shutdown', 'index', 'search'):
            raise QwenRuntimeError('Unsupported Qwen action')
        state = self._state()
        health = self._probe(state)
        if not health and action in ('health', 'unload', 'shutdown'):
            return {'ok': True, 'worker_running': False, 'resident': False, 'ready': False, 'identity': self.identity}
        if action == 'health' and health:
            return health
        return self._call(state if health else self._start(), action, payload)
