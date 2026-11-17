"""Serialized JSON bridge to an isolated, persistent optional-engine runtime."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

from .config import ConfigError


class EngineProcess:
    def __init__(self, python, worker, settings, timeout=600):
        self.python, self.worker = str(python), Path(worker)
        self.settings, self.timeout = Path(settings), timeout
        self.process = None
        self.lock = threading.RLock()
        self.responses = None
        self.log = None

    def _start(self):
        self.settings.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.settings.with_suffix('.log').open('ab')
        env = os.environ | {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
        # Detached Windows launchers do not inherit the initiating shell's PYTHONPATH.
        # Propagate only an editable source tree, never another venv's site-packages.
        source = Path(__file__).resolve().parents[1]
        if (source.parent / 'pyproject.toml').is_file():
            env['PYTHONPATH'] = str(source) + os.pathsep + env.get('PYTHONPATH', '')
        self.process = subprocess.Popen([self.python, str(self.worker), '--config', str(self.settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf-8',
            bufsize=1, env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        responses = self.responses = queue.Queue()
        stream, log = self.process.stdout, self.log
        def receive():
            try:
                while True:
                    line = stream.readline(8 * 1024 * 1024 + 1)
                    if not line:
                        break
                    if len(line) > 8 * 1024 * 1024:
                        responses.put(ConfigError('Engine response exceeds 8 MiB'))
                        break
                    if not line.startswith('BTI_ENGINE_JSON:'):
                        # Native extensions and subprocesses can bypass Python's
                        # redirect_stdout. Keep their diagnostics out of the protocol.
                        if not log.closed:
                            log.write(line.encode('utf-8', errors='replace'))
                            log.flush()
                        continue
                    responses.put(json.loads(line[len('BTI_ENGINE_JSON:'):]))
            except Exception as exc:
                responses.put(exc)
            finally:
                responses.put(ConfigError('Engine process exited; inspect ' + str(self.settings.with_suffix('.log'))))
        threading.Thread(target=receive, daemon=True, name='bti-engine-output').start()

    def request(self, action, **arguments):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._dispose()
                self._start()
            try:
                self.process.stdin.write(json.dumps({'action': action, **arguments}, ensure_ascii=False, allow_nan=False) + '\n')
                self.process.stdin.flush()
                response = self.responses.get(timeout=self.timeout)
                if isinstance(response, Exception):
                    self._dispose()
                    raise ConfigError('Engine transport failed; retry after reconciliation: ' + str(response))
                if not response.get('ok'):
                    raise ConfigError(response.get('error', 'Engine request failed'))
                return response['result']
            except queue.Empty as exc:
                self._dispose()
                raise ConfigError('Engine request timed out; index must be reconciled before reuse') from exc
            except (BrokenPipeError, OSError) as exc:
                self._dispose()
                raise ConfigError('Engine connection failed; inspect engine log') from exc

    def _dispose(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

    def close(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.stdin.write('{"action":"shutdown"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._dispose()
