"""Authenticated loopback server; owns the encoder and serializes computation."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading

from .qwen_encoder import QwenEncoder
from .qwen_runtime import atomic_json, file_lock, runtime_identity


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, config, encoder_factory=QwenEncoder, store_factory=None):
        self.identity, self.token = config['identity'], config['token']
        expected = runtime_identity(self.identity['data_dir'], sys.executable,
            self.identity['model_path'], self.identity['models_dir'])
        if self.identity != expected:
            raise ValueError('Worker executable/profile identity differs')
        self.encoder = encoder_factory(self.identity['model_path'], self.identity['models_dir'])
        self.compute = threading.Lock()
        self.connections = threading.BoundedSemaphore(16)
        self.store, self.store_factory = None, store_factory
        self.last_error = None
        super().__init__(('127.0.0.1', 0), Handler)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def dispatch(self, action, payload):
        allowed = {'health': {'load'}, 'load': set(), 'unload': set(), 'shutdown': set(),
                   'index': {'unit_id'}, 'search': {'unit_id', 'query', 'limit', 'paths'}}
        if action not in allowed or set(payload) - allowed[action]:
            raise ValueError('Unsupported action or payload fields')
        if action == 'health':
            if payload.get('load', False) is not False:
                raise ValueError('health cannot load the encoder; use load explicitly')
            state = self.encoder.health()
            return {**state, 'ready': state.get('ready', False) and not self.compute.locked(),
                'busy': self.compute.locked(), 'worker_running': True, 'pid': os.getpid(),
                'error': self.last_error or state.get('error')}
        if not self.compute.acquire(blocking=False):
            raise ValueError('busy: worker is computing; retry after the current operation')
        try:
            if action in ('load', 'unload'):
                value = getattr(self.encoder, action)()
            elif action == 'shutdown':
                self.encoder.unload()
                threading.Thread(target=self.shutdown, daemon=True).start()
                value = {'shutdown': True}
            else:
                unit = payload.get('unit_id')
                if not isinstance(unit, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', unit):
                    raise ValueError('Invalid unit_id')
                if action == 'search':
                    if not isinstance(payload.get('query'), str) or not payload['query'].strip() or len(payload['query']) > 32000:
                        raise ValueError('Search requires a bounded nonempty query')
                    if type(payload.get('limit', 10)) is not int or not 1 <= payload.get('limit', 10) <= 100:
                        raise ValueError('Search limit must be from 1 to 100')
                    paths = payload.get('paths')
                    if paths is not None and (not isinstance(paths, list) or len(paths) > 1000 or any(not isinstance(p, str)
                            or not p or len(p) > 1024 or ':' in p or '\\' in p or p.startswith('/') or '..' in p.split('/') for p in paths)):
                        raise ValueError('Search paths must stay within the unit')
                if self.store is None:
                    if self.store_factory is None:
                        from .backends.qwen import QwenIndexStore
                        self.store_factory = QwenIndexStore
                    self.store = self.store_factory(Path(self.identity['data_dir']), self.encoder)
                value = {'index': self.store.index(unit)} if action == 'index' else {'results': self.store.search(
                    unit, payload['query'], limit=payload.get('limit', 10), paths=payload.get('paths'))}
            self.last_error = None
            return value
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            self.compute.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def do_POST(self):
        if self.path != '/rpc' or not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + self.server.token):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 65536:
                raise ValueError('Request body exceeds the local RPC limit')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or request.pop('identity', None) != self.server.identity:
                raise ValueError('Worker handshake identity/profile differs')
            action = request.pop('action', None)
            result = {'ok': True, **self.server.dispatch(action, request)}
        except Exception as exc:
            result = {'ok': False, 'error': str(exc)}
        data = json.dumps({**result, 'identity': self.server.identity}, ensure_ascii=False, allow_nan=False).encode('utf-8')
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (OSError, TimeoutError):
            pass  # Client timeout never kills a running index or another process.


def serve(config, encoder_factory=QwenEncoder, store_factory=None):
    directory = Path(config['identity']['data_dir']) / 'runtime' / 'qwen'
    state_path = directory / 'state.json'
    with file_lock(directory / 'worker.lock', timeout=30):
        server = WorkerServer(config, encoder_factory, store_factory)
        atomic_json(state_path, {'identity': server.identity, 'token': server.token,
                                'port': server.server_port, 'pid': os.getpid()})
        try:
            server.serve_forever(poll_interval=.1)
        finally:
            server.server_close()
            server.encoder.unload()
            try:
                if json.loads(state_path.read_text(encoding='utf-8')).get('token') == server.token:
                    state_path.unlink()
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    serve(json.loads(args.config.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
