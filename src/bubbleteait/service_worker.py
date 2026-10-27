"""Private loopback RPC facade for the BTI coordinator."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

from .config import ConfigError
from .qwen_runtime import file_lock
from .registry import _atomic_json
from .service import Coordinator, config_from_payload, config_payload


def serve(launch):
    data = json.loads(Path(launch).read_text(encoding='utf-8'))
    config = config_from_payload(data['config'])
    coordinator = Coordinator(config)
    directory = config.data_dir / 'runtime/service'
    with file_lock(directory / 'service.lock'):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != '/rpc' or not hmac.compare_digest(
                        self.headers.get('Authorization', ''), 'Bearer ' + data['token']):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 1024 * 1024:
                        raise ConfigError('Invalid request length')
                    request = json.loads(self.rfile.read(length))
                    action = request.get('action')
                    if action == 'status':
                        result = coordinator.status()
                    elif action == 'config':
                        result = {'settings': config_payload(coordinator.config),
                                  'sources': list(map(str, coordinator.config.loaded_files)),
                                  'error': coordinator.config_error}
                    elif action == 'workspace_projects':
                        result = coordinator.workspace_projects(request['workspace'])
                    elif action == 'register':
                        result = coordinator.register(request['path'], manual=True)
                    elif action == 'unwatch':
                        result = coordinator.unwatch(request['path'])
                    elif action == 'select':
                        if request.get('manual_sync'):
                            with coordinator.cv:
                                coordinator._entry(request['path'])['manual_sync'] = True
                        result = coordinator.select(request['path'], lease=request['lease'],
                            freshness=request.get('freshness', 'available'), timeout_ms=request.get('timeout_ms', 30000),
                            manual_sync=request.get('manual_sync', False))
                    elif action == 'release':
                        coordinator.release(request['lease'])
                        result = {'state': 'released'}
                    elif action == 'heartbeat':
                        result = {'state': 'renewed', 'found': coordinator.heartbeat(request['lease'])}
                    elif action == 'stop':
                        coordinator.stop.set()
                        threading.Thread(target=server.shutdown, daemon=True).start()
                        result = {'state': 'stopping'}
                    else:
                        raise ConfigError('Unsupported service action')
                    response = {'ok': True, 'result': result}
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    response = {'ok': False, 'error': str(exc)}
                encoded = json.dumps(response, default=str, ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        state_file = directory / 'state.json'
        _atomic_json(state_file, {'port': server.server_port, 'token': data['token'],
                                 'pid': os.getpid(), 'data_dir': str(config.data_dir.resolve())})
        os.chmod(state_file, 0o600)
        coordinator.start()
        try:
            server.serve_forever(poll_interval=.25)
        finally:
            coordinator.stop.set()
            with coordinator.cv:
                coordinator.cv.notify_all()
            # Hold the writer identity until an in-flight immutable build has finished.
            coordinator.thread.join()
            server.server_close()
            if state_file.exists() and json.loads(state_file.read_text())['token'] == data['token']:
                state_file.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    serve(args.config)


if __name__ == '__main__':
    main()
