"""Stock CocoIndex Code client bridge. One private daemon, many project indexes."""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding='utf-8'))
    runtime = Path(settings['runtime'])
    for key in list(os.environ):
        if key.startswith('COCOINDEX_CODE_'):
            del os.environ[key]
    os.environ.update(COCOINDEX_CODE_DIR=str(runtime), COCOINDEX_CODE_RUNTIME_DIR=str(runtime),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with contextlib.redirect_stdout(sys.stderr):
        import msgspec
        from cocoindex_code import client
        if importlib.metadata.version('cocoindex-code') != '0.2.42':
            raise ValueError('This adapter requires cocoindex-code 0.2.42')
    if args.stop:
        with contextlib.redirect_stdout(sys.stderr):
            client.stop()
        return
    stop_daemon = False
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request['action'] == 'shutdown':
                    stop_daemon = True
                    break
                with contextlib.redirect_stdout(sys.stderr):
                    if request['action'] == 'index':
                        value = client.index(request['project'])
                        result = msgspec.to_builtins(value)
                        status = msgspec.to_builtins(client.project_status(request['project']))
                        if (status.get('progress') or {}).get('num_errors', 0):
                            raise ValueError('CocoIndex reported indexing errors')
                        result['index'] = status
                        from cocoindex_code.settings import load_project_settings
                        from cocoindex_code.file_walk import build_matcher, iter_included_files
                        root = Path(request['project'])
                        selected = load_project_settings(root)
                        matcher = build_matcher(root, selected.include_patterns, selected.exclude_patterns,
                                                selected.max_file_size)
                        result['files'] = sorted(relative.as_posix() for _, relative in iter_included_files(root, root, matcher))
                    elif request['action'] == 'search':
                        result = msgspec.to_builtins(client.search(request['project'], request['query'],
                            limit=request['limit'], paths=request.get('paths')))
                    else:
                        raise ValueError('Unsupported CocoIndex action')
                if result.get('success') is False:
                    raise ValueError(result.get('message') or 'CocoIndex operation failed')
                response = {'ok': True, 'result': result}
            except Exception as exc:
                response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            print('BTI_ENGINE_JSON:' + json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        if stop_daemon:
            with contextlib.redirect_stdout(sys.stderr):
                client.stop()


if __name__ == '__main__':
    main()
