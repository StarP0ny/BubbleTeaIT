"""Fixed standalone entrypoint used by the detached Windows WMI launcher."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    # WMI starts from the user's environment, independent of MCP pipes/PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    original = sys.stdout, sys.stderr
    try:
        with (args.config.parent / 'worker.log').open('a', encoding='utf-8', buffering=1) as log:
            sys.stdout = sys.stderr = log
            try:
                from bubbleteait.qwen_worker import serve
                import json
                serve(json.loads(args.config.read_text(encoding='utf-8')))
            except Exception:
                import traceback
                traceback.print_exc(file=log)
                raise SystemExit(1)
    finally:
        sys.stdout, sys.stderr = original


if __name__ == '__main__':
    main()
