"""Windows detached entry point; local source path, no shell interpolation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from bubbleteait.service_worker import main

if __name__ == '__main__':
    # WMI does not inherit the caller's output pipes.
    import json
    launch = Path(sys.argv[sys.argv.index('--config') + 1])
    data = json.loads(launch.read_text(encoding='utf-8'))
    log = Path(data['config']['data_dir']) / 'runtime/service/service.log'
    with log.open('a', encoding='utf-8', buffering=1) as stream:
        sys.stdout = sys.stderr = stream
        main()
