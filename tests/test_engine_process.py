"""Native diagnostics must not corrupt or permanently poison the JSON bridge."""
from pathlib import Path
import sys
import tempfile
import unittest

from bubbleteait.config import ConfigError
from bubbleteait.engine_process import EngineProcess


class EngineProcessTests(unittest.TestCase):
    def test_native_stdout_noise_is_logged_and_framed_response_is_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root/'worker.py'
            worker.write_text('''import sys,json
print("native diagnostic",flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['action']=='shutdown':break
 if r['action']=='bad':print('BTI_ENGINE_JSON:{invalid',flush=True)
 else:print('BTI_ENGINE_JSON:'+json.dumps({'ok':True,'result':{'echo':r['action']}}),flush=True)
''', encoding='utf-8')
            settings = root/'bridge.json'
            settings.write_text('{}')
            process = EngineProcess(sys.executable, worker, settings, 10)
            try:
                self.assertEqual(process.request('first'), {'echo': 'first'})
                with self.assertRaisesRegex(ConfigError, 'transport failed'):
                    process.request('bad')
                self.assertEqual(process.request('recovered'), {'echo': 'recovered'})
            finally:
                process.close()
            self.assertIn('native diagnostic', settings.with_suffix('.log').read_text())
