"""Persistent runtime lifecycle/auth/busy tests, with no ML or GPU dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from bubbleteait.qwen_runtime import QwenRuntimeClient, QwenRuntimeError, atomic_json, file_lock
from bubbleteait.qwen_worker import serve


class FakeEncoder:
    def __init__(self, *args):
        self.resident, self.load_count = False, 0

    def health(self):
        return {'resident': self.resident, 'ready': self.resident, 'load_count': self.load_count, 'memory': {}, 'error': None}

    def load(self):
        if not self.resident:
            self.load_count += 1
            self.resident = True
        return self.health()

    def unload(self):
        self.resident = False
        return self.health()


class FakeStore:
    entered, release = threading.Event(), threading.Event()

    def __init__(self, data_dir, encoder):
        self.encoder = encoder

    def index(self, unit_id):
        self.encoder.load()
        self.entered.set()
        if unit_id == 'slow':
            self.release.wait(5)
        return {'unit_id': unit_id, 'status': 'ready'}

    def search(self, unit_id, query, **kwargs):
        return [{'unit_id': unit_id, 'query': query}]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.threads, self.starts = [], []
        FakeStore.entered.clear()
        FakeStore.release.clear()
        self.client = self.new_client()
        self.mock = patch.object(QwenRuntimeClient, '_spawn', side_effect=self.launch)
        self.mock.start()
        self.addCleanup(self.cleanup)

    def new_client(self):
        return QwenRuntimeClient(self.root / 'data', sys.executable, self.root / 'models/qwen', models_dir=self.root / 'models', timeout_seconds=3)

    def launch(self, launch, environment):
        config = json.loads(launch.read_text(encoding='utf-8'))
        self.starts.append((launch, {'env': environment}))
        thread = threading.Thread(target=serve, args=(config, FakeEncoder, FakeStore), daemon=True)
        thread.start()
        self.threads.append(thread)
        class Process:
            def poll(self):
                return None if thread.is_alive() else 1
        return Process()

    def cleanup(self):
        FakeStore.release.set()
        try:
            self.client.request('shutdown')
        finally:
            for thread in self.threads:
                thread.join(3)
            self.mock.stop()
            self.temp.cleanup()

    def wait_stopped(self):
        for thread in self.threads:
            thread.join(3)
        self.assertFalse(self.client.state_path.exists())

    def test_health_is_lazy_and_two_clients_share_one_resident_worker(self):
        self.assertFalse(self.client.request('health')['worker_running'])
        self.assertFalse(self.client.runtime_dir.exists())
        self.client.request('load')
        other = self.new_client()
        other.request('load')
        self.assertEqual(other.request('health')['load_count'], 1)
        self.assertEqual(len(self.starts), 1)
        environment = self.starts[0][1]['env']
        self.assertIn(str(Path(__file__).resolve().parents[1] / 'src'), environment['PYTHONPATH'])
        self.client.request('unload')
        self.assertFalse(other.request('health')['resident'])
        other.request('load')
        self.assertEqual(self.client.request('health')['load_count'], 2)

    def test_busy_rejects_second_compute_but_health_remains_available(self):
        self.client.request('load')
        errors = []
        def indexing():
            try:
                self.client.request('index', unit_id='slow')
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=indexing)
        thread.start()
        self.assertTrue(FakeStore.entered.wait(2))
        health = self.new_client().request('health')
        self.assertTrue(health['busy'])
        self.assertFalse(health['ready'])
        with self.assertRaisesRegex(QwenRuntimeError, 'busy'):
            self.client.request('search', unit_id='unit', query='question')
        FakeStore.release.set()
        thread.join(3)
        self.assertEqual(errors, [])

    def test_shutdown_restarts_cleanly_without_stopping_foreign_processes(self):
        self.client.request('load')
        self.client.request('shutdown')
        self.wait_stopped()
        self.assertFalse(self.client.request('health')['worker_running'])
        self.client.request('load')
        self.assertEqual(len(self.starts), 2)
        self.assertEqual(self.client.request('health')['load_count'], 1)

    def test_concurrent_startup_is_atomic_and_reuses_one_worker(self):
        barrier, states, errors = threading.Barrier(2), [], []
        def start():
            try:
                barrier.wait(2)
                states.append(self.new_client()._start())
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=start) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(4)
        self.assertEqual(errors, [])
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0]['token'], states[1]['token'])
        self.assertEqual(len(self.starts), 1)

    def test_stale_state_can_restart_but_held_worker_lock_cannot(self):
        self.client.request('load')
        old = self.client._state()
        self.client.request('shutdown')
        self.wait_stopped()
        atomic_json(self.client.state_path, old)  # Simulate crash leaving its state behind.
        with file_lock(self.client.runtime_dir / 'worker.lock'):
            with self.assertRaisesRegex(QwenRuntimeError, 'busy'):
                self.client.request('load')
        self.assertEqual(len(self.starts), 1)
        self.client.request('load')
        self.assertEqual(len(self.starts), 2)
        self.assertNotEqual(old['token'], self.client._state()['token'])

    def test_auth_and_wrong_profile_are_rejected_without_worker_replacement(self):
        self.client.request('load')
        state = self.client._state()
        with self.assertRaisesRegex(QwenRuntimeError, 'HTTP403'):
            self.client._call({**state, 'token': '0' * 64}, 'health', {})
        wrong = {**state, 'identity': {**state['identity'], 'profile_hash': 'wrong'}}
        atomic_json(self.client.state_path, wrong)
        try:
            with self.assertRaisesRegex(QwenRuntimeError, 'identity/profile'):
                self.new_client().request('load')
        finally:
            atomic_json(self.client.state_path, state)
        self.assertEqual(len(self.starts), 1)
        self.assertTrue(self.client.request('health')['resident'])

    def test_rpc_has_no_arbitrary_paths_commands_or_payload_fields(self):
        self.client.request('load')
        for action, payload in [('exec', {}), ('index', {'unit_id': '../outside'}),
                ('load', {'model_path': '/outside'}), ('search', {'unit_id': 'x', 'query': 'q', 'paths': ['../outside']})]:
            with self.subTest(action=action):
                with self.assertRaises(QwenRuntimeError):
                    self.client.request(action, **payload)

    def test_client_import_has_no_torch_numpy_or_sentence_transformers(self):
        # Actual fresh interpreter, unlike this process where other tests may import numpy.
        self.mock.stop()
        try:
            code = "import sys; import bubbleteait.qwen_runtime; assert not any(n in sys.modules for n in ('torch','numpy','sentence_transformers'))"
            result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            self.mock.start()


@unittest.skipUnless(sys.platform == 'win32', 'Windows MCP Job Object regression')
class WindowsJobLifecycleTests(unittest.TestCase):
    def test_lazy_worker_survives_mcp_stdio_parent_job_close_without_loading_ml(self):
        import ctypes
        from ctypes import wintypes
        try:
            import anyio
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.os.win32 import utilities as windows_mcp
        except ImportError:
            self.skipTest('MCP SDK is required for the Windows lifecycle regression')
        if not windows_mcp.win32job:
            self.skipTest('MCP SDK Windows Job Object support is required for this regression')
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handles = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, models = root / 'data', root / 'models'
            client = QwenRuntimeClient(data, sys.executable, models / 'qwen', models_dir=models, timeout_seconds=30)
            source = str(Path(__file__).resolve().parents[1] / 'src')
            # The short-lived MCP facade launches the real stdlib worker lazily;
            # no encoder.load or model files are required by this regression.
            code = ('import sys,json; from bubbleteait.qwen_runtime import QwenRuntimeClient; '
                f'c=QwenRuntimeClient({str(data)!r},sys.executable,{str(models / "qwen")!r},models_dir={str(models)!r}); '
                'c._start(); print(json.dumps({"jsonrpc":"2.0","method":"bti/worker-ready"}),flush=True); sys.stdin.read()')
            parameters = StdioServerParameters(command=sys.executable, args=['-X', 'utf8', '-c', code],
                env=os.environ | {'PYTHONPATH': source}, cwd=source)
            async def exercise():
                async with stdio_client(parameters) as (read, _):
                    self.assertTrue(windows_mcp._process_jobs, 'SDK must own a real kill-on-close Job')
                    with anyio.fail_after(40):
                        message = await read.receive()
                    self.assertNotIsInstance(message, Exception)
                    before = client.request('health')
                    self.assertTrue(before['worker_running'])
                    self.assertFalse(before['resident'])
                    handle = kernel.OpenProcess(0x100000, False, before['pid'])  # SYNCHRONIZE only, never terminate.
                    self.assertTrue(handle)
                    handles.append(handle)
                # The SDK has now closed its KILL_ON_JOB_CLOSE handle. A direct
                # CREATE_NO_WINDOW child dies here; a WMI-owned worker survives.
                after = client.request('health')
                self.assertTrue(after['worker_running'])
                self.assertEqual(after['pid'], before['pid'])
                self.assertEqual(after['load_count'], 0)
            try:
                anyio.run(exercise)
            finally:
                client.request('shutdown')
                deadline = time.monotonic() + 5
                while client.state_path.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                for handle in handles:
                    try:
                        self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
                    finally:
                        kernel.CloseHandle(handle)


if __name__ == '__main__':
    unittest.main()
