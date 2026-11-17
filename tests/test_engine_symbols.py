"""Real build-command preservation and source-boundary checks around Serena."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import os
import subprocess

from test_engine_coordinator import EngineFixture
from bubbleteait.config import ConfigError
from bubbleteait.engine_projects import selected_project
from bubbleteait.engine_symbols import SymbolSession, command_arguments


class SymbolBridge:
    def __init__(self, *args):
        self.process = None
        self.value = []
        self.calls = []

    def request(self, action, **arguments):
        self.calls.append(arguments)
        return self.value

    def close(self):
        pass


class EngineSymbolTests(EngineFixture, unittest.TestCase):

    def session(self, server='clangd'):
        root = self.config.project_dir
        (root / 'config.h').write_text('#define THRESHOLD 37\n')
        self.config.projects['controller']['include'] = ['**/*.c', '**/*.h']
        database = self.root / 'build/compile_commands.json'
        database.parent.mkdir(exist_ok=True)
        database.write_text(json.dumps([{'directory': str(root), 'file': str(root/'main.c'),
            'arguments': [sys.executable, '-DHWVER_CONTROLLER', '-I', str(root), '-march=vendor',
                          '-DHEADER_PATH="' + (root/'config.h').as_posix() + '"', '-c', str(root/'main.c')]}]))
        settings = self.config.projects['controller']
        settings.update(compile_commands=database, remove_flags=['-march=*'], add_flags=['-march=rv32imac'])
        self.config.engines['serena'] = {'executable': sys.executable, 'language_server': sys.executable,
            'python_executable': sys.executable, 'server': server, 'timeout_seconds': 10, 'query_drivers': []}
        _, project, _ = selected_project(self.config)
        session = SymbolSession(self.config, root, project, settings, project.plan(self.config.data_dir),
                                self.coordinator.engine, 'fixture', process_factory=SymbolBridge)
        self.addCleanup(session.close)
        return session

    def test_real_compile_macros_and_include_layout_are_preserved(self):
        session = self.session()
        commands = json.loads((session.view/'compile_commands.json').read_text())
        argv = commands[0]['arguments']
        self.assertIn('-DHWVER_CONTROLLER', argv)
        self.assertNotIn('-march=vendor', argv)
        self.assertIn('-march=rv32imac', argv)
        self.assertIn(str(session.view), argv)
        self.assertTrue((session.view/'config.h').is_file())
        self.assertTrue(any((session.view/'config.h').as_posix() in a for a in argv))
        self.assertFalse((self.config.project_dir/'.serena').exists())
        self.assertFalse(session.profile['build_verified'])

    def test_ccls_is_selected_without_changing_build_commands_to_clang(self):
        session = self.session('ccls')
        settings = json.loads((session.view/'.serena/project.yml').read_text())
        self.assertEqual(settings['language_servers'], ['cpp_ccls'])
        commands = json.loads((session.view/'compile_commands.json').read_text())
        self.assertEqual(commands[0]['arguments'][0], sys.executable)

    def test_foreign_result_is_rejected_even_beyond_output_limit(self):
        session = self.session()
        session.process.value = [
            {'relative_path': 'main.c', 'name_path': 'threshold', 'body_location': {'start_line': 0, 'end_line': 0}},
            {'relative_path': '../reference/main.c', 'name_path': 'threshold'}]
        with self.assertRaisesRegex(ConfigError, 'outside'):
            session.query({'symbol_operation': 'definition', 'path': 'main.c', 'name': 'threshold', 'limit': 1})

    def test_symbol_coordinates_and_text_are_checked_against_original_bytes(self):
        session = self.session()
        session.process.value = [{'relative_path': 'main.c', 'name_path': 'threshold',
                                  'body_location': {'start_line': 0, 'end_line': 0}, 'body': 'int threshold = 999;'}]
        with self.assertRaisesRegex(ConfigError, 'does not match'):
            session.query({'symbol_operation': 'definition', 'path': 'main.c', 'name': 'threshold'})

    def test_file_reference_container_uses_verified_file_lines_without_fake_symbol_span(self):
        session = self.session()
        session.process.value = {'main.c': {'File': [{'name_path': 'main',
                                  'body_location': {'start_line': 0, 'end_line': 3}}]}}
        result = session.query({'symbol_operation': 'references', 'path': 'main.c', 'name': 'threshold'})
        self.assertEqual(result['results'][0]['location_kind'], 'file_context')
        self.assertEqual(result['results'][0]['lines'], [1, 1])

    def test_command_line_parsing_preserves_windows_toolchain_paths(self):
        if sys.platform != 'win32':
            self.skipTest('Windows command-line quoting')
        argv = command_arguments('"C:\\Tool Chain\\bin\\gcc.exe" -I"C:\\source tree\\include" -c "C:\\source tree\\main.c"')
        self.assertEqual(argv, ['C:\\Tool Chain\\bin\\gcc.exe', '-IC:\\source tree\\include', '-c', 'C:\\source tree\\main.c'])

    def test_language_server_launcher_preserves_stdio_and_exit_status(self):
        env = os.environ | {'BTI_LANGUAGE_SERVER': sys.executable,
            'BTI_LANGUAGE_SERVER_ARGS': json.dumps(['-u', '-c',
                'import sys;print("READY",flush=True);print("ECHO:"+sys.stdin.readline(),flush=True)'])}
        result = subprocess.run([sys.executable, '-c', 'from bubbleteait.language_server_launcher import main;main()'],
            input='request\n', capture_output=True, text=True, env=env, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'READY\nECHO:request\n\n')
