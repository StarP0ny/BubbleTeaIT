"""Wire-level tests of the MCP facade; the index status is a text fixture."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bubbleteait.registry import IndexRecipe, Registry, SourceSpec


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'Install the mcp extra')
class MCPStdioTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_external_client_search_read_status_and_reject_traversal(self):
        project = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('bti_mcp_probe', project / 'tools/probe_mcp.py')
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'fixture'
            source.mkdir()
            (root / '.local').mkdir()
            # Read/status/text must remain usable if the optional ML runtime
            # is unavailable; constructing it during initialize would fail.
            (root / '.local/config.toml').write_text(
                '[backends.cocoindex]\npython_executable="missing-bti-runtime-python-12345"\n', encoding='utf-8')
            (source / 'control.c').write_text('int heatingAllowed(int trip) {\n    return trip == 0;\n}\n', encoding='utf-8')
            registry = Registry(root / 'data')
            registry.register(SourceSpec('fixture', 'module', source, include=('**/*.c',)))
            unit = registry.snapshot('fixture', IndexRecipe('text-fixture', 'none', 'none', 'lines'))
            status = root / 'data/indexes' / unit['unit_id'] / 'status.json'
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({'schema_version': 1, 'unit_id': unit['unit_id'],
                'recipe_hash': unit['recipe_hash'], 'backend': 'text-fixture', 'status': 'ready'}), encoding='utf-8')
            registry.compose('fixture-project', [unit['unit_id']], require_ready=True)
            with patch.dict(os.environ, {'BTI_DATA_DIR': str(root/'data'), 'BTI_MODELS_DIR': str(root/'models')}):
                result = await probe.probe(root, 'fixture-project', 'heatingAllowed', 'text', root/'stderr.log', timeout=20)
            self.assertEqual(result['status'], 'passed')
            self.assertTrue(result['traversal_rejected'])
            self.assertEqual(result['tools'], ['code_symbols', 'context_search', 'index_status', 'list_files', 'list_sources', 'project_info', 'read_source', 'read_sources',
                                               'resolve_sources', 'search', 'sync_project', 'sync_status', 'watch_status'])
            found = result['search']['structuredContent']['results']
            self.assertEqual(found[0]['unit_id'], unit['unit_id'])
            self.assertIn('heatingAllowed', found[0]['content'])


if __name__ == '__main__':
    unittest.main()
