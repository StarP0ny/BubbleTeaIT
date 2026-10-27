"""Request manifest caching must never hide corruption outside selected files."""
import unittest
from unittest.mock import patch

import test_direct as fixtures
from bubbleteait.registry import RegistryError


class DirectValidationTests(unittest.TestCase):
    def test_navigation_and_citable_reads_need_no_embedding_backend(self):
        fixture = fixtures.DirectTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        unit = fixture.unit('plain-source', {'src/control.c': 'int threshold = 37;\r\n'})
        with patch('bubbleteait.direct.make_backend', side_effect=AssertionError('Unexpected embedding initialization')):
            direct = fixture.direct(unit)
            sources = direct.list_sources()
            files = direct.list_files(source_ids=[sources['entries'][0]['source_id']])
            self.assertEqual([entry['path'] for entry in files['entries']], ['src/control.c'])
            result = direct.read_many([files['entries'][0]['read']], include_catalog=False)
            excerpt = result['excerpts'][0]
            self.assertEqual(excerpt['text'], 'int threshold = 37;\r\n')
            canonical = direct.resolve([excerpt['id']])[0]
            self.assertEqual((canonical['unit_id'], canonical['file_path']), (unit['unit_id'], 'src/control.c'))

    def test_unselected_unit_corruption_rolls_back_publication_and_is_not_cached(self):
        fixture = fixtures.DirectTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        selected = fixture.unit('selected', {'main.c': 'int first;\nint second;\n'})
        unselected = fixture.unit('unselected', {'other.c': 'int unrelated;\n'})
        direct = fixture.direct(selected, unselected)
        request = {'source_id': 'selected', 'path': 'main.c', 'start': 1, 'end': 1}
        issued = direct.read_many([request])['excerpts'][0]['id']
        ledger_before = dict(direct._ledger)
        snapshot = fixture.registry.data_dir / 'sources' / unselected['unit_id'] / 'files' / 'other.c'
        original = snapshot.read_bytes()
        pack = direct._pack

        def corrupt_after_packing(*args, **kwargs):
            result = pack(*args, **kwargs)
            snapshot.write_bytes(b'int corrupted;\n')
            return result

        # The unselected unit changes after source reads and ledger staging.
        # Exit validation must reject the entire response and undo new IDs.
        with patch.object(direct, '_pack', side_effect=corrupt_after_packing):
            with self.assertRaisesRegex(RegistryError, 'Snapshot bytes do not match'):
                direct.read_many([{**request, 'start': 2, 'end': 2}])
        self.assertEqual(direct._ledger, ledger_before)
        self.assertEqual(direct.service._depth, 0)
        self.assertIsNone(direct.service.registry.manifests)
        self.assertFalse(direct.last_profile['succeeded'])

        # A new request must inspect the whole composition again, even though
        # its issued evidence refers exclusively to the unchanged selected unit.
        with self.assertRaisesRegex(RegistryError, 'Snapshot bytes do not match'):
            direct.resolve([issued])
        self.assertEqual(direct._ledger, ledger_before)
        snapshot.write_bytes(original)
        self.assertEqual(direct.resolve([issued])[0]['unit_id'], selected['unit_id'])
        self.assertIsNone(direct.service.registry.manifests)


if __name__ == '__main__':
    unittest.main()
