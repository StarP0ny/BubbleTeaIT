"""Read-only search over one pinned, ready composition."""
from __future__ import annotations

import copy

from .registry import _digest
from .search import SearchError, SearchService


class PinnedSearchService(SearchService):
    def __init__(self, data_dir, composition, backend=None):
        super().__init__(data_dir, backend)
        self.pinned = copy.deepcopy(composition)
        self.load_composition(self.pinned['composition_id'])

    def load_composition(self, composition_id):
        c = self.pinned
        if composition_id != c['composition_id'] or c.get('ready') is not True or c.get('schema_version') != 1:
            raise SearchError('Pinned search requires a pinned ready composition')
        if c.get('configuration_hash') != _digest(c['configuration']):
            raise SearchError('Pinned composition configuration fingerprint differs')
        refs = c['units']
        fingerprint = _digest({'units': [{'unit_id': r['unit_id'], 'recipe_hash': r['recipe_hash']} for r in refs],
                               'configuration_hash': c['configuration_hash']})
        if c.get('composition_hash') != fingerprint or not refs:
            raise SearchError('Pinned composition fingerprint differs')
        seen = set()
        for r in refs:
            u = self.registry.load_unit(r['unit_id'])
            if u['source_id'] in seen or r['source_id'] != u['source_id'] or r['recipe_hash'] != u['recipe_hash']:
                raise SearchError('Pinned unit identity differs')
            seen.add(u['source_id'])
            if r['index_status'] != 'ready' or self.registry.index_status(r['unit_id'])['status'] != 'ready':
                raise SearchError('Pinned index is no longer ready')
        return copy.deepcopy(c)
