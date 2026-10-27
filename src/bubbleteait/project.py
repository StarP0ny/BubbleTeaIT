"""Declarative component trees; runtime settings never inherit from dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import tomllib

from .config import ConfigError, _boolean, _string, _strings, _table
from .registry import Registry, SourceSpec, _digest, _identifier, _relative


def discover_project(directory: Path | str) -> Path | None:
    """Find the nearest manifest, stopping at the current repository boundary."""
    start = Path(directory).resolve()
    for folder in (start, *start.parents):
        candidate = folder / 'bti.toml'
        if candidate.is_file():
            return candidate
        if (folder / '.git').exists():
            break
    return None


@dataclass(frozen=True)
class Project:
    name: str
    path: Path
    sources: tuple[SourceSpec, ...]
    components: tuple[dict, ...]
    config_hash: str
    context: dict

    def plan(self, data_dir) -> dict:
        registry = Registry(data_dir)
        selections = []
        owners = {}
        for spec in self.sources:
            selection = registry.preview(spec)
            selection['selection'] = asdict(spec) | {'root': str(spec.root)}
            for entry in selection['files']:
                path = (spec.root / entry['path']).resolve()
                if path in owners:
                    raise ConfigError(f'Overlapping sources: {owners[path]} and {spec.source_id}: {path}')
                owners[path] = spec.source_id
            selections.append(selection)
        return {'project': self.name, 'project_config': str(self.path), 'config_hash': self.config_hash,
                'components': list(self.components), 'sources': selections,
                'file_count': sum(item['file_count'] for item in selections),
                'ready': all(item['ready'] for item in selections)}


def _patterns(value, key):
    values = _strings(value, key)
    if not values:
        raise ConfigError(f'{key}: expected at least one pattern')
    for pattern in values:
        _relative(pattern)
    return values


def load_project(path: Path | str) -> Project:
    """Resolve explicit dependencies, variants and source-local narrowing rules."""
    path = Path(path).resolve()
    components, sources, active, seen, identities = [], [], [], {}, {}

    def visit(filename, variant=None, narrowing=None, parent=None):
        filename = filename.resolve()
        if filename in active:
            raise ConfigError('Component config cycle: ' + ' -> '.join(map(str, [*active, filename])))
        if len(active) >= 64:
            raise ConfigError('Component config nesting exceeds 64')
        selection = {'variant': variant, **(narrowing or {})}
        if filename in seen:
            if seen[filename] != selection:
                raise ConfigError(f'Conflicting selections of component: {filename}')
            return
        try:
            raw_bytes = filename.read_bytes()
            raw = tomllib.loads(raw_bytes.decode('utf-8-sig'))
        except (OSError, ValueError) as error:
            raise ConfigError(f'Cannot load project config {filename}: {error}') from error
        _table(raw, str(filename), {'schema_version', 'component', 'source', 'sources', 'dependencies', 'variants', 'context'})
        if type(raw.get('schema_version')) is not int or raw['schema_version'] != 1:
            raise ConfigError(f'{filename}: schema_version must be 1')
        component = _table(raw.get('component'), 'component', {'name', 'kind'})
        name = _identifier(component.get('name'), 'component.name')
        kind = _string(component.get('kind', 'module'), 'component.kind')
        if kind not in {'project', 'module', 'sdk', 'documents'}:
            raise ConfigError('component.kind must be project, module, sdk or documents')
        if name in identities and identities[name] != filename:
            raise ConfigError(f'Duplicate component name {name}: {filename} and {identities[name]}')
        identities[name] = filename
        context = _table(raw.get('context', {}), 'context', set(raw.get('context', {})) if isinstance(raw.get('context', {}), dict) else set())
        # Validate that context is finite JSON before any registry mutation.
        _digest(context)
        variants = raw.get('variants', {})
        if not isinstance(variants, dict):
            raise ConfigError('variants must be a table')
        for key, value in variants.items():
            _identifier(key, 'variant')
            _table(value, f'variants.{key}', {'include', 'exclude'})
            for option, patterns in value.items():
                _patterns(patterns, f'variants.{key}.{option}')
        if variant is not None and variant not in variants:
            raise ConfigError(f'Unknown variant {variant} in {filename}')
        selected_variant = variants.get(variant, {})
        declared = dict(raw.get('sources', {})) if isinstance(raw.get('sources', {}), dict) else None
        if declared is None:
            raise ConfigError('sources must be a table')
        if 'source' in raw:
            if 'source' in declared:
                raise ConfigError('source and sources.source cannot both be declared')
            declared['source'] = raw['source']
        if not declared:
            raise ConfigError(f'{filename}: declare source or sources')
        if (variant is not None or narrowing) and 'source' not in declared:
            raise ConfigError(f'{filename}: variants and consumer filters require a primary source')
        dependencies = raw.get('dependencies', {})
        if not isinstance(dependencies, dict):
            raise ConfigError('dependencies must be a table')
        active.append(filename)
        seen[filename] = selection
        components.append({'name': name, 'kind': kind, 'config': str(filename),
                           'sha256': hashlib.sha256(raw_bytes).hexdigest(),
                           'parent': parent, 'selection': selection})
        for local_name, value in declared.items():
            _identifier(local_name, 'source name')
            item = _table(value, f'sources.{local_name}',
                          {'path', 'include', 'exclude', 'attach', 'respect_gitignore', 'role', 'kind'})
            root = (filename.parent / _string(item.get('path', '.'), 'source.path')).resolve()
            groups = []
            excludes = list(_strings(item.get('exclude', []), 'source.exclude'))
            # Variants and consumer overrides only narrow the primary source.
            if local_name == 'source':
                for restriction in (selected_variant, narrowing or {}):
                    if 'include' in restriction:
                        groups.append(_patterns(restriction['include'], 'selection.include'))
                    excludes.extend(_strings(restriction.get('exclude', []), 'selection.exclude'))
            source_kind = _string(item.get('kind', 'module' if kind == 'sdk' else kind), 'source.kind')
            sources.append(SourceSpec(_identifier(name + '.' + local_name, 'source_id'), source_kind, root,
                include=_patterns(item.get('include', ['**/*']), 'source.include'), exclude=tuple(excludes),
                role=_string(item.get('role', 'sdk' if kind == 'sdk' else 'docs' if kind == 'documents' else 'code'), 'source.role'),
                respect_gitignore=_boolean(item.get('respect_gitignore', True), 'source.respect_gitignore'),
                attach_paths=_strings(item.get('attach', []), 'source.attach'), include_groups=tuple(groups)))
        for alias, value in dependencies.items():
            _identifier(alias, 'dependency name')
            item = _table(value, f'dependencies.{alias}', {'path', 'variant', 'include', 'exclude'})
            dependency = (filename.parent / _string(item.get('path'), 'dependency.path')).resolve()
            if dependency.is_dir():
                dependency /= 'bti.toml'
            selected = item.get('variant')
            if selected is not None:
                _string(selected, 'dependency.variant')
            visit(dependency, selected, {k: v for k, v in item.items() if k in {'include', 'exclude'}}, name)
        active.pop()
        return name, context

    name, context = visit(path)
    # Component roots are owned by their own sources even with broad parent globs.
    configs = [Path(component['config']) for component in components]
    normalized = []
    for spec in sources:
        exclusions = list(spec.exclude)
        for config in configs:
            child = config.parent
            if child != spec.root and child.is_relative_to(spec.root):
                exclusions.append(child.relative_to(spec.root).as_posix() + '/**')
        for other in sources:
            if other.root != spec.root and other.root.is_relative_to(spec.root):
                exclusions.append(other.root.relative_to(spec.root).as_posix() + '/**')
        normalized.append(replace(spec, exclude=tuple(sorted(set(exclusions)))))
    fingerprint = _digest({'components': components, 'sources': [asdict(s) | {'root': str(s.root)} for s in normalized], 'context': context})
    return Project(name, path, tuple(normalized), tuple(components), fingerprint, context)


def sync_project(project, config, recipe, backend, *, plan=None):
    """Publish all ready units together, preserving the previous composition on error."""
    from .qwen_runtime import file_lock
    with file_lock(config.data_dir / 'metadata/project-sync.lock'):
        return _sync_project(project, config, recipe, backend, plan=plan)


def _sync_project(project, config, recipe, backend, *, plan=None):
    plan = project.plan(config.data_dir) if plan is None else plan
    if plan['config_hash'] != project.config_hash or not plan['ready']:
        raise ConfigError('Project preflight failed; inspect bti plan before indexing')
    registry = Registry(config.data_dir)
    units = []
    for spec, selection in zip(project.sources, plan['sources'], strict=True):
        registry.register(spec)
        unit = registry.snapshot(spec.source_id, recipe)
        if unit['files'] != selection['files']:
            raise ConfigError('Source changed after preflight; rerun sync: ' + spec.source_id)
        status = registry.index_status(unit['unit_id'])
        if status['status'] != 'ready':
            backend.index(unit['unit_id'])
        units.append(unit['unit_id'])
    if load_project(project.path).config_hash != project.config_hash:
        raise ConfigError('Project configuration changed during sync; previous composition retained')
    # Indexing may take minutes. Recheck every selected source, including earlier units,
    # before changing the published alias. New/deleted files matter as much as edits.
    current = project.plan(config.data_dir)
    if [s['files'] for s in current['sources']] != [s['files'] for s in plan['sources']]:
        raise ConfigError('Source changed during sync; previous composition retained; retry')
    if load_project(project.path).config_hash != project.config_hash:
        raise ConfigError('Project configuration changed during sync; previous composition retained')
    configuration = project.context | {'source_roots': {s.source_id: str(s.root) for s in project.sources},
         'bti': {'config_hash': project.config_hash, 'components': list(project.components),
                 'project_root': str(project.path.parent)}}
    references = [registry.load_unit(unit_id) for unit_id in units]
    composition_hash = _digest({'units': [{'unit_id': u['unit_id'], 'recipe_hash': u['recipe_hash']} for u in references],
                                'configuration_hash': _digest(configuration)})
    # Publish the immutable address before switching the convenient project alias.
    pinned_id = project.name[:55] + '-' + composition_hash
    pinned = registry.compose(pinned_id, units, require_ready=True, configuration=configuration)
    composition = registry.compose(project.name, units, require_ready=True, configuration=configuration)
    return {'project': project.name, 'composition_id': project.name,
            'composition_hash': composition['composition_hash'], 'pinned_composition_id': pinned_id,
            'pinned_composition_hash': pinned['composition_hash'], 'units': units, 'ready': True}


def project_info(config):
    """Describe the selected project without starting models or scanning the corpus."""
    from .search import SearchService
    if config.project_config is None:
        return {'configured': False, 'hint': 'Create bti.toml or pass --project-file; explicit composition_id remains supported'}
    project = load_project(config.project_config)
    composition_path = config.data_dir / 'metadata/compositions' / (project.name + '.json')
    result = {'configured': True, 'project': project.name, 'project_config': str(project.path),
              'config_hash': project.config_hash, 'components': list(project.components),
              'sources': [{'source_id': s.source_id, 'role': s.role} for s in project.sources],
              'ready': False, 'next': 'Call context_search/read_sources with freshness=wait, or run bti sync; plan is optional diagnostics'}
    if composition_path.exists():
        composition = SearchService(config.data_dir).load_composition(project.name)
        pinned_id = project.name[:55] + '-' + composition['composition_hash']
        pinned = SearchService(config.data_dir).load_composition(pinned_id)
        result.update(ready=True, composition_id=pinned_id, composition_hash=pinned['composition_hash'],
                      project_alias=project.name,
                      config_changed=composition['configuration'].get('bti', {}).get('config_hash') != project.config_hash,
                      next='Call context_search directly or read_sources with a known path; composition_id optionally pins this version; check source_changed')
    return result
