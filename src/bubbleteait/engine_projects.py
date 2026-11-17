"""Reuse BTI source selection for live engines without editing consumer projects."""
from dataclasses import asdict
from pathlib import Path

from .config import ConfigError
from .project import Project, load_project
from .registry import SourceSpec, _digest


def selected_project(config, name_or_path=None):
    value = str(name_or_path) if name_or_path is not None else str(config.project_dir)
    if value in config.projects:
        name, item = value, config.projects[value]
        root = Path(item['path']).resolve()
    else:
        root = Path(value).resolve()
        matches = [(n, p) for n, p in config.projects.items() if Path(p['path']).resolve() == root]
        if len(matches) > 1:
            raise ConfigError('Several project registrations have the same root')
        name, item = matches[0] if matches else (None, {})
    if not root.is_dir():
        raise ConfigError(f'Project directory is unavailable: {root}')
    filename = Path(item.get('project_file', root / 'bti.toml'))
    if 'include' not in item:
        project = load_project(filename)
        if name and name != project.name:
            project = Project(name, project.path, project.sources, project.components, project.config_hash, project.context)
    else:
        from .project import _patterns
        source = SourceSpec((name or root.name) + '.source', 'project', root,
            include=_patterns(item['include'], 'projects.include'),
            exclude=tuple(item.get('exclude', [])), role='code',
            respect_gitignore=item.get('respect_gitignore', True), attach_paths=tuple(item.get('attach', [])))
        project = Project(name or root.name, filename, (source,), (),
            _digest(asdict(source) | {'root': str(root)}), {})
    return root, project, item


def workspace_catalog(config, workspace):
    if workspace not in config.workspaces:
        raise ConfigError(f'Unknown workspace: {workspace}')
    settings = config.workspaces[workspace]
    if 'projects' in settings:
        candidates = list(settings['projects'])
    else:
        from .service import discover_workspace
        candidates = [str(p.parent) for p in discover_workspace(settings)]
    projects, errors, seen = [], [], set()
    for candidate in candidates:
        try:
            root, project, item = selected_project(config, candidate)
            if project.name in seen:
                raise ConfigError('Duplicate project name: ' + project.name)
            seen.add(project.name)
            projects.append({'project': project.name, 'path': str(root), 'description': item.get('description', '')})
        except ValueError as exc:
            errors.append({'path': str(candidate), 'error': str(exc)})
    return {'projects': projects, 'errors': errors}
