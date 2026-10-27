"""Create data directories without modifying existing contents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from .config import Config


DATA_SUBDIRECTORIES = ("sources", "indexes", "metadata", "derived", "jobs", "logs")


@dataclass(frozen=True)
class Directory:
    name: str
    path: str
    created: bool


def initialize_data(config: Config) -> tuple[Directory, ...]:
    """Create the configured layout, preserving all existing files.

    Preflight catches file/directory collisions before making changes. A later
    OS failure may leave already-created empty directories; rerunning is safe.
    No rollback or recursive removal is performed.
    """
    targets = [("data_dir", config.data_dir)]
    targets.extend((name, config.data_dir / name) for name in DATA_SUBDIRECTORIES)
    targets.append(("models_dir", config.models_dir))
    for name, path in targets:
        ancestor = path
        while not ancestor.exists() and ancestor.parent != ancestor:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise NotADirectoryError(f"{name}: expected directory, found a file: {ancestor}")

    result = []
    for name, path in targets:
        existed = path.is_dir()
        path.mkdir(parents=True, exist_ok=True)
        result.append(Directory(name, str(path), not existed))
    return tuple(result)


def as_json(directories: tuple[Directory, ...]) -> dict:
    return {"ok": True, "directories": [asdict(directory) for directory in directories]}
