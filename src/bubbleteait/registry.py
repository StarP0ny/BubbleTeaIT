"""Portable source definitions, immutable byte snapshots and project composition.

The registry does not embed, parse or generate text. Git is read-only provenance;
content hashes are computed from selected raw bytes even for a clean checkout.
Composition references whole units and never merges backend database files.

Public API::

    registry = Registry(data_dir)
    registry.register(SourceSpec("application", "project", Path("checkout")))
    manifest = registry.snapshot("application", recipe)
    registry.load_unit(manifest["unit_id"], verify_files=True)
    registry.compose("release", [manifest["unit_id"]], configuration={})

By default composition is a draft. Publishing with ``require_ready=True`` needs
every backend's ``indexes/<unit_id>/status.json`` to have schema_version=1,
status='ready', matching unit_id and recipe_hash. This is a single-writer API:
JSON replacement/publication is atomic, but concurrent registry edits require
the coordinator's writer lock (planned in the update layer).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import uuid


DEFAULT_EXCLUDES = tuple(
    f"**/{name}/**"
    for name in (
        ".venv", "venv", "vendor", "build", "dist", "node_modules", ".local",
        "__pycache__", ".pytest_cache", ".ruff_cache", "models", "indexes",
    )
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UNIT_ID = re.compile(r"u-[0-9a-f]{64}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class RegistryError(ValueError):
    """Source, snapshot or composition is invalid or inconsistent."""


class SnapshotChangedError(RegistryError):
    """The selected source changed while a snapshot was being built."""


@dataclass(frozen=True)
class SourceSpec:
    """Logical source and explicit file selection, respecting Git ignores.

    ``include``/``exclude`` are POSIX relative glob patterns: ``*`` matches one
    path segment and ``**`` zero or more segments. Default excluded directory
    names apply below the source root. ``attach_paths`` explicitly opts a file
    or subtree into selection despite Git ignores and automatic exclusions;
    include globs and explicit excludes still apply. Attachments stay inside
    root. Selected symlinks are rejected, not dereferenced.
    """

    source_id: str
    kind: str
    root: Path
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    role: str = "unknown"
    respect_gitignore: bool = True
    attach_paths: tuple[str, ...] = ()
    include_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class IndexRecipe:
    """All vector/chunk-affecting settings belong in this fingerprint.

    Pin ``embedding_revision`` to an actual model revision. Options include
    document/query instructions, normalization and other backend parameters.
    The registry records, but does not download or validate, model revisions.
    """

    backend: str
    embedding_model: str
    embedding_revision: str
    chunker: str
    schema_version: str = "1"
    options: Mapping[str, Any] = field(default_factory=dict)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RegistryError("Metadata must contain finite JSON values") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise RegistryError(f"Invalid {label}: expected 1-128 portable identifier characters")
    return value


def _unit_identifier(value: str) -> str:
    if not isinstance(value, str) or not _UNIT_ID.fullmatch(value):
        raise RegistryError("Invalid unit_id")
    return value


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or any(character in value for character in ("\\", ":", "\0")):
        raise RegistryError("Expected a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise RegistryError("Path traversal or an absolute path is not allowed")
    return path.as_posix()


@lru_cache(maxsize=8192)
def _matches(path: str, pattern: str) -> bool:
    parts, patterns = tuple(path.split("/")), tuple(pattern.split("/"))

    @lru_cache(maxsize=None)
    def match(pi: int, gi: int) -> bool:
        if gi == len(patterns):
            return pi == len(parts)
        if patterns[gi] == "**":
            return match(pi, gi + 1) or (pi < len(parts) and match(pi + 1, gi))
        return pi < len(parts) and fnmatchcase(parts[pi], patterns[gi]) and match(pi + 1, gi + 1)

    return match(0, 0)


def _json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RegistryError(f"Cannot read metadata: {path}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _recipe(recipe: IndexRecipe) -> dict:
    value = asdict(recipe)
    for name in ("backend", "embedding_model", "embedding_revision", "chunker", "schema_version"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise RegistryError(f"Recipe {name} must be a nonempty string")
    if not isinstance(value["options"], dict):
        raise RegistryError("Recipe options must be a JSON object")
    _canonical(value)
    return value


def _checked_spec(spec: SourceSpec) -> SourceSpec:
    _identifier(spec.source_id, "source_id")
    if spec.kind not in {"project", "module", "documents"}:
        raise RegistryError("Source kind must be project, module or documents")
    if not isinstance(spec.role, str) or not spec.role.strip():
        raise RegistryError("Source role must be a nonempty string")
    root = Path(spec.root).resolve()
    if not root.is_dir():
        raise RegistryError(f"Source directory does not exist: {root}")
    if ".git" in root.parts:
        raise RegistryError("Git metadata cannot be a source root")
    if not isinstance(spec.include_groups, (list, tuple)):
        raise RegistryError("include_groups must be arrays of relative globs")
    for patterns in (spec.include, spec.exclude, *spec.include_groups):
        if not isinstance(patterns, (list, tuple)):
            raise RegistryError("Include and exclude must be arrays of relative globs")
        for pattern in patterns:
            _relative(pattern)
    if not spec.include:
        raise RegistryError("At least one include pattern is required")
    if type(spec.respect_gitignore) is not bool:
        raise RegistryError("respect_gitignore must be a boolean")
    if not isinstance(spec.attach_paths, (tuple, list)):
        raise RegistryError("attach_paths must be an array of paths")
    attachments = []
    for value in spec.attach_paths:
        if not isinstance(value, str) or not value or "\0" in value:
            raise RegistryError("Attachment must be a nonempty path")
        path = Path(value)
        path = path if path.is_absolute() else root / path
        # Use an absolute lexical path first; resolve alone would hide links.
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(root):
            raise RegistryError("Attachment must stay inside the source root; register another source for outside paths")
        relative = path.relative_to(root).as_posix()
        if relative != ".":
            _safe_file(root, relative)
        if ".git" in path.relative_to(root).parts:
            raise RegistryError("Git metadata cannot be attached")
        if not path.exists():
            raise RegistryError(f"Attachment does not exist: {value}")
        attachments.append(relative)
    return SourceSpec(spec.source_id, spec.kind, root, tuple(spec.include), tuple(spec.exclude),
                      spec.role, spec.respect_gitignore, tuple(sorted(set(attachments))),
                      tuple(tuple(group) for group in spec.include_groups))


def _safe_file(root: Path, relative: str) -> Path:
    if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
        raise RegistryError("File root must not be a symlink/junction")
    path = root / _relative(relative)
    # Reject links in every component; resolving only the final file misses a
    # linked parent directory. This also prevents crafted snapshot manifests.
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise RegistryError(f"Selected symlink/junction is not supported: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise RegistryError(f"Selected path escapes source root: {relative}")
    return path


def _read_file(path: Path) -> tuple[bytes, dict]:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RegistryError(f"Selected path is not a regular file: {path}")
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or len(content) != after.st_size:
        raise SnapshotChangedError(f"Source changed while reading: {path}")
    return content, {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}


def _git_provenance(root: Path) -> dict:
    if not shutil.which("git"):
        return {"available": False, "reason": "git executable unavailable"}

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, timeout=30, check=False)

    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode:
            return {"available": False, "reason": "not a Git working tree"}
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "-z", "--untracked-files=normal", "--", ".")
        submodules = git("submodule", "status", "--recursive")
        return {
            "available": True,
            "root": top.stdout.decode("utf-8", "surrogateescape").strip(),
            "head": head.stdout.decode("ascii", "replace").strip() if head.returncode == 0 else None,
            "dirty": bool(status.stdout) if status.returncode == 0 else None,
            "status": status.stdout.decode("utf-8", "surrogateescape").split("\0")[:-1],
            "submodules": submodules.stdout.decode("utf-8", "surrogateescape").splitlines(),
            "status_available": status.returncode == 0,
            "submodules_available": submodules.returncode == 0,
        }
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "reason": "Git provenance command failed"}


class _IgnorePolicy:
    """Ask Git itself about ignores, including nested repositories and negation.

    A disposable bare repository supplies Git semantics for a directory that
    has no repository; no .git directory is written into the source. Record
    relevant rule files, including missing ones, for checked capture.
    """

    def __init__(self, root: Path):
        if not shutil.which("git"):
            raise RegistryError("Respecting .gitignore requires Git; install Git or explicitly use --no-respect-gitignore")
        self.root = root
        self.rules: dict[str, str | None] = {}
        self.contexts: dict[Path, tuple[Path, list[str], Path | None]] = {}
        self.temporary = tempfile.TemporaryDirectory(prefix="bti-ignore-")
        self.fallback: list[str] | None = None

    def close(self):
        self.temporary.cleanup()

    @staticmethod
    def _run(command: list[str], *args: str, input: bytes | None = None):
        try:
            return subprocess.run([*command, *args], input=input, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RegistryError("Git ignore check failed") from exc

    def _record(self, path: Path):
        value = self._rule_fingerprint(path)
        key = str(path)
        if key in self.rules and self.rules[key] != value:
            raise SnapshotChangedError(f"Ignore rules changed during capture: {path}")
        self.rules[key] = value

    @staticmethod
    def _rule_fingerprint(path: Path) -> str | None:
        # Git does not follow .gitignore symlinks. Never read their targets
        # while recording the checked-capture state either.
        if path.is_symlink():
            return "symlink:" + os.readlink(path)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def _context(self, directory: Path):
        if directory in self.contexts:
            return self.contexts[directory]
        # Reuse the parent context unless this directory starts a nested repo.
        parent = directory.parent
        if directory != self.root and parent in self.contexts and not (directory / ".git").exists():
            context = self.contexts[parent]
        else:
            command = ["git", "--no-optional-locks", "-C", str(directory)]
            top = self._run(command, "rev-parse", "--show-toplevel")
            if top.returncode == 0:
                worktree = Path(os.fsdecode(top.stdout).strip()).resolve()
                gitdir = self._run(command, "rev-parse", "--absolute-git-dir")
                if gitdir.returncode:
                    raise RegistryError("Cannot locate Git metadata for ignore rules")
                context = (worktree, ["git", "--no-optional-locks", "-C", str(worktree)],
                           Path(os.fsdecode(gitdir.stdout).strip()))
            else:
                if self.fallback is None:
                    gitdir = Path(self.temporary.name) / "rules.git"
                    initialized = self._run(["git"], "init", "--bare", "--quiet", str(gitdir))
                    if initialized.returncode:
                        raise RegistryError("Cannot initialize temporary Git ignore context")
                    self.fallback = ["git", "--no-optional-locks", "-C", str(self.root),
                                     f"--git-dir={gitdir}", f"--work-tree={self.root}"]
                context = (self.root, self.fallback, None)
            worktree, context_command, gitdir = context
            if gitdir is not None:
                # worktree-specific config and shared config both affect ignores.
                self._record(gitdir / "config")
                self._record(gitdir / "config.worktree")
                excludes = self._run(context_command, "rev-parse", "--git-path", "info/exclude")
                if excludes.returncode == 0:
                    exclude_path = Path(os.fsdecode(excludes.stdout).strip())
                    self._record(exclude_path if exclude_path.is_absolute() else worktree / exclude_path)
            global_excludes = self._run(context_command, "config", "--path", "--get", "core.excludesFile")
            if global_excludes.returncode == 0:
                exclude_path = Path(os.fsdecode(global_excludes.stdout).strip())
                self._record(exclude_path if exclude_path.is_absolute() else worktree / exclude_path)
            else:
                config_home = os.environ.get("XDG_CONFIG_HOME")
                if config_home:
                    self._record(Path(config_home) / "git/ignore")
                else:
                    try:
                        self._record(Path.home() / ".config/git/ignore")
                    except RuntimeError:
                        pass  # No home directory in an intentionally minimal environment.
        self.contexts[directory] = context
        worktree, _, _ = context
        current = directory
        while current.is_relative_to(worktree):
            self._record(current / ".gitignore")
            if current == worktree:
                break
            current = current.parent
        return context

    def ignored(self, directory: Path, names: list[tuple[str, bool]]) -> set[str]:
        if not names:
            return set()
        worktree, command, _ = self._context(directory)
        paths = {}
        for name, is_directory in names:
            relative = (directory / name).relative_to(worktree).as_posix()
            paths[relative + ("/" if is_directory else "")] = name
        response = self._run(command, "check-ignore", "--no-index", "-z", "--stdin",
                             input=b"\0".join(os.fsencode(path) for path in paths) + b"\0")
        if response.returncode not in (0, 1):
            raise RegistryError("Git ignore check failed: " + response.stderr.decode("utf-8", "replace").strip())
        return {paths[os.fsdecode(path)] for path in response.stdout.split(b"\0") if path}

    def verify(self):
        for name, expected in self.rules.items():
            path = Path(name)
            actual = self._rule_fingerprint(path)
            if actual != expected:
                raise SnapshotChangedError(f"Ignore rules changed during capture: {path}")


class Registry:
    """Metadata at data_dir/metadata; byte snapshots at sources/<unit_id>/files.

    Paths of live checkouts exist only in registration/provenance, not in unit
    identity. An identical logical source and recipe can therefore be reused
    after relocating a checkout. Sources need not be Git repositories.
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir).resolve()
        self.sources_file = self.data_dir / "metadata/sources.json"

    def _sources(self) -> dict:
        if not self.sources_file.exists():
            return {"schema_version": 1, "sources": {}}
        value = _json(self.sources_file)
        if value.get("schema_version") != 1 or not isinstance(value.get("sources"), dict):
            raise RegistryError("Unsupported source registry schema")
        return value

    def register(self, spec: SourceSpec) -> dict:
        """Add/update a source definition without touching its checkout."""
        spec = _checked_spec(spec)
        self._check_source_boundary(spec)
        value = asdict(spec)
        value["root"] = str(spec.root)
        registry = self._sources()
        registry["sources"][spec.source_id] = value
        _atomic_json(self.sources_file, registry)
        return value

    def list_sources(self) -> list[dict]:
        return [value for _, value in sorted(self._sources()["sources"].items())]

    def source(self, source_id: str) -> SourceSpec:
        """Return the current registered source, including local path mapping."""
        _identifier(source_id, "source_id")
        value = self._sources()["sources"].get(source_id)
        if value is None:
            raise RegistryError(f"Unknown source_id: {source_id}")
        try:
            spec = _checked_spec(SourceSpec(**value))
            self._check_source_boundary(spec)
            return spec
        except TypeError as exc:
            raise RegistryError(f"Invalid registered source: {source_id}") from exc

    def _check_source_boundary(self, spec: SourceSpec) -> None:
        if spec.root.is_relative_to(self.data_dir):
            raise RegistryError("Source root cannot be the registry data directory or a directory inside it")

    def _inventory(self, spec: SourceSpec, *, skipped: list | None = None) -> list[dict]:
        selected = []
        policy = _IgnorePolicy(spec.root) if spec.respect_gitignore else None

        def attached(relative: str) -> bool:
            return any(path == "." or relative == path or relative.startswith(path + "/") for path in spec.attach_paths)

        def leads_to_attachment(relative: str) -> bool:
            return any(path.startswith(relative + "/") for path in spec.attach_paths)

        def excluded(relative: str, *, directory: bool = False) -> bool:
            candidate = relative + "/_" if directory else relative
            explicit = any(_matches(relative, pattern) or (directory and pattern.endswith("/**") and _matches(candidate, pattern)) for pattern in spec.exclude)
            automatic = any(_matches(candidate, pattern) for pattern in DEFAULT_EXCLUDES)
            return explicit or (automatic and not attached(relative) and not (directory and leads_to_attachment(relative)))

        def fail_walk(error: OSError) -> None:
            raise error

        try:
            for directory, dirs, files in os.walk(spec.root, followlinks=False, onerror=fail_walk):
                parent = Path(directory)
                candidates = [(name, True) for name in dirs if name != ".git"] + [(name, False) for name in files if name != ".git"]
                # Parent repositories cannot check paths inside a submodule.
                # Check that boundary as a directory, then use its own context.
                ignored = policy.ignored(parent, candidates) if policy else set()
                kept = []
                for name in sorted(dirs):
                    path = parent / name
                    relative = path.relative_to(spec.root).as_posix()
                    if name == ".git" or path.resolve() == self.data_dir or excluded(relative, directory=True):
                        if skipped is not None:
                            skipped.append({"path": relative + "/", "reason": "excluded_or_internal"})
                        continue
                    if name in ignored and not attached(relative) and not leads_to_attachment(relative):
                        if skipped is not None:
                            skipped.append({"path": relative + "/", "reason": "gitignore"})
                        continue
                    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                        raise RegistryError(f"Source symlink/junction directory is not supported: {relative}")
                    kept.append(name)
                dirs[:] = kept
                for name in sorted(files):
                    relative = (parent / name).relative_to(spec.root).as_posix()
                    reason = ("excluded_or_internal" if name == ".git" or excluded(relative) else
                              "gitignore" if name in ignored and not attached(relative) else
                              "not_included" if not all(any(_matches(relative, pattern) for pattern in group)
                                                        for group in (spec.include, *spec.include_groups)) else None)
                    if reason:
                        if skipped is not None:
                            skipped.append({"path": relative, "reason": reason})
                        continue
                    path = _safe_file(spec.root, relative)
                    if path.resolve().is_relative_to(self.data_dir):
                        continue
                    _, entry = _read_file(path)
                    selected.append({"path": relative, **entry})
            if policy:
                policy.verify()
            self._last_selection_rules = dict(policy.rules) if policy else {}
        finally:
            if policy:
                policy.close()
        return sorted(selected, key=lambda entry: entry["path"])

    def preview(self, spec: SourceSpec) -> dict:
        """Inspect the exact selection and UTF-8 readiness without registration or ML."""
        spec = _checked_spec(spec)
        self._check_source_boundary(spec)
        skipped = []
        files = self._inventory(spec, skipped=skipped)
        rules = dict(self._last_selection_rules)
        errors = []
        for entry in files:
            raw, checked = _read_file(_safe_file(spec.root, entry["path"]))
            if checked != {"sha256": entry["sha256"], "size": entry["size"]}:
                raise SnapshotChangedError("Source changed during preview: " + entry["path"])
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as error:
                errors.append({"path": entry["path"], "reason": "not_utf8", "byte_offset": error.start})
            if b"\0" in raw:
                errors.append({"path": entry["path"], "reason": "binary_nul"})
        return {"source_id": spec.source_id, "root": str(spec.root), "role": spec.role,
                "files": files, "file_count": len(files), "skipped": skipped,
                "ignore_rules": rules, "encoding_errors": errors,
                "warnings": (["empty_source"] if not files else ["small_source"] if len(files) < 3 else []),
                "ready": bool(files) and not errors}

    def snapshot(self, source_id: str, recipe: IndexRecipe) -> dict:
        """Publish a checked immutable byte snapshot; embeddings remain missing.

        A first inventory, copy-time digest check and final inventory detect
        additions/deletions/edits during capture. This is checked capture rather
        than an OS-level filesystem freeze. Failure never publishes a new unit.
        A failed staging directory is retained for diagnosis, without ready
        manifest; it is never returned as a usable snapshot.
        """
        spec = self.source(source_id)
        recipe_value = _recipe(recipe)
        recipe_hash = _digest(recipe_value)
        files = self._inventory(spec)
        selection_rules = dict(self._last_selection_rules)
        if not files:
            raise RegistryError(f"No files selected for source: {source_id}")
        content_hash = _digest(files)
        identity = {"source_id": source_id, "kind": spec.kind, "role": spec.role, "content_hash": content_hash, "recipe_hash": recipe_hash}
        unit_id = "u-" + _digest(identity)
        target = self.data_dir / "sources" / unit_id
        if target.exists():
            return self.load_unit(unit_id, verify_files=True)

        staging = self.data_dir / "sources" / (".staging-" + uuid.uuid4().hex)
        (staging / "files").mkdir(parents=True)
        for entry in files:
            path = _safe_file(spec.root, entry["path"])
            content, copied = _read_file(path)
            if copied != {"sha256": entry["sha256"], "size": entry["size"]}:
                raise SnapshotChangedError(f"Source changed before copying: {entry['path']}")
            destination = staging / "files" / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if self._inventory(spec) != files or self._last_selection_rules != selection_rules:
            raise SnapshotChangedError("Selected source files changed during capture")

        manifest = {
            "schema_version": 1, "unit_id": unit_id, **identity,
            "recipe": recipe_value, "created_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_ready": True, "index_status": "missing", "files": files,
            "provenance": {"source_root": str(spec.root), "git": _git_provenance(spec.root),
                           "selection": {"include": list(spec.include), "exclude": list(spec.exclude),
                                         "respect_gitignore": spec.respect_gitignore,
                                         "attach_paths": list(spec.attach_paths)}},
        }
        if spec.include_groups:
            manifest["provenance"]["selection"]["include_groups"] = [list(group) for group in spec.include_groups]
        _atomic_json(staging / "manifest.json", manifest)
        try:
            staging.rename(target)
        except OSError:
            if not target.exists():
                raise
            return self.load_unit(unit_id, verify_files=True)
        return manifest

    def load_unit(self, unit_id: str, *, verify_files: bool = True) -> dict:
        """Validate manifest identity and optionally every snapshot byte."""
        _unit_identifier(unit_id)
        root = self.data_dir / "sources" / unit_id
        if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
            raise RegistryError("Snapshot root must not be a symlink")
        manifest = _json(root / "manifest.json")
        if manifest.get("schema_version") != 1 or manifest.get("unit_id") != unit_id or manifest.get("snapshot_ready") is not True:
            raise RegistryError("Snapshot manifest is not ready or has invalid identity")
        recipe_hash = _digest(manifest.get("recipe"))
        if manifest.get("recipe_hash") != recipe_hash:
            raise RegistryError("Snapshot recipe hash does not match manifest")
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise RegistryError("Snapshot file manifest is empty or invalid")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
                raise RegistryError("Invalid snapshot file entry")
            relative = _relative(entry["path"])
            if relative in seen:
                raise RegistryError("Duplicate snapshot file path")
            seen.add(relative)
            if not isinstance(entry["sha256"], str) or not _HASH.fullmatch(entry["sha256"]) or type(entry["size"]) is not int or entry["size"] < 0:
                raise RegistryError("Invalid snapshot file hash or size")
            path = _safe_file(root / "files", relative)
            if verify_files:
                try:
                    _, actual = _read_file(path)
                except OSError as exc:
                    raise RegistryError(f"Snapshot file is unavailable: {relative}") from exc
                if actual != {"sha256": entry["sha256"], "size": entry["size"]}:
                    raise RegistryError(f"Snapshot bytes do not match: {relative}")
        if entries != sorted(entries, key=lambda item: item["path"]):
            raise RegistryError("Snapshot file list is not canonical")
        if manifest.get("content_hash") != _digest(entries):
            raise RegistryError("Snapshot content hash does not match manifest")
        identity = {key: manifest.get(key) for key in ("source_id", "kind", "role", "content_hash", "recipe_hash")}
        if "u-" + _digest(identity) != unit_id:
            raise RegistryError("Snapshot unit identity does not match manifest")
        return manifest

    def index_status(self, unit_id: str) -> dict:
        """Validate a backend status record against the immutable source recipe."""
        manifest = self.load_unit(unit_id)
        path = self.data_dir / "indexes" / unit_id / "status.json"
        if not path.exists():
            return {"unit_id": unit_id, "status": "missing", "recipe_hash": manifest["recipe_hash"]}
        status = _json(path)
        if status.get("schema_version") != 1 or status.get("unit_id") != unit_id or status.get("recipe_hash") != manifest["recipe_hash"]:
            raise RegistryError("Backend status does not match source unit and recipe")
        if status.get("status") not in {"building", "ready", "error"}:
            raise RegistryError("Invalid backend index status")
        if status.get("backend") != manifest["recipe"]["backend"]:
            raise RegistryError("Backend status names a different backend")
        return status

    def compose(self, composition_id: str, unit_ids: Sequence[str], *, configuration: Mapping[str, Any] | None = None, require_ready: bool = False) -> dict:
        """Save exact unit references and configuration fingerprint, never merge.

        Drafts always carry ready=false, even if all indexes currently exist.
        Publishing requires require_ready=True and every selected index ready.
        An existing composition is replaced atomically only after validation.
        """
        _identifier(composition_id, "composition_id")
        if not unit_ids or isinstance(unit_ids, (str, bytes)):
            raise RegistryError("Composition needs a nonempty array of unit IDs")
        if len(set(unit_ids)) != len(unit_ids):
            raise RegistryError("Composition contains duplicate unit IDs")
        units = [self.load_unit(unit_id) for unit_id in unit_ids]
        source_ids = [unit["source_id"] for unit in units]
        if len(set(source_ids)) != len(source_ids):
            raise RegistryError("A composition must select one exact version per logical source")
        states = [self.index_status(unit["unit_id"]) for unit in units]
        all_ready = all(state["status"] == "ready" for state in states)
        if require_ready and not all_ready:
            raise RegistryError("Composition cannot be published: some indexes are not ready")
        config = dict(configuration or {})
        references = [
            {"unit_id": unit["unit_id"], "source_id": unit["source_id"], "recipe_hash": unit["recipe_hash"], "index_status": state["status"]}
            for unit, state in zip(units, states, strict=True)
        ]
        value = {
            "schema_version": 1, "composition_id": composition_id,
            "ready": require_ready and all_ready, "units": references,
            "configuration": config, "configuration_hash": _digest(config),
        }
        value["composition_hash"] = _digest({"units": [{"unit_id": unit["unit_id"], "recipe_hash": unit["recipe_hash"]} for unit in references], "configuration_hash": value["configuration_hash"]})
        _atomic_json(self.data_dir / "metadata/compositions" / f"{composition_id}.json", value)
        return value
