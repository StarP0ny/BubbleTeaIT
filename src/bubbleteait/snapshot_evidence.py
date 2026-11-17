"""Validate snapshot search evidence at the service and native worker boundaries."""
from __future__ import annotations

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import stat
import sys

from .registry import RegistryError, _relative, _safe_file


class SearchError(RegistryError):
    """Composition, filter, source or backend response is not usable."""


def _path_matches(connection: sqlite3.Connection, path: str, patterns: list[str] | None) -> bool:
    return patterns is None or any(connection.execute("SELECT ? GLOB ?", (path, pattern)).fetchone()[0] for pattern in patterns)


def _bytes(path: Path, expected_hash: str | None = None) -> tuple[bytes, str]:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise SearchError("Source path is not a regular file")
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise SearchError(f"Source file is unavailable: {path}") from exc
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise SearchError("Source changed while it was being read; retry")
    if len(content) != after.st_size:
        raise SearchError("Source size changed while it was being read; retry")
    digest = hashlib.sha256(content).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise SearchError("Snapshot source hash does not match its manifest")
    return content, digest


def _text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SearchError("Source is not UTF-8; an explicit decoding adapter is required") from exc


def snapshot_file(data_dir: Path, manifest: dict, relative: str) -> tuple[dict, bytes, str]:
    _relative(relative)
    entry = next((entry for entry in manifest["files"] if entry["path"] == relative), None)
    if entry is None:
        raise SearchError("Requested source file is outside the snapshot whitelist")
    root = data_dir / "sources" / manifest["unit_id"] / "files"
    content, digest = _bytes(_safe_file(root, relative), entry["sha256"])
    return entry, content, digest


def verify_result(data_dir: Path, manifest: dict, row: dict) -> dict:
    """Return the original snapshot slice and reject contradictory evidence."""
    if (not isinstance(row, dict) or not {"file_path", "start_line", "end_line", "content"} <= row.keys()
            or not isinstance(row["file_path"], str) or not isinstance(row["content"], str)):
        raise SearchError("Backend returned an invalid source result")
    if "score" in row:
        score = row["score"]
        if type(score) not in (int, float) or not -sys.float_info.max <= score <= sys.float_info.max:
            raise SearchError("Backend result score must be a finite number")
    _, raw, digest = snapshot_file(data_dir, manifest, row["file_path"])
    proof = {key: manifest[key] for key in ("unit_id", "source_id", "recipe_hash")}
    proof["source_sha256"] = digest
    if any(key in row and row[key] != value for key, value in proof.items()):
        raise SearchError("Backend result source identity differs from the snapshot")
    lines = _text(raw).splitlines(keepends=True)
    start, end = row["start_line"], row["end_line"]
    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
        raise SearchError("Backend result has invalid source coordinates")
    excerpt = "".join(lines[start - 1:end])
    # Some workers normalize CRLF and omit the last line terminator.
    candidate = row["content"].replace("\r\n", "\n").removesuffix("\n")
    if not candidate.strip() or candidate != excerpt.replace("\r\n", "\n").removesuffix("\n"):
        raise SearchError("Backend result content does not match source coordinates")
    return {**row, **proof, "content": excerpt}


def verify_worker_results(data_dir: Path, manifest: dict, rows, limit: int, paths: list[str] | None,
                          *, native_paths: bool = False) -> list[dict]:
    if not isinstance(rows, list) or len(rows) > limit:
        raise SearchError("Worker returned an invalid result list")
    verified = []
    with closing(sqlite3.connect(":memory:")) as connection:
        for row in rows:
            if not isinstance(row, dict) or "score" not in row:
                raise SearchError("Worker returned an unscored result")
            if native_paths and isinstance(row.get("file_path"), str):
                row = {**row, "file_path": row["file_path"].replace("\\", "/")}
            value = verify_result(data_dir, manifest, row)
            if not _path_matches(connection, value["file_path"], paths):
                raise SearchError("Backend violated its pre-retrieval path filter")
            verified.append(value)
    return verified
