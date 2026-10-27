"""Search exact published compositions and read whitelisted source coordinates.

Qwen exact cosine scores share a global ranking only for identical recipes.
Other federated output interleaves per-unit ranks in composition order; it
never sorts scores from different models together. The text channel ranks
matching lines by substring counts, not BM25. Path filters use SQLite GLOB.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any, Sequence

from .registry import Registry, RegistryError, _digest, _identifier, _relative, _safe_file


class SearchError(RegistryError):
    """Composition, filter, source or backend response is not usable."""


def _filter_values(values: Sequence[str] | None, label: str) -> list[str] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) or not value for value in values):
        raise SearchError(f"{label} must be an array of nonempty strings")
    return list(values)


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
    digest = hashlib.sha256(content).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise SearchError("Snapshot source hash does not match its manifest")
    return content, digest


def _text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SearchError("Source is not UTF-8; an explicit decoding adapter is required") from exc


class SearchService:
    """Read-only composition search; optional backend supplies vector retrieval.

    ``backend.search(unit_id, query, limit=..., paths=...)`` must apply paths
    before top-k. Returning an out-of-filter row is an error, not post-filtered
    recovery. Roles/source_ids select immutable units before backend calls.
    Empty filter arrays select nothing. No endpoint is constructed implicitly.
    """

    def __init__(self, data_dir: Path | str, backend: Any = None):
        self.data_dir = Path(data_dir).resolve()
        self.registry = Registry(self.data_dir)
        self.backend = backend

    def load_composition(self, composition_id: str) -> dict:
        """Validate fingerprints, exact refs and current readiness of all units."""
        _identifier(composition_id, "composition_id")
        path = _safe_file(self.data_dir, f"metadata/compositions/{composition_id}.json")
        try:
            composition = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SearchError(f"Cannot load composition: {composition_id}") from exc
        if not isinstance(composition, dict) or composition.get("schema_version") != 1 or composition.get("composition_id") != composition_id:
            raise SearchError("Composition schema or identity is invalid")
        if composition.get("ready") is not True:
            raise SearchError("Composition is a draft; publish it after its indexes are ready")
        configuration = composition.get("configuration")
        if not isinstance(configuration, dict) or composition.get("configuration_hash") != _digest(configuration):
            raise SearchError("Composition configuration fingerprint does not match")
        roots = configuration.get("source_roots", {})
        if not isinstance(roots, dict) or any(not isinstance(root, str) or not Path(root).is_absolute() for root in roots.values()):
            raise SearchError("configuration.source_roots must map source IDs to absolute paths")
        refs = composition.get("units")
        if not isinstance(refs, list) or not refs:
            raise SearchError("Composition must contain a nonempty unit array")
        seen_units, seen_sources = set(), set()
        for reference in refs:
            if not isinstance(reference, dict) or not {"unit_id", "source_id", "recipe_hash", "index_status"} <= reference.keys():
                raise SearchError("Invalid composition unit reference")
            manifest = self.registry.load_unit(reference["unit_id"])
            if reference["source_id"] != manifest["source_id"] or reference["recipe_hash"] != manifest["recipe_hash"]:
                raise SearchError("Composition reference does not match its source recipe")
            if reference["unit_id"] in seen_units or reference["source_id"] in seen_sources:
                raise SearchError("Composition mixes duplicate units or logical source versions")
            seen_units.add(reference["unit_id"])
            seen_sources.add(reference["source_id"])
            if reference["index_status"] != "ready" or self.registry.index_status(reference["unit_id"])["status"] != "ready":
                raise SearchError("Composition index is no longer ready")
        fingerprint = _digest({"units": [{"unit_id": ref["unit_id"], "recipe_hash": ref["recipe_hash"]} for ref in refs], "configuration_hash": composition["configuration_hash"]})
        if composition.get("composition_hash") != fingerprint:
            raise SearchError("Composition fingerprint does not match its unit references")
        return composition

    def _snapshot_file(self, manifest: dict, relative: str) -> tuple[dict, bytes, str]:
        _relative(relative)
        entry = next((entry for entry in manifest["files"] if entry["path"] == relative), None)
        if entry is None:
            raise SearchError("Requested source file is outside the snapshot whitelist")
        root = self.data_dir / "sources" / manifest["unit_id"] / "files"
        content, digest = _bytes(_safe_file(root, relative), entry["sha256"])
        return entry, content, digest

    def _decorate(self, manifest: dict, row: dict, rank: int) -> dict:
        if not isinstance(row, dict) or not {"file_path", "start_line", "end_line", "content"} <= row.keys() or not isinstance(row["content"], str):
            raise SearchError("Backend returned an invalid source result")
        if "score" in row and (isinstance(row["score"], bool) or not isinstance(row["score"], (int, float)) or not math.isfinite(row["score"])):
            raise SearchError("Backend result score must be a finite number")
        entry, content, _ = self._snapshot_file(manifest, row["file_path"])
        lines = _text(content).splitlines(keepends=True) or [""]
        start, end = row["start_line"], row["end_line"]
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise SearchError("Backend result has invalid source coordinates")
        excerpt = "".join(lines[start - 1:end]).replace("\r\n", "\n")
        candidate = row["content"].replace("\r\n", "\n")
        if not candidate.strip() or candidate.strip() not in excerpt.strip():
            raise SearchError("Backend result content does not match source coordinates")
        return {
            **row, "unit_id": manifest["unit_id"], "source_id": manifest["source_id"],
            "source_sha256": entry["sha256"], "recipe_hash": manifest["recipe_hash"],
            "role": manifest["role"], "kind": manifest["kind"],
            "provenance": manifest["provenance"], "unit_rank": rank,
        }

    def _text_rows(self, manifest: dict, query: str, limit: int, paths: list[str] | None, connection: sqlite3.Connection) -> list[dict]:
        terms = set(re.findall(r"\w+", query.casefold()))
        if not terms:
            raise SearchError("Text query must contain an identifier or word")
        candidates = []
        for entry in manifest["files"]:
            if not _path_matches(connection, entry["path"], paths):
                continue
            _, content, _ = self._snapshot_file(manifest, entry["path"])
            try:
                lines = _text(content).splitlines(keepends=True)
            except SearchError:
                continue  # Binary/non-UTF8 files need a separate parsing recipe.
            seen_ranges = set()
            for index, line in enumerate(lines):
                folded = line.casefold()
                hits = sum(term in folded for term in terms)
                if not hits:
                    continue
                start, end = max(1, index - 1), min(len(lines), index + 3)
                if (start, end) in seen_ranges:
                    continue
                seen_ranges.add((start, end))
                candidates.append({
                    "file_path": entry["path"], "start_line": start, "end_line": end,
                    "content": "".join(lines[start - 1:end]), "score": hits,
                    "score_kind": "matching_query_terms_on_line",
                    "backend": "snapshot-text",
                })
        candidates.sort(key=lambda row: (-row["score"], row["file_path"], row["start_line"], row["end_line"]))
        return candidates[:limit]

    def search(self, composition_id: str, query: str, *, limit: int = 10, channel: str = "vector", roles: Sequence[str] | None = None, source_ids: Sequence[str] | None = None, paths: Sequence[str] | None = None) -> dict:
        """Search selected units, preserving their recipe and per-unit ranking."""
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 100:
            raise SearchError("Search needs a nonempty query and integer limit from 1 to 100")
        if channel not in {"vector", "text"}:
            raise SearchError("Search channel must be vector or text")
        roles = _filter_values(roles, "roles")
        source_ids = _filter_values(source_ids, "source_ids")
        paths = _filter_values(paths, "paths")
        for pattern in paths or []:
            _relative(pattern)
        composition = self.load_composition(composition_id)
        selected = []
        for reference in composition["units"]:
            manifest = self.registry.load_unit(reference["unit_id"])
            if (roles is None or manifest["role"] in roles) and (source_ids is None or manifest["source_id"] in source_ids):
                selected.append(manifest)
        groups = []
        with closing(sqlite3.connect(":memory:")) as connection:
            for manifest in selected:
                if paths == [] or (paths is not None and not any(_path_matches(connection, entry["path"], paths) for entry in manifest["files"])):
                    continue
                if channel == "vector":
                    if self.backend is None:
                        raise SearchError("Vector search requires an explicitly configured backend")
                    rows = self.backend.search(manifest["unit_id"], query, limit=limit, paths=paths)
                else:
                    rows = self._text_rows(manifest, query, limit, paths, connection)
                if not isinstance(rows, list) or len(rows) > limit:
                    raise SearchError("Backend returned an invalid result list")
                group = []
                for rank, row in enumerate(rows, start=1):
                    decorated = self._decorate(manifest, row, rank)
                    if not _path_matches(connection, decorated["file_path"], paths):
                        raise SearchError("Backend violated its pre-retrieval path filter")
                    group.append(decorated)
                groups.append(group)
        # The new Qwen backend returns normalized exact cosines. Units from
        # precisely the same recipe share an embedding space; selecting each
        # unit's top-k is sufficient for the global top-k. Keep legacy/mixed
        # recipes and non-cosine channels on the existing rank interleave.
        comparable = (channel == "vector" and bool(selected)
            and all(manifest["recipe"]["backend"] == "qwen-dense" for manifest in selected)
            and len({manifest["recipe_hash"] for manifest in selected}) == 1
            and all(row.get("score_kind") == "cosine" and "score" in row
                    for group in groups for row in group))
        if comparable:
            ordered = sorted((row for group in groups for row in group),
                key=lambda row: (-row["score"], row["unit_id"], row["file_path"],
                                 row["start_line"], row["end_line"]))
            results = [{**row, "rank": rank} for rank, row in enumerate(ordered[:limit], start=1)]
        else:
            results = []
            for rank in range(max((len(group) for group in groups), default=0)):
                for group in groups:
                    if rank < len(group):
                        results.append({**group[rank], "rank": len(results) + 1})
                        if len(results) == limit:
                            break
                if len(results) == limit:
                    break
        # Never return a successful mixed-generation result if status changed
        # during retrieval. Backend reader/writer locking handles each unit.
        if self.load_composition(composition_id)["composition_hash"] != composition["composition_hash"]:
            raise SearchError("Composition changed during retrieval; retry")
        return {
            "state": "ready", "composition_id": composition_id,
            "composition_hash": composition["composition_hash"], "channel": channel,
            "ranking": "global_cosine_same_recipe" if comparable else "per_unit_rank_interleave",
            "scores_comparable_across_units": comparable,
            "selected_units": [manifest["unit_id"] for manifest in selected], "results": results,
        }

    def read_source(self, unit_id: str, file_path: str, *, start_line: int = 1, end_line: int | None = None, mode: str = "snapshot", composition_id: str | None = None) -> dict:
        """Read 1-based inclusive source coordinates from an exact unit.

        Working roots come from composition.configuration.source_roots[source_id]
        when explicit, otherwise the unit's capture provenance. Current registry
        roots are deliberately ignored: they may point at a different version.
        Snapshot reads remain usable when the live checkout is absent/blocked;
        working_state explains why working_sha256/source_changed is unknown.
        """
        if mode not in {"snapshot", "working"}:
            raise SearchError("Read mode must be snapshot or working")
        if type(start_line) is not int or start_line < 1 or (end_line is not None and (type(end_line) is not int or end_line < start_line)):
            raise SearchError("Source coordinates must be positive inclusive line numbers")
        manifest = self.registry.load_unit(unit_id)
        composition = self.load_composition(composition_id) if composition_id is not None else None
        if composition is not None and not any(ref["unit_id"] == unit_id for ref in composition["units"]):
            raise SearchError("Requested unit does not belong to this composition")
        entry, snapshot_bytes, snapshot_hash = self._snapshot_file(manifest, file_path)
        root_value = None
        if composition is not None:
            root_value = composition["configuration"].get("source_roots", {}).get(manifest["source_id"])
        if root_value is None:
            root_value = manifest["provenance"].get("source_root")
        working_bytes, working_hash, working_state = None, None, "unavailable"
        if isinstance(root_value, str) and Path(root_value).is_absolute():
            try:
                working_path = _safe_file(Path(root_value), file_path)
                if not Path(root_value).is_dir() or not working_path.exists():
                    working_state = "missing"
                else:
                    working_bytes, working_hash = _bytes(working_path)
                    working_state = "available"
            except RegistryError:
                if mode == "working":
                    raise
                working_state = "blocked"
        if mode == "working" and working_bytes is None:
            raise SearchError(f"Working source is {working_state}; snapshot remains available")
        content = snapshot_bytes if mode == "snapshot" else working_bytes
        lines = _text(content).splitlines(keepends=True) or [""]
        end = len(lines) if end_line is None else end_line
        if not start_line <= end <= len(lines):
            raise SearchError("Requested source coordinates exceed file length")
        return {
            "state": "ready", "unit_id": unit_id, "source_id": manifest["source_id"],
            "composition_id": composition_id,
            "composition_hash": composition["composition_hash"] if composition is not None else None,
            "file_path": entry["path"], "mode": mode, "role": manifest["role"],
            "start_line": start_line, "end_line": end, "total_lines": len(lines),
            "content": "".join(lines[start_line - 1:end]),
            "source_sha256": snapshot_hash, "working_sha256": working_hash,
            "used_sha256": snapshot_hash if mode == "snapshot" else working_hash,
            "working_state": working_state,
            "source_changed": working_hash != snapshot_hash if working_hash is not None else (True if working_state == "missing" else None),
            "recipe_hash": manifest["recipe_hash"], "provenance": manifest["provenance"],
        }
