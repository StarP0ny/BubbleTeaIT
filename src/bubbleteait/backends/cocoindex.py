"""Offline CocoIndex Code adapter for immutable registry snapshots.

No ML package is imported by this module. A JSON subprocess uses its own
runtime, model and daemon. All results are checked against snapshot hashes and
source coordinates before reaching the caller. Scores belong to one recipe.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from ..registry import Registry
from .base import BackendCapabilities


class BackendError(ValueError):
    """Backend is unavailable, busy, inconsistent or returned invalid data."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".status-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def _unit_lock(path: Path):
    """OS advisory lock releases on process exit; readers take the same lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BackendError("Index is busy; retry after the current operation") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class CocoIndexBackend:
    """Blocking adapter; CLI/MCP coordinator can run calls as background jobs.

    ``worker_path`` defaults to this checkout's tools directory. Installed
    distributions should configure the explicit path to the packaged checkout.
    ``models_dir`` should be the configured BTI_MODELS_DIR (E:\\models locally).
    """

    capabilities = BackendCapabilities(name="cocoindex-code")

    def __init__(self, data_dir: Path | str, python_executable: Path | str, model_path: Path | str,
                 *, models_dir: Path | str | None = None, worker_path: Path | str | None = None,
                 timeout_seconds: float = 600):
        self.data_dir = Path(data_dir).resolve()
        executable = shutil.which(str(python_executable))
        if executable is None:
            raise BackendError(f"Python executable was not found: {python_executable}")
        self.python_executable = Path(executable).resolve()
        self.model_path = Path(model_path).resolve()
        self.models_dir = Path(models_dir).resolve() if models_dir is not None else self.model_path.parent
        if not self.model_path.is_relative_to(self.models_dir):
            raise BackendError("Model directory must be contained in models_dir")
        self.worker_path = Path(worker_path).resolve() if worker_path is not None else Path(__file__).resolve().parents[3] / "tools/cocoindex_worker.py"
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise BackendError("timeout_seconds must be positive and finite")
        self.timeout_seconds = timeout_seconds
        self.registry = Registry(self.data_dir)

    def _context(self, unit_id: str) -> tuple[dict, Path, str]:
        manifest = self.registry.load_unit(unit_id, verify_files=True)
        recipe = manifest["recipe"]
        if recipe["backend"] != "cocoindex-code" or recipe["chunker"] != "cocoindex-code-default@0.2.42":
            raise BackendError("Unsupported backend or chunker; use cocoindex-code-default@0.2.42")
        if set(recipe["options"]) - {"indexing_params", "query_params"}:
            raise BackendError("Unsupported recipe options")
        for key in ("indexing_params", "query_params"):
            value = recipe["options"].get(key, {})
            if not isinstance(value, dict) or set(value) - {"prompt_name", "normalize_embeddings"}:
                raise BackendError(f"Unsupported {key}")
        model_manifest_path = self.model_path / "bti-model-manifest.json"
        try:
            model = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BackendError("A complete local bti-model-manifest.json is required") from exc
        if not isinstance(model, dict):
            raise BackendError("Local model manifest must be a JSON object")
        if model.get("status") != "complete" or (model.get("repository"), model.get("revision")) != (recipe["embedding_model"], recipe["embedding_revision"]):
            raise BackendError("Model identity/revision does not match the index recipe")
        files = model.get("files")
        if not isinstance(files, list) or not files:
            raise BackendError("Model manifest has no files")
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"] or not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise BackendError("Invalid file entry in local model manifest")
            relative = Path(entry["path"])
            path = self.model_path / relative
            if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(self.model_path):
                raise BackendError("Invalid local model file path")
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise BackendError(f"Model file hash mismatch: {relative}")
        model_hash = hashlib.sha256(model_manifest_path.read_bytes()).hexdigest()
        root = self.data_dir / "indexes" / unit_id
        if root.is_symlink() or not root.resolve().is_relative_to(self.data_dir):
            raise BackendError("Backend index directory escapes the data directory")
        if (root / "runtime/recovery-required.json").exists():
            raise BackendError("Private daemon needs recovery after a failed timeout cleanup")
        return manifest, root, model_hash

    def _stage(self, manifest: dict, root: Path) -> dict:
        workspace = root / "workspace"
        if workspace.is_symlink() or not workspace.resolve().is_relative_to(root):
            raise BackendError("Backend workspace escapes the index directory")
        workspace.mkdir(parents=True, exist_ok=True)
        expected = {item["path"] for item in manifest["files"]}
        actual = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file() and ".cocoindex_code" not in path.relative_to(workspace).parts}
        if actual - expected:
            raise BackendError("Backend workspace contains files outside snapshot whitelist")
        empty_files = []
        for entry in manifest["files"]:
            relative = Path(entry["path"])
            if ".cocoindex_code" in relative.parts:
                raise BackendError("Snapshot uses a reserved backend metadata directory")
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or not target.resolve().is_relative_to(workspace):
                raise BackendError("Backend workspace contains a link outside its root")
            source = self.data_dir / "sources" / manifest["unit_id"] / "files" / relative
            if not target.exists():
                shutil.copyfile(source, target)
            if _sha256(target) != entry["sha256"]:
                raise BackendError("Backend workspace differs from immutable snapshot")
            text = target.read_text(encoding="utf-8", errors="strict")
            if not text.strip():
                empty_files.append(entry["path"])
        settings = workspace / ".cocoindex_code" / "settings.yml"
        if settings.parent.is_symlink() or not settings.parent.resolve().is_relative_to(workspace):
            raise BackendError("Backend metadata escapes the workspace")
        settings.parent.mkdir(exist_ok=True)
        # JSON is valid YAML; no YAML dependency is needed in BubbleTeaIT.
        _json_write(settings, {"include_patterns": ["**/*"], "exclude_patterns": ["**/.cocoindex_code/**"]})
        return {"whitelist_files": len(manifest["files"]), "empty_count": len(empty_files),
                "zero_chunk_files": empty_files}

    def _run(self, request: dict, root: Path) -> dict:
        payload = {**request, "runtime_dir": str(root / "runtime"), "database_dir": str(root / "database"),
                   "workspace": str(root / "workspace"), "models_dir": str(self.models_dir), "model_path": str(self.model_path)}
        command = [str(self.python_executable), str(self.worker_path)]
        environment = os.environ | {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        try:
            completed = subprocess.run(command, input=json.dumps(payload), text=True, encoding="utf-8", capture_output=True,
                                       timeout=self.timeout_seconds, env=environment, cwd=self.data_dir)
        except subprocess.TimeoutExpired as exc:
            try:
                cleanup = subprocess.run(command, input=json.dumps({**payload, "action": "stop"}), text=True, encoding="utf-8",
                                         capture_output=True, timeout=30, env=environment, cwd=self.data_dir)
                if cleanup.returncode:
                    raise RuntimeError("Daemon stop failed")
            except (OSError, subprocess.SubprocessError, RuntimeError):
                _json_write(root / "runtime/recovery-required.json", {"reason": "timeout cleanup failed"})
            raise BackendError("CocoIndex worker timed out") from exc
        (root / "worker-stderr.log").write_text(completed.stderr, encoding="utf-8")
        try:
            response = json.loads(completed.stdout)
        except ValueError as exc:
            raise BackendError("CocoIndex worker returned invalid JSON; inspect worker-stderr.log") from exc
        if not isinstance(response, dict) or completed.returncode != 0 or response.get("ok") is not True:
            raise BackendError(response.get("error", "CocoIndex worker failed") if isinstance(response, dict) else "Invalid worker response")
        return response

    def index(self, unit_id: str) -> dict:
        manifest, root, model_hash = self._context(unit_id)
        with _unit_lock(root / "writer.lock"):
            status = {"schema_version": 1, "unit_id": unit_id, "recipe_hash": manifest["recipe_hash"],
                      "backend": "cocoindex-code", "model_manifest_sha256": model_hash,
                      "whitelist_files": len(manifest["files"])}
            _json_write(root / "status.json", {**status, "status": "building"})
            try:
                status.update(self._stage(manifest, root))
                response = self._run({"action": "index", "recipe": manifest["recipe"]}, root)
                # Upstream total_files is COUNT(DISTINCT file_path) in the
                # chunk table, not the number of source files processed.
                total_files = response.get("index", {}).get("total_files")
                status["total_files"] = total_files
                expected = status["whitelist_files"] - status["empty_count"]
                if type(total_files) is not int or total_files != expected:
                    raise BackendError("Indexed file count differs from non-empty snapshot whitelist; "
                                       f"expected {expected}, got {total_files}")
                self.registry.load_unit(unit_id, verify_files=True)
                status.update(status="ready", updated_at=datetime.now(timezone.utc).isoformat(), worker=response)
            except Exception as exc:
                _json_write(root / "status.json", {**status, "status": "error", "error": str(exc)})
                raise
            _json_write(root / "status.json", status)
            return status

    def search(self, unit_id: str, query: str, *, limit: int = 10, paths: list[str] | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 100:
            raise BackendError("Search requires a query and integer limit from 1 to 100")
        if paths is not None and (not isinstance(paths, list) or not all(isinstance(path, str) for path in paths)):
            raise BackendError("paths must be a list of relative path filters")
        if paths is not None and any(not path or ":" in path or "\\" in path or path.startswith("/") or ".." in path.split("/") for path in paths):
            raise BackendError("paths filters must stay within the snapshot")
        manifest, root, model_hash = self._context(unit_id)
        with _unit_lock(root / "writer.lock"):
            status = self.registry.index_status(unit_id)
            if status.get("status") != "ready" or status.get("model_manifest_sha256") != model_hash:
                raise BackendError("Index is missing, busy, stale or belongs to different model files")
            self._stage(manifest, root)
            response = self._run({"action": "search", "recipe": manifest["recipe"], "query": query, "limit": limit, "paths": paths}, root)
            entries = {entry["path"]: entry for entry in manifest["files"]}
            results = []
            rows = response.get("results")
            if not isinstance(rows, list) or len(rows) > limit:
                raise BackendError("Worker returned an invalid results list")
            for row in rows:
                if not isinstance(row, dict) or not {"file_path", "start_line", "end_line", "content", "score"} <= row.keys() or not isinstance(row["file_path"], str) or not isinstance(row["content"], str):
                    raise BackendError("Worker returned a malformed search result")
                relative = row["file_path"].replace("\\", "/")
                entry = entries.get(relative)
                if entry is None:
                    raise BackendError("Search result is outside the snapshot whitelist")
                source = self.data_dir / "sources" / unit_id / "files" / relative
                lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
                start, end = row["start_line"], row["end_line"]
                if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
                    raise BackendError("Search result has invalid source coordinates")
                content = row["content"].replace("\r\n", "\n")
                if not content.strip() or content.strip() not in "".join(lines[start-1:end]).strip():
                    raise BackendError("Search content does not match its source coordinates")
                if isinstance(row["score"], bool) or not isinstance(row["score"], (int, float)) or not math.isfinite(row["score"]):
                    raise BackendError("Search result has invalid score")
                results.append({**row, "file_path": relative, "unit_id": unit_id, "source_id": manifest["source_id"],
                                "source_sha256": entry["sha256"], "recipe_hash": manifest["recipe_hash"], "backend": "cocoindex-code"})
            self.registry.load_unit(unit_id, verify_files=True)
            return results
