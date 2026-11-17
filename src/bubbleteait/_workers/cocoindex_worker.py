"""One JSON request/response, run by an isolated cocoindex-code[full] runtime.

The parent owns snapshot validation, staging and the unit writer lock. This
worker configures only its private daemon; it never uses the user's default
CocoIndex daemon and stops its own daemon before reporting completion.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def execute(request: dict) -> dict:
    runtime = Path(request["runtime_dir"]).resolve()
    workspace = Path(request["workspace"]).resolve()
    database = Path(request["database_dir"]).resolve()
    models = Path(request["models_dir"]).resolve()
    if not runtime.is_absolute() or not workspace.is_absolute():
        raise ValueError("Expected absolute worker paths")
    for path in [runtime, database, models / ".hf-cache", models / ".tmp"]:
        path.mkdir(parents=True, exist_ok=True)
    for name in list(os.environ):
        if name.startswith("COCOINDEX_CODE_"):
            del os.environ[name]
    os.environ.update({
        "COCOINDEX_CODE_DIR": str(runtime), "COCOINDEX_CODE_RUNTIME_DIR": str(runtime),
        "COCOINDEX_CODE_DB_PATH_MAPPING": f"{workspace}={database}",
        "HF_HOME": str(models / ".hf-cache"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TMP": str(models / ".tmp"), "TEMP": str(models / ".tmp"), "TMPDIR": str(models / ".tmp"),
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "PYTHONUTF8": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    })
    # These imports must follow environment setup; settings read environment.
    import msgspec
    import yaml
    from cocoindex_code import client

    version = importlib.metadata.version("cocoindex-code")
    if version != "0.2.42":
        raise ValueError(f"Expected cocoindex-code 0.2.42, installed {version}")
    if request["action"] == "stop":
        return {"ok": True, "response": msgspec.to_builtins(client.stop())}
    options = request["recipe"]["options"]
    settings = {
        "embedding": {"provider": "sentence-transformers", "model": request["model_path"], "device": "cpu",
                      "indexing_params": options.get("indexing_params", {}), "query_params": options.get("query_params", {})},
        "daemon": {"idle_timeout_minutes": 1},
    }
    (runtime / "global_settings.yml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    result = None
    operation_error = None
    try:
        if request["action"] == "index":
            progress = []
            response = client.index(str(workspace), on_progress=lambda value: progress.append(msgspec.to_builtins(value)))
            if not response.success:
                raise RuntimeError(response.message or "CocoIndex index failed")
            status = msgspec.to_builtins(client.project_status(str(workspace)))
            if (status.get("progress") or {}).get("num_errors", 0) or (progress and progress[-1].get("num_errors", 0)):
                raise RuntimeError("CocoIndex reported indexing errors")
            result = {"ok": True, "response": msgspec.to_builtins(response), "index": status,
                      "last_progress": progress[-1] if progress else None}
        elif request["action"] == "search":
            response = client.search(str(workspace), request["query"], limit=request["limit"], paths=request.get("paths"))
            if not response.success:
                raise RuntimeError(response.message or "CocoIndex search failed")
            result = {"ok": True, **msgspec.to_builtins(response)}
        else:
            raise ValueError("Unknown worker action")
    except Exception as exc:
        operation_error = exc
    finally:
        try:
            client.stop()
        except Exception as exc:
            if operation_error is None:
                operation_error = RuntimeError(f"Private daemon shutdown failed: {exc}")
    if operation_error is not None:
        raise operation_error
    return {**result, "versions": {name: importlib.metadata.version(name) for name in
            ("cocoindex-code", "cocoindex", "sentence-transformers", "torch")}}


def main() -> None:
    try:
        request = json.load(sys.stdin)
        # Imported libraries must not contaminate the stdout JSON protocol.
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(request)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
