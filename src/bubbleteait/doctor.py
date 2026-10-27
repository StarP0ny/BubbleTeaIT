"""Read-only diagnostics. Network probes require explicit opt-in."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Config, ModelConfig


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class Report:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return not any(check.status == "error" for check in self.checks)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}


def _directory(name: str, path: Path) -> Check:
    if path.exists():
        if not path.is_dir():
            return Check(name, "error", f"Expected a directory: {path}")
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            return Check(name, "error", f"Directory is not accessible for runtime data: {path}")
        return Check(name, "ok", str(path))
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        return Check(name, "error", f"Cannot create directory under {parent}: {path}")
    return Check(name, "warning", f"Not created yet: {path}; accessible ancestor: {parent}")


def _model_problem(model: ModelConfig) -> str | None:
    try:
        if not model.path.is_dir():
            return f"Model directory is missing: {model.path}"
        missing = [file for file in model.files if not (model.path / file).is_file()]
        if missing:
            return f"Missing declared model files at {model.path}: {', '.join(missing)}"
        if not model.files and not any(model.path.iterdir()):
            return f"Model directory is empty: {model.path}"
    except OSError:
        return f"Model directory cannot be inspected: {model.path}"
    return None


def run_doctor(config: Config, *, probe_endpoints: bool = False, env: Mapping[str, str] | None = None) -> Report:
    env = os.environ if env is None else env
    checks = [Check("python", "ok" if sys.version_info >= (3, 12) else "error", sys.version.split()[0]), _directory("data_dir", config.data_dir), _directory("models_dir", config.models_dir)]
    checks.append(Check("configuration", "ok", ", ".join(str(path) for path in config.loaded_files) or "Defaults; no configuration files loaded"))
    for distribution in ("cocoindex-code", "cocoindex", "mcp", "sentence-transformers"):
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            checks.append(Check(f"package:{distribution}", "skipped", "Optional distribution is not installed"))
        else:
            checks.append(Check(f"package:{distribution}", "ok", f"Installed {installed}; runtime integration not tested"))
    if not config.models:
        checks.append(Check("models", "warning", "No embedding model profiles configured"))
    for model in config.models:
        if not model.enabled:
            checks.append(Check(f"model:{model.name}", "skipped", "Disabled"))
            continue
        problem = _model_problem(model)
        if problem:
            checks.append(Check(f"model:{model.name}", "error" if model.required else "warning", problem))
        else:
            checks.append(Check(f"model:{model.name}", "ok", f"Local files present at {model.path}; runtime loading not tested"))
    for backend_name, backend in (("cocoindex", config.cocoindex), ("qwen", config.qwen)):
        if backend is None:
            continue
        errors = []
        if not shutil.which(backend.python_executable, path=env.get("PATH", os.defpath)):
            errors.append(f"Worker Python executable is missing: {backend.python_executable}")
        model = next((model for model in config.models if model.name == backend.model_profile), None)
        if model is None:
            errors.append(f"Unknown model profile: {backend.model_profile}")
        elif not model.enabled:
            errors.append(f"Selected model profile is disabled: {backend.model_profile}")
        else:
            problem = _model_problem(model)
            if problem:
                errors.append(problem)
        if backend_name == "qwen" and not errors:
            from .application import qwen_model
            try:
                qwen_model(config)
            except ValueError as exc:
                errors.append(str(exc))
        detail = "; ".join(errors) if errors else f"Worker executable and enabled model profile {backend.model_profile} are present; worker environment and model loading not tested"
        checks.append(Check(f"backend:{backend_name}", "error" if errors else "ok", detail))
    if config.llm is None:
        checks.append(Check("llm", "skipped", "No generation endpoint configured"))
    else:
        llm = config.llm
        key = env.get(llm.api_key_env, "") if llm.api_key_env else ""
        if llm.api_key_env and not key:
            checks.append(Check("llm:credentials", "error", f"Environment variable is missing: {llm.api_key_env}"))
        elif not probe_endpoints:
            checks.append(Check("llm", "skipped", f"Configured {llm.base_url}; use --probe-endpoints for GET /models (no generation)"))
        else:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            try:
                with urlopen(Request(llm.base_url + "/models", headers=headers), timeout=llm.timeout_seconds) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    raise ValueError("Expected an OpenAI-compatible models list")
                ids = [item.get("id") for item in payload["data"] if isinstance(item, dict)]
                if llm.model and llm.model not in ids:
                    checks.append(Check("llm", "error", "Endpoint responded, but configured model is not listed"))
                else:
                    checks.append(Check("llm", "ok", f"GET /models responded; {len(ids)} listed models; tool calling not tested"))
            except HTTPError as exc:
                checks.append(Check("llm", "error", f"Models endpoint returned HTTP {exc.code}"))
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                checks.append(Check("llm", "error", f"Models probe failed ({type(exc).__name__}); check endpoint and server logs"))
    for server in config.mcp:
        name = f"mcp:{server.name}"
        if not server.enabled:
            checks.append(Check(name, "skipped", "Disabled"))
            continue
        errors = []
        if server.cwd and not server.cwd.is_dir():
            errors.append(f"Working directory is missing: {server.cwd}")
        absent_env = [key for key in server.env_names if not env.get(key)]
        if absent_env:
            errors.append(f"Missing environment variables: {', '.join(absent_env)}")
        if server.transport == "stdio" and not shutil.which(server.command or "", path=env.get("PATH", os.defpath)):
            errors.append(f"Executable is missing: {server.command}")
        if errors:
            checks.append(Check(name, "error" if server.required else "warning", "; ".join(errors)))
        else:
            checks.append(Check(name, "ok", f"{server.transport} configuration available; protocol handshake not tested"))
    return Report(tuple(checks))
