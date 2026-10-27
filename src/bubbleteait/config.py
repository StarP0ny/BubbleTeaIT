"""Typed configuration with explicit layers and no machine-specific defaults.

Precedence: defaults < configs/bti.toml < .local/config.toml < BTI_* env.
Relative paths belong to the file that defined them. Missing implicit files
are allowed; a missing explicitly requested file is an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import tomllib
import sys
from typing import Any, Mapping
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """Configuration cannot be interpreted safely."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: Path
    required: bool = False
    enabled: bool = True
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 5.0


@dataclass(frozen=True)
class MCPConfig:
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    url: str | None = None
    enabled: bool = True
    required: bool = False
    env_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CocoIndexConfig:
    python_executable: str
    model_profile: str = "e5-small"
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class QwenConfig:
    python_executable: str
    model_profile: str = "qwen-0.6b"
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class SymbolsConfig:
    executable: str
    clangd_path: Path
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    backend: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    executable: str | None = None
    reasoning_effort: str = "low"


@dataclass(frozen=True)
class ResearchConfig:
    default_profile: str
    skill_path: Path
    profiles: tuple[ResearchProfile, ...]


@dataclass(frozen=True)
class Config:
    project_dir: Path
    data_dir: Path
    models_dir: Path
    models: tuple[ModelConfig, ...] = ()
    llm: LLMConfig | None = None
    mcp: tuple[MCPConfig, ...] = ()
    loaded_files: tuple[Path, ...] = ()
    cocoindex: CocoIndexConfig | None = None
    qwen: QwenConfig | None = None
    research: ResearchConfig | None = None
    symbols: SymbolsConfig | None = None
    project_config: Path | None = None
    service_file: Path | None = None
    sync: dict = field(default_factory=dict)
    workspaces: dict = field(default_factory=dict)
    embedding_profile: str | None = None
    embedding_override: str | None = None
    sync_overrides: dict = field(default_factory=dict)
    embedding_profiles: dict = field(default_factory=dict)


def _string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key}: expected a nonempty string")
    return value


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{key}: expected a boolean")
    return value


def _strings(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{key}: expected an array of strings")
    return tuple(_string(item, key) for item in value)


def _table(value: Any, key: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{key}: expected a table")
    unknown = value.keys() - allowed
    if unknown:
        raise ConfigError(f"{key}: unknown options: {', '.join(sorted(unknown))}")
    return value


def _expand(value: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = next(group for group in match.groups() if group is not None)
        if name not in env:
            raise ConfigError(f"Undefined environment variable in path: {name}")
        return env[name]

    return re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([^%]+)%", replace, value)


def _path(value: Any, base: Path, env: Mapping[str, str], key: str) -> Path:
    expanded = _expand(_string(value, key), env)
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _url(value: Any, key: str) -> str:
    raw = _string(value, key).rstrip("/")
    try:
        parts = urlsplit(raw)
        _ = parts.port  # Validate a malformed or out-of-range port.
    except ValueError as exc:
        raise ConfigError(f"{key}: invalid URL") from exc
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ConfigError(f"{key}: expected an absolute HTTP(S) URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ConfigError(f"{key}: credentials, query and fragment are not supported; use api_key_env")
    return raw


def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def _read(path: Path, env: Mapping[str, str]) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot load {path}: {exc}") from exc
    _table(raw, str(path), {"paths", "models", "llm", "mcp", "backends", "research", "symbols",
                           "service", "sync", "embedding", "workspaces"})
    if 'service' in raw:
        item = _table(raw['service'], 'service', {'data_dir'})
        raw.setdefault('paths', {}).update(item)
    if 'sync' in raw:
        item = _table(raw['sync'], 'sync', {'enabled', 'debounce_ms', 'reconcile_interval_seconds', 'poll_interval_seconds'})
        if 'enabled' in item:
            _boolean(item['enabled'], 'sync.enabled')
        for key, low, high in [('debounce_ms', 0, 60000), ('reconcile_interval_seconds', 1, 86400),
                               ('poll_interval_seconds', 1, 86400)]:
            if key in item and (type(item[key]) is not int or not low <= item[key] <= high):
                raise ConfigError(f'sync.{key}: expected integer in [{low}, {high}]')
    if 'embedding' in raw:
        embedding = _table(raw['embedding'], 'embedding', {'default_profile', 'profiles'})
        profiles = embedding.get('profiles', {})
        if not isinstance(profiles, dict):
            raise ConfigError('embedding.profiles: expected a table')
        selected = _string(embedding.get('default_profile'), 'embedding.default_profile')
        if selected not in profiles:
            raise ConfigError('embedding.default_profile: unknown profile')
        for name, entry in profiles.items():
            item = _table(entry, f'embedding.profiles.{name}',
                          {'backend', 'model_path', 'python_executable', 'device'})
            if item.get('backend') not in {'qwen-dense', 'cocoindex-code'}:
                raise ConfigError(f'embedding.profiles.{name}: unsupported backend')
            if item.get('device', 'auto') != 'auto':
                raise ConfigError('Only device="auto" is currently supported by embedding adapters')
            raw.setdefault('models', {})[name] = {'path': item.get('model_path')}
        # Keep both adapters configured for historical units, prefer the selected profile for its adapter.
        ordered = [name for name in profiles if name != selected] + [selected]
        for name in ordered:
            item = profiles[name]
            adapter = 'qwen' if item['backend'] == 'qwen-dense' else 'cocoindex'
            raw.setdefault('backends', {})[adapter] = {
                'python_executable': item.get('python_executable', sys.executable), 'model_profile': name}
    if 'workspaces' in raw:
        if not isinstance(raw['workspaces'], dict):
            raise ConfigError('workspaces: expected a table')
        for name, entry in raw['workspaces'].items():
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', name):
                raise ConfigError('Invalid workspace name')
            item = _table(entry, f'workspaces.{name}', {'path', 'discover', 'exclude', 'watch'})
            item['path'] = _path(item.get('path'), path.parent, env, f'workspaces.{name}.path')
            for key, default in [('discover', ['*/bti.toml']), ('exclude', [])]:
                patterns = _strings(item.get(key, default), f'workspaces.{name}.{key}')
                if any(Path(p).is_absolute() or '..' in p.split('/') or '\\' in p for p in patterns):
                    raise ConfigError('Workspace patterns must stay inside its root and use POSIX paths')
                item[key] = list(patterns)
            item['watch'] = _boolean(item.get('watch', True), f'workspaces.{name}.watch')
    if "symbols" in raw:
        symbols = _table(raw["symbols"], "symbols", {"executable", "clangd_path", "timeout_seconds"})
        if "executable" in symbols:
            command = _expand(_string(symbols["executable"], "symbols.executable"), env)
            symbols["executable"] = (str(_path(command, path.parent, env, "symbols.executable"))
                                     if "/" in command or "\\" in command else command)
        if "clangd_path" in symbols:
            symbols["clangd_path"] = _path(symbols["clangd_path"], path.parent, env, "symbols.clangd_path")
    if "research" in raw:
        research = _table(raw["research"], "research", {"default_profile", "skill_path", "profiles"})
        if "skill_path" in research:
            research["skill_path"] = _path(research["skill_path"], path.parent, env, "research.skill_path")
        profiles = research.get("profiles", {})
        if not isinstance(profiles, dict):
            raise ConfigError("research.profiles: expected a table")
        for name, entry in profiles.items():
            item = _table(entry, f"research.profiles.{name}",
                          {"backend", "model", "base_url", "api_key_env", "executable", "reasoning_effort"})
            if "executable" in item:
                command = _expand(_string(item["executable"], f"research.profiles.{name}.executable"), env)
                item["executable"] = (str(_path(command, path.parent, env, f"research.profiles.{name}.executable"))
                                      if "/" in command or "\\" in command else command)
    if "backends" in raw:
        backends = _table(raw["backends"], "backends", {"cocoindex", "qwen"})
        for name, backend in backends.items():
            item = _table(backend, f"backends.{name}", {"python_executable", "model_profile", "timeout_seconds"})
            if "python_executable" in item:
                command = _string(item["python_executable"], f"backends.{name}.python_executable")
                if "/" in command or "\\" in command:
                    item["python_executable"] = str(_path(command, path.parent, env, f"backends.{name}.python_executable"))
    if "paths" in raw:
        paths = _table(raw["paths"], "paths", {"data_dir", "models_dir"})
        for key in paths:
            paths[key] = _path(paths[key], path.parent, env, f"paths.{key}")
    for section in ("models", "mcp"):
        if section not in raw:
            continue
        if not isinstance(raw[section], dict):
            raise ConfigError(f"{section}: expected a table")
        for name, entry in raw[section].items():
            if not isinstance(entry, dict):
                raise ConfigError(f"{section}.{name}: expected a table")
            key = "path" if section == "models" else "cwd"
            if key in entry:
                entry[key] = _path(entry[key], path.parent, env, f"{section}.{name}.{key}")
            if section == "mcp" and "command" in entry:
                command = _string(entry["command"], f"mcp.{name}.command")
                if command.startswith(("./", "../", ".\\", "..\\")):
                    entry["command"] = str(_path(command, path.parent, env, f"mcp.{name}.command"))
    for name, item in raw.get('embedding', {}).get('profiles', {}).items():
        command = item.get('python_executable', sys.executable)
        item['python_executable'] = (str(_path(command, path.parent, env, f'embedding.profiles.{name}.python_executable'))
                                     if '/' in command or '\\' in command else command)
        item['model_path'] = raw['models'][name]['path']
    return raw


def load_config(
    project_dir: Path | str = ".",
    *,
    config_path: Path | str | None = None,
    local_config_path: Path | str | None = None,
    project_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    env = os.environ if env is None else env
    root = Path(project_dir).resolve()
    from .project import discover_project
    manifest = ((root / project_file).resolve() if project_file is not None else discover_project(root))
    if project_file is not None and not manifest.is_file():
        raise ConfigError(f'Cannot load project config: {manifest}')
    if manifest is not None:
        root = manifest.parent
    shared = _path(str(config_path), root, env, "config") if config_path is not None else root / "configs/bti.toml"
    local = _path(str(local_config_path), root, env, "local-config") if local_config_path is not None else root / ".local/config.toml"
    merged: dict[str, Any] = {}
    loaded: list[Path] = []
    user_config = (Path(env['LOCALAPPDATA']) / 'BubbleTeaIT/config.toml' if 'LOCALAPPDATA' in env else
                   Path(env['XDG_CONFIG_HOME']) / 'bubbleteait/config.toml' if 'XDG_CONFIG_HOME' in env else None)
    service_file = user_config.with_name('config-service.toml') if user_config is not None else None
    if config_path is not None:
        # An explicitly selected new service config is authoritative; legacy configs retain their layers.
        candidate = _read(shared, env)
        if any(key in candidate for key in ('service', 'embedding', 'sync', 'workspaces')) or shared.name == 'config-service.toml':
            service_file = shared
    if service_file is not None and service_file.exists():
        layers = [(service_file, True)]
    else:
        service_file = None
        layers = ([(user_config, False)] if user_config is not None else []) + [(shared, config_path is not None), (local, local_config_path is not None)]
    for path, explicit in layers:
        if path.exists() or explicit:
            _merge(merged, _read(path, env))
            loaded.append(path)

    paths = merged.get("paths", {})
    if "BTI_DATA_DIR" in env:
        default_data = _path(env["BTI_DATA_DIR"], root, env, "BTI_DATA_DIR")
    elif "data_dir" in paths:
        default_data = paths["data_dir"]
    elif "LOCALAPPDATA" in env:
        default_data = Path(env["LOCALAPPDATA"]) / "BubbleTeaIT"
    elif "XDG_DATA_HOME" in env:
        default_data = Path(env["XDG_DATA_HOME"]) / "bubbleteait"
    else:
        default_data = Path.home() / ".local/share/bubbleteait"
    data_dir = default_data.resolve()
    models_dir = _path(env["BTI_MODELS_DIR"], root, env, "BTI_MODELS_DIR") if "BTI_MODELS_DIR" in env else paths.get("models_dir", data_dir / "models")

    models: list[ModelConfig] = []
    for name, item in merged.get("models", {}).items():
        _table(item, f"models.{name}", {"path", "required", "enabled", "files"})
        files = _strings(item.get("files", []), f"models.{name}.files")
        if any(Path(file).is_absolute() or ".." in Path(file).parts for file in files):
            raise ConfigError(f"models.{name}.files: expected paths contained in the model directory")
        models.append(ModelConfig(name, item.get("path", models_dir / name), _boolean(item.get("required", False), f"models.{name}.required"), _boolean(item.get("enabled", True), f"models.{name}.enabled"), files))

    llm = None
    if "llm" in merged:
        item = _table(merged["llm"], "llm", {"base_url", "model", "api_key_env", "timeout_seconds"})
        timeout = item.get("timeout_seconds", 5.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ConfigError("llm.timeout_seconds: expected a number in (0, 300]")
        llm = LLMConfig(_url(item.get("base_url"), "llm.base_url"), _string(item["model"], "llm.model") if "model" in item else None, _string(item["api_key_env"], "llm.api_key_env") if "api_key_env" in item else None, float(timeout))

    mcp: list[MCPConfig] = []
    for name, item in merged.get("mcp", {}).items():
        _table(item, f"mcp.{name}", {"transport", "command", "args", "cwd", "url", "enabled", "required", "env_names"})
        transport = item.get("transport", "stdio")
        if transport not in {"stdio", "http", "sse"}:
            raise ConfigError(f"mcp.{name}.transport: expected stdio, http or sse")
        command = _string(item["command"], f"mcp.{name}.command") if "command" in item else None
        url = _url(item["url"], f"mcp.{name}.url") if "url" in item else None
        enabled = _boolean(item.get("enabled", True), f"mcp.{name}.enabled")
        if enabled and ((transport == "stdio" and not command) or (transport != "stdio" and not url)):
            raise ConfigError(f"mcp.{name}: {transport} requires {'command' if transport == 'stdio' else 'url'}")
        mcp.append(MCPConfig(name, transport, command, _strings(item.get("args", []), f"mcp.{name}.args"), item.get("cwd"), url, enabled, _boolean(item.get("required", False), f"mcp.{name}.required"), _strings(item.get("env_names", []), f"mcp.{name}.env_names")))
    selected_backends = {}
    for name, cls, default_profile in (("cocoindex", CocoIndexConfig, "e5-small"),
                                       ("qwen", QwenConfig, "qwen-0.6b")):
        if name not in merged.get("backends", {}):
            continue
        item = merged["backends"][name]
        timeout = item.get("timeout_seconds", 600.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
            raise ConfigError(f"backends.{name}.timeout_seconds: expected a number in (0, 3600]")
        selected_backends[name] = cls(
            _string(item.get("python_executable"), f"backends.{name}.python_executable"),
            _string(item.get("model_profile", default_profile), f"backends.{name}.model_profile"),
            float(timeout))
    research = None
    if "research" in merged:
        item = merged["research"]
        default_profile = _string(item.get("default_profile"), "research.default_profile")
        skill_path = item.get("skill_path")
        if not isinstance(skill_path, Path):
            raise ConfigError("research.skill_path: expected an explicit skill file path")
        profiles = []
        for name, profile in item.get("profiles", {}).items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
                raise ConfigError("research profile names must be simple identifiers")
            key = f"research.profiles.{name}"
            backend = _string(profile.get("backend"), f"{key}.backend")
            if backend not in {"codex", "openai-compatible", "codex-native", "openai-native"}:
                raise ConfigError(f"{key}.backend: expected codex, openai-compatible, codex-native or openai-native")
            model = _string(profile.get("model"), f"{key}.model")
            base_url = _url(profile["base_url"], f"{key}.base_url") if "base_url" in profile else None
            api_key_env = _string(profile["api_key_env"], f"{key}.api_key_env") if "api_key_env" in profile else None
            if api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
                raise ConfigError(f"{key}.api_key_env: expected an environment variable name")
            executable = profile.get("executable")
            if backend in {"codex", "codex-native"} and (not executable or base_url is not None or api_key_env is not None):
                raise ConfigError(f"{key}: codex requires executable and uses its own configured authentication")
            if backend in {"openai-compatible", "openai-native"} and (base_url is None or executable is not None):
                raise ConfigError(f"{key}: openai-compatible requires base_url and does not use executable")
            effort = _string(profile.get("reasoning_effort", "low"), f"{key}.reasoning_effort")
            if effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
                raise ConfigError(f"{key}.reasoning_effort: unsupported effort value")
            profiles.append(ResearchProfile(name, backend, model, base_url, api_key_env, executable, effort))
        if default_profile not in {profile.name for profile in profiles}:
            raise ConfigError("research.default_profile must select a configured research profile")
        research = ResearchConfig(default_profile, skill_path, tuple(profiles))
    symbols = None
    if "symbols" in merged:
        item = merged["symbols"]
        timeout = item.get("timeout_seconds", 120.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
            raise ConfigError("symbols.timeout_seconds: expected a number in (0, 600]")
        clangd = item.get("clangd_path")
        if not isinstance(clangd, Path):
            raise ConfigError("symbols.clangd_path: expected an explicit executable path")
        symbols = SymbolsConfig(_string(item.get("executable"), "symbols.executable"), clangd, float(timeout))
    return Config(root, data_dir, models_dir, tuple(models), llm, tuple(mcp), tuple(loaded),
                  selected_backends.get("cocoindex"), selected_backends.get("qwen"), research, symbols, manifest,
                  service_file, merged.get('sync', {}), merged.get('workspaces', {}),
                  merged.get('embedding', {}).get('default_profile'),
                  embedding_profiles=merged.get('embedding', {}).get('profiles', {}))
