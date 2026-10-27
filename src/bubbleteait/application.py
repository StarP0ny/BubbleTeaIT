"""Lazy backend selection follows each immutable unit, not the default model."""
import json

from .config import Config, ConfigError


def _model(config: Config, profile: str):
    model = next((item for item in config.models if item.name == profile and item.enabled), None)
    if model is None:
        raise ConfigError(f"Enabled embedding model profile is missing: {profile}")
    return model


def qwen_model(config: Config):
    """Validate the fixed profile's local metadata without loading the model."""
    from .backends.qwen import MODEL_REPOSITORY, MODEL_REVISION
    if config.qwen is None:
        raise ConfigError("Configure backends.qwen before controlling the Qwen worker")
    model = _model(config, config.qwen.model_profile)
    if not model.path.resolve().is_relative_to(config.models_dir.resolve()):
        raise ConfigError("Qwen model directory must be contained in models_dir")
    manifest_path = model.path / "bti-model-manifest.json"
    if not manifest_path.resolve().is_relative_to(model.path.resolve()):
        raise ConfigError("Qwen model manifest must stay inside the model directory")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError("Qwen requires a readable complete local model manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ConfigError("Qwen requires a complete local model manifest")
    if (manifest.get("repository"), manifest.get("revision")) != (MODEL_REPOSITORY, MODEL_REVISION):
        raise ConfigError("Qwen profile requires the pinned Qwen3-Embedding-0.6B repository and revision")
    return model


def make_qwen_client(config: Config):
    from .qwen_runtime import QwenRuntimeClient
    model = qwen_model(config)
    return QwenRuntimeClient(config.data_dir, config.qwen.python_executable, model.path,
                             models_dir=config.models_dir, timeout_seconds=config.qwen.timeout_seconds)


class ConfiguredBackend:
    """Route exact unit recipes; adding Qwen leaves existing E5 units usable."""

    def __init__(self, config: Config):
        from .registry import Registry
        self.config = config
        self.registry = Registry(config.data_dir)
        self._backends = {}

    def _for_unit(self, unit_id: str):
        name = self.registry.load_unit(unit_id)["recipe"]["backend"]
        if name in self._backends:
            return self._backends[name]
        config = self.config
        if name == "cocoindex-code":
            from .backends.cocoindex import CocoIndexBackend
            settings = config.cocoindex
            if settings is None:
                raise ConfigError("Configure backends.cocoindex to access this unit")
            model = _model(config, settings.model_profile)
            backend = CocoIndexBackend(config.data_dir, settings.python_executable, model.path,
                                       timeout_seconds=settings.timeout_seconds, models_dir=config.models_dir)
        elif name == "qwen-dense":
            from .backends.qwen import QwenBackend
            settings = config.qwen
            model = qwen_model(config)
            backend = QwenBackend(config.data_dir, settings.python_executable, model.path,
                                  timeout_seconds=settings.timeout_seconds, models_dir=config.models_dir)
        else:
            raise ConfigError(f"Unsupported unit backend: {name}")
        self._backends[name] = backend
        return backend

    def index(self, unit_id: str) -> dict:
        return self._for_unit(unit_id).index(unit_id)

    def search(self, unit_id: str, query: str, *, limit: int = 10, paths: list[str] | None = None) -> list[dict]:
        return self._for_unit(unit_id).search(unit_id, query, limit=limit, paths=paths)


def make_backend(config: Config):
    if config.cocoindex is None and config.qwen is None:
        raise ConfigError("Configure backends.cocoindex or backends.qwen before indexing or vector search")
    return ConfiguredBackend(config)
