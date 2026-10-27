"""Command line entry point; unavailable capabilities are not advertised."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .config import ConfigError, load_config
from .doctor import run_doctor
from .layout import as_json, initialize_data


def _recipe(config, filename=None):
    from .registry import IndexRecipe, RegistryError
    if filename:
        try:
            recipe = IndexRecipe(**json.loads(Path(filename).read_text(encoding="utf-8")))
        except (TypeError, ValueError) as exc:
            raise RegistryError("Recipe file must contain IndexRecipe fields") from exc
        if recipe.backend == "qwen-dense":
            from .application import qwen_model
            from .backends.qwen import validate_recipe
            qwen_model(config)
            validate_recipe(recipe)
        return recipe
    if config.qwen is not None and (config.embedding_profile is None or config.embedding_profile == config.qwen.model_profile):
        from .application import qwen_model
        from .backends.qwen import default_recipe
        qwen_model(config)
        return default_recipe()
    profile = config.cocoindex.model_profile if config.cocoindex else "e5-small"
    model = next((item for item in config.models if item.name == profile and item.enabled), None)
    if model is None:
        raise ConfigError(f"Enabled embedding model profile is missing: {profile}")
    manifest = json.loads((model.path / "bti-model-manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ConfigError("The embedding model does not have a complete manifest")
    if any(not isinstance(manifest.get(key), str) or not manifest[key] for key in ("repository", "revision")):
        raise ConfigError("The embedding manifest must identify its repository and revision")
    if manifest["repository"] != "intfloat/multilingual-e5-small":
        raise ConfigError("The default recipe is E5-specific; supply --recipe for another model")
    return IndexRecipe("cocoindex-code", manifest["repository"], manifest["revision"],
                       "cocoindex-code-default@0.2.42", options={
                           "indexing_params": {"prompt_name": "passage"},
                           "query_params": {"prompt_name": "query"}})


def _backend(config):
    from .application import make_backend
    return make_backend(config)


def _source_command(args, config):
    from .registry import Registry, SourceSpec
    registry = Registry(config.data_dir)
    if args.command == "source":
        if args.source_command == "list":
            return registry.list_sources()
        return registry.register(SourceSpec(args.source_id, args.kind, Path(args.path),
            include=tuple(args.include or ["**/*"]), exclude=tuple(args.exclude or []), role=args.role,
            respect_gitignore=args.respect_gitignore, attach_paths=tuple(args.attach_path or [])))
    if args.command == "index":
        manifest = registry.snapshot(args.source_id, _recipe(config, args.recipe))
        if args.snapshot_only:
            return manifest
        return _backend(config).index(manifest["unit_id"])
    if args.command == "compose":
        configuration = json.loads(Path(args.configuration).read_text(encoding="utf-8")) if args.configuration else {}
        if not isinstance(configuration, dict):
            raise ConfigError("Composition configuration must be a JSON object")
        return registry.compose(args.composition_id, args.unit, configuration=configuration,
                                require_ready=not args.draft)
    if args.command == "status":
        return registry.index_status(args.unit)
    if args.command == "symbols":
        import asyncio
        from .symbols import SymbolService
        return asyncio.run(SymbolService(config, args.composition_id).query(
            args.operation, args.source_id, args.path, name=args.name,
            limit=args.limit, max_chars=args.max_chars))
    if args.command in {"list-sources", "list-files"}:
        from .direct import DirectService
        direct = DirectService(config, args.composition_id)
        common = dict(query=args.query, roles=args.role, offset=args.offset,
                      limit=args.limit, max_chars=args.max_chars)
        if args.command == "list-sources":
            return direct.list_sources(**common)
        return direct.list_files(**common, source_ids=args.source_id, paths=args.path)
    if args.command in {"search", "read-source"}:
        from .search import SearchService
        backend = _backend(config) if args.command == "search" and args.channel == "vector" else None
        service = SearchService(config.data_dir, backend=backend)
        if args.command == "search":
            return service.search(args.composition_id, args.query, limit=args.limit,
                channel=args.channel, roles=args.role, source_ids=args.source_id, paths=args.path)
        return service.read_source(args.unit, args.path, start_line=args.start_line,
            end_line=args.end_line, mode=args.mode, composition_id=args.composition_id)
    raise ConfigError("Unsupported source command")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bti", description="BubbleTeaIT: context tools for AI agents")
    parser.add_argument("--version", action="version", version=f"BubbleTeaIT {__version__}")
    parser.add_argument("--project-dir", default=".", help="Configuration root (default: current directory)")
    parser.add_argument("--config", help="Shared TOML configuration; explicit paths must exist")
    parser.add_argument("--local-config", help="Local TOML overrides; default: .local/config.toml")
    parser.add_argument("--project-file", help="Explicit component-tree manifest (default: nearest bti.toml)")
    commands = parser.add_subparsers(dest="command", required=True)
    from .service_cli import add_commands
    add_commands(commands)
    for name, help_text in (("plan", "Preview hierarchical source selection and UTF-8 errors without models"),
                            ("sync", "Index the configured component tree and publish a ready composition"),
                            ("project-info", "Show selected project and exact published composition")):
        project_command = commands.add_parser(name, help=help_text)
        project_command.add_argument("--json", action="store_true")
        if name == "sync":
            project_command.add_argument("--recipe", help="Explicit index recipe JSON")
            project_command.add_argument('--wait', action='store_true', help='Wait for the shared coordinator')
            project_command.add_argument('--timeout-ms', type=int, help='Total CLI wait limit; default waits until complete')
    doctor = commands.add_parser("doctor", help="Check configuration without launching models or MCP servers")
    doctor.add_argument("--json", action="store_true", help="Output structured JSON")
    doctor.add_argument("--probe-endpoints", action="store_true", help="Probe generation provider via GET /models; never request completion")
    initialize = commands.add_parser("init", help="Create runtime data directories; preserve existing contents")
    initialize.add_argument("--json", action="store_true", help="Output structured JSON")
    source = commands.add_parser("source", help="Register source roots and explicit file selection")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    source_add = source_commands.add_parser("add")
    source_add.add_argument("source_id")
    source_add.add_argument("--kind", choices=["project", "module", "documents"], required=True)
    source_add.add_argument("--path", required=True)
    source_add.add_argument("--include", action="append", help="Repeatable POSIX relative glob")
    source_add.add_argument("--exclude", action="append")
    source_add.add_argument("--respect-gitignore", action=argparse.BooleanOptionalAction, default=True,
                            help="Respect Git ignore rules (default); include globs do not override ignores")
    source_add.add_argument("--attach-path", action="append",
                            help="Explicit file/subtree within source; overrides ignores and automatic exclusions, not --exclude")
    source_add.add_argument("--role", default="unknown")
    source_add.add_argument("--json", action="store_true")
    source_list = source_commands.add_parser("list")
    source_list.add_argument("--json", action="store_true")
    index = commands.add_parser("index", help="Snapshot a registered source, then build its index")
    index.add_argument("source_id")
    index.add_argument("--snapshot-only", action="store_true", help="Capture bytes without invoking embeddings")
    index.add_argument("--recipe", help="Explicit JSON index recipe (default: Qwen when configured, otherwise E5)")
    index.add_argument("--json", action="store_true")
    compose = commands.add_parser("compose", help="Publish exact references to ready indexes")
    compose.add_argument("composition_id")
    compose.add_argument("--unit", action="append", required=True)
    compose.add_argument("--configuration", help="JSON object with product/configuration context")
    compose.add_argument("--draft", action="store_true", help="Save unready references as a draft")
    compose.add_argument("--json", action="store_true")
    status = commands.add_parser("status", help="Validate a snapshot and its backend status")
    status.add_argument("--unit", required=True)
    status.add_argument("--json", action="store_true")
    search = commands.add_parser("search", help="Retrieve context from an exact ready composition")
    search.add_argument("query")
    search.add_argument("--project", "--composition", dest="composition_id")
    search.add_argument("--channel", choices=["vector", "text"], default="vector")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--role", action="append")
    search.add_argument("--source-id", action="append")
    search.add_argument("--path", action="append")
    search.add_argument("--json", action="store_true")
    search.add_argument('--freshness', choices=['available', 'wait'])
    search.add_argument('--workspace', help='Search a configured workspace through the shared service')
    search.add_argument('--projects', nargs='+', help='Restrict workspace search to named projects')
    search.add_argument('--max-chars', type=int, default=12000, help='Compact workspace response budget')
    symbols = commands.add_parser("symbols", help="Navigate pinned code with standalone read-only Serena, without LLMs")
    symbols.add_argument("--project", "--composition", dest="composition_id")
    symbols.add_argument("--source-id", required=True)
    symbols.add_argument("--path", required=True)
    symbols.add_argument("--operation", choices=["overview", "definition", "references"], default="definition")
    symbols.add_argument("--name")
    symbols.add_argument("--limit", type=int, default=20)
    symbols.add_argument("--max-chars", type=int, default=8000)
    symbols.add_argument("--json", action="store_true")
    for name, default_limit, default_chars in (("list-sources", 20, 6000), ("list-files", 40, 8000)):
        navigation = commands.add_parser(name, help="Browse pinned sources/files without loading an embedding model")
        navigation.add_argument("--project", "--composition", dest="composition_id")
        navigation.add_argument("--query", help="Literal case-insensitive source/path substring")
        navigation.add_argument("--role", action="append")
        navigation.add_argument("--offset", type=int, default=0)
        navigation.add_argument("--limit", type=int, default=default_limit)
        navigation.add_argument("--max-chars", type=int, default=default_chars)
        if name == "list-files":
            navigation.add_argument("--source-id", action="append", help="Logical source ID or returned alias")
            navigation.add_argument("--path", action="append", help="POSIX GLOB filter")
        navigation.add_argument("--json", action="store_true")
    read = commands.add_parser("read-source", help="Read original snapshot/current text with hash comparison")
    read.add_argument("--unit", required=True)
    read.add_argument("--path", required=True)
    read.add_argument("--start-line", type=int, default=1)
    read.add_argument("--end-line", type=int)
    read.add_argument("--mode", choices=["snapshot", "working"], default="snapshot")
    read.add_argument("--project", "--composition", dest="composition_id")
    read.add_argument("--json", action="store_true")
    worker = commands.add_parser("worker", help="Control the configured persistent Qwen worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    for action in ("status", "load", "unload", "stop"):
        control = worker_commands.add_parser(action)
        control.add_argument("--json", action="store_true")
    mcp = commands.add_parser("mcp", help="Run the stdio MCP facade (requires the mcp extra)")
    mcp.add_argument('--workspace', help='Configured workspace scope')
    args = parser.parse_args(argv)
    try:
        configuration_root = (getattr(args, 'path', None) or args.project_dir
                              if args.command in {'watch', 'unwatch'} else args.project_dir)
        config = load_config(configuration_root, config_path=getattr(args, 'service_config', None) or args.config, local_config_path=args.local_config,
                             project_file=args.project_file)
        if args.command in {'service', 'watch', 'unwatch', 'workspace'}:
            from .service_cli import run
            return run(args, config)
        if args.command == 'sync' and not args.recipe:
            if config.project_config is None:
                raise ConfigError('No bti.toml found')
            from .service import ServiceClient
            from .service_cli import output
            client = ServiceClient(config)
            import time
            if args.timeout_ms is not None and args.timeout_ms < 0:
                raise ConfigError('--timeout-ms must be nonnegative')
            deadline = None if args.timeout_ms is None else time.monotonic() + args.timeout_ms / 1000
            try:
                while True:
                    remaining = 60000 if deadline is None else max(0, min(60000, int((deadline - time.monotonic()) * 1000)))
                    result = client.select(config.project_config.parent, freshness='wait', timeout_ms=remaining, manual_sync=True)
                    if result['state'] not in {'updating', 'initializing'} or deadline is not None and time.monotonic() >= deadline:
                        break
            finally:
                client.close()
            output(result, args.json)
            return 0 if result['state'] == 'ready' else 2
        if args.command == 'sync' and args.wait and args.recipe:
            raise ConfigError('--recipe is supported by standalone sync; configure a service embedding profile for --wait')
        if args.command == 'search' and args.workspace:
            if args.composition_id:
                raise ConfigError('--workspace conflicts with --composition/--project')
            from .direct import DirectService
            from .direct_session import DirectSession
            from .service_cli import output
            session = DirectSession(config, lambda cid: DirectService(config, cid), workspace=args.workspace)
            try:
                result = session.search(args.query, projects=args.projects, freshness=args.freshness or 'available',
                    limit=args.limit, max_chars=args.max_chars, roles=args.role, source_ids=args.source_id,
                    paths=args.path, include_catalog=False)
            finally:
                session.client.close()
            output(result, args.json)
            return 0 if result.get('state') == 'ready' else 2
        if args.command == 'search' and args.projects:
            raise ConfigError('--projects requires --workspace')
        if args.command == 'search' and args.composition_id and args.freshness:
            raise ConfigError('--freshness applies to live project search; omit --composition/--project')
        if args.command in {"plan", "sync", "project-info"}:
            from .project import load_project, project_info, sync_project
            if args.command == "project-info":
                result = project_info(config)
            else:
                if config.project_config is None:
                    raise ConfigError("No bti.toml found; use --project-file")
                project = load_project(config.project_config)
                result = project.plan(config.data_dir)
                if args.command == "sync":
                    if not result['ready']:
                        print(json.dumps(result, ensure_ascii=False, indent=2))
                        return 2
                    print(f"Preflight: {len(result['sources'])} sources, {result['file_count']} files; indexing", file=sys.stderr)
                    result = sync_project(project, config, _recipe(config, args.recipe), _backend(config), plan=result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get('ready', True) or args.command == 'project-info' else 2
        needs_composition = args.command in {"search", "symbols", "list-sources", "list-files"}
        if needs_composition and not args.composition_id:
            if args.command == 'search' and args.freshness:
                if config.project_config is None:
                    raise ConfigError('No bti.toml found')
                from .service import ServiceClient
                client = ServiceClient(config)
                try:
                    selected = client.select(config.project_config.parent, freshness=args.freshness)
                finally:
                    client.close()
                if selected['state'] != 'ready':
                    from .service_cli import output
                    output(selected, args.json)
                    return 2
                args.composition_id = selected['composition_id']
            from .project import project_info
            info = project_info(config)
            if not info.get('ready'):
                raise ConfigError("Select --project/--composition or publish bti.toml with bti sync")
            if info.get('config_changed'):
                raise ConfigError("Project configuration changed; run bti plan/sync or select an explicit --composition")
            args.composition_id = info['composition_id']
        if args.command == "mcp":
            try:
                from .mcpserver import build_server
                build_server(config, workspace=args.workspace).run(transport="stdio")
            except ImportError as exc:
                raise ConfigError("MCP dependencies are missing; install the 'mcp' extra") from exc
            return 0
        if args.command == "worker":
            from .application import make_qwen_client
            action = {"status": "health", "stop": "shutdown"}.get(args.worker_command, args.worker_command)
            response = make_qwen_client(config).request(action)
            print(json.dumps(response, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"source", "index", "compose", "status", "search", "read-source", "list-sources", "list-files", "symbols"}:
            print(json.dumps(_source_command(args, config), ensure_ascii=False, indent=2))
            return 0
        if args.command == "init":
            directories = initialize_data(config)
            if args.json:
                print(json.dumps(as_json(directories), ensure_ascii=False, indent=2))
            else:
                for directory in directories:
                    state = "created" if directory.created else "exists"
                    print(f"[{state}] {directory.name}: {directory.path}")
            return 0
        report = run_doctor(config, probe_endpoints=args.probe_endpoints)
    except (ValueError, OSError, RuntimeError) as exc:
        name = "initialization" if args.command == "init" and isinstance(exc, OSError) else "configuration"
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "checks": [{"name": name, "status": "error", "detail": str(exc)}]}, ensure_ascii=False, indent=2))
        else:
            print(f"[error] {name}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        for check in report.checks:
            print(f"[{check.status}] {check.name}: {check.detail}")
        print("Doctor: ready for configured checks" if report.ok else "Doctor: errors found")
    return 0 if report.ok else 1
