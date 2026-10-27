"""MCP context facade over exact snapshots with optional automatic synchronization."""
from functools import wraps
import inspect
from typing import Annotated, Any, Literal

from .config import Config, ConfigError


def build_server(config: Config, *, workspace: str | None = None):
    from pydantic import BaseModel, ConfigDict, Field
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp_types import ToolAnnotations
    from .application import make_backend
    from .registry import Registry, RegistryError
    from .search import SearchService

    service = SearchService(config.data_dir)
    registry = Registry(config.data_dir)
    direct_services = {}
    symbol_services = {}

    class IssuedRead(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        id: Annotated[str, Field(min_length=1)]

    class SourceRead(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        source_id: Annotated[str, Field(min_length=1)] | None = None
        path: Annotated[str, Field(min_length=1)]
        start: Annotated[int, Field(ge=1)] | None = None
        end: Annotated[int, Field(ge=1)] | None = None
        project: str | None = None
        unit_id: Annotated[str, Field(strict=True, pattern=r'^u-[0-9a-f]{64}$',
                                     description='Optional exact-version guard from code_symbols; rejects a stale locator after composition publication')] | None = None

    ReadRequest = Annotated[str, Field(strict=True, min_length=1)] | IssuedRead | SourceRead
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
    project_read = ToolAnnotations(read_only_hint=config.project_config is None and workspace is None,
                                   destructive_hint=False, open_world_hint=False)

    def anticipated_errors(function):
        """Keep actionable domain diagnostics visible through the SDK error boundary."""
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapped(*args, **kwargs):
                try:
                    return await function(*args, **kwargs)
                except (RegistryError, ConfigError) as error:
                    raise ToolError(str(error)) from error
            return async_wrapped
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except (RegistryError, ConfigError) as error:
                raise ToolError(str(error)) from error
        return wrapped

    def direct_service(composition_id):
        from .direct import DirectService
        current = service.load_composition(composition_id)
        cached = direct_services.get(composition_id)
        if cached is None or cached.composition['composition_hash'] != current['composition_hash']:
            cached = DirectService(config, composition_id)
            if len(direct_services) >= 8:
                direct_services.pop(next(iter(direct_services)))
            direct_services[composition_id] = cached
        return cached
    from .direct_session import DirectSession
    session = DirectSession(config, direct_service, workspace)
    server = MCPServer('BubbleTeaIT', instructions=(
        'Search only registered compositions of exact source versions. '
        'Call context_search directly for the configured project, or read_sources for a known project-relative path. '
        'composition_id pins a version and is optional for the configured project. '
        'Use freshness=wait after changing source files or bti.toml to wait for automatic indexing. '
        'Search excerpts are already citable; read again only to expand context. '
        'Use list_sources/list_files only when discovery is needed. '
        'Read the original source before drawing conclusions. '
        'A retrieved test is not an executed test. Source roots cannot be added through search.'))

    @server.tool(structured_output=True, annotations=read_only)
    @anticipated_errors
    def project_info() -> dict[str, Any]:
        """Inspect the configured component tree and published exact version without loading models. Optional discovery: context_search/read_sources can select the configured project directly and synchronize with freshness=wait."""
        from .project import project_info as describe
        return describe(config)

    @server.tool(structured_output=True, annotations=read_only)
    @anticipated_errors
    def search(composition_id: str, query: str, limit: int = 10,
               channel: Literal['vector', 'text'] = 'vector',
               roles: list[str] | None = None, source_ids: list[str] | None = None,
               paths: list[str] | None = None) -> dict[str, Any]:
        """Retrieve source fragments from exact ready units in a composition."""
        selected = (SearchService(config.data_dir, backend=make_backend(config))
                    if channel == 'vector' else service)
        return selected.search(composition_id, query, limit=limit, channel=channel,
                              roles=roles, source_ids=source_ids, paths=paths)

    @server.tool(structured_output=True, annotations=read_only)
    @anticipated_errors
    def read_source(unit_id: str, file_path: str, start_line: int = 1,
                    end_line: int | None = None, mode: Literal['snapshot', 'working'] = 'snapshot',
                    composition_id: str | None = None) -> dict[str, Any]:
        """Read a whitelisted original source and report changes from its snapshot."""
        return service.read_source(unit_id, file_path, start_line=start_line,
            end_line=end_line, mode=mode, composition_id=composition_id)

    @server.tool(structured_output=True, annotations=read_only)
    @anticipated_errors
    def index_status(unit_id: str) -> dict[str, Any]:
        """Validate a unit's snapshot identity, recipe and current index status."""
        return registry.index_status(unit_id)

    @server.tool(structured_output=True, annotations=project_read)
    @anticipated_errors
    def list_sources(composition_id: str | None = None, query: str | None = None,
                     roles: list[str] | None = None,
                     offset: Annotated[int, Field(strict=True, ge=0)] = 0,
                     limit: Annotated[int, Field(strict=True, ge=1, le=200)] = 20,
                     max_chars: Annotated[int, Field(strict=True, ge=512, le=64000)] = 6000,
                     project: str | None = None) -> dict[str, Any]:
        """List pinned source aliases, roles and file counts without embeddings. query is a literal name/role substring. Continue with next_offset using unchanged filters and composition_hash. configuration_status describes declared metadata, not verified active build configuration. Use returned source_id in list_files/context_search/read_sources."""
        scopes, state = session.scopes(composition_id, project=project)
        if not scopes:
            return state
        if len(scopes) != 1:
            raise ConfigError('Select one project for list_sources')
        return scopes[0][1].list_sources(query=query, roles=roles,
                                                          offset=offset, limit=limit, max_chars=max_chars)

    @server.tool(structured_output=True, annotations=project_read)
    @anticipated_errors
    def list_files(composition_id: str | None = None, query: str | None = None,
                   source_ids: list[str] | None = None, paths: list[str] | None = None,
                   roles: list[str] | None = None,
                   offset: Annotated[int, Field(strict=True, ge=0)] = 0,
                   limit: Annotated[int, Field(strict=True, ge=1, le=200)] = 40,
                   max_chars: Annotated[int, Field(strict=True, ge=512, le=64000)] = 8000,
                   project: str | None = None) -> dict[str, Any]:
        """Find whitelisted snapshot paths without embeddings. query is a literal path substring: query='Core/ch32v2/' finds that subtree. paths are POSIX GLOB filters: paths=['Core/ch32v2/**'] matches files below that directory; a bare directory name matches no files. Keep source_id to distinguish same paths in different products. Continue with next_offset and unchanged filters/composition_hash. Each text file supplies a ready read request of <=120 lines for read_sources; binary files have read=null. Listing is navigation, not citable source text."""
        scopes, state = session.scopes(composition_id, project=project)
        if not scopes:
            return state
        if len(scopes) != 1:
            raise ConfigError('Select one project for list_files')
        return scopes[0][1].list_files(query=query, source_ids=source_ids,
            paths=paths, roles=roles, offset=offset, limit=limit, max_chars=max_chars)

    @server.tool(structured_output=True, annotations=project_read)
    @anticipated_errors
    def context_search(composition_id: str | None = None, query: str = '',
                       limit: Annotated[int, Field(strict=True, ge=1, le=8)] = 6,
                       max_chars: Annotated[int, Field(strict=True, ge=512, le=16000)] = 12000,
                       roles: list[str] | None = None, source_ids: list[str] | None = None,
                       paths: list[str] | None = None,
                       include_catalog: Annotated[bool | None, Field(strict=True)] = None,
                       project: str | None = None, workspace: str | None = None,
                       projects: list[str] | None = None,
                       compositions: list[dict[str, str]] | None = None,
                       freshness: Literal['available', 'wait'] = 'available',
                       timeout_ms: Annotated[int, Field(strict=True, ge=0, le=60000)] = 30000) -> dict[str, Any]:
        """Search the configured project directly; no discovery call is required. composition_id pins an exact version; workspace/projects select cross-project context. freshness=wait waits for automatic synchronization after source/config edits; updating/error contains no fresh search results. Returned original text and IDs are already citable; read_sources only expands context. roles/source_ids/paths are explicit filters, mentioning a path in query is not a filter. max_chars bounds compact JSON, coverage describes returned windows rather than selection completeness; next_read is ready to use, omitted_reads are suggestions, not evidence."""
        if not query.strip():
            raise ConfigError('query must not be empty')
        return session.search(query, composition_id=composition_id, project=project, workspace=workspace,
            projects=projects, compositions=compositions, freshness=freshness, timeout_ms=timeout_ms,
            limit=limit, max_chars=max_chars, roles=roles, source_ids=source_ids, paths=paths,
            include_catalog=(composition_id is not None if include_catalog is None else include_catalog))

    @server.tool(structured_output=True, annotations=project_read)
    @anticipated_errors
    def read_sources(composition_id: str | None = None,
                     requests: Annotated[list[ReadRequest], Field(min_length=1, max_length=20)] = None,
                     max_chars: Annotated[int, Field(strict=True, ge=512, le=24000)] = 16000,
                     include_catalog: Annotated[bool | None, Field(strict=True)] = None,
                     project: str | None = None, workspace: str | None = None,
                     freshness: Literal['available', 'wait'] = 'available',
                     timeout_ms: Annotated[int, Field(strict=True, ge=0, le=60000)] = 30000) -> dict[str, Any]:
        """Read 1–20 known paths or issued IDs without prior list/resolve calls. {path} reads the first <=120 lines; {path,start,end} reads an exact inclusive range. Without source_id paths are project-relative or absolute within declared roots; with source_id paths are relative to that source. IDs preserve their original snapshot across automatic updates; use a path with freshness=wait to read a new version. An invalid request rejects the batch; explicit ranges are never silently clipped. Output text/IDs are citable; omitted_reads are unissued suggestions. Specify project for a workspace path; cross-project IDs can be batched."""
        if not requests:
            raise ConfigError('requests must contain 1..20 reads')
        normalized = [request.model_dump(exclude_none=True) if isinstance(request, BaseModel) else request for request in requests]
        def issued(request):
            return isinstance(request, str) or isinstance(request, dict) and set(request) == {'id'}
        # Mixing an issued ID with a path would resolve the path in the old
        # preferred composition selected by the ID. Require an explicit pin.
        if composition_id is None and any(issued(request) for request in normalized) \
                and not all(issued(request) for request in normalized):
            raise ConfigError('Do not mix issued evidence IDs and path requests; pass composition_id explicitly')
        identifiers = [r if isinstance(r, str) else r.get('id') for r in normalized]
        if (composition_id is None and all(identifier in session.ledger for identifier in identifiers)
                and len({session.ledger[i][0].composition_id for i in identifiers}) > 1):
            if project or workspace or freshness != 'available':
                raise ConfigError('A batch of versioned IDs conflicts with live selectors/freshness')
            return session.read_origins(normalized, max_chars=max_chars, include_catalog=bool(include_catalog))
        names = {r['project'] for r in normalized if isinstance(r, dict) and 'project' in r}
        if len(names) > 1 or names and project and names != {project}:
            raise ConfigError('Select one project per read_sources batch')
        project = project or (next(iter(names)) if names else None)
        normalized = [{k: v for k, v in r.items() if k != 'project'} if isinstance(r, dict) else r for r in normalized]
        scopes, state = session.read_scope(normalized, composition_id, project=project, workspace=workspace,
                                           freshness=freshness, timeout_ms=timeout_ms)
        if not scopes:
            return state
        if len(scopes) != 1:
            raise ConfigError('Select one project per read_sources batch')
        name, direct, sync = scopes[0]
        if composition_id is None:
            return session.invoke(direct, 'read_many', normalized,
                {'max_chars': max_chars, 'include_catalog': bool(include_catalog)}, name, sync)
        result = direct.read_many(normalized, max_chars=max_chars,
            include_catalog=(composition_id is not None if include_catalog is None else include_catalog))
        session.remember(result, direct, name)
        return result if composition_id is not None else {**result, 'state': 'ready', 'project': name, 'sync': sync}

    @server.tool(structured_output=True, annotations=read_only)
    @anticipated_errors
    def resolve_sources(composition_id: str | None = None, ids: list[str] = None) -> dict[str, Any]:
        """Resolve <=80 issued IDs into verified unit/path/lines/SHA using their original versions. Additional reading/resolution is not required to cite search excerpts. IDs live in this MCP session; an explicit composition_id is a compatibility guard."""
        if not isinstance(ids, list) or len(ids) > 80:
            raise ConfigError('ids must be an array of at most 80 issued IDs')
        return session.resolved(ids, composition_id)

    @server.tool(structured_output=True, annotations=read_only)
    @anticipated_errors
    def watch_status(project: str | None = None, workspace: str | None = None) -> dict[str, Any]:
        """List projects watched by the shared service, queue states and errors; never loads embedding weights."""
        return session.status(project=project, workspace=workspace)

    @server.tool(structured_output=True, annotations=read_only)
    @anticipated_errors
    def sync_status(project: str | None = None, workspace: str | None = None) -> dict[str, Any]:
        """Inspect the shared synchronization queue without starting an embedding model."""
        return session.status(project=project, workspace=workspace)

    @server.tool(structured_output=True, annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
    @anticipated_errors
    def sync_project(wait: Annotated[bool, Field(strict=True)] = True,
                     timeout_ms: Annotated[int, Field(strict=True, ge=0, le=60000)] = 30000,
                     project: str | None = None, workspace: str | None = None) -> dict[str, Any]:
        """Synchronize the configured project into derived indexes; source files are never edited."""
        return session.sync(wait=wait, timeout_ms=timeout_ms, project=project, workspace=workspace)

    @server.tool(structured_output=True, annotations=project_read)
    @anticipated_errors
    async def code_symbols(operation: Literal['overview', 'definition', 'references'],
                           path: str, source_id: str | None = None, name: str | None = None,
                           limit: Annotated[int, Field(strict=True, ge=1, le=40)] = 20,
                           max_chars: Annotated[int, Field(strict=True, ge=512, le=16000)] = 8000,
                           composition_id: str | None = None, project: str | None = None,
                           freshness: Literal['available', 'wait'] = 'available',
                           timeout_ms: Annotated[int, Field(strict=True, ge=0, le=60000)] = 30000) -> dict[str, Any]:
        """Navigate exact pinned C/C++ sources with standalone read-only Serena; configure [symbols] first. Use source_id/path from list_files, and name for definition/references. Returns verified navigation locators and ready source reads, not issued evidence IDs; read_sources obtains citable excerpts. References may identify enclosing symbols rather than exact call sites. compile_context=unknown means active macros/build variant and reference completeness are unverified. Separate product compositions keep conflicting module versions apart. No LLM or embedding inference is used."""
        from .symbols import SymbolService
        if composition_id is None:
            scopes, state = session.scopes(project=project, freshness=freshness, timeout_ms=timeout_ms)
            if not scopes:
                return state
            if len(scopes) != 1:
                raise ConfigError('Select one project for code_symbols')
            _, direct, _ = scopes[0]
            composition_id = direct.composition_id
            request = direct.path_request({'path': path, **({'source_id': source_id} if source_id else {})})
            source_id, path = request['source_id'], request['path']
        elif project or freshness != 'available':
            raise ConfigError('Explicit composition_id conflicts with live selectors/freshness')
        elif source_id is None:
            request = direct_service(composition_id).path_request({'path': path})
            source_id, path = request['source_id'], request['path']
        current = service.load_composition(composition_id)
        selected = symbol_services.get(composition_id)
        if selected is None or selected.composition['composition_hash'] != current['composition_hash']:
            selected = SymbolService(config, composition_id)
            if len(symbol_services) >= 8:
                symbol_services.pop(next(iter(symbol_services)))
            symbol_services[composition_id] = selected
        return await selected.query(operation, source_id, path, name=name, limit=limit, max_chars=max_chars)

    return server
