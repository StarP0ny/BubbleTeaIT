"""One public MCP connection for coordinated live engines."""
from functools import wraps
from typing import Any, Literal

from .config import ConfigError
from .engine_client import EngineClient


def build_engine_server(config, workspace=None):
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp_types import ToolAnnotations
    client = EngineClient(config, workspace)
    server = MCPServer('BubbleTeaIT', instructions=(
        'Use context_search directly; project discovery is optional. The default project is configured by the workspace. '
        'Select projects explicitly for cross-project comparisons. Results identify the project, original path, lines and SHA. '
        'Pass an excerpt’s next_read directly to read_sources for its function or neighboring lines at the indexed SHA. '
        'Use freshness=wait after edits to wait for index reconciliation. A stale excerpt is not the current file. '
        'Use code_symbols for definitions/references; compilation-database coverage and parser limitations are reported. '
        'Retrieved tests are not executed tests. Tools do not edit project sources.'))
    annotations = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

    def errors(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except (ValueError, OSError) as exc:
                raise ToolError(str(exc)) from exc
        return invoke

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def project_info(project: str | None = None) -> dict[str, Any]:
        """Describe selected projects and engines without loading models."""
        return client.call('info', project=project)

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def context_search(query: str, project: str | None = None, projects: list[str] | None = None,
                       limit: int = 6, max_chars: int = 12000, roles: list[str] | None = None,
                       source_ids: list[str] | None = None, paths: list[str] | None = None,
                       freshness: Literal['available', 'wait'] = 'available', timeout_ms: int = 30000) -> dict[str, Any]:
        """Search stock CocoIndex Code indexes through the shared coordinator. Projects retain separate rankings.

        Excerpts already contain original text, paths, inclusive lines and source SHA. Discovery is optional.
        available may return a labelled older index; wait requests byte reconciliation. Omitted projects include errors.
        """
        return client.call('search', query=query, project=project, projects=projects, limit=limit, max_chars=max_chars,
            roles=roles, source_ids=source_ids, paths=paths, freshness=freshness, timeout_ms=timeout_ms)

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def read_sources(requests: list[dict[str, Any] | str], project: str | None = None,
                     max_chars: int = 16000) -> dict[str, Any]:
        """Read current selected paths without embeddings, or re-read issued IDs at their original source version.

        Pass the complete next_read object from a search hit as tool arguments for its pinned context.
        Path requests: {path, source_id?, start?, end?, sha256?}. SHA rejects a stale navigation locator.
        ID requests: an issued string or {id}. Do not mix IDs and paths. IDs belong to this MCP session.
        """
        if any(isinstance(r, dict) and 'anchor_id' in r for r in requests):
            if not project or not all(isinstance(r, dict) and isinstance(r.get('anchor_id'), str) and 'id' not in r for r in requests):
                raise ConfigError('Pinned reads require a project and a locator with anchor_id for every request')
            return client.call('read', project=project, ids=[r['anchor_id'] for r in requests],
                               anchors=requests, max_chars=max_chars)
        identifiers = [r if isinstance(r, str) else r.get('id') for r in requests]
        if any(identifiers):
            if not all(identifiers) or project:
                raise ConfigError('Do not mix evidence IDs with live path/project selectors')
            return client.call('read', ids=identifiers, max_chars=max_chars)
        return client.call('read', requests=requests, project=project, max_chars=max_chars)

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def resolve_sources(ids: list[str]) -> dict[str, Any]:
        """Resolve issued evidence IDs to their original project/path/lines/SHA."""
        return client.call('resolve', ids=ids)

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def list_files(project: str | None = None, paths: list[str] | None = None,
                   roles: list[str] | None = None, source_ids: list[str] | None = None,
                   offset: int = 0, limit: int = 40) -> dict[str, Any]:
        """Browse selected original files without starting the embedding model."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise ConfigError('Invalid offset/limit')
        result = client.call('files', project=project, paths=paths, roles=roles, source_ids=source_ids)
        files = result['files']
        return {**result, 'files': files[offset:offset+limit], 'total': len(files)}

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def code_symbols(operation: Literal['overview', 'definition', 'references'], path: str,
                     project: str | None = None, source_id: str | None = None, name: str | None = None,
                     limit: int = 20, max_chars: int = 8000) -> dict[str, Any]:
        """Use a persistent, isolated Serena session with configured clangd/ccls and real compile commands.

        Returns source-verified navigation locators and ready reads. Compilation database does not prove a successful build.
        """
        return client.call('symbols', symbol_operation=operation, path=path, project=project,
                           source_id=source_id, name=name, limit=limit, max_chars=max_chars)

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def watch_status() -> dict[str, Any]:
        """Inspect watched projects and engine states without loading models."""
        return client.status()

    @server.tool(structured_output=True, annotations=annotations)
    @errors
    def sync_status() -> dict[str, Any]:
        """Inspect the shared coordinator queue and errors."""
        return client.status()

    @server.tool(structured_output=True,
                 annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
    @errors
    def sync_project(project: str | None = None, projects: list[str] | None = None,
                     wait: bool = True, timeout_ms: int = 30000) -> dict[str, Any]:
        """Request incremental reconciliation; modifies derived data only."""
        return client.call('sync', project=project, projects=projects,
                           freshness='wait' if wait else 'available', timeout_ms=timeout_ms)

    return server
