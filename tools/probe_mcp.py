"""Real stdio MCP client for a registered BubbleTeaIT composition."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


def structured(result):
    if result.is_error:
        raise RuntimeError('MCP tool reported an error; inspect the saved response')
    if isinstance(result.structured_content, dict):
        return result.structured_content
    for item in result.content:
        if item.type == 'text':
            value = json.loads(item.text)
            if isinstance(value, dict):
                return value
    raise RuntimeError('Expected a structured tool result')


async def probe(root: Path, composition: str, query: str, channel: str,
                log_path: Path, *, timeout: float = 180):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    args = ['-m', 'bubbleteait', '--project-dir', str(root), 'mcp']
    params = StdioServerParameters(command=sys.executable, args=args, cwd=str(root),
        env=os.environ | {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'real_stdio_mcp', 'composition_id': composition, 'channel': channel}
    async with asyncio.timeout(timeout):
        with log_path.open('x', encoding='utf-8') as stderr:
            async with stdio_client(params, errlog=stderr) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timeout) as session:
                    initialized = await session.initialize()
                    listed = await session.list_tools()
                    names = {tool.name for tool in listed.tools}
                    if names != {'project_info', 'search', 'read_source', 'index_status', 'context_search', 'read_sources', 'resolve_sources', 'list_sources', 'list_files', 'code_symbols',
                                 'sync_project', 'sync_status', 'watch_status'}:
                        raise RuntimeError(f'Unexpected tools: {sorted(names)}')
                    report['initialize'] = initialized.model_dump(mode='json', by_alias=True)
                    report['tools'] = sorted(names)
                    searched = await session.call_tool('search', {
                        'composition_id': composition, 'query': query, 'channel': channel, 'limit': 5})
                    report['search'] = searched.model_dump(mode='json', by_alias=True)
                    hits = structured(searched)['results']
                    if not hits:
                        raise RuntimeError('Probe query returned no fixture/source evidence')
                    hit = hits[0]
                    sources = await session.call_tool('list_sources', {'composition_id': composition})
                    report['list_sources'] = sources.model_dump(mode='json', by_alias=True)
                    structured(sources)
                    files = await session.call_tool('list_files', {
                        'composition_id': composition, 'source_ids': [hit['source_id']],
                        'query': hit['file_path']})
                    report['list_files'] = files.model_dump(mode='json', by_alias=True)
                    entries = structured(files)['entries']
                    if not any(entry['path'] == hit['file_path'] and entry.get('read') for entry in entries):
                        raise RuntimeError('Navigation did not return the source path and a bounded read request')
                    status = await session.call_tool('index_status', {'unit_id': hit['unit_id']})
                    report['index_status'] = status.model_dump(mode='json', by_alias=True)
                    if structured(status)['status'] != 'ready':
                        raise RuntimeError('Result unit is not ready')
                    original = await session.call_tool('read_source', {
                        'unit_id': hit['unit_id'], 'file_path': hit['file_path'],
                        'start_line': hit['start_line'], 'end_line': hit['end_line'],
                        'composition_id': composition})
                    report['read_source'] = original.model_dump(mode='json', by_alias=True)
                    structured(original)
                    rejected = await session.call_tool('read_source', {
                        'unit_id': hit['unit_id'], 'file_path': '../outside.txt'})
                    if not rejected.is_error:
                        raise RuntimeError('Traversal was not rejected')
                    report['traversal_rejected'] = True
    report['status'] = 'passed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path.cwd())
    parser.add_argument('--composition', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--channel', choices=['vector', 'text'], default='text')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    log_path = args.output.with_suffix('.stderr.log')
    if args.output.exists() or log_path.exists():
        parser.error('Output or stderr evidence already exists; choose a new --output path')
    result = asyncio.run(probe(args.project_dir.resolve(), args.composition, args.query,
                              args.channel, log_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in (
        'kind', 'composition_id', 'channel', 'status', 'tools', 'traversal_rejected')}, indent=2))


if __name__ == '__main__':
    main()
