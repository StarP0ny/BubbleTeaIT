"""Opt-in real CLI/MCP acceptance against disposable projects and a private service.

Uses installed engines and local weights from --config. Never edits a consumer
checkout or shares the configured service/index namespace.
"""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from bubbleteait.config import load_config
from bubbleteait.service import ServiceClient


def toml_text(mapping):
    lines = []
    def section(table, prefix=()):
        if prefix:
            lines.append('[' + '.'.join(json.dumps(k) for k in prefix) + ']')
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(json.dumps(key) + '=' + json.dumps(value, default=str, ensure_ascii=False))
        lines.append('')
        for key, value in table.items():
            if isinstance(value, dict):
                section(value, (*prefix, key))
    section(mapping)
    return '\n'.join(lines)


def decoded(result):
    if getattr(result, 'is_error', False):
        raise AssertionError(str(result.content))
    value = getattr(result, 'structured_content', None)
    if value is not None:
        return value
    return json.loads(next(c.text for c in result.content if getattr(c, 'type', None) == 'text'))


async def probe(config, filename, primary, reference, compiler):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable, args=['-m', 'bubbleteait', '--config', str(filename),
        '--project-dir', str(primary), 'mcp', '--workspace', 'work'], env=os.environ.copy())
    checks = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            checks['tools'] = sorted(t.name for t in tools.tools)
            async def call(tool_name, **arguments):
                return decoded(await session.call_tool(tool_name, arguments))
            current = await call('read_sources', requests=[{'path': 'main.c'}])
            assert current['excerpts'] and '37' in current['excerpts'][0]['text']
            state = await call('watch_status')
            assert not state['engines']['cocoindex']['bridge_running']
            checks['read_without_embeddings'] = True
            start = time.monotonic()
            first = await call('context_search', query='Read the configured threshold', freshness='wait', timeout_ms=60000)
            assert first['state'] == 'ready', first
            assert {e['project'] for e in first['excerpts']} == {'controller'}
            identifier = first['excerpts'][0]['id']
            checks['first_search_seconds'] = round(time.monotonic()-start, 3)
            cross = await call('context_search', query='Read the configured threshold',
                projects=['controller', 'reference'], freshness='wait', timeout_ms=60000)
            assert cross['state'] == 'ready', cross
            assert {e['project'] for e in cross['excerpts']} == {'controller', 'reference'}
            checks['cross_project_search'] = True
            before = (await call('watch_status'))['engines']['cocoindex']['bridge_pid']
            cli = await asyncio.to_thread(subprocess.run, [sys.executable, '-m', 'bubbleteait', '--config', str(filename),
                '--project-dir', str(primary), 'search', 'threshold', '--workspace', 'work', '--projects',
                'controller', 'reference', '--freshness', 'wait'], capture_output=True, text=True, encoding='utf-8', timeout=90)
            assert cli.returncode == 0, cli.stderr + cli.stdout
            assert json.loads(cli.stdout)['state'] == 'ready'
            assert (await call('watch_status'))['engines']['cocoindex']['bridge_pid'] == before
            checks['cli_and_mcp_share_runtime'] = True
            async with stdio_client(params) as (read2, write2):
                async with ClientSession(read2, write2) as second:
                    await second.initialize()
                    refused = await second.call_tool('read_sources', {'requests': [identifier]})
                    assert getattr(refused, 'is_error', False)
                    foreign = await second.call_tool('read_sources', first['excerpts'][0]['next_read'])
                    assert getattr(foreign, 'is_error', False)
                    other = decoded(await second.call_tool('context_search', {'query': 'threshold', 'freshness': 'wait'}))
                    assert other['state'] == 'ready'
                    assert (await call('watch_status'))['engines']['cocoindex']['bridge_pid'] == before
            checks['two_clients_and_session_id_isolation'] = True
            data = (primary / 'main.c').read_text().replace('37', '42')
            (primary / 'main.c').write_text(data, encoding='utf-8')
            (primary / 'added.c').write_text('int added_threshold = 99;\n', encoding='utf-8')
            edited = await call('context_search', query='threshold', freshness='wait', timeout_ms=60000)
            assert edited['state'] == 'ready', edited
            files = await call('list_files')
            assert any(f['path'] == 'added.c' for f in files['files'])
            old = await call('read_sources', requests=[identifier])
            assert old['excerpts'][0]['source_changed']
            assert '37' in old['excerpts'][0]['text']
            expanded = (await call('read_sources', **first['excerpts'][0]['next_read']))['excerpts'][0]
            assert expanded['source_changed'] and expanded['generation'] == first['excerpts'][0]['generation']
            assert '37' in expanded['text'] and expanded['sha256'] == first['excerpts'][0]['sha256']
            (primary / 'added.c').unlink()
            deleted = await call('context_search', query='threshold', freshness='wait', timeout_ms=60000)
            assert deleted['state'] == 'ready'
            assert all(e['path'] != 'added.c' for e in deleted['excerpts'])
            checks['add_edit_delete_and_pinned_read'] = True
            if 'serena' in config.engines:
                defined = await call('code_symbols', operation='definition', path='main.c', name='read_threshold')
                assert defined['results'], defined
                again = await call('code_symbols', operation='references', path='main.c', name='threshold')
                assert again['results'], again
                assert again['session_reused'], again
                checks['serena_definition_references_and_reuse'] = True
                checks['language_server'] = defined['server']
    client = ServiceClient(config)
    client.request('stop')
    deadline = time.monotonic() + 60
    while client.probe() and time.monotonic() < deadline:
        await asyncio.sleep(.25)
    assert client.probe() is None
    client.ensure()
    restarted = await asyncio.to_thread(subprocess.run, [sys.executable, '-m', 'bubbleteait', '--config', str(filename),
        '--project-dir', str(primary), 'search', 'threshold', '--workspace', 'work', '--freshness', 'wait'],
        capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert restarted.returncode == 0, restarted.stderr + restarted.stdout
    checks['restart_persisted_index'] = True
    client.close()
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Installed engine paths/weights; service data dir is NOT reused')
    parser.add_argument('--compiler', help='Existing GCC executable for the disposable C fixture')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    installed = load_config(Path.cwd(), config_path=args.config)
    # Retain the isolated fixture, logs and receipts, including failed attempts.
    with contextlib.nullcontext(tempfile.mkdtemp(prefix='bti-engines-')) as temporary:
        root = Path(temporary)
        projects = {}
        for name in ('controller', 'reference'):
            path = root / name
            path.mkdir()
            (path / 'main.c').write_text('int threshold = 37;\nint read_threshold(void) { return threshold; }\n', encoding='utf-8')
            projects[name] = {'path': path, 'include': ['**/*.c']}
        if 'serena' in installed.engines:
            if not args.compiler:
                parser.error('--compiler is required when Serena is configured')
            primary = projects['controller']['path']
            database = root / 'build/compile_commands.json'
            database.parent.mkdir()
            arguments = [str(Path(args.compiler).resolve()), '-std=gnu11', '-DPRODUCT_CONTROLLER=1',
                         '-I', str(primary), '-c', str(primary / 'main.c')]
            database.write_text(json.dumps([{'directory': str(primary), 'file': str(primary/'main.c'),
                                             'arguments': arguments}]), encoding='utf-8')
            subprocess.run([arguments[0], '-fsyntax-only', str(primary/'main.c')], check=True, capture_output=True)
            projects['controller']['compile_commands'] = database
        filename = root / 'config-service.toml'
        filename.write_text(toml_text({'service': {'data_dir': root/'data'},
            'paths': {'models_dir': installed.models_dir}, 'engines': installed.engines, 'projects': projects,
            'workspaces': {'work': {'path': root, 'projects': list(projects), 'default_project': 'controller', 'watch': False}},
            'sync': {'debounce_ms': 0, 'poll_interval_seconds': 1}}), encoding='utf-8')
        config = load_config(projects['controller']['path'], config_path=filename)
        try:
            checks = asyncio.run(probe(config, filename, projects['controller']['path'], projects['reference']['path'], args.compiler))
            Path(args.output).write_text(json.dumps({'state': 'complete', 'checks': checks}, indent=2), encoding='utf-8')
            print(json.dumps(checks, ensure_ascii=False))
        finally:
            client = ServiceClient(config)
            client.request('stop')
            deadline = time.monotonic()+60
            while client.probe() and time.monotonic() < deadline:
                time.sleep(.25)
            client.close()


if __name__ == '__main__':
    main()
