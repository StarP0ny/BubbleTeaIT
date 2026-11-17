"""Keep one read-only stock Serena MCP session alive across JSON bridge calls."""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import sys


async def serve(settings):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    parameters = StdioServerParameters(command=settings['executable'], args=[
        'start-mcp-server', '--project', settings['view'], '--context', settings['context'],
        '--agent-interface', 'tools', '--mode', 'planning', '--mode', 'no-memories',
        '--enable-web-dashboard', 'false', '--open-web-dashboard', 'false', '--enable-gui-log-window', 'false'],
        cwd=settings['directory'], env=os.environ | {'SERENA_HOME': settings['home'], 'PYTHONUTF8': '1',
            'SERENA_USAGE_REPORTING': 'false', 'BTI_LANGUAGE_SERVER': settings['language_server'],
            'BTI_LANGUAGE_SERVER_ARGS': json.dumps(settings['ls_args'])})
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            if {t.name for t in listed.tools} != set(settings['tools']):
                raise ValueError('Serena tools differ from the read-only allowlist')
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                try:
                    request = json.loads(line)
                    if request['action'] == 'shutdown':
                        break
                    if request.get('tool') not in settings['tools']:
                        raise ValueError('Unsupported Serena tool')
                    result = await session.call_tool(request['tool'], request['arguments'])
                    if getattr(result, 'is_error', False):
                        messages = [c.text for c in result.content if getattr(c, 'type', None) == 'text']
                        detail = messages[0].splitlines()[0][:500] if messages else 'inspect bridge.log'
                        raise ValueError('Serena tool failed: ' + detail)
                    value = getattr(result, 'structured_content', None)
                    if isinstance(value, dict) and set(value) == {'result'}:
                        value = value['result']
                    if value is None:
                        texts = [c.text for c in result.content if getattr(c, 'type', None) == 'text']
                        if len(texts) != 1:
                            raise ValueError('Serena must return one structured result')
                        value = texts[0]
                    if isinstance(value, str):
                        value = json.loads(value)
                    response = {'ok': True, 'result': value}
                except Exception as exc:
                    response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
                print('BTI_ENGINE_JSON:' + json.dumps(response, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text(encoding='utf-8'))
    asyncio.run(serve(settings))


if __name__ == '__main__':
    main()
