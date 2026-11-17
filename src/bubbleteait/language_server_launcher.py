"""Pass project-owned arguments to a real LSP executable; no shell involved."""
import json
import os
import sys
import subprocess


def main():
    executable = os.environ['BTI_LANGUAGE_SERVER']
    arguments = json.loads(os.environ.get('BTI_LANGUAGE_SERVER_ARGS', '[]'))
    command = [executable, *arguments, *sys.argv[1:]]
    if os.name == 'nt':
        # Windows execv spawns and exits: the LSP owner would see an already-dead
        # server process. Keep the launcher alive while forwarding native handles.
        raise SystemExit(subprocess.call(command, stdin=sys.stdin.buffer, stdout=sys.stdout.buffer,
                                        stderr=sys.stderr.buffer, creationflags=subprocess.CREATE_NO_WINDOW))
    os.execv(executable, command)
