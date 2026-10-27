"""Local Windows WMI launcher, outside a caller's kill-on-close Job Object.

CREATE_NO_WINDOW alone does not detach Job membership. Windows documents that
Win32_Process.Create does not inherit the caller's Job; cscript only performs
that fixed local call and exits. No shell, network, scheduler or SDK mutation.
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _vbs_string(value):
    if any(char in value for char in '\r\n\0'):
        raise ValueError('Invalid Windows launcher path')
    return '"' + value.replace('"', '""') + '"'


def spawn_worker(executable, launcher, config, data_dir):
    command = subprocess.list2cmdline([str(executable), '-X', 'utf8', str(launcher), '--config', str(config)])
    script = Path(config).parent / 'launch-worker.vbs'
    script.write_text('\n'.join([
        'Option Explicit',
        'Dim services, startup, code, pid',
        'Set services = GetObject("winmgmts:{impersonationLevel=impersonate}!\\\\.\\root\\cimv2")',
        'Set startup = services.Get("Win32_ProcessStartup").SpawnInstance_',
        'startup.ShowWindow = 0',
        'startup.CreateFlags = 16777216',  # BREAKAWAY from WMI provider Job; ShowWindow=0 hides the worker.
        'code = services.Get("Win32_Process").Create(' + _vbs_string(command) + ', ' + _vbs_string(str(data_dir)) + ', startup, pid)',
        'If IsNull(pid) Then pid = 0',
        'WScript.Echo CStr(code) & " " & CStr(pid)',
        'If code <> 0 Then WScript.Quit 1',
    ]) + '\n', encoding='utf-16')
    cscript = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/cscript.exe'
    result = subprocess.run([str(cscript), '//NoLogo', str(script)], stdin=subprocess.DEVNULL,
        capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
    stdout, stderr = result.stdout.decode('oem', errors='replace'), result.stderr.decode('oem', errors='replace')
    parts = stdout.strip().split()
    if result.returncode or len(parts) != 2 or parts[0] != '0' or not parts[1].isdigit():
        raise ValueError('Windows local WMI worker launch failed; check WSH/WMI policy: '
                         + (stderr.strip() or stdout.strip() or str(result.returncode)))
    return int(parts[1])
