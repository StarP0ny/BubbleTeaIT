param(
    [string]$PythonExecutable = (Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'),
    [switch]$AddToUserPath
)
$ErrorActionPreference = 'Stop'
$resolvedPython = (Resolve-Path -LiteralPath $PythonExecutable).Path
& $resolvedPython -m bubbleteait --version
if ($LASTEXITCODE -ne 0) { throw 'BTI is not installed in the selected Python environment' }
$launcherDirectory = Join-Path $env:LOCALAPPDATA 'BubbleTeaIT/bin'
New-Item -ItemType Directory -Force -Path $launcherDirectory | Out-Null
$launcher = Join-Path $launcherDirectory 'bti.cmd'
$launcherContent = "@echo off`r`n`"$resolvedPython`" -X utf8 -m bubbleteait %*`r`n"
if (Test-Path -LiteralPath $launcher) {
    if ((Get-Content -LiteralPath $launcher -Raw) -ne $launcherContent) {
        throw "Existing launcher differs; inspect it before replacing: $launcher"
    }
} else {
    [System.IO.File]::WriteAllText($launcher, $launcherContent, [System.Text.UTF8Encoding]::new($false))
}
if ($AddToUserPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { $_ })
    if ($entries -notcontains $launcherDirectory) {
        [Environment]::SetEnvironmentVariable('Path', (($entries + $launcherDirectory) -join ';'), 'User')
    }
    if (($env:Path -split ';') -notcontains $launcherDirectory) {
        $env:Path = "$env:Path;$launcherDirectory"
    }
}
Write-Output "CLI launcher: $launcher"
if ($AddToUserPath) { Write-Output 'User PATH updated. Restart existing terminals/agent clients to inherit it.' }
