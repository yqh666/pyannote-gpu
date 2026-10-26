param(
    [string]$PythonExe = 'python',
    [switch]$RunInference,
    [string]$Audio = ''
)

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$ffmpegBin = Join-Path $repo 'tools\ffmpeg-shared\bin'

& (Join-Path $repo 'install_ffmpeg_native.ps1') -RepoRoot $repo -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { throw 'Runtime preparation failed.' }

$env:PYTHONPATH = (Join-Path $repo 'src')
$env:PYTHONHOME = ''
$env:PYTHONNOUSERSITE = '1'
$env:PYANNOTE_METRICS_ENABLED = '0'
$env:PATH = $ffmpegBin + ';' + $env:PATH

$args = @((Join-Path $repo 'scripts\smoke_test.py'))
if ($RunInference) { $args += '--run' }
if ($Audio) { $args += @('--audio', $Audio) }
& $PythonExe @args
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
