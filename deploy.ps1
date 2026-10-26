param(
    [ValidateSet('check','run','batch','serve')][string]$Mode = 'check',
    [string]$PythonExe = 'python',
    [string]$Audio,
    [string]$InputDir,
    [string]$OutputDir,
    [int]$NumSpeakers = 0,
    [string]$Model = 'pyannote/speaker-diarization-community-1',
    [ValidateSet('torch','shared-frames','tensorrt')][string]$Backend = 'torch',
    [string]$EnginePath,
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo 'src'
$env:PYTHONHOME = ''
$env:PYTHONNOUSERSITE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:PYANNOTE_METRICS_ENABLED = '0'
& (Join-Path $repo 'install_ffmpeg_native.ps1') -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { throw 'Runtime preparation failed.' }
if ($Mode -eq 'check') {
    $checkArgs = @('--backend', $Backend)
    if ($EnginePath) { $checkArgs += @('--engine', $EnginePath) }
    & $PythonExe (Join-Path $repo 'scripts/check_environment.py') @checkArgs
} else {
    $invokeArgs = @('--model', $Model, '--device', $Device, '--backend', $Backend, '--offline')
    if ($EnginePath) { $invokeArgs += @('--engine', $EnginePath) }
    switch ($Mode) {
        'serve' {
            if (-not $OutputDir) { $OutputDir = Join-Path $repo 'outputs/service' }
            & $PythonExe -m pyannote_dual.service @invokeArgs --data-dir $OutputDir --port $Port
        }
        'batch' {
            if (-not $InputDir -or -not $OutputDir) { throw 'batch requires -InputDir and -OutputDir.' }
            if ($NumSpeakers -gt 0) { $invokeArgs += @('--num-speakers', $NumSpeakers) }
            & $PythonExe -m pyannote_dual.batch @invokeArgs --input-dir $InputDir --output-dir $OutputDir
        }
        'run' {
            if (-not $Audio) { $Audio = Join-Path $repo 'example/huawei_test.wav' }
            if (-not $OutputDir) { $OutputDir = Join-Path $repo 'outputs/deployed_run' }
            if ($NumSpeakers -gt 0) { $invokeArgs += @('--num-speakers', $NumSpeakers) }
            & $PythonExe -m pyannote_dual.cli @invokeArgs --audio $Audio --output-dir $OutputDir
        }
    }
}
if ($LASTEXITCODE -ne 0) { throw "$Mode failed with exit code $LASTEXITCODE" }
