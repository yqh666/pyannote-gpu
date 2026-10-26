param(
    [ValidateSet('torch','shared-frames','tensorrt')][string]$Backend = 'tensorrt',
    [string]$PythonExe = '',
    [string]$OutputDir = ''
)
# Run the bundled huawei_test example through the selected GPU backend.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = if ($OutputDir) { $OutputDir } else { Join-Path $Root "example\output" }
$Python = if ($PythonExe) {
    $PythonExe
} elseif ($env:PYTHON_EXE_OVERRIDE) {
    $env:PYTHON_EXE_OVERRIDE
} elseif ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    "python"
}

$FfmpegBin = Join-Path $Root "tools\ffmpeg-shared\bin"
if (-not (Test-Path -LiteralPath (Join-Path $FfmpegBin "avcodec-62.dll"))) {
    throw "FFmpeg runtime is missing. Run install_ffmpeg_native.ps1 first."
}

$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONHOME = ""
$env:PYTHONNOUSERSITE = "1"
$env:PYANNOTE_METRICS_ENABLED = "0"
$env:PATH = $FfmpegBin + ";" + $env:PATH

& $Python (Join-Path $Root "scripts\check_environment.py") --backend $Backend
if ($LASTEXITCODE -ne 0) { throw "Environment check failed." }

& $Python (Join-Path $Root "scripts\run_community1.py") `
    --audio (Join-Path $Root "example\huawei_test.wav") `
    --output-dir $Target `
    --num-speakers 3 `
    --device cuda `
    --backend $Backend `
    --offline
if ($LASTEXITCODE -ne 0) { throw "Diarization failed." }
