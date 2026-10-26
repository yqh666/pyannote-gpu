param(
    [string]$RepoRoot = $PSScriptRoot,
    [string]$PythonExe = 'python',
    [switch]$InstallIntoEnvironment
)

$ErrorActionPreference = 'Stop'
$ffmpegArchive = Join-Path $RepoRoot 'vendor\ffmpeg\ffmpeg-8.0.1-full_build-shared.zip'
$torchcodecWheel = Join-Path $RepoRoot 'vendor\torchcodec\torchcodec-0.9.1-cp311-cp311-win_amd64.whl'
$ffmpegSha256 = 'e4a40b46e3a3f3e5f2ed28352bd1fd6c733ae3808a8743682ef7f0d4e20c7d51'
$torchcodecSha256 = '508e0ce6da1e4af186b131cdbdd3d998bc4313e9f3a3f5b78f64aa5f9685ed88'
$toolsDir = Join-Path $RepoRoot 'tools'
$extractDir = Join-Path $toolsDir 'ffmpeg-8.0.1-full_build-shared'
$ffmpegDest = Join-Path $toolsDir 'ffmpeg-shared'

foreach ($path in @($ffmpegArchive, $torchcodecWheel)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing runtime file: $path. See vendor\ffmpeg\README.md and README.md."
    }
}
if ((Get-FileHash -LiteralPath $ffmpegArchive -Algorithm SHA256).Hash.ToLower() -ne $ffmpegSha256) {
    throw 'FFmpeg archive SHA256 mismatch.'
}
if ((Get-FileHash -LiteralPath $torchcodecWheel -Algorithm SHA256).Hash.ToLower() -ne $torchcodecSha256) {
    throw 'TorchCodec wheel SHA256 mismatch.'
}

New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
$ffmpegBinDest = Join-Path $ffmpegDest 'bin'
if (-not (Test-Path -LiteralPath (Join-Path $ffmpegBinDest 'avcodec-62.dll'))) {
    Expand-Archive -LiteralPath $ffmpegArchive -DestinationPath $extractDir -Force
    $ffmpegBinSource = Get-ChildItem -LiteralPath $extractDir -Recurse -Directory -Filter 'bin' |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'avcodec-62.dll') } |
        Select-Object -First 1
    if (-not $ffmpegBinSource) { throw 'Could not find an FFmpeg shared bin directory in the archive.' }
    New-Item -ItemType Directory -Force -Path $ffmpegBinDest | Out-Null
    Copy-Item -Path (Join-Path $ffmpegBinSource.FullName '*') -Destination $ffmpegBinDest -Recurse -Force
}
if ($InstallIntoEnvironment) {
    & $PythonExe -m pip install --force-reinstall --no-deps $torchcodecWheel
    if ($LASTEXITCODE -ne 0) { throw 'TorchCodec wheel installation failed.' }
} else {
    $overlayDir = Join-Path $toolsDir 'torchcodec-overlay'
    if (-not (Test-Path -LiteralPath (Join-Path $overlayDir 'torchcodec\__init__.py'))) {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($torchcodecWheel, $overlayDir)
    }
}

$env:PYTHONPATH = (Join-Path $RepoRoot 'src')
$env:PYTHONHOME = ''
$env:PYTHONNOUSERSITE = '1'
$env:PYANNOTE_METRICS_ENABLED = '0'
$env:PATH = $ffmpegBinDest + ';' + $env:PATH

& (Join-Path $ffmpegBinDest 'ffmpeg.exe') -hide_banner -version | Select-Object -First 4
& $PythonExe (Join-Path $RepoRoot 'scripts\smoke_test.py') --audio (Join-Path $RepoRoot 'example\huawei_test.wav')
if ($LASTEXITCODE -ne 0) { throw 'TorchCodec/FFmpeg verification failed.' }

Write-Host ''
Write-Host 'FFmpeg/TorchCodec runtime installed and verified.'
Write-Host "FFmpeg bin:  $ffmpegBinDest"
Write-Host "TorchCodec:  $torchcodecWheel"
