param(
    [string]$PythonExe = 'python',
    [switch]$Download
)
$ErrorActionPreference = 'Stop'
$env:PYTHONHOME = ''
$env:PYTHONNOUSERSITE = '1'
$installArgs = @((Join-Path $PSScriptRoot 'scripts/install_tensorrt.py'))
if ($Download) { $installArgs += '--download' }
& $PythonExe @installArgs
if ($LASTEXITCODE -ne 0) { throw 'TensorRT project-local installation failed.' }
