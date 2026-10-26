@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
set "PYTHON_EXE=python"
set "BACKEND=tensorrt"
set "FFMPEG_BIN=%ROOT%tools\ffmpeg-shared\bin"

if defined PYTHON_EXE_OVERRIDE set "PYTHON_EXE=%PYTHON_EXE_OVERRIDE%"
if defined PYANNOTE_BACKEND_OVERRIDE set "BACKEND=%PYANNOTE_BACKEND_OVERRIDE%"
if defined CONDA_PREFIX set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"

set "PYTHONPATH=%ROOT%src"
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
set "PYANNOTE_METRICS_ENABLED=0"
set "PATH=%FFMPEG_BIN%;%PATH%"

if not exist "%FFMPEG_BIN%\avcodec-62.dll" (
  echo Missing FFmpeg shared runtime.
  echo Run: powershell -ExecutionPolicy Bypass -File "%ROOT%install_ffmpeg_native.ps1"
  pause
  exit /b 1
)

"%PYTHON_EXE%" "%ROOT%scripts\check_environment.py" --backend "%BACKEND%"
if errorlevel 1 goto :error

"%PYTHON_EXE%" "%ROOT%scripts\run_community1.py" ^
  --audio "%ROOT%example\huawei_test.wav" ^
  --output-dir "%ROOT%example\output" ^
  --num-speakers 3 ^
  --device cuda ^
  --backend "%BACKEND%" ^
  --offline
if errorlevel 1 goto :error

echo.
echo Done. Output: %ROOT%example\output
pause
exit /b 0

:error
echo.
echo Failed. See ENVIRONMENT.md for setup and troubleshooting.
pause
exit /b 1
