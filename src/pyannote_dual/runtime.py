# -*- coding: utf-8 -*-
"""FFmpeg and TorchCodec runtime discovery."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FFMPEG_BIN = REPO_ROOT / "tools" / "ffmpeg-shared" / "bin"
_DLL_HANDLES = {}


def activate_runtime(ffmpeg_bin: str | Path | None = None) -> Path:
    """Expose the required FFmpeg shared DLLs before importing torchcodec."""
    ffmpeg_dir = Path(ffmpeg_bin or DEFAULT_FFMPEG_BIN).expanduser().resolve()
    codec_dll = ffmpeg_dir / "avcodec-62.dll"
    if not codec_dll.is_file():
        raise RuntimeError(
            f"FFmpeg shared DLLs were not found in {ffmpeg_dir}. "
            "Run: powershell -ExecutionPolicy Bypass -File .\\install_ffmpeg_native.ps1"
        )

    overlay = REPO_ROOT / "tools" / "torchcodec-overlay"
    if (overlay / "torchcodec" / "__init__.py").is_file() and str(overlay) not in sys.path:
        sys.path.insert(0, str(overlay))
    if os.name == "nt" and str(ffmpeg_dir) not in _DLL_HANDLES:
        _DLL_HANDLES[str(ffmpeg_dir)] = os.add_dll_directory(str(ffmpeg_dir))
    if str(ffmpeg_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    if str(ffmpeg_dir) not in sys.path:
        sys.path.append(str(ffmpeg_dir))
    return ffmpeg_dir


def runtime_versions() -> dict:
    """Return installed TorchCodec and FFmpeg library versions."""
    activate_runtime()
    import torchcodec
    try:
        from torchcodec import get_ffmpeg_library_versions
    except ImportError:
        from torchcodec._core import get_ffmpeg_library_versions

    return {
        "torchcodec": getattr(torchcodec, "__version__", "unknown"),
        "torchcodec_path": str(Path(torchcodec.__file__).resolve()),
        "ffmpeg": get_ffmpeg_library_versions(),
    }
