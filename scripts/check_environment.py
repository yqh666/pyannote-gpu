# -*- coding: utf-8 -*-
"""Print versions and verify FFmpeg/TorchCodec without running the model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PACKAGES = [
    "pyannote.audio",
    "torch",
    "torchaudio",
    "torchcodec",
    "huggingface_hub",
    "numpy",
]


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['torch', 'shared-frames', 'tensorrt'], default='torch')
    parser.add_argument('--engine', default=None)
    args = parser.parse_args()
    from pyannote_dual.runtime import activate_runtime, runtime_versions

    ffmpeg_bin = activate_runtime()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    for package in PACKAGES:
        print(f"{package}: {version(package)}")

    print(f"FFmpeg bin: {ffmpeg_bin}")

    try:
        from torchcodec.decoders import AudioDecoder
        print(f"FFmpeg libraries: {runtime_versions()['ffmpeg']}")
        example = REPO_ROOT / "example" / "huawei_test.wav"
        print(f"Example audio: {example} ({'OK' if example.is_file() else 'MISSING'})")
        print(f"AudioDecoder import: {AudioDecoder.__name__}")
    except Exception as exc:
        print(f"TorchCodec check failed: {exc}")
        raise

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"CUDA check failed: {exc}")
        raise

    if args.backend == 'tensorrt':
        from pyannote_dual.trt_backend import DEFAULT_ENGINE, import_tensorrt
        trt = import_tensorrt()
        engine = Path(args.engine or DEFAULT_ENGINE).resolve()
        metadata_path = engine.with_suffix('.json')
        if not engine.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f'TensorRT engine or metadata missing: {engine}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        with engine.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != metadata['engine_sha256']:
                raise RuntimeError('TensorRT engine checksum does not match its metadata')
        if metadata['tensorrt_version'] != trt.__version__:
            raise RuntimeError('TensorRT engine version does not match the installed runtime')
        if not torch.cuda.is_available() or metadata['gpu_name'] != torch.cuda.get_device_name(0):
            raise RuntimeError('TensorRT engine was built for a different GPU')
        print(f"TensorRT: {trt.__version__}")
        print(f"TensorRT engine: {engine}")
        print(f"TensorRT precision: {metadata['precision']}")


if __name__ == "__main__":
    main()
