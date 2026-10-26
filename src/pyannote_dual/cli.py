# -*- coding: utf-8 -*-
"""Command line interface for the dual-output Community-1 pipeline."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .output import write_bundle
from .pipeline import MODEL_ID, Community1Diarizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode with FFmpeg/TorchCodec, run pyannote Community-1 once, and write "
            "complete and non-conflict diarization."
        )
    )
    parser.add_argument("--audio", required=True, help="Input audio file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: outputs/<audio-stem>.",
    )
    parser.add_argument("--model", default=MODEL_ID, help="Model id or local model directory.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("torch", "shared-frames", "tensorrt"), default="torch")
    parser.add_argument("--engine", default=None, help="TensorRT engine file; built on this GPU.")
    parser.add_argument("--offline", action="store_true", help="Use the Hugging Face cache only.")
    parser.add_argument("--ffmpeg-bin", default=None, help="Directory containing FFmpeg shared DLLs.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Decode audio crops on demand instead of preloading the entire file.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-speakers", type=int, default=None)
    group.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audio = Path(args.audio).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "outputs" / audio.stem).resolve()
    )

    diarizer = Community1Diarizer.from_pretrained(
        model=args.model,
        token=args.token,
        device=args.device,
        offline=args.offline,
        ffmpeg_bin=args.ffmpeg_bin,
        backend=args.backend,
        engine_path=args.engine,
    )
    started = time.perf_counter()
    bundle = diarizer.process(
        audio,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        streaming=args.streaming,
    )
    elapsed = time.perf_counter() - started
    summary = write_bundle(bundle, output_dir, inference_seconds=elapsed)

    print(f"Audio: {summary['audio']}")
    print(f"Model: {summary['model']}")
    print(f"Device: {summary['device']}")
    print(f"Speakers: {summary['num_speakers']} {summary['speakers']}")
    print(
        "Complete: "
        f"{summary['complete']['segment_count']} segments, "
        f"{summary['complete']['overlap_seconds']:.3f}s overlap"
    )
    print(
        "Non-conflict: "
        f"{summary['non_conflict']['segment_count']} segments, "
        f"{summary['non_conflict']['overlap_seconds']:.3f}s overlap"
    )
    print(f"Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()
