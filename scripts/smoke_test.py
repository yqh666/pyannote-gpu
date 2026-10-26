# -*- coding: utf-8 -*-
"""Decode the sample through TorchCodec and optionally run model inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyannote_dual.runtime import activate_runtime, runtime_versions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=str(REPO_ROOT / "example" / "huawei_test.wav"))
    parser.add_argument("--run", action="store_true", help="Also load Community-1 and run inference.")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    args = parser.parse_args()

    activate_runtime()
    from torchcodec.decoders import AudioDecoder

    audio = Path(args.audio).resolve()
    result = {
        "runtime": runtime_versions(),
        "decoded": None,
    }
    if audio.is_file():
        decoder = AudioDecoder(str(audio))
        samples = decoder.get_all_samples()
        result["decoded"] = {
            "audio": str(audio),
            "codec": decoder.metadata.codec,
            "sample_rate": decoder.metadata.sample_rate,
            "channels": decoder.metadata.num_channels,
            "shape": list(samples.data.shape),
            "duration_seconds": decoder.metadata.duration_seconds_from_header,
        }
    elif args.run:
        raise FileNotFoundError(f"Inference audio does not exist: {audio}")
    else:
        result["decode_skipped"] = f"Optional sample audio not found: {audio}"

    if args.run:
        from pyannote_dual import Community1Diarizer

        diarizer = Community1Diarizer.from_pretrained(model=args.model)
        bundle = diarizer.process(args.audio, num_speakers=3)
        result["inference"] = bundle.summary()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
