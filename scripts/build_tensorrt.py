"""Build the Community-1 WeSpeaker CUDA frame encoder on this GPU."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pyannote_dual import Community1Diarizer, MODEL_ID
from pyannote_dual.trt_backend import DEFAULT_ENGINE, build_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--engine', default=None)
    parser.add_argument('--max-batch', type=int, default=16)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()
    diarizer = Community1Diarizer.from_pretrained(model=args.model, device='cuda', offline=True)
    model = diarizer._pipeline._embedding.model_
    engine_path = args.engine or DEFAULT_ENGINE.with_name(f'wespeaker_frames_{args.precision}.engine')
    metadata = build_engine(model, path=engine_path, max_batch=args.max_batch,
                            precision=args.precision, duration=diarizer._pipeline._segmentation.duration)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
