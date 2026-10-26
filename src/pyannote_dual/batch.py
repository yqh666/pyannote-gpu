"""Sequential file processing with one resident model and per-file failures."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .pipeline import Community1Diarizer, MODEL_ID
from .output import write_bundle

AUDIO_SUFFIXES = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.mp4', '.mkv', '.webm'}


def run_batch(diarizer, input_dir, output_dir, **options):
    source, target = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError(f'Input directory does not exist: {source}')
    if target == source or source in target.parents:
        raise ValueError('Place the output directory outside the input directory')
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise ValueError(f'No supported audio files in {source}')
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for audio in files:
        relative = audio.relative_to(source)
        # Keep the extension in the directory name: foo.wav and foo.mp3 cannot collide.
        destination = target / relative.parent / relative.name
        start = time.perf_counter()
        try:
            bundle = diarizer.process(audio, **options)
            summary = write_bundle(bundle, destination, time.perf_counter() - start)
            record = {'audio': str(audio), 'status': 'succeeded', 'summary': summary}
        except Exception as exc:
            record = {'audio': str(audio), 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        records.append(record)
        print(json.dumps({'audio': str(audio), 'status': record['status']}, ensure_ascii=False), flush=True)
        (target / 'batch_summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--backend', choices=['torch', 'shared-frames', 'tensorrt'], default='torch')
    parser.add_argument('--engine', default=None)
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--min-speakers', type=int)
    parser.add_argument('--max-speakers', type=int)
    parser.add_argument('--streaming', action='store_true')
    args = parser.parse_args()
    diarizer = Community1Diarizer.from_pretrained(model=args.model, device=args.device,
                                               offline=args.offline, backend=args.backend,
                                               engine_path=args.engine)
    records = run_batch(diarizer, args.input_dir, args.output_dir, num_speakers=args.num_speakers,
                        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
                        streaming=args.streaming)
    if any(row['status'] == 'failed' for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
