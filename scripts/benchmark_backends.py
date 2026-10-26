"""Compare official, shared-frame and optional TensorRT execution on real audio."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pyannote_dual import Community1Diarizer
from pyannote_dual.output import write_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=str(ROOT / "example/huawei_test.wav"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/backend_benchmark"))
    parser.add_argument("--num-speakers", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--engine", default=None)
    args = parser.parse_args()
    root = Path(args.output_dir)
    diarizer = Community1Diarizer.from_pretrained(device="cuda", offline=True)
    import numpy as np
    import torch
    from pyannote_dual.shared_frames import enable_shared_frames
    captured = {}
    def hook(name, artifact, **kwargs):
        if name == "embeddings" and "completed" not in kwargs:
            captured[diarizer.backend] = artifact.copy()
    original_apply = diarizer._pipeline.apply
    def apply_with_hook(*a, **kw):
        return original_apply(*a, hook=hook, **kw)
    diarizer._pipeline.apply = apply_with_hook
    rows = []
    reference = None
    backends = ['torch', 'shared-frames'] + (['tensorrt'] if args.tensorrt else [])
    for backend in backends:
        if backend in {"shared-frames", "tensorrt"}:
            enable_shared_frames(diarizer._pipeline)
            diarizer.backend = backend
        if backend == 'tensorrt':
            from pyannote_dual.trt_backend import TensorRTFrames, DEFAULT_ENGINE
            diarizer._pipeline.shared_frame_encoder = TensorRTFrames(args.engine or DEFAULT_ENGINE,
                                                                    diarizer._pipeline._embedding.model_)
        for index in range(args.runs + 1):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            bundle = diarizer.process(args.audio, num_speakers=args.num_speakers)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            actual = [[s.as_dict() for s in getattr(bundle, mode)] for mode in ("complete", "non_conflict")]
            if reference is None:
                reference = actual
            row = {"backend": backend, "warmup": index == 0, "seconds": elapsed,
                   "segments_equal": actual == reference,
                   "complete_count": len(bundle.complete), "exclusive_count": len(bundle.non_conflict),
                   "exclusive_overlap": bundle.non_conflict_overlap_seconds,
                   "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6}
            rows.append(row)
            print(json.dumps(row), flush=True)
        write_bundle(bundle, root / backend, inference_seconds=elapsed)
    a = captured['torch']
    metrics = {'shape': list(a.shape), 'all_segments_equal': all(row['segments_equal'] for row in rows)}
    for backend in backends[1:]:
        b = captured[backend]
        finite = np.isfinite(a) & np.isfinite(b)
        metrics[backend] = {'same_nan_mask': bool(np.array_equal(np.isnan(a), np.isnan(b))),
                            'embedding_max_abs_difference': float(np.max(np.abs(a[finite] - b[finite])))}
    result = {"runs": rows, "comparison": metrics}
    (root / 'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(metrics), flush=True)
    if not metrics['all_segments_equal']:
        raise SystemExit('Output differences detected; inspect before enabling this backend by default.')


if __name__ == '__main__':
    main()
