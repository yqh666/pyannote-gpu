"""Check TensorRT frame/pooled embedding numerics across dynamic batch sizes."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pyannote_dual import Community1Diarizer
from pyannote_dual.trt_backend import TensorRTFrames, DEFAULT_ENGINE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', default=str(DEFAULT_ENGINE))
    args = parser.parse_args()
    diarizer = Community1Diarizer.from_pretrained(device='cuda', offline=True)
    model = diarizer._pipeline._embedding.model_
    encoder = TensorRTFrames(args.engine, model)
    import torch
    torch.manual_seed(20261019)
    # Match the pipeline's strict FP32 reproducibility setting rather than the
    # raw PyTorch convolution module's TF32 default.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    with torch.inference_mode():
        for batch, silence in [(1, True), (1, False), (7, False), (16, False)]:
            shape = (batch, 1, int(model.audio.sample_rate * encoder.metadata['duration']))
            waves = torch.zeros(shape, device='cuda') if silence else torch.randn(shape, device='cuda') * .03
            expected = model.forward_frames(waves)
            actual = encoder(waves)
            weights = torch.rand(batch, 589, device='cuda')
            a = model.forward_embedding(expected, weights=weights)
            b = model.forward_embedding(actual, weights=weights)
            row = {'batch': batch, 'silence': silence,
                   'frame_max_abs_difference': (expected-actual).abs().max().item(),
                   'embedding_max_abs_difference': (a-b).abs().max().item()}
            rows.append(row)
            print(json.dumps(row), flush=True)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(b, a, atol=2e-5, rtol=2e-4)
        try:
            encoder(torch.zeros(1, 1, model.audio.sample_rate, device='cuda'))
        except ValueError:
            invalid_shape_rejected = True
        else:
            raise AssertionError('Wrong-duration input must be rejected')
    result = {'verified': True, 'engine': str(encoder.path), 'reference_tf32': False, 'cases': rows,
              'wrong_duration_rejected': invalid_shape_rejected}
    target = ROOT / 'outputs/tensorrt_frames_verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
