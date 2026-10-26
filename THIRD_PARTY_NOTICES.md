# Third-party notices

This project uses the following third-party software and model:

- pyannote.audio 4.0.7, MIT License.
- PyTorch, BSD-style License.
- TorchCodec 0.9.1, BSD-style License.
- FFmpeg 8.0.1 full shared build, provided by the upstream FFmpeg project and
  the build distributor. Review the exact build configuration and license terms
  before redistribution.
- Hugging Face Hub, Apache License 2.0.
- Optional NVIDIA TensorRT 10.13.3.9 and CUDA runtime 12.9.79 are governed by
  their NVIDIA software license agreements; the original license files remain
  in the downloaded wheels and installed package metadata. ONNX and ml_dtypes
  retain their upstream license files as well. Download sources and hashes are
  recorded in vendor/tensorrt/dependencies.json.
- pyannote/speaker-diarization-community-1 and its component model weights are
  governed by the model card and license on Hugging Face.

Model weights are intentionally not redistributed by this repository. Users must
review and accept the upstream Community-1 terms before downloading or using the
weights.

The locally tested huawei_test.wav contains real speech and is excluded from the
GitHub source repository. Include it in a separate delivery only when you have
the right to redistribute it.
