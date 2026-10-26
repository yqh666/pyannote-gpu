"""TensorRT WeSpeaker frame encoder; official feature extraction/pooling retained."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = ROOT / 'models/tensorrt/wespeaker_frames_fp32.engine'
_DLL_HANDLES = []


def activate_gpu_packages():
    packages = ROOT / 'tools/gpu-packages'
    if packages.is_dir() and str(packages) not in sys.path:
        sys.path.insert(0, str(packages))
    import torch
    if os.name == 'nt':
        candidates = [packages / 'tensorrt_libs', packages / 'tensorrt_cu12_libs',
                      packages / 'nvidia/cuda_runtime/bin', packages / 'nvidia/cuda_runtime/lib',
                      Path(torch.__file__).parent / 'lib']
        for path in candidates:
            if path.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(path)))
                if str(path) not in os.environ.get('PATH', '').split(os.pathsep):
                    os.environ['PATH'] = str(path) + os.pathsep + os.environ.get('PATH', '')


def import_tensorrt():
    activate_gpu_packages()
    try:
        import tensorrt_bindings as trt
    except ModuleNotFoundError:
        try:
            import tensorrt as trt
        except ModuleNotFoundError as exc:
            raise RuntimeError('TensorRT dependencies missing. Run install_tensorrt.ps1 first.') from exc
    return trt


def weights_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.resnet.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TensorRTFrames:
    """A single execution context, used sequentially on PyTorch's CUDA stream."""
    def __init__(self, path, model, *, check_weights=True):
        import torch
        self.trt = trt = import_tensorrt()
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Build the TensorRT engine first: {self.path}')
        self.metadata = json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'))
        engine_bytes = self.path.read_bytes()
        if hashlib.sha256(engine_bytes).hexdigest() != self.metadata['engine_sha256']:
            raise ValueError('Engine checksum differs from its metadata; rebuild the engine')
        if self.metadata['tensorrt_version'] != trt.__version__:
            raise ValueError('Engine TensorRT version differs; rebuild on this machine')
        if self.metadata['gpu_name'] != torch.cuda.get_device_name():
            raise ValueError('Engine was built for a different GPU; rebuild on this machine')
        if check_weights and self.metadata['weights_sha256'] != weights_digest(model):
            raise ValueError('Engine weights do not match the loaded model; rebuild the engine')
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError('TensorRT engine could not be deserialized')
        if (self.engine.num_io_tensors != 2
                or self.engine.get_tensor_dtype('fbank') != trt.float32
                or self.engine.get_tensor_dtype('frames') != trt.float32):
            raise ValueError('Engine must expose FP32 fbank input and frames output')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT execution context could not be created')
        self.model = model
        self.stream = torch.cuda.Stream()

    def __call__(self, waveforms):
        import torch
        fbank = self.model.compute_fbank(waveforms).contiguous()
        if fbank.dtype != torch.float32 or fbank.device.type != 'cuda':
            raise ValueError('TensorRT encoder requires CUDA FP32 fbank input')
        if (not 1 <= fbank.shape[0] <= self.metadata['max_batch']
                or list(fbank.shape[1:]) != self.metadata['fbank_dimensions']):
            raise ValueError(f'Unsupported fbank shape {tuple(fbank.shape)}; rebuild the engine profile')
        if not self.context.set_input_shape('fbank', tuple(fbank.shape)):
            raise ValueError(f'Unsupported fbank shape {tuple(fbank.shape)}; rebuild the engine profile')
        shape = tuple(self.context.get_tensor_shape('frames'))
        if any(dim < 1 for dim in shape):
            raise RuntimeError(f'Unresolved TensorRT output shape: {shape}')
        current = torch.cuda.current_stream(fbank.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            frames = torch.empty(shape, device=fbank.device, dtype=torch.float32)
        if not self.context.set_tensor_address('fbank', fbank.data_ptr()):
            raise RuntimeError('Could not bind fbank CUDA buffer')
        if not self.context.set_tensor_address('frames', frames.data_ptr()):
            raise RuntimeError('Could not bind frames CUDA buffer')
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        fbank.record_stream(self.stream)
        current.wait_stream(self.stream)
        frames.record_stream(current)
        return frames


def build_engine(model, *, path=DEFAULT_ENGINE, duration=10.0, max_batch=16, precision='fp32'):
    import torch
    trt = import_tensorrt()
    import onnx
    if precision not in {'fp32', 'fp16'} or max_batch < 1:
        raise ValueError('Unsupported precision or invalid max_batch')
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    class Encoder(torch.nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet
        def forward(self, fbank):
            return self.resnet.forward_frames(fbank)

    with torch.inference_mode():
        sample = torch.zeros(1, 1, int(model.audio.sample_rate * duration), device='cuda')
        features = model.compute_fbank(sample)
        dimensions = list(features.shape[1:])
        onnx_path = path.with_suffix('.onnx')
        torch.onnx.export(Encoder(model.resnet).eval(), features, str(onnx_path),
                          input_names=['fbank'], output_names=['frames'],
                          dynamic_axes={'fbank': {0: 'batch'}, 'frames': {0: 'batch'}},
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(str(onnx_path))
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024**2)
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = 3
    profile = builder.create_optimization_profile()
    profile.set_shape('fbank', [1, *dimensions], [max_batch, *dimensions], [max_batch, *dimensions])
    config.add_optimization_profile(profile)
    print(json.dumps({'building': str(path), 'precision': precision, 'fbank_shape': dimensions,
                      'max_batch': max_batch, 'tensorrt': trt.__version__}), flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    path.write_bytes(bytes(serialized))
    metadata = {'tensorrt_version': trt.__version__, 'torch_version': torch.__version__,
                'gpu_name': torch.cuda.get_device_name(), 'weights_sha256': weights_digest(model),
                'precision': precision, 'max_batch': max_batch, 'fbank_dimensions': dimensions,
                'duration': duration, 'engine_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'onnx_sha256': hashlib.sha256(onnx_path.read_bytes()).hexdigest()}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata
