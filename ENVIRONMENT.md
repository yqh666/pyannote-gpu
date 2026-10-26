# 环境说明

本仓库固定使用 **FFmpeg 8.0.1 shared + TorchCodec 0.9.1** 解码音频，并把原生文件路径交给
`pyannote/speaker-diarization-community-1`。整个运行链路不使用其它音频读取库。

## 1. 已验证环境

| 项目 | 已验证值 |
|---|---|
| 操作系统 | Windows 10/11 x64 |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| CUDA | 12.6 |
| Python | 3.11.16 |
| Conda 环境 | Python 3.11 独立环境（名称可自定） |
| pyannote.audio | 4.0.7 |
| torch | 2.9.1+cu126 |
| torchaudio | 2.9.1+cu126 |
| TorchCodec | 0.9.1 |
| FFmpeg | 8.0.1 full shared |
| huggingface_hub | 1.30.0 |
| numpy | 2.4.6 |
| 模型 | `pyannote/speaker-diarization-community-1` |

CPU 也能运行，只是速度更慢。Community-1 官方独占输出
`exclusive_speaker_diarization` 要求 pyannote.audio 4.x；本仓库固定为已验证的 4.0.7。

2026-10-19 增加可选 TensorRT 后端：TensorRT 10.13.3.9（CUDA 12）、CUDA runtime 12.9.79、
ONNX 1.19.1、ml_dtypes 0.5.3，安装到 `tools/gpu-packages/`；本机驱动保持 566.24，
PyTorch 保持 cu126。TensorRT 运行依赖和 ONNX 工具总下载约 1.540 GB；
FP32 引擎已在当前 GPU 构建并验收，详见 [docs/TENSORRT.md](docs/TENSORRT.md)。

## 2. 从零创建环境

```powershell
conda create -n pyannote-community1 python=3.11 -y
conda activate pyannote-community1
python -m pip install --upgrade pip
```

### NVIDIA GPU

```powershell
pip install torch==2.9.1+cu126 torchaudio==2.9.1+cu126 `
  --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

### 仅 CPU

```powershell
pip install torch==2.9.1 torchaudio==2.9.1
pip install pyannote.audio==4.0.7 torchcodec==0.9.1 huggingface_hub==1.30.0 numpy==2.4.6
```

## 3. 安装 FFmpeg 和 TorchCodec

GitHub 仓库保留 TorchCodec wheel；FFmpeg 压缩包需按 `vendor/ffmpeg/README.md` 另行准备：

```text
vendor/ffmpeg/ffmpeg-8.0.1-full_build-shared.zip
vendor/torchcodec/torchcodec-0.9.1-cp311-cp311-win_amd64.whl
```

执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install_ffmpeg_native.ps1
```

脚本完成以下工作：

1. 从内置压缩包释放 FFmpeg。
2. 自动定位含 `avcodec-62.dll` 的 shared bin。
3. 复制到 `tools/ffmpeg-shared/bin`。
4. 将内置 TorchCodec 0.9.1 wheel 解包到项目内 `tools/torchcodec-overlay`，不修改 Conda 环境。只有显式指定 `-InstallIntoEnvironment` 才会 pip 覆盖安装。
5. 把 FFmpeg bin 加入当前进程的 `PATH`。
6. 本地存在 `example/huawei_test.wav` 时，通过 TorchCodec 的 `AudioDecoder` 做实际解码；GitHub 克隆未提供真实语音时跳过。

安装后的关键路径：

```text
tools/ffmpeg-shared/bin/ffmpeg.exe
tools/ffmpeg-shared/bin/avcodec-62.dll
```

默认使用 `tools/torchcodec-overlay/torchcodec` 内的核心 DLL；运行入口自动激活 overlay。

## 4. 运行时环境变量

仓库内的批处理和 PowerShell 启动脚本会自动设置：

```text
PATH=<repo>	oolsfmpeg-sharedin;...
PYTHONPATH=<repo>src
PYTHONHOME=
PYTHONNOUSERSITE=1
PYANNOTE_METRICS_ENABLED=0
```

自定义 FFmpeg 目录时，可在命令行传入：

```powershell
--ffmpeg-bin D:\ffmpeg\bin
```

传入目录中必须存在 `avcodec-62.dll` 及其它 FFmpeg shared 依赖。

## 5. 模型授权与下载

Community-1 是 gated 模型：

1. 登录 Hugging Face。
2. 打开 <https://huggingface.co/pyannote/speaker-diarization-community-1>。
3. 接受模型许可。
4. 创建具有读取权限的 access token。
5. 设置 `HF_TOKEN`，不要把真实 token 提交到 Git。

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"
python scripts\run_community1.py --audio D:\audio\meeting.wav --output-dir outputs\meeting --num-speakers 3
```

首次运行会下载权重。权重已经缓存时可以增加 `--offline`。如果模型已保存为本地目录，
直接传 `--model D:\models\speaker-diarization-community-1`。

## 6. 音频解码

TorchCodec 通过 FFmpeg 8.0.1 直接读取音频文件，pyannote 接收文件路径。支持格式由内置
FFmpeg 构建决定，已验证 WAV 和 MP3。常见输入还包括 FLAC、M4A、AAC、OGG、Opus、
WMA、MP4、MKV 和 WebM。

本机验收音频参数（输入 WAV 不提交 GitHub）：

| 项目 | 值 |
|---|---|
| 时长 | 181.648254 秒 |
| 采样率 | 44,100 Hz |
| 声道 | 2 |
| 位深 | 16-bit PCM |
| 容器/编码 | WAV / PCM |

## 7. 一次推理，两种输出

pyannote.audio 4.0.7 的 Community-1 返回对象包含：

```python
output.speaker_diarization              # 完整版，可重叠
output.exclusive_speaker_diarization    # 非冲突版，官方独占时间轴
```

本仓库在同一个 `pipeline(...)` 调用后同时读取这两个字段，因此不会为了生成两套结果而重复
推理。

## 8. 完整复现命令

```powershell
conda activate pyannote-community1
cd <仓库根目录>
$env:HF_TOKEN = "hf_xxxxxxxx"

pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\install_ffmpeg_native.ps1
powershell -ExecutionPolicy Bypass -File .\verify_install.ps1

python scripts\run_community1.py `
  --audio D:\audio\meeting.wav `
  --output-dir outputs\meeting `
  --num-speakers 3

python -m unittest discover -s tests -v
```

预期结果：

- 完整版 56 段，重叠约 9.500625 秒；
- 非冲突版 60 段，重叠 0 秒；
- 输出包含 `complete`、`non_conflict` 和 `summary.json`。

## 9. 常见问题

### 找不到 `avcodec-62.dll`

处理：

```powershell
powershell -ExecutionPolicy Bypass -File .\install_ffmpeg_native.ps1
```

也可以显式指定：

```powershell
--ffmpeg-bin D:\ffmpeg\bin
```

### TorchCodec 导入失败或 DLL 初始化失败

先确认：

```powershell
conda activate pyannote-community1
python -c "import torch, torchcodec; print(torch.__version__, torchcodec.__version__)"
```

再运行安装脚本准备仓库内置 TorchCodec 0.9.1 overlay，并用 `scripts/check_environment.py` 检查实际加载路径。

### 401/403 或 gated repo

原因：没有接受许可或 token 无权限。处理：接受许可，重新生成 token，设置 `HF_TOKEN`。

### `exclusive_speaker_diarization` 不存在

原因：pyannote.audio 版本过旧或使用非 Community-1 管线。处理：

```powershell
pip install pyannote.audio==4.0.7
```

### CUDA 不可用

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

若显示 `False`，先纠正 PyTorch/CUDA 安装；也可显式加 `--device cpu` 运行。

### 显存不足

优先使用 `--device cpu`、`--streaming`，或关闭其它占用 GPU 的进程。

### 说话人数不准

模型自动计数偶尔偏低。若已知人数，显式指定：

```powershell
--num-speakers 3
```

人数约束影响完整版和独占版，因为两者来自同一次聚类。
