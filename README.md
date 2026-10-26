# pyannote Community-1 双输出说话人分离

## 交付版本概览（更新 2026-10-19）

本项目已提供单文件、批处理和本地 HTTP 服务，服务启动后只加载一次模型并持续复用。
交付版本同时包含两条已验收的 GPU 推理路径：

- `tensorrt`：已完成 TensorRT 10.13.3.9 接入，推荐在已安装运行库并为目标 GPU 构建引擎后使用；
- `torch`：官方 PyTorch CUDA 路径，作为默认兼容后端，不要求安装 TensorRT。

HTTP 服务使用 Python 标准库，不依赖 FastAPI、uvicorn 等额外服务框架。接收方完成下方的
Python 环境、模型和运行库准备后，可在已激活的环境中启动 TensorRT 服务：

```powershell
.\deploy.ps1 -Mode serve -PythonExe python -Backend tensorrt
```

模型就绪后监听 `http://127.0.0.1:8765`。支持上传音频、排队推理、查询状态和下载两套结果。
完整命令、接口和实际验收结果见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。
启动脚本默认离线加载模型；FFmpeg、TorchCodec 和 TensorRT 运行库均可放在项目目录中，
不要求写入接收方的 Python 环境。TensorRT 固定依赖总下载约 1.540 GB，当前样例热运行平均
从 6.54 秒降至 1.86 秒，双输出分段一致。安装、构建引擎和换机步骤见
[TensorRT 安装与使用](docs/TENSORRT.md)。`shared-frames` 是不依赖 TensorRT 的另一种可选加速后端。

本仓库使用 **FFmpeg + TorchCodec** 从原生音频文件路径解码并运行预训练模型
`pyannote/speaker-diarization-community-1`。一次推理同时导出两种说话人时间轴：

| 输出 | pyannote 字段 | 特点 | 推荐用途 |
|---|---|---|---|
| 完整版 `complete` | `speaker_diarization` | 保留多人同时说话的区间，允许重叠 | 重叠分析、DER 评估、完整记录 |
| 非冲突版 `non_conflict` | `exclusive_speaker_diarization` | 官方“独占”时间轴，任一时间点最多一个说话人 | ASR、字幕、会议纪要和其它单说话人下游 |

非冲突版不是后处理猜测，而是 Community-1 在同一个聚类结果上生成的官方
`exclusive_speaker_diarization`。两套结果同时保留，便于按任务选择。

## 解码路径

```text
音频文件
  -> FFmpeg 8.0.1 共享库
  -> TorchCodec 0.9.1 AudioDecoder
  -> pyannote Community-1 原生文件路径
  -> 一次推理
  -> complete + non_conflict
```

GitHub 源码仓库保留与 Python 3.11 / torch 2.9.1 匹配的 `TorchCodec 0.9.1` wheel，
但不提交大型 FFmpeg 压缩包。按 `vendor/ffmpeg/README.md` 放入已校验归档后，安装脚本会释放
FFmpeg、将 TorchCodec 解包为项目内 overlay；提供测试音频时还会完成实际解码自检。

## 仓库内容

```text
pyannote-community1-dual-output/
├─ README.md
├─ ENVIRONMENT.md
├─ requirements.txt
├─ pyproject.toml
├─ install_ffmpeg_native.ps1
├─ install_tensorrt.ps1
├─ deploy.ps1
├─ verify_install.ps1
├─ run_example.bat
├─ run_example.ps1
├─ vendor/
│  ├─ ffmpeg/README.md         # FFmpeg 归档名称、来源与 SHA256
│  ├─ torchcodec/torchcodec-0.9.1-cp311-cp311-win_amd64.whl
│  └─ tensorrt/dependencies.json
├─ src/pyannote_dual/
│  ├─ runtime.py               # FFmpeg / TorchCodec 运行时
│  ├─ pipeline.py              # Community-1 双输出提取
│  ├─ output.py                # TXT / RTTM / JSON 写出
│  ├─ shared_frames.py         # 共享声纹帧特征优化
│  ├─ trt_backend.py           # TensorRT 引擎构建与执行
│  ├─ batch.py                 # 递归批处理入口
│  ├─ service.py               # 本地 HTTP 服务
│  └─ cli.py                   # 单文件入口
├─ scripts/
│  ├─ build_tensorrt.py
│  ├─ verify_tensorrt_frames.py
│  ├─ verify_deployment.py
│  └─ benchmark_backends.py
├─ tests/
├─ example/
│  └─ output/                  # 已跑出的双输出示例；真实语音不提交 GitHub
└─ docs/
   ├─ DEPLOYMENT.md
   ├─ GITHUB.md
   ├─ TENSORRT.md
   ├─ OUTPUT_FORMAT.md
   ├─ VERIFIED_RUN.md
   └─ 交付报告.md
```

## 快速开始

### 1. 创建 Python 环境

```powershell
conda create -n pyannote-community1 python=3.11 -y
conda activate pyannote-community1
python -m pip install --upgrade pip
```

### 2. 安装 PyTorch

NVIDIA GPU：

```powershell
pip install torch==2.9.1+cu126 torchaudio==2.9.1+cu126 `
  --index-url https://download.pytorch.org/whl/cu126
```

仅 CPU：

```powershell
pip install torch==2.9.1 torchaudio==2.9.1
```

### 3. 安装 Python 依赖和原生运行时

先按 [vendor/ffmpeg/README.md](vendor/ffmpeg/README.md) 将 FFmpeg 归档放入
`vendor/ffmpeg/`。GitHub 仓库不包含这个大型第三方文件。

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\install_ffmpeg_native.ps1
```

安装脚本会：

- 校验 SHA256 并释放 FFmpeg 8.0.1 shared 包；
- 将仓库内 TorchCodec 0.9.1 wheel 解包到 `tools/torchcodec-overlay`；
- 加载 FFmpeg 共享库；
- 如果本地存在 `example/huawei_test.wav`，通过 TorchCodec 实际解码；不存在时跳过音频解码。

只做环境自检：

```powershell
powershell -ExecutionPolicy Bypass -File .\verify_install.ps1
```

### 4. 准备模型

首次下载 Community-1 前，需要在 Hugging Face 接受模型许可并设置 token：

```powershell
$env:HF_TOKEN = "hf_你的令牌"
```

首次下载模型时，直接调用允许联网的 Python 入口：

```powershell
$env:PYTHONPATH = ".\src"
python -m pyannote_dual.cli --audio D:\audio\meeting.wav --output-dir outputs\first_run --num-speakers 3 --device cuda
```

也可以在设置 token 后用统一验收脚本完成模型推理：

```powershell
powershell -ExecutionPolicy Bypass -File .\verify_install.ps1 -RunInference -Audio D:\audio\meeting.wav
```

下载并验证一次后，`deploy.ps1` 会从缓存离线加载。交付目录已附模型时，也可通过 `-Model`
直接指向模型目录，无需 Hugging Face token：

```powershell
.\deploy.ps1 -Mode run -PythonExe python -Backend torch `
  -Model D:\models\speaker-diarization-community-1 `
  -Audio D:\audio\meeting.wav -OutputDir outputs\meeting -NumSpeakers 3
```

### 5. 安装并构建 TensorRT 后端

交付包包含 `tools/gpu-wheels/` 时，安装过程不联网：

```powershell
.\install_tensorrt.ps1 -PythonExe python
python scripts\build_tensorrt.py
python scripts\verify_tensorrt_frames.py
```

如果交付包不携带 wheel 缓存，可显式下载已固定并校验的约 1.540 GB 依赖：

```powershell
.\install_tensorrt.ps1 -PythonExe python -Download
python scripts\build_tensorrt.py
```

TensorRT 引擎与 GPU、TensorRT 版本和模型权重绑定，接收方换机器后需要重新构建。

### 6. 运行 TensorRT 双输出示例

```powershell
.\deploy.ps1 -Mode run -PythonExe python -Backend tensorrt `
  -Audio D:\audio\meeting.wav -OutputDir outputs\meeting -NumSpeakers 3
```

启动常驻 HTTP 服务：

```powershell
.\deploy.ps1 -Mode serve -PythonExe python -Backend tensorrt -Port 8765
```

## 输出

```text
<output-dir>/
├─ complete/
│  ├─ segments.json
│  ├─ diarization.txt
│  └─ diarization.rttm
├─ non_conflict/
│  ├─ segments.json
│  ├─ diarization.txt
│  └─ diarization.rttm
└─ summary.json
```

`complete` 允许出现重叠。`non_conflict` 中任意两条时间区间不重叠，但相邻边界可以相等。
详细字段见 [docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md)。

## 已跑示例

本机验收使用的 `huawei_test.wav` 是 44.1 kHz、双声道、181.648 秒的真实测试音频；
为避免公开真实语音，GitHub 仓库只保留其结果样例，不提交输入 WAV。
在 RTX 4060 Laptop 8 GB、CUDA 12.6、`pyannote.audio 4.0.7` 环境下，
指定 `--num-speakers 3` 的已有验证结果为：

| 指标 | 完整版 | 非冲突版 |
|---|---:|---:|
| 说话人数 | 3 | 3 |
| 段数 | 56 | 60 |
| 重叠秒数 | 9.500625 | 0.000000 |
| 重叠区间 | 有 | 无 |
| 官方 torch 基线 | 约 6.54 秒 | 与完整版一次推理产生 |
| TensorRT FP32 | 约 1.86 秒 | 与完整版一次推理产生 |

示例文件是实际输出，不是手工编写的格式样例。复现记录见
[docs/VERIFIED_RUN.md](docs/VERIFIED_RUN.md)。

## 常用参数

```text
--audio PATH             输入音频
--output-dir PATH        输出目录，默认 outputs/<音频名>
--model ID_OR_PATH       HF 模型 ID 或本地模型目录
--token TOKEN            HF token，也可用 HF_TOKEN 环境变量
--device auto|cpu|cuda   默认 auto
--backend torch|shared-frames|tensorrt  默认 torch
--engine PATH            自定义本机 TensorRT 引擎
--ffmpeg-bin PATH        自定义 FFmpeg shared bin 目录
--streaming              按需解码，不一次预载全部音频
--num-speakers N         已知人数时最稳
--min-speakers N         最少人数
--max-speakers N         最多人数
--offline                仅用本地 Hugging Face 缓存
```

`--num-speakers` 与 `--min-speakers` 互斥。输入格式由 FFmpeg 8.0.1 支持，包括 WAV、
MP3、FLAC、M4A、AAC、OGG、Opus、WMA、MP4、MKV 和 WebM。

## 验收检查

不加载模型即可验证示例输出契约：

```powershell
python -m unittest discover -s tests -v
```

测试会确认：

- 完整版确实包含重叠；
- 非冲突版没有任何重叠；
- 所有分段都有正时长且不超出音频范围；
- 两种输出使用同一组说话人标签。

## 交付打包说明

如果通过压缩包或移动硬盘进行授权的离线交付，建议保留以下内容：

- 全部源代码、`docs/`、`tests/`、`vendor/` 和根目录 PowerShell 脚本；
- `tools/gpu-wheels/`：约 1.540 GB，使接收方无需重新下载 TensorRT 固定依赖；
- 有权再分发时才包含 `example/huawei_test.wav`；`example/output/` 可作为已验证输出样例；
- 需要提交验收证据时，保留 `outputs/tensorrt_benchmark_2026-10-19/`、
  `outputs/deployment_verify_tensorrt_20261019_200314/` 和 `outputs/final_delivery_verify/`。

`.gitignore` 默认排除了真实音频、FFmpeg 压缩包、`tools/gpu-wheels/`、`tools/gpu-packages/`、
`models/tensorrt/` 和 `outputs/`，因此只提交 Git 仓库不会包含这些大文件和验收产物。接收方可用 wheel 重新生成
`tools/gpu-packages/`；TensorRT 引擎应在接收方 GPU 上重新构建，不应把当前机器的 engine
视为跨机器通用文件。

模型权重不在本仓库中。接收方需要自行接受 Community-1 许可并下载，或在许可允许的前提下
接收单独提供的本地模型目录，再通过 `-Model` 指定。完整交付包和纯源码 Git 仓库的范围不同，
提交时应在材料清单中注明是否包含 TensorRT wheel、模型和验收输出。

## 重要说明

- Community-1 是 gated 模型，首次下载必须先接受许可并提供 token。
- 模型权重没有放进本仓库，避免违反许可和扩大仓库体积；仓库支持 Hugging Face 模型 ID 和本地模型目录。
- 非冲突版解决的是“时间轴冲突”，不修复说话人识别或聚类错误。
- 自动人数估计偏少时，指定 `--num-speakers` 通常更可靠。
- GitHub 发布步骤及源码仓库与离线包的区别见 [docs/GITHUB.md](docs/GITHUB.md)。
