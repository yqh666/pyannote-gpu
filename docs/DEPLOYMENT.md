# Windows GPU 端到端部署

更新：2026-10-26。本项目部署的是说话人分离：输入音频，输出带时间戳的文件内说话人编号。
不包含真实人名识别、ASR 文本或视频座位绑定。程序默认值为官方 Community-1 + PyTorch CUDA，
便于尚未构建本机引擎的接收方启动；交付演示和已验证加速路径使用 `-Backend tensorrt`。
TensorRT 主运行库 wheel 为 1,518,425,990 字节，约 1.52 GB；加入 Python 绑定、CUDA runtime、
ONNX 和 ml_dtypes 后，已核实的五个固定文件总下载量为 1,539,565,572 字节
（约 1.540 GB / 1.434 GiB），已下载、安装并通过本机 GPU 验收。
不需完整 CUDA Toolkit。可用 `-Backend tensorrt` 启用，安装、引擎构建和实测见 [TENSORRT.md](TENSORRT.md)。

## 接收方准备清单

GitHub 仓库不是完整离线运行包。运行以下命令前，接收方应完成：

1. 创建 Python 3.11 环境并安装 PyTorch、pyannote 等基础依赖；
2. 按 `vendor/ffmpeg/README.md` 下载 94,371,734 字节的 FFmpeg 归档并放入指定目录；
3. 接受 Community-1 模型许可并完成一次联网下载，或准备合规的本地模型目录；
4. 仅在选择 `tensorrt` 后端时，下载 1,539,565,572 字节的固定依赖并在目标 GPU 构建引擎；
5. 提供自己的输入音频。GitHub 不包含真实验收音频。

Python/PyTorch/pyannote 和模型权重的下载量不包含在 TensorRT 的 1.540 GB 中。

## 完成准备后运行

先进入交付项目根目录；下列相对路径均以该目录为基准。
以下命令中的 `python` 指接收方当前已激活 Python 3.11 环境里的解释器，不绑定环境名称或安装路径。
PowerShell 入口默认 `--offline`，模型从已授权的缓存或 `-Model` 指定的本地目录读取。

```powershell
# 运行时与 GPU 自检
.\deploy.ps1 -Mode check -PythonExe python -Backend tensorrt

# 单个文件；示例已知 3 人，未知人数时省略 -NumSpeakers
.\deploy.ps1 -Mode run -PythonExe python -Backend tensorrt -NumSpeakers 3

# 自己的音频
.\deploy.ps1 -Mode run -PythonExe python -Backend tensorrt -Audio D:\audio\meeting.wav -OutputDir D:\results\meeting

# 递归批处理，同一个模型依次处理目录中的媒体文件
.\deploy.ps1 -Mode batch -PythonExe python -Backend tensorrt -InputDir D:\audio -OutputDir D:\results

# 常驻模型的 HTTP 服务
.\deploy.ps1 -Mode serve -PythonExe python -Backend tensorrt -Port 8765
```

`-OutputDir` 在 serve 模式表示任务数据目录，默认 `outputs/service`。批处理的输出目录必须在
输入目录外，每个文件保留相对目录与扩展名，避免同名 WAV/MP3 覆盖。某个文件失败会记录错误，
继续后续文件，最终 `batch_summary.json` 汇总，存在失败则进程退出码为 1。

`-Device cuda` 为默认；GPU 不可用时明确报错。需要 CPU 时显式传 `-Device cpu`。
尚未安装 TensorRT 或尚未为目标 GPU 构建引擎时，改用 `-Backend torch`。
完整离线包可以自带 FFmpeg；GitHub 克隆版本必须先另行准备 FFmpeg 归档。安装脚本将 FFmpeg 与
固定 TorchCodec wheel 解包到 `tools/`，脚本可重复执行，默认不强制重装 Conda 环境中的 TorchCodec。
`install_ffmpeg_native.ps1 -InstallIntoEnvironment` 是显式覆盖安装选项，通常无需使用。

## 服务调用

服务仅绑定本机回环地址。启动后 `GET /health` 返回 `status=ready`、模型、设备和 backend。
HTTP 使用标准库，无 FastAPI/uvicorn 依赖。单个工作线程复用一个模型，任务顺序执行。

在另一个 PowerShell 窗口上传并等待结果：

```powershell
$baseUrl = 'http://127.0.0.1:8765'
Invoke-RestMethod "$baseUrl/health"
$submitted = Invoke-RestMethod -Method Post `
  -Uri "$baseUrl/jobs?filename=meeting.wav&num_speakers=3" `
  -ContentType 'application/octet-stream' `
  -InFile 'D:\audio\meeting.wav'
do {
  Start-Sleep -Milliseconds 500
  $job = Invoke-RestMethod "$baseUrl$($submitted.status_url)"
} while ($job.status -in @('receiving', 'queued', 'running'))
if ($job.status -eq 'failed') { throw $job.error }
$result = Invoke-RestMethod "$baseUrl$($submitted.result_url)"
$result.summary
$result.non_conflict.segments
```

| 方法与路径 | 行为 |
|---|---|
| `GET /health` | 模型就绪状态 |
| `POST /jobs?filename=x.wav&num_speakers=3` | 原始文件字节上传，Content-Length 必填；成功为 202 |
| `GET /jobs/{id}` | receiving / queued / running / succeeded / failed，以及错误或 summary |
| `GET /jobs/{id}/result` | summary、complete 和 non_conflict 的 JSON；未成功返回 409 |
| `GET /jobs/{id}/artifacts/complete/segments.json` | 下载单个结果文件 |
| `GET /jobs/{id}/artifacts/non_conflict/diarization.txt` | 下载非冲突版 TXT |
| `GET /jobs/{id}/artifacts/summary.json` | 下载汇总 |

两个模式均支持 `segments.json`、`diarization.txt`、`diarization.rttm`。
人数未知可省略人数参数，或使用 `min_speakers` / `max_speakers`，不能与 `num_speakers` 同时提供。
`streaming=true` 可按需解码。默认最大上传 512 MiB、最多 8 个未完成任务；队列满返回 429，超限返回 413。
直接运行模块可调整限制：

```powershell
$env:PYTHONPATH = '.\src'
python -m pyannote_dual.service --offline --device cuda --max-pending 4 --max-upload-mb 256
```

任务状态和输入/输出保留在数据目录中。服务重启后仍可查询已完成任务；中断任务标为 failed，
不会自动重复推理。用 Ctrl+C 停止服务时等待已排队任务处理完毕。输入和结果由使用者按需归档或清理。

## 结果契约

一次官方推理同时生成 `speaker_diarization` 与 `exclusive_speaker_diarization`，分别写入
`complete/`、`non_conflict/`。后者必须使用官方字段，字段缺失即报错。说话人编号仅在文件内有意义。
每个模式输出 JSON/TXT/RTTM，另有 summary 记录 backend、设备、时长、段数、重叠秒数与文件位置。
字段详情见 OUTPUT_FORMAT.md。

## 现有机器的验收证据

`scripts/verify_deployment.py` 使用真实 CUDA 模型与本机 HTTP 请求，不使用替身模型。
2026-10-17 验收记录：`outputs/deployment_verify_20261017_195124/verification.json`。

- 上传坏音频得到 failed，随后上传正常音频成功，服务工作线程继续运行。
- 181.648254 秒音频：完整 56 段、重叠 9.500625 秒；非冲突 60 段、重叠 0。
- HTTP 结果和批处理结果的全部 `(start, end, speaker)` 与既有 verify_run 基线完全一致。
- 六个 HTTP 产物均可下载；TXT 与基线逐字节相同，RTTM 除上传输入文件名外逐字节相同。
- 非法人数、路径式文件名、未知任务与越界产物请求得到错误响应。
- 8 项单元测试覆盖输出契约、排队限制、任务失败隔离、重启状态与批处理路径冲突。
- `deploy.ps1 -Mode run -NumSpeakers 3` 也已从运行时准备到写出结果完整运行成功，产物位于 `outputs/deployed_run`。

复验：

```powershell
python -m unittest discover -s tests -v
python scripts\verify_deployment.py
```

## 可选优化

默认 `torch` 保持官方计算路径。`-Backend shared-frames` 复用同一音频窗的 WeSpeaker 帧特征，
各说话人的掩码与池化仍分别计算，聚类与双输出仍使用官方实现；不需要下载新依赖。
该后端仅适用于本项目已固定的 Community-1 / WeSpeakerResNet34 推理，后续训练或换模型应使用 torch。

本机该测试音频的两次热运行：torch 平均 6.486 秒，shared-frames 平均 2.627 秒（约 2.47 倍）；
PyTorch 峰值已分配显存约 1709 MB / 897 MB（不含驱动和其它程序）。六次结果分段完全一致，
embedding 最大绝对差约 4.62e-7。证据在 `outputs/backend_benchmark/benchmark.json`。
这仅代表该音频，尚无跨数据集 DER 验收，因此保持显式可选。

### TensorRT（2026-10-19 已验收）

TensorRT 10.13.3.9 的 FP32 引擎用于 WeSpeaker 卷积帧特征提取，并结合上述共享帧计算。
原有分割、fbank、掩码与池化、聚类、完整和 exclusive 双输出继续复用。启动：

```powershell
.\deploy.ps1 -Mode run -PythonExe python -Backend tensorrt -NumSpeakers 3 -OutputDir outputs\tensorrt_run
.\deploy.ps1 -Mode serve -PythonExe python -Backend tensorrt
```

同一 181.648254 秒音频、每个后端预热一次后再运行三次，推理与结果对象封装平均耗时：

| 后端 | 平均耗时 | 相对官方 torch |
|---|---:|---:|
| torch | 6.538 秒 | 1.00 倍 |
| shared-frames | 2.592 秒 | 2.52 倍 |
| tensorrt FP32 + 共享帧 | 1.860 秒 | 3.52 倍 |

不含模型加载与结果写盘。TensorRT 比 shared-frames 进一步提速约 1.39 倍；
12 次完整/非冲突分段及标签完全一致，embedding 最大绝对差约 2.18e-6。
记录：`outputs/tensorrt_benchmark_2026-10-19/benchmark.json`。

真实 HTTP 与批处理的全部起止时间和标签也与 `verify_run` 基线一致，产物下载与失败恢复通过。
记录：`outputs/deployment_verify_tensorrt_20261019_200314/verification.json`。
HTTP 首次正常任务的处理与封装耗时 2.507 秒，不含模型加载、上传与排队。
批量 1/7/16 的 FP32 数值检查见 `outputs/tensorrt_frames_verification.json`。
已有 AMI ES2011c 前 120 秒也通过三后端的分段与标签一致性复验，记录在
`outputs/tensorrt_ami_ES2011c_120s/benchmark.json`；未下载新数据。

默认仍为 torch。引擎在目标 GPU 上构建，换机器须重建；该引擎只加速声纹卷积部分，
不是整个 pyannote 流程都转成 TensorRT。更大规模的 DER 验收仍待开展。

## 换机器部署

本轮已验证的是当前 Windows + RTX 4060 环境，尚未在 Linux、容器或其它 GPU 上验收。
新机器先按 ENVIRONMENT.md 创建 Python 3.11 / torch 2.9.1+cu126 / pyannote.audio 4.0.7 环境；
本轮没有重新下载这些包。离线使用时须准备已授权模型及其 segmentation、embedding、plda 子目录，
通过 `-Model D:\models\speaker-diarization-community-1` 指定，或准备完整 HF 缓存。
模型许可和现有音频的再分发约束见 THIRD_PARTY_NOTICES.md。
