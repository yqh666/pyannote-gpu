# TensorRT 可选后端

TensorRT CUDA 12 主运行库 wheel 为 1,518,425,990 字节，约 1.52 GB。加入 Python 绑定、
CUDA runtime、ONNX 和 ml_dtypes 后，本项目固定的五个文件总下载量为 1,539,565,572 字节，
约 1.540 GB / 1.434 GiB。本项目固定使用 TensorRT 10.13.3.9，依赖安装在项目的
`tools/gpu-packages/`，不覆盖 Conda 环境。

## 下载与安装

固定清单位于 `vendor/tensorrt/dependencies.json`，含来源、字节数与 SHA256。

| 文件 | 下载大小 |
|---|---:|
| TensorRT CUDA 12 运行库 | 1,518,425,990 字节 |
| TensorRT CP311 Windows 绑定 | 877,325 字节 |
| CUDA runtime 12.9.79 | 3,591,604 字节 |
| ONNX 1.19.1 | 16,464,319 字节 |
| ml_dtypes 0.5.3 | 206,334 字节 |
| 合计 | 1,539,565,572 字节，约 1.540 GB / 1.434 GiB |

不需要完整 CUDA Toolkit。上述下载量不包含已有的 Python、PyTorch、pyannote 和模型权重。
下载的 wheel、解包后的 DLL、构建的 ONNX 和引擎会分别占用磁盘空间。
本机这三类文件合计约 3.96 GB，不含原有环境和验收音频；网络下载仍为上表的 1.540 GB。

交付包包含 `tools/gpu-wheels/` 时，可离线安装：

```powershell
.\install_tensorrt.ps1 -PythonExe python
```

接收方尚无缓存时，显式允许下载固定清单：

```powershell
.\install_tensorrt.ps1 -PythonExe python -Download
```

脚本支持断点续传，校验大小和哈希；pip 只安装清单中的本地文件，不自动解析或下载其它依赖。
本清单适用于 Windows x64 / Python 3.11。

## 构建本机引擎

```powershell
python scripts\build_tensorrt.py
```

使用交付的本地模型目录时：

```powershell
python scripts\build_tensorrt.py --model D:\models\speaker-diarization-community-1
```

默认构建 FP32（关闭 TF32），引擎路径为 `models/tensorrt/wespeaker_frames_fp32.engine`，同时写出 ONNX
和 JSON 元数据。只导出 WeSpeaker ResNet 的帧特征卷积计算；fbank、说话人掩码、池化、
分割、聚类和双输出继续使用原有实现。每个音频窗的卷积特征由多个说话人复用。

引擎支持 10 秒音频窗、批量 1～16。换 GPU、TensorRT 版本或模型权重后须在目标机器重建；
加载时校验 GPU 名称、版本、模型权重及引擎哈希。推理不会自动联网或自动构建引擎。

## 使用

```powershell
.\deploy.ps1 -Mode run -PythonExe python -Backend tensorrt -NumSpeakers 3 -OutputDir outputs\tensorrt_run
.\deploy.ps1 -Mode serve -PythonExe python -Backend tensorrt
.\deploy.ps1 -Mode batch -PythonExe python -Backend tensorrt -InputDir D:\audio -OutputDir D:\results
```

可用 `-EnginePath` 指定其它本机引擎。Python 模块入口对应 `--backend tensorrt --engine PATH`。
服务单工作线程顺序使用一个执行上下文；不要让多个线程同时调用同一个 diarizer 对象。

## 本机验收结果

Windows、RTX 4060 Laptop 8 GB、驱动 566.24、torch 2.9.1+cu126。
`huawei_test.wav` 为 181.648254 秒、指定 3 人；各后端预热一次再热运行三次：

| 后端 | 热运行平均 |
|---|---:|
| 官方 torch | 6.538 秒 |
| shared-frames | 2.592 秒 |
| TensorRT FP32 + 共享帧 | 1.860 秒 |

TensorRT 相对 torch 约 3.52 倍，相对 shared-frames 约 1.39 倍。
计时包括音频解码、推理与结果对象封装，不含模型加载及写盘。12 次完整/非冲突分段和标签
全部相同；真实样例 embedding 最大绝对差约 2.18e-6。
证据：`outputs/tensorrt_benchmark_2026-10-19/benchmark.json`。

已通过真实 HTTP 与批处理验收：完整 56 段、非冲突 60 段，与旧基线的全部起止时间和标签
一致；坏音频 failed 后正常任务 succeeded，六个产物均能下载。HTTP 首次正常任务的
处理与封装为 2.507 秒。临时验收服务已关闭。
证据：`outputs/deployment_verify_tensorrt_20261019_200314/verification.json`。

另用已有 AMI ES2011c 的前 120 秒（4 人约束）复验，三后端各运行两次，全部分段与标签
一致；torch 热运行 3.949 秒、TensorRT 1.010 秒，embedding 最大差约 1.06e-6。
未下载新数据；该短片段验证回归一致性，不是 AMI 全集 DER 评估。
证据：`outputs/tensorrt_ami_ES2011c_120s/benchmark.json`。

FP32 数值检查涵盖静音、随机输入、批量 1/7/16；与关闭 TF32 的 PyTorch 参考相比，
最大帧特征差约 6.63e-6，最大池化 embedding 差约 1.12e-6，错误窗长被拒绝。
证据：`outputs/tensorrt_frames_verification.json`。原有 8 项单元测试亦通过。

## 复验

```powershell
python scripts\verify_tensorrt_frames.py
python scripts\benchmark_backends.py --tensorrt --runs 3 --output-dir outputs\tensorrt_benchmark_2026-10-19
python scripts\verify_deployment.py --backend tensorrt
```

帧特征检查涵盖静音、随机输入、批量 1/7/16 及错误窗长拒绝。基准比较官方 torch、共享帧
PyTorch 与 TensorRT 的预热及热运行，检查完整和非冲突输出的全部起止时间及标签。
HTTP/批处理验收还检查坏音频后的恢复和产物下载。

默认后端仍为 `torch`。有限样例的一致性检查不能替代跨数据集 DER 验收；TensorRT
数值计算与 PyTorch 可能存在微小差异。基准中的 `peak_allocated_mb` 只统计 PyTorch
分配器，不能代表 TensorRT 的总显存占用。
