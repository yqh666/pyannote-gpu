# huawei_test 验证记录

日期：2026-09-11

## 输入

| 项目 | 值 |
|---|---|
| 文件 | `example/huawei_test.wav` |
| SHA-256 | `fb81a44768b6f51088a997e0d4f5d3bcbcc4836e688f7b08c16386f5627ddac2` |
| 时长 | 181.648254 秒 |
| 采样率 | 44,100 Hz |
| 声道 | 2 |
| 位深 | 16-bit PCM |
| 推理人数 | 指定为 3 |

## 环境

| 项目 | 值 |
|---|---|
| 系统 | Windows |
| GPU | NVIDIA RTX 4060 Laptop 8 GB |
| CUDA | 12.6 |
| Python | 3.11.16 |
| torch | 2.9.1+cu126 |
| torchaudio | 2.9.1+cu126 |
| pyannote.audio | 4.0.7 |
| TorchCodec | 0.9.1 |
| FFmpeg | 8.0.1 full shared |
| huggingface_hub | 1.30.0 |
| numpy | 2.4.6 |
| 模型 | `pyannote/speaker-diarization-community-1` |

## 解码验证

输入采用原生文件路径交给 pyannote。文件先由 FFmpeg 8.0.1 shared 和
TorchCodec 0.9.1 的 `AudioDecoder` 验证可读。

| 项目 | 值 |
|---|---|
| 编码 | PCM 16-bit |
| 采样率 | 44,100 Hz |
| 声道 | 2 |
| 解码张量 | `(2, 8010688)` |

## 结果

| 指标 | 完整版 | 非冲突版 |
|---|---:|---:|
| 输出字段 | `speaker_diarization` | `exclusive_speaker_diarization` |
| 说话人数 | 3 | 3 |
| 段数 | 56 | 60 |
| 语音并集时长 | 65.458125 秒 | 65.458125 秒 |
| 重叠时长 | 9.500625 秒 | 0.000000 秒 |
| 官方 torch 推理耗时 | 约 7.0 秒 | 同一次推理，无额外耗时 |
| 官方 torch RTF | 约 0.039 | 同一次推理 |

示例结果的固定文件位于：

```text
example/output/complete/
example/output/non_conflict/
example/output/summary.json
```

## 契约检查

```powershell
python -m unittest discover -s tests -v
```

检查重点：

1. 完整版至少存在一个重叠区间。
2. 非冲突版不存在重叠区间。
3. 两套输出时间范围合法。
4. 两套输出说话人集合一致。

## 说明

完整版和非冲突版属于同一个 Community-1 聚类结果的两种时间轴表达。非冲突版会把重叠帧
分配给其中一个说话人，因此并不增加识别能力，也不会修复人数估计错误。

本文保留早期官方 torch 验证记录。2026-10-19 的 TensorRT 验收结果见
[TENSORRT.md](TENSORRT.md)：相同输入的三次热运行平均 1.860 秒，完整与非冲突分段均与
官方 torch 基线一致。
