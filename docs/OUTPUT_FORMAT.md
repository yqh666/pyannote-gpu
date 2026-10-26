# 输出格式

每次推理写出两个独立目录，二者来自同一次 Community-1 调用。

## 目录

```text
output/
├─ complete/
│  ├─ diarization.txt
│  ├─ diarization.rttm
│  └─ segments.json
├─ non_conflict/
│  ├─ diarization.txt
│  ├─ diarization.rttm
│  └─ segments.json
└─ summary.json
```

## TXT

制表符分隔，无表头：

```text
0.031	1.195	SPEAKER_00
5.245	6.562	SPEAKER_00
```

字段依次是开始秒数、结束秒数、说话人标签。

## RTTM

标准 RTTM：

```text
SPEAKER huawei_test 1 0.031 1.165 <NA> <NA> SPEAKER_00 <NA> <NA>
```

第 4 字段为起点，第 5 字段为持续时长。

## segments.json

```json
{
  "mode": "complete",
  "model": "pyannote/speaker-diarization-community-1",
  "device": "cuda",
  "audio": "D:\\path\\huawei_test.wav",
  "duration_seconds": 181.648254,
  "num_speakers": 3,
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "segment_count": 56,
  "overlap_seconds": 9.500625,
  "speech_seconds": 65.458125,
  "segments": [
    {"start": 0.031, "end": 1.1953, "speaker": "SPEAKER_00"}
  ]
}
```

`mode` 为 `complete` 或 `non_conflict`。`overlap_seconds` 是所有说话时段之和减去
时间轴覆盖并集后的时长。非冲突版应为 0。
`speech_seconds` 是这些分段覆盖的语音并集时长，不重复统计重叠。
`summary.json` 另含 `backend`（`torch`、`shared-frames` 或 `tensorrt`），用于追溯执行方式。

## summary.json

`summary.json` 同时记录两套输出的段数、重叠秒数和模型信息，并列出六个产物路径。
示例：

```json
{
  "backend": "tensorrt",
  "complete": {
    "segment_count": 56,
    "overlap_seconds": 9.500625,
    "has_overlap": true
  },
  "non_conflict": {
    "segment_count": 60,
    "overlap_seconds": 0.0,
    "has_overlap": false
  }
}
```

## 下游选择

- ASR/字幕：使用 `non_conflict`，保证每个音频时刻只路由给一个说话人。
- 重叠分析：使用 `complete`，检查多人同时说话。
- 比较实验：同时保存两份并计算 `overlap_seconds` 差异。
