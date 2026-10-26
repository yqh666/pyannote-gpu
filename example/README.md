# huawei_test 示例

GitHub 仓库包含：

- `output/complete/`：完整版，保留重叠，共 56 段。
- `output/non_conflict/`：官方独占版，无时间冲突，共 60 段。
- `output/summary.json`：两套输出的指标汇总。

本机验收使用了 `huawei_test.wav`（181.648 秒、44.1 kHz、双声道），但它包含真实语音，
因此被 `.gitignore` 排除，不随 GitHub 源码发布。复现时请提供自己的音频，或在有权再分发时
通过单独的离线交付包提供该文件。

运行前先完成 FFmpeg/TorchCodec 安装：

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\install_ffmpeg_native.ps1
```

复现命令：

```powershell
python scripts\run_community1.py `
  --audio D:\audio\meeting.wav `
  --output-dir outputs\meeting `
  --num-speakers 3
```

验证已有输出：

```powershell
python -m unittest discover -s tests -v
```

本机验收音频 SHA-256（用于核对单独交付的授权副本）：

```text
fb81a44768b6f51088a997e0d4f5d3bcbcc4836e688f7b08c16386f5627ddac2
```
