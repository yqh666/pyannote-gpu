# GitHub 发布清单

本仓库按“可审查源码仓库”准备。GitHub 中保留代码、文档、测试、固定依赖清单和小型结果样例；
不提交真实语音、大型运行库、已解包依赖、模型权重和 GPU 专用 TensorRT 引擎。

## 推送前检查

```powershell
git status --short
git diff --cached --stat
git ls-files | ForEach-Object { Get-Item -LiteralPath $_ } |
  Sort-Object Length -Descending |
  Select-Object -First 20 FullName, Length
```

确认输出中不包含：

- `.env`、token、私钥、凭据文件；
- `example/*.wav` 等真实语音；
- `tools/gpu-wheels/`、`tools/gpu-packages/`；
- `models/tensorrt/`、模型权重和 `outputs/`；
- `vendor/ffmpeg/*.zip` 大型第三方二进制。

## 首次提交

```powershell
git add .
git status --short
git commit -m "Initial GitHub-ready release"
```

在 GitHub 创建一个空仓库后，再添加其真实 URL 并推送：

```powershell
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

不要把真实 token 写进命令、README 或远程 URL。公开仓库建议开启 GitHub 的 secret scanning、
push protection 和 Dependabot alerts。GitHub 的大文件与安全说明：

- <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>
- <https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning>

## GitHub 仓库与离线交付包的区别

GitHub 克隆后，使用者需要：

1. 自行准备 FFmpeg 归档，要求见 `vendor/ffmpeg/README.md`；
2. 提供自己的测试音频；
3. 接受 Community-1 许可并下载模型，或提供许可允许的本地模型目录；
4. 仅在选择可选 TensorRT 后端时，按 `docs/TENSORRT.md` 下载 1,539,565,572 字节
   （约 1.540 GB / 1.434 GiB）的固定依赖，并在目标 GPU 上构建引擎。

其中 Python/PyTorch/pyannote 与 Community-1 模型权重属于基础运行准备，不包含在上述
TensorRT 下载量中。GitHub 当前跟踪文件约 2.34 MB；另行准备的 FFmpeg 归档为
94,371,734 字节。完整下载范围见根目录 `README.md` 的“下载内容清单”。

完整离线交付包可以额外携带 FFmpeg 归档和 `tools/gpu-wheels/`，但仍不应把当前 GPU 的
TensorRT engine 当作跨机器通用文件。
