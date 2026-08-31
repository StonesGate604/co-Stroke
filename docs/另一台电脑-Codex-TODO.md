# co-Stroke 跨电脑接续 TODO

> 给另一台电脑上的 Codex：请按顺序执行下面的任务。项目当前稳定里程碑为 v4.1，主要模型是“矢量笔画 Transformer + 部分画布 CNN”的多模态自回归模型。

## 目标

- 从 GitHub 克隆 co-Stroke；
- 下载 Quick, Draw! 的 `cat` 简化版 NDJSON 训练数据；
- 恢复 v4.1 模型 checkpoint；
- 运行测试；
- 启动网页和本地模型服务；
- 验证浏览器中的 AI Continue 确实调用 v4.1，而不是 mock fallback。

## TODO

### 1. 克隆并打开项目

```powershell
git clone https://github.com/StonesGate604/co-Stroke.git
cd co-Stroke
git checkout master
git pull --ff-only
```

在 Codex 中打开刚克隆的本地 `co-Stroke` 文件夹，并先阅读：

- `README.zh-CN.md`
- `docs/v4.1-multimodal-architecture.md`
- 本文件

### 2. 创建 Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch
```

若电脑有 NVIDIA GPU，应根据该电脑的 CUDA 环境从 PyTorch 官方安装页选择匹配的安装命令，不要盲目沿用旧电脑的 CUDA wheel。

### 3. 下载 Quick, Draw! 猫类别训练数据

推荐直接运行项目同步脚本。它会下载 Quick, Draw! 猫数据和当前正式 v4.1 checkpoint，并验证 checkpoint 的 SHA-256：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_project_assets.ps1
```

仓库当前是私有仓库。同步脚本会安全地复用 Git Credential Manager 中已有的 GitHub 登录凭据，不会把 token 写进项目。因为另一台电脑需要先成功克隆这个私有仓库，所以正常情况下凭据已经存在；若脚本提示没有 GitHub credential，请先在该电脑上完成 GitHub 登录。

如需同时下载 v3.1、v4.0.1、v4.1 三个正式 checkpoint：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_project_assets.ps1 `
  -IncludeAllFormalCheckpoints
```

以下是只下载训练数据时的手动方式：

```powershell
New-Item -ItemType Directory -Force data\quickdraw
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/quickdraw_dataset/full/simplified/cat.ndjson" `
  -OutFile "data\quickdraw\cat.ndjson"
Get-Item data\quickdraw\cat.ndjson | Select-Object FullName, Length
```

预期文件路径：`data\quickdraw\cat.ndjson`。该文件约 76 MB，属于训练数据，不在 Git 仓库中。

### 4. 恢复训练好的 v4.1 checkpoint

上一步的同步脚本会从 GitHub Release 下载下面的文件并放到正确位置：

```text
runs/stroke-multimodal-v41-cat/checkpoint.pt
```

检查：

```powershell
Get-Item runs\stroke-multimodal-v41-cat\checkpoint.pt |
  Select-Object FullName, Length
```

当前正式 checkpoint 大约为 27.4 MB。普通 `git clone` 不会获得它，因为仓库的 `.gitignore` 忽略了 `runs/` 和 `*.pt`。

如需从断点继续训练，还应另外复制：

```text
runs/stroke-multimodal-v41-cat/latest.pt
runs/stroke-multimodal-v41-cat/config.json
```

### 5. 运行测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

记录测试总数与失败信息；不要在测试失败时直接改模型结构，先定位是环境、路径还是代码问题。

### 6. 启动静态网页

在第一个终端中运行：

```powershell
.\.venv\Scripts\python.exe -m http.server 8000
```

打开：

```text
http://127.0.0.1:8000/public/
```

### 7. 启动 v4.1 模型服务

在第二个终端中运行（有可用 CUDA 时）：

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py `
  --checkpoint runs\stroke-multimodal-v41-cat\checkpoint.pt `
  --device cuda
```

没有 CUDA 时改为：

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py `
  --checkpoint runs\stroke-multimodal-v41-cat\checkpoint.pt `
  --device cpu
```

健康检查地址：

```text
http://127.0.0.1:8787/health
```

确认响应中的模型类型为 `stroke-multimodal-v4.1`，再回到网页测试 AI Continue。若模型服务不可用，前端可能退回 mock stroke，因此不能只凭网页按钮可点击就判断模型已加载。

### 8. 可选：重新训练 v4.1

正式训练命令依赖旧版 v4 checkpoint 进行初始化：

```powershell
.\.venv\Scripts\python.exe -u scripts\train_stroke_multimodal_v41.py `
  --data data\quickdraw\cat.ndjson `
  --out-dir runs\stroke-multimodal-v41-cat `
  --max-drawings 70000 `
  --epochs 12 `
  --batch-size 192 `
  --init-checkpoint runs\stroke-relational-v4-cat\checkpoint.pt `
  --device cuda
```

若没有 `runs/stroke-relational-v4-cat/checkpoint.pt`，不要直接声称已复现原训练条件；可以从头训练，但那是不同实验，需要使用新的输出目录并记录配置。

## 完成标准

- `git status` 没有意外生成的大文件；
- `data/quickdraw/cat.ndjson` 存在；
- v4.1 `checkpoint.pt` 存在；
- 单元测试通过；
- `http://127.0.0.1:8000/public/` 可访问；
- `http://127.0.0.1:8787/health` 报告 v4.1；
- AI Continue 的请求由本地模型服务响应，而不是 mock fallback。
