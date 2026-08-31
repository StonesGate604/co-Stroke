# co-Stroke

[English](README.md) | 简体中文

**co-Stroke 是一个以创作过程为中心的人–AI 绘画实验。** 它不只把绘画看作一张完成的图像，而是把它表示为一组有顺序的笔画。每次绘画过程都可以被记录、回放、倒退、编辑、由人类继续、由自回归模型继续，并在未来转换为实体绘图运动。

这个项目提出的不只是机器学习问题，也是一个人机交互问题：

> 当每一笔都可以被记录、回放、预测和续写时，人类和 AI 应该怎样共享同一个创作过程的控制权？

co-Stroke 源自 NYU ITP Code Your Way 课程中的 **AI Drawing Studio** 项目。早期项目探索了浏览器绘画过程的记录与回放；co-Stroke 进一步将技术表示聚焦到笔画序列，并把交互设计推进到混合主动式（mixed-initiative）、轮流进行的人–AI 协同创作。

## 目录

- [为什么使用笔画序列？](#为什么使用笔画序列)
- [HCI 与研究定位](#hci-与研究定位)
- [当前状态](#当前状态)
- [交互模型](#交互模型)
- [功能](#功能)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [界面使用方法](#界面使用方法)
- [本地模型 API](#本地模型-api)
- [co-stroke.json v0.1](#co-strokejson-v01)
- [多模态关系笔画模型 v4.1](#多模态关系笔画模型-v41)
- [连续 stroke-5 Transformer v3](#连续-stroke-5-transformer-v3)
- [人类优先上下文策略 v3.1](#人类优先上下文策略-v31)
- [模型基线](#模型基线)
- [训练](#训练)
- [测试](#测试)
- [项目结构](#项目结构)
- [已知限制](#已知限制)
- [研究机会](#研究机会)
- [路线图](#路线图)

## 为什么使用笔画序列？

多数生成式图像系统以最终的栅格图像为目标，中间的创作过程往往不可见：用户输入提示词，模型返回一个完成结果。co-Stroke 从另一个前提出发：

```text
绘画 = 随时间发生的一组有序动作
```

每一笔都有作者、几何形状、样式、持续时间、时间线位置，以及它与前序动作的关系。保留这些过程信息后，系统可以支持许多仅生成最终图像时难以表达的交互：

- 回放一幅画的构建过程；
- 查看每一笔由谁贡献；
- 在 AI 每完成一笔之后暂停；
- 让用户从当前历史继续绘制；
- 回到较早的笔画并替换后续内容；
- 比较不同的人–AI 轮流创作策略；
- 导出完整过程，用于后续分析或实体执行。

当前 v4.1 模型在完整笔画层面进行自回归预测，同时读取精确的矢量几何和当前部分画布的像素表示：

```text
可见矢量笔画 + 部分画布像素 -> 下一条完整笔画
```

早期 v3 使用了与 SketchRNN stroke-5 相关的动作流表示，该模型仍作为基线保留。但 co-Stroke 不只是一个生成模型；它的重点是完整的交互系统，模型只是共享时间线中的一个参与者。

## HCI 与研究定位

co-Stroke 位于以下研究方向的交叉位置：

- **人–AI 交互（Human–AI Interaction）**：用户如何理解、指挥、中断和回应 AI 协作者；
- **协同创造系统（Co-Creative Systems）**：人类和模型共同贡献时，能动性与作者身份如何协商；
- **创造力支持工具（Creativity Support Tools）**：界面如何支持探索，同时不取代用户的创作劳动；
- **混合主动式交互（Mixed-Initiative Interaction）**：控制权如何在人类和 AI 之间转移；
- **以过程为中心的交互**：时间记录如何支持回放、反思、修订和分支；
- **具身交互与人机协作（未来方向）**：共享的数字笔画历史如何转换为机械臂运动。

当前原型体现了几项设计立场：

1. **协作的基本单位是笔画，而不是完整图像。**
2. **AI 输出必须可以被中断。** 默认界面每次只请求一条 AI 笔画。
3. **贡献来源应当保持可见。** 人类笔画为黑色，AI 笔画为粉色；每个时间线步骤保留作者元数据。
4. **编辑历史本身是一种有意义的创作行为。** 回退后由人类或 AI 添加新笔画时，旧的未来会被截断。
5. **人类输入不应从模型上下文中消失。** v4.1 推理策略保护人类/非 AI 笔画，只保留近期 AI 输出，并为最近的人类笔画提供独立栅格通道。
6. **系统不应鼓励长时间的 AI 自主生成。** 连续生成 12 条 AI 笔画后，界面会要求用户先添加一条人类笔画。

本仓库目前包含一个可以运行的研究原型，但尚未完成正式用户研究。它可以作为研究控制感、作者感、主动权、信任、惊喜、创作能动性与协作策略的平台。

## 当前状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 浏览器绘画界面 | 已实现 | 指针输入、画笔、橡皮擦、实时预览、撤销和清空 |
| 笔画时间线 | 已实现 | 步骤图标、播放、暂停、重置、点击/拖动定位和横向滚动 |
| 回退后编辑 | 已实现 | 新的人类或 AI 笔画会截断选定位置之后的笔画 |
| 贡献来源显示 | 已实现 | 人类和 AI 使用不同颜色，并保留作者元数据 |
| JSON 加载 | 部分实现 | **Load** 当前只加载内置 `simple-house.json`，还不是任意文件选择器 |
| JSON 导出 | 已实现 | 将当前会话下载为格式化的 `co-stroke.json` v0.1 数据 |
| 本地 AI 续画 | 已实现 | 浏览器调用 `127.0.0.1:8787` 上的 Python/PyTorch 服务 |
| 离线模拟续画 | 已实现 | 无法连接本地模型时使用简单回退笔画 |
| 多模态关系笔画模型 v4.1 | 已实现并启用 | 笔画 Transformer + 部分画布 CNN；每回合贡献一条完整笔画 |
| 人类优先上下文压缩 | 已实现 | 最多 24 个笔画单元；保护人类几何，只保留最近六条 AI 笔画 |
| 候选采样与重排序 | 内部已实现 | 每次采样 16 个候选并按几何连接/重叠评分；界面仍只显示选中结果 |
| 自动化测试 | 已实现 | 14 项测试，覆盖旧模型、栅格通道、目标泄漏、梯度、重采样与上下文策略 |
| Quick, Draw! 转换 CLI | 占位 | `scripts/convert_quickdraw.py` 尚未执行真正转换 |
| 多候选选择界面 | 计划中 | 服务内部已经重排候选，但尚未向用户开放比较与人工选择 |
| 持久化时间线分支 | 计划中 | v0.1 仍采用线性“截断未来”策略 |
| 机械臂导出 | 计划中 | 数据结构为未来的校准路径转换器预留了空间 |
| 正式 HCI 用户研究 | 尚未开展 | 研究问题、实验协议与参与者研究仍是后续工作 |

## 交互模型

原型使用一条由所有参与者共享的线性时间线：

```text
人类笔画 -> 人类笔画 -> AI 笔画 -> 人类笔画 -> ...
```

每一条已提交的笔画都是一个时间线步骤。用户可以回放序列、跳转到之前的步骤，并从该位置继续。

### 轮流创作

用户按下 **AI Continue** 后才会发起 AI 回合。浏览器当前请求参数为：

```text
maxStrokes: 1
maxPointsPerStroke: 32
sampling steps: 64
temperature: 0.55
```

这种设置有意让 AI 以渐进方式参与。用户可以看到每一次贡献，然后决定自己继续画、再次请求 AI、回退、擦除或清空。

### 回退与编辑

时间线使用以下策略：

```text
branchPolicy: truncate-future-on-edit
```

如果当前可见历史结束在步骤 `k`，此时加入新的人类或 AI 笔画，步骤 `k` 后面的笔画会先被删除：

```text
原来：A -> B -> C -> D
回退：A -> B
编辑：A -> B -> E
```

因此，当前原型支持历史修订，但还不会把 `C -> D` 保存成可命名的另一个分支。

### 视觉来源标记

- 人类画笔颜色：`#111111`；
- AI 画笔颜色：`#ff44aa`；
- 默认画笔宽度：`4 px`；
- 橡皮擦宽度：`10 px`；
- Sequence 面板显示标题、类别、当前步骤和当前笔画作者；
- 时间线图标可区分画笔、橡皮擦、AI，以及未来的机器人作者。

### AI 连续生成限制

界面会统计可见历史末尾连续出现的 AI 笔画数。达到 12 条后，系统暂停新的 AI 请求并提示用户添加一条人类笔画。这是交互策略而不是模型本身的限制，其目的是保持共同创作循环，避免模型无限使用自己的输出继续生成。

## 功能

### 画布与输入

- `960 x 640` HTML Canvas；
- 取值范围为 `[0, 1]` 的归一化指针坐标；
- 支持鼠标、数位笔和其他兼容 Pointer Events 的输入；
- 记录每个点相对于本条笔画起点的时间 `t`，单位为毫秒；
- 记录压力 `p`；设备不能提供有效压力时使用 `0.5`；
- 移除与前一点距离小于 `0.002` 归一化单位的中间事件；
- 使用二次曲线插值使浏览器渲染更加平滑；
- 使用 Canvas 2D 的 `destination-out` 合成模式实现擦除。

### 时间线

- 每条笔画对应一个步骤；
- 自动播放速度约为每 `420 ms` 显示一条笔画；
- 支持暂停和重置；
- 点击步骤图标可直接定位；
- 点击或拖动时间线轨道可寻找最近步骤；
- 长序列支持横向滚动；
- 显示当前步骤光标以及作者/工具图标。

### 会话操作

- 加载内置房屋示例；
- 将完整绘画过程导出为 JSON；
- 撤销最后一条已存储笔画；
- 清空并创建新的猫类别会话；
- 从较早的时间线步骤继续编辑；
- 选择由人类或 AI 继续下一笔。

### 模型集成

- 本地 HTTP 模型服务；
- 健康检查与续画端点；
- 为独立静态前端提供 CORS；
- 自动识别 checkpoint 类型；
- 支持当前 v4.1 模型以及 v4/v3/v2/v1 checkpoint；
- 服务不可用或响应无效时自动切换到客户端回退；
- 在界面中显示推理上下文统计信息。

## 系统架构

```text
┌────────────────────────────── 浏览器 ──────────────────────────────────┐
│                                                                        │
│  指针输入 ──> 归一化笔画 ──> co-stroke 会话                            │
│                                  │                                     │
│                                  ├──> Canvas 渲染器                     │
│                                  ├──> 时间线播放器/视图                 │
│                                  ├──> JSON 导出                         │
│                                  └──> AI adapter                        │
│                                           │                            │
└───────────────────────────────────────────┼────────────────────────────┘
                                            │ POST /continue
                                            ▼
┌────────────────────────── 本地 Python 服务 ────────────────────────────┐
│ checkpoint 加载 -> 人类优先笔画压缩                                  │
│       ├─> 矢量笔画 Encoder + 关系 Transformer ─┐                      │
│       └─> 三通道部分画布 + CNN Encoder ────────┼─> 融合 -> GMM 笔画解码│
└───────────────────────────────────────────┬────────────────────────────┘
                                            │ co-Stroke 笔画
                                            ▼
                                     浏览器共享时间线
```

### 前端组件

| 文件 | 职责 |
| --- | --- |
| `public/index.html` | 应用外壳、工具栏、画布、序列信息、AI 控制和时间线结构 |
| `src/app.js` | 共享状态、控件绑定、人/AI 回合、导出、撤销和连续生成限制 |
| `src/drawing-input.js` | 指针捕获、坐标归一化、时间/压力、预览和点压缩 |
| `src/timeline-player.js` | 播放、定位、Canvas 2D 渲染和逐笔变化事件 |
| `src/timeline-view.js` | 时间线图标、滚动、拖动定位、当前光标和来源显示 |
| `src/stroke-format.js` | 绘画/笔画创建、旧格式兼容、验证、截断和序列化 |
| `src/ai-adapter.js` | 本地 HTTP adapter、响应验证和模拟回退 |
| `src/styles.css` | 布局与视觉样式 |

### Python 组件

| 文件 | 职责 |
| --- | --- |
| `scripts/train_stroke_multimodal_v41.py` | 当前 v4.1 部分画布渲染、CNN/矢量融合、损失、训练与采样 |
| `scripts/train_stroke_relational_v4.py` | 纯矢量完整笔画关系模型 v4 基线 |
| `scripts/train_stroke5_transformer.py` | 历史连续 stroke-5 v3 数据集、转换、模型、损失、训练与采样 |
| `scripts/serve_stroke_model.py` | checkpoint 加载、推理、上下文压缩、HTTP API 与响应转换 |
| `scripts/train_sketchgpt_primitives.py` | 方向–长度 primitive token v2 基线 |
| `scripts/train_stroke_transformer.py` | 早期量化 `(dx, dy, pen)` 基线 |
| `scripts/convert_quickdraw.py` | 为转换器预留的入口，目前仍是占位脚本 |

## 快速开始

### 环境要求

- 支持 JavaScript Modules、Canvas 2D 和 Pointer Events 的现代浏览器；
- Python 3；
- 运行模型训练、测试或本地推理服务时需要 PyTorch；
- 使用真实 AI 续画时需要训练好的 checkpoint。

前端不需要构建步骤。Lucide 图标通过 jsDelivr 加载，因此图标加载需要网络；HTML 中仍保留文字回退内容。

### 1. 不启动模型运行界面

在项目根目录运行：

```powershell
python -m http.server 8000
```

打开：

```text
http://localhost:8000/public/
```

不要直接通过 `file://` 打开 `public/index.html`。前端使用 JavaScript Modules，并会请求内置示例文件，因此应通过 HTTP 提供服务。

即使不启动模型服务，也可以绘画、擦除、回放、定位、撤销、清空和导出。此时按下 **AI Continue** 会使用模拟续画，以便测试交互循环。

### 2. 使用本地 v4.1 模型运行

保持静态服务器运行，在第二个终端中执行：

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py
```

默认 checkpoint 路径：

```text
runs/stroke-multimodal-v41-cat/checkpoint.pt
```

健康检查地址：`http://127.0.0.1:8787/health`。

服务默认只绑定 `127.0.0.1:8787`。可以指定其他 checkpoint、设备、主机或端口：

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py `
  --checkpoint runs\stroke-multimodal-v41-cat\checkpoint.pt `
  --device cuda `
  --host 127.0.0.1 `
  --port 8787
```

没有 CUDA 时可以使用 `--device cpu`，但推理速度可能较慢。

## 界面使用方法

1. 在画布上绘制一条或多条黑色笔画。
2. 按下 **AI Continue**，请求一条粉色的模型笔画。
3. 自己继续绘画，或再次请求 AI。
4. 使用时间线回放过程或定位到更早的步骤。
5. 在较早位置添加新笔画，以替换原来的未来。
6. 按下 **Export**，将完整过程保存为 JSON。

| 控件 | 行为 |
| --- | --- |
| **Load** | 加载 `data/examples/simple-house.json` |
| **Export** | 下载 `<drawing-title>.json` |
| **Undo** | 删除最后一条已存储笔画 |
| **Clear** | 创建类别为 `cat` 的空会话 |
| **Pen** | 创建人类作者的黑色画笔笔画 |
| **Eraser** | 创建会从渲染结果中移除像素的橡皮擦笔画 |
| **Play** | 逐笔推进序列 |
| **Pause** | 停止播放 |
| **Reset** | 定位到步骤 `0`，但不删除数据 |
| **AI Continue** | 截断不可见未来、压缩上下文并请求一条 AI 笔画 |

## 本地模型 API

模型服务使用 Python 标准库中的 `ThreadingHTTPServer`，提供两个端点。

### `GET /health`

响应示例：

```json
{
  "ok": true,
  "model": "local-stroke-multimodal-v4.1-cat",
  "version": "epoch-12",
  "model_type": "stroke-multimodal-v4.1",
  "checkpoint": "runs/stroke-multimodal-v41-cat/checkpoint.pt",
  "device": "cuda",
  "context_policy": "stroke-multimodal-human-priority-v4.1"
}
```

实际 epoch 取决于加载的 checkpoint。

### `POST /continue`

浏览器发送完整绘画数据和当前选择的时间线边界：

```json
{
  "drawing": {
    "version": "0.1.0",
    "category": "cat",
    "strokes": []
  },
  "currentStep": 0,
  "options": {
    "maxStrokes": 1,
    "maxPointsPerStroke": 32,
    "steps": 64,
    "temperature": 0.55,
    "categoryHint": "cat",
    "style": { "color": "#ff44aa", "width": 4 }
  }
}
```

服务只使用 `drawing.strokes[:currentStep]`，然后采样续画，并返回可以直接加入时间线的 co-Stroke 笔画，以及模型和上下文元数据。可选的 `options.seed` 会在采样前设置 Python 与 PyTorch 随机种子；省略时服务会生成随机种子。

v4.1 响应中的上下文信息示例：

```json
{
  "context": {
    "policy": "stroke-multimodal-human-priority-v4.1",
    "unit": "strokes",
    "maxActions": 24,
    "visibleStrokes": 8,
    "humanStrokesUsed": 4,
    "aiStrokesUsed": 4,
    "candidateCount": 16,
    "selectedOverlap": 0.0625,
    "selectedNearestDistance": 0.01,
    "droppedAIStrokes": 0,
    "compacted": true
  }
}
```

本地开发服务器允许任意来源跨域访问（`Access-Control-Allow-Origin: *`）；公开部署前应重新评估这项配置。

### 客户端回退行为

以下情况会从 `LocalStrokeModelAdapter` 切换到 `MockStrokeModelAdapter`：HTTP 请求失败、服务返回非 2xx 状态，或响应不包含 `strokes` 数组。模拟 adapter 会在当前可见历史终点附近创建一条三个点组成的短粉色笔画。它只是界面回退，不是训练后的模型，也不会返回上下文压缩统计。

## co-stroke.json v0.1

`co-stroke.json` 是 UI、模型服务、示例、导出会话、测试和未来实体输出工具共享的人类可读交换格式。

正式 schema 位于 [`schemas/co-stroke.schema.json`](schemas/co-stroke.schema.json)，详细设计说明位于 [`docs/co-stroke-json-v0.1.md`](docs/co-stroke-json-v0.1.md)。

### 顶层绘画对象

```json
{
  "schema": "https://co-stroke.local/schema/co-stroke-v0.1.json",
  "version": "0.1.0",
  "id": "drawing_...",
  "title": "cat-session",
  "category": "cat",
  "createdAt": "2026-06-01T00:00:00.000Z",
  "updatedAt": "2026-06-01T00:00:00.000Z",
  "source": { "type": "human-session", "name": "local browser session" },
  "canvas": {
    "width": 960,
    "height": 640,
    "coordinateSystem": "normalized",
    "background": "#ffffff"
  },
  "timeline": {
    "unit": "stroke",
    "currentStep": 0,
    "branchPolicy": "truncate-future-on-edit"
  },
  "strokes": []
}
```

绘画对象必需字段为 `version`、`id`、`title`、`canvas` 和 `strokes`。schema 允许额外字段，因此未来可以加入研究记录元数据，而不必立即破坏交换格式。

### 笔画对象

```json
{
  "id": "stroke_001",
  "author": { "type": "human", "id": "local-user" },
  "tool": "pen",
  "style": {
    "color": "#111111",
    "width": 4,
    "opacity": 1,
    "lineCap": "round",
    "lineJoin": "round"
  },
  "timing": { "startMs": 0, "durationMs": 400 },
  "points": [
    { "x": 0.25, "y": 0.70, "t": 0, "p": 0.5 },
    { "x": 0.55, "y": 0.70, "t": 400, "p": 0.5 }
  ],
  "metadata": { "insertedAtStep": 0 }
}
```

schema 支持的作者类型为 `human`、`ai`、`dataset` 和 `robot`；v0.1 支持的工具为 `pen` 与 `eraser`。

### 坐标策略

交换格式保存绝对归一化坐标：

```text
x: 0..1，从左到右
y: 0..1，从上到下
```

归一化坐标使会话不依赖特定屏幕分辨率，简化了缩放回放，也为未来的纸张/机械臂标定提供统一输入。当前 v4.1 模型会将浏览器画布 letterbox 到正方形模型空间，并按完整笔画处理坐标；历史 v3 tokenizer 则把坐标转换为全局相对 `(dx, dy, pen-state)` 动作。

### 时间与压力

点的 `t` 是相对于单条笔画开始时刻的时间。系统会在可用时记录压力 `p`，但当前渲染器和模型尚未使用压力改变宽度或生成结果。存在最后一点时间时，`durationMs` 使用该时间；否则格式归一化代码会估算持续时间。

### 兼容性归一化

`src/stroke-format.js` 支持当前嵌套的 `author` 和 `style`，也会兼容旧版字符串 `author`、顶层 `color` 与 `size` 字段。导出数据统一使用 `0.1.0` 格式和归一化坐标。

## 多模态关系笔画模型 v4.1

V4.1 使用两种同步表示预测下一条完整笔画：

```text
可见笔画 -> 形状/空间/作者/顺序 embedding -> 关系 Transformer
可见笔画 -> 3 x 64 x 64 部分画布         -> CNN 栅格 Encoder
                                             -> 融合上下文
                                             -> 下一笔起点 GMM
                                             -> 相对形状 GMM
```

每条矢量笔画先映射到正方形 Quick, Draw! 模型空间，再按等弧长重采样为 16 个点。三个栅格通道分别保存全部可见笔画、AI 笔画和最近一条人类/非 AI 笔画。训练时只渲染选中的上下文，被隐藏的目标笔画绝不会出现在输入图像中。

训练样本混合了有序前缀续写与随机子集缺失笔画补全。栅格分支使用独立输出头的辅助目标，迫使 CNN 学到有用的视觉状态，同时避免把预训练 v4 输出头拉离原有分布。矢量分支从 v4.0.1 热启动，其微调学习率是新模块的四分之一。

当前本地猫模型使用 69,872 张已识别的 Quick, Draw! 草图，包含 6,837,054 个参数，训练 12 个 epoch。融合主分支的验证损失达到 `-3.7397`，纯矢量 v4.0.1 约为 `-3.6114`。这个似然提升证明栅格画布提供了额外预测信息，但并不等同于视觉质量已经理想。人工测试中仍能看到重复耳朵/弧线，以及长时间 AI 自回归后的退化。

详细设计与训练记录见 [`docs/v4.1-multimodal-architecture.md`](docs/v4.1-multimodal-architecture.md)。

## 连续 stroke-5 Transformer v3

v3 基线是使用简化版 Quick, Draw! 猫类别草图训练的因果 Transformer。

### 动作表示

每个原始点会产生一个 `(dx, dy, pen)` 动作。pen 状态如下：

| 数值 | 名称 | 含义 |
| --- | --- | --- |
| `0` | `PEN_CONTINUE` | 到达新点后继续落笔绘制 |
| `1` | `PEN_LIFT` | 到达新点后抬笔 |
| `2` | `PEN_END` | 绘画结束 |

决定“移动到下一个点时是否可见”的是**前一个动作**的 pen 状态。这个细节保留了从一条笔画终点到下一条笔画起点之间不可见的抬笔移动。如果删除该移动，独立笔画的空间位置就会丢失。

动作流在概念上从画布中心 `(0.5, 0.5)` 开始，最后一条笔画后附加 `(0, 0, 2)` 的 `PEN_END` 动作。

### 数据预处理

v3 数据加载器会：

1. 读取简化版 Quick, Draw! NDJSON；
2. 默认只保留被识别的绘画；
3. 将坐标从 `0..255` 归一化到 `0..1`；
4. 把绝对点转换成全局相对位移；
5. 保留笔画之间的抬笔移动；
6. 添加绘画结束状态；
7. 跳过长度超过 `max_len` 的序列，而不是截断其几何形状；
8. 计算整个数据集的坐标标准差；
9. 训练时使用该尺度归一化坐标目标。

训练样本保存在内存中，因此数据集大小会影响启动时间和内存占用。

### 模型架构

模型输入由以下三项相加：连续 `(dx, dy)` 的线性投影、三种 pen 状态的可学习 embedding、位置 embedding。随后通过 pre-norm 因果 `TransformerEncoder`。

模型有两个输出头：

1. 用于联合 `(dx, dy)` 分布的 **20 分量全协方差二维高斯混合密度头**；
2. 用于 pen 状态的 **三分类头**。

坐标头对每个混合分量预测权重 `π`、均值 `μx/μy`、标准差 `σx/σy` 和相关系数 `ρ`。联合建模避免把水平和垂直运动当作彼此无关的分类选择。

### 当前训练配置

| 参数 | 数值 |
| --- | ---: |
| 请求的 recognized drawings | 70,000 |
| 验证集比例 | 0.04 |
| 最大序列长度 | 192 actions |
| batch size | 64 |
| epochs | 12 |
| 模型宽度 | 384 |
| Transformer 层数 | 6 |
| attention heads | 6 |
| dropout | 0.2 |
| 高斯混合分量 | 20 |
| learning rate | 0.0001 |
| weight decay | 0.01 |
| seed | 7 |
| optimizer | AdamW |
| 学习率调度 | cosine annealing，最低为初始值的 10% |
| gradient clipping | max norm 1.0 |
| mixed precision | CUDA 默认开启，可用 `--no-amp` 关闭 |

当前本地猫类别配置的坐标尺度约为 `0.12495403`。它由所选训练数据计算并保存在 checkpoint 中，不应直接用于其他数据集。

### 目标函数与采样

总损失为坐标负对数似然与 pen 状态交叉熵之和。Padding 会被 mask，坐标损失不计算 `PEN_END` 目标。即使 Transformer 使用 CUDA autocast，概率密度计算仍保持 float32。

推理时，temperature 会同时影响高斯分量选择、分量方差（乘以 `sqrt(temperature)`）和 pen 状态采样。生成位置限制在 `[0.02, 0.98]`，服务端单次位移限制在 `[-0.6, 0.6]`。达到请求笔画数、点数限制、步骤预算或生成 `PEN_END` 时停止。如果有用笔画在预算结束时仍未闭合，服务会补成 `PEN_LIFT`；如果没有得到可解码折线，v3 最多重试三次。

## 人类优先上下文策略 v3.1

浏览器指针输入通常比简化版 Quick, Draw! 数据密集。如果直接发送原始历史，模型窗口会很快填满，近期 AI 输出还可能挤掉用户最初的绘画。服务因此会在 tokenization 前压缩可见历史。

### 策略常量

```text
模型 max_len:                     192
可用历史动作预算:                 191
为人类/非 AI 预留预算:            最多 120 actions
保留的近期 AI 上限:               6 strokes
每条人类笔画点数上限:             16
每条 AI 笔画点数上限:             12
```

### 压缩流程

1. 忽略少于两个点的笔画。
2. 将 AI 笔画与所有非 AI 笔画分开。
3. 按近似相等弧长对选定折线重采样。
4. 为非 AI 几何信息分配最多 120 个动作。
5. 剩余容量最多保留最近六条 AI 笔画。
6. 如果仅保留各笔画端点也超出预算，则在整幅画中均匀选择笔画，并尽量保留首尾。
7. 在转换为 stroke-5 之前，将选中笔画恢复到原时间顺序。

代码会把所有非 `ai` 作者——包括 `human`、`dataset` 和 `robot`——都视为受保护的上下文。界面将其显示为“human”，因为一般交互会话主要包含人类与 AI 两类作者。

响应会报告原始与保留动作数、保留笔画数、丢弃的 AI 笔画数以及是否发生压缩。浏览器显示类似 `Context: 92 human + 48 AI / 191` 的摘要，悬停后可以查看策略细节。

## 模型基线

服务会自动识别多类 checkpoint，而不需要修改浏览器通信协议。

### v4.1：多模态关系笔画模型

- `scripts/train_stroke_multimodal_v41.py`；
- checkpoint 类型为 `stroke-multimodal-v4.1`；
- 矢量笔画 Transformer + 三通道部分画布 CNN；
- 使用混合密度头预测完整笔画的起点和相对形状；
- 当前默认模型。

### v4.0.1：纯矢量关系笔画模型

- `scripts/train_stroke_relational_v4.py`；
- checkpoint 类型为 `stroke-relational-v4`；
- 完整笔画 embedding 与双向关系注意力；
- 16 候选几何重排序与正方形画布 letterbox 映射；
- 保存为 Git tag `v0.4.0.1`。

### v3：连续 stroke-5 Transformer

- `scripts/train_stroke5_transformer.py`；
- checkpoint 类型为 `stroke5-transformer-v3`；
- 连续联合 `(dx, dy)` 混合密度预测；
- 单独的 pen 分类预测；
- 保留笔画间不可见移动；
- 历史动作级基线。

### v2：方向–长度 primitive Transformer

- `scripts/train_sketchgpt_primitives.py`；
- checkpoint 类型为 `sketchgpt-segments-v2`；
- 将 48 个方向之一与 8 个长度区间之一组合成单个 token；
- 使用 `BOS`、`STROKE_END`、`EOS` 和 `PAD`；
- 采样时使用重复方向惩罚；
- 用于缓解早期重复方向编码造成的直线循环问题。

### v1：量化位移 Transformer

- `scripts/train_stroke_transformer.py`；
- checkpoint 类型为 `stroke-transformer`；
- 将 `dx` 和 `dy` 分别量化为 121 个区间；
- 分别预测 `dx`、`dy` 与 pen 分布；
- 是用于打通数据、模型和前端循环的第一版基线。

缺少 `token_encoding` 字段的旧 primitive checkpoint 会按 `repeated-direction` 旧格式解释。

## 训练

训练脚本需要简化版 Quick, Draw! NDJSON。当前命令使用 `data/quickdraw/cat.ndjson`。原始 NDJSON、checkpoint 和 `runs/` 都被 Git 忽略，因此克隆仓库不会自动获得训练数据和本地模型权重。

### 训练当前多模态 v4.1 模型

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

最佳 checkpoint 按融合主分支的验证损失选择，而不是按“主任务 + 栅格辅助任务”的总目标选择。当前训练选中 epoch 12，`val_loss = -3.739664`。没有 CUDA 时可以使用 `--device cpu`，并按需添加 `--no-amp`，但完整训练会明显更慢。

### 训练连续 v3 模型

```powershell
.\.venv\Scripts\python.exe scripts\train_stroke5_transformer.py `
  --data data\quickdraw\cat.ndjson `
  --out-dir runs\stroke5-transformer-v3-cat `
  --max-drawings 70000 `
  --epochs 12 `
  --batch-size 64 `
  --d-model 384 `
  --layers 6 `
  --heads 6 `
  --mixtures 20 `
  --dropout 0.2
```

其他参数包括 `--include-unrecognized`、`--val-fraction`、`--max-len`、`--lr`、`--weight-decay`、`--seed`、`--device` 与 `--no-amp`。

训练输出：

| 文件 | 含义 |
| --- | --- |
| `config.json` | 完整配置与计算得到的坐标尺度 |
| `latest.pt` | 最近一个 epoch 的 checkpoint |
| `checkpoint.pt` | 当前验证损失最低的 checkpoint |
| `sample.json` | co-Stroke 格式的无条件自回归样本 |

控制台会报告训练集与验证集的总损失、坐标损失、pen 损失、学习率、参数量、坐标尺度，以及因序列过长而跳过的绘画数量。

### 训练 v2 方向–长度基线

```powershell
.\.venv\Scripts\python.exe -u scripts\train_sketchgpt_primitives.py `
  --data data\quickdraw\cat.ndjson `
  --out-dir runs\sketchgpt-segments-v2-cat `
  --max-drawings 50000 `
  --max-len 192 `
  --batch-size 128 `
  --epochs 10 `
  --d-model 256 `
  --layers 6 `
  --heads 8 `
  --dropout 0.1 `
  --lr 0.0003 `
  --device cuda
```

### 可复现性说明

- 训练时 Python 与 PyTorch 默认 seed 为 `7`；
- 有 CUDA 时也会设置 CUDA seed；
- v3 CUDA 训练允许 TF32 矩阵乘法；
- 训练/验证划分使用带 seed 的 `torch.Generator`；
- 服务采样默认具有随机性，除非请求提供 `options.seed`；
- 硬件以及 PyTorch/CUDA 版本仍可能造成数值差异。

## 测试

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前测试覆盖：

- Quick, Draw! -> stroke-5 -> 折线的往返转换；
- 独立笔画锚点和抬笔移动的保留；
- `CONTINUE`、`LIFT` 与 `END` 状态位置；
- Transformer 输出维度和有限损失值；
- v4/v4.1 完整笔画输出维度与采样；
- 三通道部分画布栅格化；
- 隐藏目标笔画不会泄漏到像素输入；
- Raster Encoder 获得非零梯度；
- 弧长重采样与端点保留；
- 每条笔画的点数上限；
- 人类上下文容量预留；
- 在人类几何之前优先丢弃较旧 AI 笔画；
- 压缩后上下文不超过模型预算。

浏览器交互、schema validator 集成、视觉回归与端到端 HTTP 测试尚未加入。

## 项目结构

```text
co-Stroke/
├── README.md                              # 英文说明
├── README.zh-CN.md                        # 中文说明
├── blog/
│   └── 001-introduction.md                # 项目最初动机
├── data/
│   └── examples/simple-house.json         # 内置示例
├── docs/
│   ├── co-stroke-json-v0.1.md             # 当前格式设计
│   ├── v4-architecture.md                  # 纯矢量关系模型 v4 设计
│   ├── v4.1-multimodal-architecture.md    # 当前矢量/栅格设计与训练结果
│   ├── stroke-format.md                    # 早期简化格式说明
│   └── conversations/                      # 开发过程记录
├── public/index.html                       # 浏览器应用外壳
├── schemas/co-stroke.schema.json           # JSON Schema draft 2020-12
├── scripts/
│   ├── convert_quickdraw.py                # 占位转换器
│   ├── serve_stroke_model.py               # 本地推理 HTTP 服务
│   ├── train_stroke_multimodal_v41.py      # 当前矢量 + 栅格 v4.1 模型
│   ├── train_stroke_relational_v4.py       # 纯矢量关系 v4 基线
│   ├── train_sketchgpt_primitives.py       # primitive 基线
│   ├── train_stroke_transformer.py         # 量化位移基线
│   └── train_stroke5_transformer.py        # 连续 stroke-5 v3 基线
├── src/
│   ├── ai-adapter.js
│   ├── app.js
│   ├── drawing-input.js
│   ├── stroke-format.js
│   ├── styles.css
│   ├── timeline-player.js
│   └── timeline-view.js
└── tests/
    ├── test_context_packing.py
    ├── test_stroke5_transformer.py
    ├── test_stroke_relational_v4.py
    └── test_stroke_multimodal_v41.py
```

本地通常还会存在 `data/quickdraw/`、`runs/` 和 `.venv/`，这些目录不会提交到 Git。

## 已知限制

### 交互与界面

- **Load 不是通用导入器。** 当前始终加载内置房屋示例。
- **时间线仍是线性的。** 被替换的未来会被删除，而不是保留为分支。
- **没有持久存储。** 刷新页面会重置未导出的浏览器状态。
- **没有候选结果比较。** 界面每次只请求一个续画结果。
- **没有独立拒绝按钮。** 用户需要通过撤销或回退拒绝 AI 结果。
- **尚未进行键盘与无障碍审查。** 键盘、屏幕阅读器、对比度和触屏体验仍需评估。
- **擦除历史不是模型的语义输入。** v4/v4.1 会排除橡皮擦轨迹；模型只能看到剩余画笔几何，无法理解什么内容曾被删除。
- **压力只被记录，尚未使用。** 当前不影响渲染、生成或机械臂输出。
- **新会话类别固定。** Clear 会创建 `cat` 会话，以配合当前猫类别模型。

### 模型与数据

- **单类别 checkpoint。** 当前没有学习 category token。
- **上下文有限。** v4.1 最多保留 24 个笔画单元和最近六条 AI 笔画。
- **训练与交互存在分布差异。** Quick, Draw! 与缓慢、反复的人–AI 共创过程不同。
- **缺少显式语义规划。** 栅格输入改善了验证似然，但当前没有猫部件词表、缺失部件目标或构图计划。
- **长 rollout 仍会退化。** 模型连续使用自己的输出作为上下文时，仍会重复耳朵/弧线并生成穿越主体的长线。
- **只有几何重排序。** 16 个候选会按连接、重叠、孤立和边界塌缩评分，但尚未评价可识别性、新颖性、近期笔画相似度或用户意图。
- **栅格上下文被压缩。** CNN 将空间特征图投影为单个上下文向量，尚未通过 cross-attention 暴露空间 token。
- **缺少正式输出评价。** 已有验证 NLL，但还没有可识别性、多样性和人类偏好指标。
- **数据集全部载入内存。** 较大数据集需要较多 RAM。
- **没有资源安装器。** 数据和 checkpoint 不提交，且尚无自动下载/配置命令。

### 研究

- 尚未完成形成性研究或控制实验；
- 没有预注册假设；
- 没有参与者事件记录系统；
- 没有集成经过验证的问卷；
- 没有定性编码协议；
- 尚无正式发表结果。

这些限制用于准确说明当前研究原型所处阶段，并不表示系统已经可以直接用于正式实验。

## 研究机会

该系统可以支持以下问题：

- AI 应该在绘画过程中的什么时机介入？
- 与一次性完成相比，逐笔生成是否能保留更强的用户控制感？
- 明确标记 AI 来源并提供过程回放，会怎样影响作者身份判断？
- 什么样的可预测性与惊喜平衡最有利于创作探索？
- 保护人类上下文是否会影响信任和继续协作的意愿？
- 用户如何把回退和撤销当作与 AI 协商的机制？
- 用户应该在多个候选中选择，还是直接编辑一个建议？
- AI 自动主动介入与用户主动请求 AI 有什么体验差异？
- 机械臂执行共同作品后，用户对作品所有权的感受是否变化？

一种可能的对照研究设计：

| 实验条件 | 说明 |
| --- | --- |
| Human-only | 用户在没有模型帮助的情况下绘画 |
| One-shot AI | AI 一次生成相对完整的续画 |
| Stroke-level co-creation | 人类与 AI 通过可中断的时间线笔画轮流创作 |

行为指标可以包括完成时间、AI 请求次数、撤销/回退次数、被替换和保留的 AI 笔画、每轮长度，以及用户对不同历史的探索。体验指标可以包括控制感、作者感、能动性、创造力支持、满意度、信任与惊喜程度。访谈可以关注协作策略、冲突时刻、用户如何理解 AI 意图，以及用户如何判断某个贡献是否属于“自己的”绘画。

在正式研究前，原型还需要加入事件日志、参与者/会话 ID、可配置实验条件、知情同意与隐私处理、可复现任务提示，以及适合研究使用的数据导出流程。

## 路线图

### 近期

1. 增加任意 JSON 导入、schema 验证与可见错误提示。
2. 增加多个续画候选以及接受、拒绝和重新生成操作。
3. 将替代未来保存为可命名分支。
4. 加入适合试点研究的会话与事件日志。
5. 增加浏览器测试与端到端服务测试。
6. 在验证 NLL 之外增加草图可识别性评价。

### 模型发展

1. 增加近期 AI 笔画形状相似度与栅格占用惩罚，抑制重复结构。
2. 将 CNN 特征图单元保留为空间 token，通过 cross-attention 与笔画 token 融合。
3. 在扩展猫以外类别前，加入可识别性增益或缺失部件监督。
4. 使用损坏或模型生成的上下文训练，缩小长 rollout 的分布差异。
5. 在界面中提供多个候选供用户选择，并评价多样性、匹配度与可识别性。
6. 在训练多个类别前加入 category 条件控制。

### HCI 研究发展

1. 开展形成性访谈与 think-aloud 研究。
2. 根据观察到的用户策略改进交互。
3. 确定假设、实验条件、测量指标与分析流程。
4. 在较大规模控制实验前完成试点研究。
5. 同时报告行为证据和定性发现。

### 实体输出

机械臂导出应保持为独立的校准转换器：

```text
归一化 x/y -> 校准后的纸张坐标
笔画开始     -> 落笔
笔画结束     -> 抬笔
样式/压力    -> 可选速度或力度映射
```

将交换数据与设备运动分离，可以让浏览器、模型和实体执行器保持松耦合。

## 项目起源

co-Stroke 起源于这样一个观察：AI 辅助绘画中最有趣的问题，可能不是 AI 能否生成一张精致图像，而是用户能否看到、中断、重定向、重新解释，并共同承担图像形成过程的责任。

因此，这个项目有意同时保持技术性与交互导向：它既是一个数据格式、时间界面和自回归模型，也是一套用于研究创作能动性如何在人类、AI，以及未来的实体机器之间转移的平台。
