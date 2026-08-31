# co-Stroke

English | [简体中文](README.zh-CN.md)

**co-Stroke is a process-based human–AI drawing experiment.** It treats a drawing not only as a finished image, but as an ordered sequence of strokes that can be recorded, replayed, rewound, edited, continued by a person, continued by an autoregressive model, and eventually translated into physical drawing motion.

The project asks a human–computer interaction question as much as a machine-learning question:

> If every stroke can be recorded, replayed, predicted, and continued, how can a human and an AI share control of the same creative process?

co-Stroke grows out of **AI Drawing Studio**, a Code Your Way project at NYU ITP. The earlier project explored recording and replaying browser drawing sessions. co-Stroke narrows the technical representation to stroke sequences and expands the interaction design toward mixed-initiative, turn-based human–AI co-creation.

## Contents

- [Why stroke sequences?](#why-stroke-sequences)
- [HCI and research framing](#hci-and-research-framing)
- [Current status](#current-status)
- [Interaction model](#interaction-model)
- [Features](#features)
- [System architecture](#system-architecture)
- [Quick start](#quick-start)
- [Using the interface](#using-the-interface)
- [Local model API](#local-model-api)
- [co-stroke.json v0.1](#co-strokejson-v01)
- [Multimodal relational model v4.1](#multimodal-relational-model-v41)
- [Continuous stroke-5 Transformer v3](#continuous-stroke-5-transformer-v3)
- [Human-priority context policy v3.1](#human-priority-context-policy-v31)
- [Model baselines](#model-baselines)
- [Training](#training)
- [Testing](#testing)
- [Repository structure](#repository-structure)
- [Known limitations](#known-limitations)
- [Research opportunities](#research-opportunities)
- [Roadmap](#roadmap)

## Why stroke sequences?

Most generative image systems optimize for a final raster image. The intermediate creative process is hidden: a person supplies a prompt and receives a completed result. co-Stroke starts from a different assumption:

```text
drawing = ordered actions over time
```

Each stroke has an author, geometry, style, duration, position in the timeline, and relationship to previous actions. Preserving this process makes several interactions possible that are difficult to express with final-image generation:

- replay the construction of a drawing;
- inspect who contributed each stroke;
- pause AI generation after a single contribution;
- let a person continue from the current history;
- rewind to an earlier stroke and replace the future;
- compare different human–AI turn-taking policies;
- export the same process for later analysis or physical execution.

The current v4.1 model is autoregressive at the stroke level and conditions on
both exact vector geometry and a rasterized view of the visible partial drawing:

```text
visible vector strokes + partial-canvas pixels -> one next complete stroke
```

Earlier v3 work used a SketchRNN-related stroke-5 action stream. That model
remains available as a baseline, but the surrounding project is an interactive
system: the model is only one participant in a shared timeline.

## HCI and research framing

co-Stroke sits at the intersection of:

- **Human–AI interaction** — how people understand, direct, interrupt, and respond to an AI collaborator;
- **co-creative systems** — how agency and authorship are negotiated when both human and model contribute;
- **creativity support tools** — how an interface supports exploration without replacing the user's creative work;
- **mixed-initiative interaction** — how control moves between the human and the AI;
- **process-centered interaction** — how a temporal record supports replay, reflection, revision, and branching;
- **embodied interaction / HRI**, as a future direction — how a shared digital stroke history can become robot-arm motion.

The current prototype embodies several design positions:

1. **The unit of collaboration is a stroke, not a finished image.**
2. **AI output remains interruptible.** The default interface requests one AI stroke at a time.
3. **Provenance should remain visible.** Human strokes are black and AI strokes are pink, and timeline steps retain author metadata.
4. **Editing history is a meaningful creative action.** Drawing or requesting AI output after rewinding truncates the old future.
5. **Human input should not disappear from the model context.** The v4.1 inference policy protects human/non-AI strokes, retains only recent AI output, and gives the raster encoder a dedicated latest-human channel.
6. **Long autonomous rollouts should be discouraged.** After 12 consecutive AI strokes, the interface asks for a new human stroke.

This repository currently contains a working research prototype, not a completed user study. It can support future experiments on control, authorship, initiative, trust, surprise, creative agency, and collaboration strategy.

## Current status

| Area | Status | Notes |
| --- | --- | --- |
| Browser drawing interface | Implemented | Pointer input, pen, eraser, preview, undo, and clear |
| Stroke timeline | Implemented | Step icons, playback, pause, reset, click/drag seeking, horizontal scrolling |
| Edit-after-rewind | Implemented | New human or AI work truncates strokes after the selected point |
| Provenance display | Implemented | Human and AI strokes use different colors and author metadata |
| JSON loading | Partially implemented | **Load** currently loads the bundled `simple-house.json` example, not an arbitrary file picker |
| JSON export | Implemented | Downloads the current session as formatted `co-stroke.json` v0.1 data |
| Local AI continuation | Implemented | Browser calls a local Python/PyTorch server at `127.0.0.1:8787` |
| Offline/mock continuation | Implemented | A simple fallback appears if the local model cannot be reached |
| Multimodal relational model v4.1 | Implemented and active | Stroke Transformer + partial-canvas CNN; one complete stroke per turn |
| Human-priority context packing | Implemented | Up to 24 stroke units; human geometry protected and only six recent AI strokes retained |
| Candidate sampling/reranking | Implemented internally | Samples 16 candidates and applies geometric connection/overlap scoring; the UI still shows only the selected result |
| Automated tests | Implemented | 14 tests covering earlier models, raster channels, target-leakage prevention, gradients, resampling, and context policies |
| Quick, Draw! converter CLI | Placeholder | `scripts/convert_quickdraw.py` does not yet perform conversion |
| Multiple-candidate UI | Planned | The service reranks candidates internally; comparison and human selection are not yet exposed |
| Persistent branches | Planned | v0.1 uses a linear truncate-future policy |
| Robot-arm export | Planned | The schema is designed to support a later calibrated path converter |
| Formal HCI user study | Not yet conducted | Research questions and experimental protocols remain future work |

## Interaction model

The prototype uses one shared linear timeline:

```text
human stroke -> human stroke -> AI stroke -> human stroke -> ...
```

Every committed stroke is one timeline step. A user can replay the sequence, jump to a previous step, and continue from there.

### Turn taking

The human always initiates an AI turn by pressing **AI Continue**. The browser currently requests:

```text
maxStrokes: 1
maxPointsPerStroke: 32
sampling steps: 64
temperature: 0.55
```

This deliberately keeps AI participation incremental. The person sees each contribution and decides whether to continue drawing, ask again, rewind, erase, or clear the session.

### Rewinding and editing

The timeline uses:

```text
branchPolicy: truncate-future-on-edit
```

If the visible history ends at step `k` and a new human or AI stroke is added, strokes after `k` are removed before the new stroke is appended:

```text
before: A -> B -> C -> D
rewind: A -> B
edit:   A -> B -> E
```

The prototype therefore supports revision but does not yet preserve `C -> D` as a named branch.

### Visual provenance

- human pen strokes: `#111111`;
- AI pen strokes: `#ff44aa`;
- default pen width: `4 px`;
- eraser width: `10 px`;
- the sequence panel reports title, category, current step, and current stroke author;
- timeline icons distinguish pen, eraser, AI, and future robot-authored steps.

### AI-rollout guardrail

The UI counts consecutive AI-authored strokes at the end of the visible history. At 12 consecutive AI strokes, it pauses further AI requests and asks the user to add a human stroke. This is an interaction policy rather than a model constraint. It keeps the system in a co-creative loop instead of allowing an indefinitely self-conditioned rollout.

## Features

### Canvas and input

- `960 x 640` HTML canvas;
- normalized pointer coordinates in the range `[0, 1]`;
- mouse, pen, or other Pointer Events-compatible input;
- per-point relative time `t` in milliseconds;
- per-point pressure `p`, with a `0.5` fallback when usable device pressure is unavailable;
- point compaction that removes intermediate events closer than `0.002` normalized units;
- quadratic curve interpolation for smoother browser rendering;
- erasing through Canvas 2D `destination-out` compositing.

### Timeline

- one step per stroke;
- automatic playback at approximately one stroke every `420 ms`;
- pause and reset controls;
- click a timeline icon to seek directly to that stroke;
- drag or click the timeline track to seek to the nearest step;
- horizontal scrolling for long sequences;
- active-step cursor and author/tool icons.

### Session operations

- load the bundled house example;
- export the complete current drawing as JSON;
- undo the latest stored stroke;
- clear and return to a new empty cat session;
- edit at an earlier timeline step;
- continue with either a human or AI stroke.

### Model integration

- local HTTP model service;
- health and continuation endpoints;
- CORS headers for the separately served static frontend;
- automatic checkpoint-type detection;
- support for the current v4.1 model and v4/v3/v2/v1 checkpoints;
- client-side fallback when the service is absent or returns an invalid response;
- inference context statistics displayed in the UI.

## System architecture

```text
┌──────────────────────────────── Browser ────────────────────────────────┐
│                                                                        │
│  Pointer input ──> normalized stroke ──> co-stroke session             │
│                                             │                          │
│                                             ├──> canvas renderer        │
│                                             ├──> timeline player/view   │
│                                             ├──> JSON export            │
│                                             └──> AI adapter             │
│                                                      │                 │
└──────────────────────────────────────────────────────┼─────────────────┘
                                                       │ POST /continue
                                                       ▼
┌──────────────────────────── Local Python service ──────────────────────┐
│ checkpoint loader -> human-priority stroke packer                      │
│       ├─> vector stroke encoder + relational Transformer ─┐            │
│       └─> 3-channel partial canvas + CNN encoder ─────────┼─> fusion   │
│                                                          └─> GMM stroke│
│                                                              decoder  │
└──────────────────────────────────────────────────────┬─────────────────┘
                                                       │ co-Stroke strokes
                                                       ▼
                                             shared browser timeline
```

### Frontend components

| File | Responsibility |
| --- | --- |
| `public/index.html` | Application shell, toolbar, canvas, sequence panel, AI control, and timeline markup |
| `src/app.js` | Shared state, control bindings, human/AI turns, export, undo, and rollout guardrail |
| `src/drawing-input.js` | Pointer capture, normalized coordinates, timing/pressure, preview, and point compaction |
| `src/timeline-player.js` | Playback, seeking, Canvas 2D rendering, and stroke-by-stroke change events |
| `src/timeline-view.js` | Timeline icons, scrolling, scrubbing, active cursor, and provenance display |
| `src/stroke-format.js` | Drawing/stroke construction, legacy normalization, validation, trimming, and serialization |
| `src/ai-adapter.js` | Local HTTP adapter, response validation, and mock fallback |
| `src/styles.css` | Layout and visual design |

### Python components

| File | Responsibility |
| --- | --- |
| `scripts/train_stroke_multimodal_v41.py` | Current v4.1 partial-canvas renderer, CNN/vector fusion model, loss, training, and sampling |
| `scripts/train_stroke_relational_v4.py` | Vector-only complete-stroke relational v4 baseline |
| `scripts/train_stroke5_transformer.py` | Historical continuous stroke-5 v3 dataset, tokenizer, model, loss, training, and sampling |
| `scripts/serve_stroke_model.py` | Checkpoint loading, inference, context packing, HTTP API, and response conversion |
| `scripts/train_sketchgpt_primitives.py` | Direction/length primitive-token v2 baseline |
| `scripts/train_stroke_transformer.py` | Earlier quantized `(dx, dy, pen)` baseline |
| `scripts/convert_quickdraw.py` | Reserved converter entry point; currently a placeholder |

## Quick start

### Prerequisites

- a modern browser with JavaScript modules, Canvas 2D, and Pointer Events;
- Python 3;
- PyTorch for model training, tests, or the local inference service;
- a trained checkpoint if real AI continuation is required.

The frontend has no build step. Lucide icons are loaded from jsDelivr, so icons require network access; text fallbacks remain in the markup.

### 1. Run the interface without the model

From the project root:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/public/
```

Do not open `public/index.html` directly as a `file://` URL. The frontend uses JavaScript modules and fetches the bundled example, so it should be served over HTTP.

You can draw, erase, replay, seek, undo, clear, and export without starting the model server. If **AI Continue** cannot reach the local model, the adapter returns a small mock continuation so the interaction loop remains testable.

### 2. Run with the local v4.1 model

Keep the static server running. In a second terminal:

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py
```

The default checkpoint path is:

```text
runs/stroke-multimodal-v41-cat/checkpoint.pt
```

Verify the service at `http://127.0.0.1:8787/health`.

The default server binds only to `127.0.0.1` on port `8787`. To choose another checkpoint, device, host, or port:

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py `
  --checkpoint runs\stroke-multimodal-v41-cat\checkpoint.pt `
  --device cuda `
  --host 127.0.0.1 `
  --port 8787
```

Use `--device cpu` if CUDA is unavailable. CPU inference is supported but may be slower.

## Using the interface

1. Draw one or more black strokes on the canvas.
2. Press **AI Continue** to request one pink model-authored stroke.
3. Continue drawing yourself or request another AI turn.
4. Use the timeline to replay the construction or seek to an earlier point.
5. Add a stroke at that earlier point to replace the previous future.
6. Press **Export** to download the complete process as JSON.

| Control | Behavior |
| --- | --- |
| **Load** | Loads `data/examples/simple-house.json` |
| **Export** | Downloads `<drawing-title>.json` |
| **Undo** | Removes the latest stored stroke |
| **Clear** | Starts a fresh empty session with category `cat` |
| **Pen** | Creates a human-authored black pen stroke |
| **Eraser** | Creates an eraser stroke that removes rendered pixels |
| **Play** | Advances through the sequence one stroke at a time |
| **Pause** | Stops playback |
| **Reset** | Seeks to step `0` without deleting data |
| **AI Continue** | Truncates hidden future strokes, packs context, and requests one AI stroke |

## Local model API

The model service uses the standard-library `ThreadingHTTPServer` and exposes two routes.

### `GET /health`

Example response:

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

The exact epoch depends on the loaded checkpoint.

### `POST /continue`

The browser sends the complete drawing plus the selected timeline boundary:

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

The server considers only `drawing.strokes[:currentStep]`, samples a continuation, and returns append-ready co-Stroke objects plus model and context metadata. Optional `options.seed` seeds Python and PyTorch before sampling. When omitted, the service chooses a random seed.

The v4.1 response includes context fields such as:

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

The server enables permissive CORS (`Access-Control-Allow-Origin: *`) for local development; review this configuration before public deployment.

### Client fallback behavior

`LocalStrokeModelAdapter` falls back to `MockStrokeModelAdapter` when the HTTP request fails, the service returns a non-2xx status, or the response lacks a `strokes` array. The mock creates a short three-point pink stroke near the visible endpoint. It is an interface fallback, not a learned model, and does not return context statistics.

## co-stroke.json v0.1

`co-stroke.json` is the human-readable interchange format shared by the UI, model service, examples, exported sessions, tests, and future physical-output tools.

The formal schema is in [`schemas/co-stroke.schema.json`](schemas/co-stroke.schema.json), and a longer note is in [`docs/co-stroke-json-v0.1.md`](docs/co-stroke-json-v0.1.md).

### Top-level drawing

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

Required drawing fields are `version`, `id`, `title`, `canvas`, and `strokes`. The schema allows additional properties so later research instrumentation can add metadata without immediately breaking the interchange format.

### Stroke object

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

Author types supported by the schema are `human`, `ai`, `dataset`, and `robot`. Tools supported in v0.1 are `pen` and `eraser`.

### Coordinate policy

The interchange format stores absolute normalized coordinates:

```text
x: 0..1, left to right
y: 0..1, top to bottom
```

Normalized coordinates make a session independent of display resolution, simplify playback, and provide a clean input to a future paper/robot calibration step. The current v4.1 runtime letterboxes the browser canvas into square model space and processes complete strokes; the historical v3 tokenizer converts coordinates into global relative `(dx, dy, pen-state)` actions.

### Timing and pressure

Point time `t` is relative to the start of the stroke. Pressure `p` is stored when available, but the current renderer and model do not use pressure to vary width or generation. `durationMs` comes from the final point time when present; otherwise normalization estimates it.

### Compatibility normalization

`src/stroke-format.js` accepts the current nested `author` and `style` form and also normalizes earlier fields such as string `author`, top-level `color`, and `size`. Exported drawings use format version `0.1.0` and normalized coordinates.

## Multimodal relational model v4.1

V4.1 predicts one complete next stroke from two synchronized representations of
the visible drawing:

```text
visible strokes -> shape/spatial/author/order embeddings -> relational Transformer
visible strokes -> 3 x 64 x 64 partial canvas           -> CNN raster encoder
                                                        -> fused context
                                                        -> next-start GMM
                                                        -> relative-shape GMM
```

Each vector stroke is normalized to the square Quick, Draw! model space and
resampled to 16 equal arc-length points. The raster channels contain all visible
strokes, AI-authored strokes, and the latest human/non-AI stroke. Training
renders only the chosen context; the hidden target stroke is never drawn into
the input image.

Training examples mix ordered-prefix completion with arbitrary-subset masked
stroke completion. A raster-only auxiliary objective with independent output
heads forces the CNN to learn useful visual state without pulling the pretrained
v4 output heads away from their existing distribution. The vector branch was
warm-started from v4.0.1 and fine-tuned at one quarter of the new-module learning
rate.

The current local cat checkpoint used 69,872 recognized Quick, Draw! sketches,
has 6,837,054 parameters, and trained for 12 epochs. Its fused main validation
loss reached `-3.7397`, compared with approximately `-3.6114` for the vector-only
v4.0.1 checkpoint. This likelihood improvement shows that the raster carries
predictive information; it does **not** establish perceptual drawing quality.
Manual tests still show repeated ear/arc motifs and degradation during long AI
rollouts.

Detailed design and training notes are in
[`docs/v4.1-multimodal-architecture.md`](docs/v4.1-multimodal-architecture.md).

## Continuous stroke-5 Transformer v3

The v3 baseline is a causal Transformer trained on simplified Quick, Draw! cat sketches.

### Action representation

Each source point contributes one `(dx, dy, pen)` action. Pen states are:

| Value | Name | Meaning |
| --- | --- | --- |
| `0` | `PEN_CONTINUE` | Continue drawing after arriving at the new point |
| `1` | `PEN_LIFT` | Lift the pen after arriving at the new point |
| `2` | `PEN_END` | End the drawing |

The **previous** action's pen state determines whether movement to the next point is visible. This preserves the invisible pen-up jump between one stroke endpoint and the next stroke anchor. Removing that jump would lose the placement of independent strokes.

The stream begins conceptually at canvas center `(0.5, 0.5)`. A `PEN_END` action `(0, 0, 2)` is appended after the final stroke.

### Dataset preprocessing

The v3 loader:

1. reads simplified Quick, Draw! NDJSON records;
2. keeps recognized drawings by default;
3. normalizes coordinates from `0..255` to `0..1`;
4. converts absolute points to global relative deltas;
5. preserves inter-stroke pen-up movement;
6. appends the end-of-drawing state;
7. skips sequences longer than `max_len` instead of truncating geometry;
8. computes one dataset-wide coordinate standard deviation;
9. divides coordinate targets by that scale during training.

Samples are held in memory, so dataset size affects startup time and RAM use.

### Model architecture

The input sums a linear projection of continuous `(dx, dy)`, a learned three-state pen embedding, and a learned position embedding. A pre-norm causal `TransformerEncoder` processes the sequence.

The output has:

1. a **20-component full-covariance bivariate Gaussian mixture-density head** for joint `(dx, dy)`;
2. a **three-class categorical head** for pen state.

For every coordinate mixture component, the model predicts weight `π`, means `μx/μy`, standard deviations `σx/σy`, and correlation `ρ`. Joint coordinate modeling avoids treating horizontal and vertical movement as unrelated categorical choices.

### Current training configuration

| Parameter | Value |
| --- | ---: |
| recognized drawings requested | 70,000 |
| validation fraction | 0.04 |
| maximum sequence length | 192 actions |
| batch size | 64 |
| epochs | 12 |
| model width | 384 |
| Transformer layers | 6 |
| attention heads | 6 |
| dropout | 0.2 |
| Gaussian mixtures | 20 |
| learning rate | 0.0001 |
| weight decay | 0.01 |
| seed | 7 |
| optimizer | AdamW |
| LR schedule | cosine annealing to 10% of initial LR |
| gradient clipping | max norm 1.0 |
| mixed precision | CUDA unless `--no-amp` is used |

The coordinate scale in the current local cat configuration is approximately `0.12495403`. It is calculated from the selected data and saved with the checkpoint, so it should not be assumed for another dataset.

### Objective and sampling

The total loss is coordinate negative log-likelihood plus pen-state cross entropy. Padding is masked; coordinate loss excludes the `PEN_END` target. Density arithmetic stays in float32 even under CUDA autocast.

At inference, temperature affects mixture selection, component variance (scaled by `sqrt(temperature)`), and pen-state sampling. Generated positions stay within `[0.02, 0.98]`; server deltas are limited to `[-0.6, 0.6]`. Sampling stops at the requested stroke count, point limit, step budget, or `PEN_END`. An unfinished useful stroke is closed with `PEN_LIFT`, and the v3 server retries up to three times if it gets no decodable polyline.

## Human-priority context policy v3.1

Browser input can be much denser than simplified Quick, Draw! data. Raw history would quickly fill the model window and could let recent AI output displace the person's original geometry. The service therefore packs visible history before tokenization.

### Policy constants

```text
model max_len:                    192
usable prior-action budget:       191
reserved human/non-AI budget:     up to 120 actions
maximum retained recent AI:       6 strokes
human points per stroke cap:      16
AI points per stroke cap:         12
```

### Packing procedure

1. Ignore strokes with fewer than two points.
2. Separate AI-authored strokes from all non-AI strokes.
3. Resample selected polylines at equal arc-length intervals.
4. Allocate up to 120 actions to non-AI geometry.
5. Use remaining capacity for at most the six most recent AI strokes.
6. If endpoints alone exceed the budget, retain evenly distributed strokes; preserve the first and last when possible.
7. Restore selected strokes to chronological order before tokenization.

The code treats every non-`ai` author—including `human`, `dataset`, and `robot`—as protected context. The UI calls this “human” because normal interactive sessions contain human and AI authors.

The response reports raw and retained action counts, retained stroke counts, dropped AI strokes, and whether compaction occurred. The browser displays a compact summary such as `Context: 92 human + 48 AI / 191`; hovering reveals policy details.

## Model baselines

The service auto-detects multiple checkpoint families without changing the
browser contract.

### v4.1: multimodal relational stroke model

- `scripts/train_stroke_multimodal_v41.py`;
- checkpoint type `stroke-multimodal-v4.1`;
- vector stroke Transformer plus three-channel partial-canvas CNN;
- predicts a complete-stroke start and relative shape with mixture-density heads;
- current default model.

### v4.0.1: vector-only relational stroke model

- `scripts/train_stroke_relational_v4.py`;
- checkpoint type `stroke-relational-v4`;
- complete-stroke embeddings with bidirectional relational attention;
- geometric 16-candidate reranking and square-canvas letterbox mapping;
- preserved as Git tag `v0.4.0.1`.

### v3: continuous stroke-5 Transformer

- `scripts/train_stroke5_transformer.py`;
- checkpoint type `stroke5-transformer-v3`;
- continuous joint `(dx, dy)` mixture-density prediction;
- categorical pen prediction;
- preserves invisible inter-stroke movement;
- historical action-level baseline.

### v2: direction–length primitive Transformer

- `scripts/train_sketchgpt_primitives.py`;
- checkpoint type `sketchgpt-segments-v2`;
- combines one of 48 directions with one of 8 length bins;
- uses `BOS`, `STROKE_END`, `EOS`, and `PAD` tokens;
- applies a repeated-direction penalty during sampling;
- reduces straight-line loops seen in the legacy repeated-direction encoding.

### v1: quantized delta Transformer

- `scripts/train_stroke_transformer.py`;
- checkpoint type `stroke-transformer`;
- quantizes `dx` and `dy` independently into 121 bins;
- predicts separate `dx`, `dy`, and pen distributions;
- served as the first complete data/model/frontend baseline.

Older primitive checkpoints without `token_encoding` are interpreted as legacy `repeated-direction` checkpoints.

## Training

Training scripts expect simplified Quick, Draw! NDJSON. Current commands use `data/quickdraw/cat.ndjson`. Raw NDJSON, checkpoints, and `runs/` are ignored by Git, so a clone does not include the training data or local weights.

### Train the current multimodal v4.1 model

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

The best checkpoint is selected by fused main validation loss, not the combined
main-plus-raster-auxiliary objective. The current run selected epoch 12 with
`val_loss = -3.739664`. Use `--device cpu` and optionally `--no-amp` when CUDA is
unavailable; full training will be substantially slower.

### Train the continuous v3 model

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

Other options include `--include-unrecognized`, `--val-fraction`, `--max-len`, `--lr`, `--weight-decay`, `--seed`, `--device`, and `--no-amp`.

Training writes:

| Output | Meaning |
| --- | --- |
| `config.json` | Resolved configuration and calculated coordinate scale |
| `latest.pt` | Most recent epoch |
| `checkpoint.pt` | Lowest validation loss so far |
| `sample.json` | Unconditional autoregressive sample in co-Stroke format |

The console reports total, coordinate, and pen losses for training and validation, learning rate, parameter count, coordinate scale, and over-length drawings skipped.

### Train the v2 direction–length baseline

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

### Reproducibility notes

- Python and PyTorch seeds default to `7` during training.
- CUDA seeds are set when available.
- TF32 matrix multiplication is enabled on CUDA for v3 training.
- train/validation splitting uses a seeded `torch.Generator`.
- server sampling is stochastic unless `options.seed` is supplied.
- exact numerical reproduction can still vary with hardware and PyTorch/CUDA versions.

## Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Current tests cover:

- Quick, Draw! -> stroke-5 -> polyline round trips;
- independent stroke anchors and pen-up jumps;
- `CONTINUE`, `LIFT`, and `END` placement;
- Transformer output shapes and finite losses;
- v4/v4.1 complete-stroke output shapes and sampling;
- three-channel partial-canvas rasterization;
- hidden-target leakage prevention;
- non-zero raster-encoder gradients;
- arc-length resampling and endpoint preservation;
- per-stroke point caps;
- reserved human-context capacity;
- dropping older AI strokes before human geometry;
- total packed context staying within budget.

Browser interaction, schema-validator integration, visual regression, and end-to-end HTTP tests have not yet been added.

## Repository structure

```text
co-Stroke/
├── README.md
├── README.zh-CN.md                        # Simplified Chinese documentation
├── blog/
│   └── 001-introduction.md              # Original project motivation
├── data/
│   └── examples/simple-house.json       # Bundled example
├── docs/
│   ├── co-stroke-json-v0.1.md           # Current format design
│   ├── v4-architecture.md                # Vector relational v4 design
│   ├── v4.1-multimodal-architecture.md  # Current vector/raster design and results
│   ├── stroke-format.md                  # Earlier compact format note
│   └── conversations/                    # Development notes
├── public/index.html                     # Browser shell
├── schemas/co-stroke.schema.json         # JSON Schema draft 2020-12
├── scripts/
│   ├── convert_quickdraw.py              # Placeholder converter
│   ├── serve_stroke_model.py             # Local inference HTTP service
│   ├── train_stroke_multimodal_v41.py    # Current vector + raster v4.1 model
│   ├── train_stroke_relational_v4.py     # Vector-only relational v4 baseline
│   ├── train_sketchgpt_primitives.py     # Primitive baseline
│   ├── train_stroke_transformer.py       # Quantized-delta baseline
│   └── train_stroke5_transformer.py      # Continuous stroke-5 v3 baseline
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

Local-only directories commonly include `data/quickdraw/`, `runs/`, and `.venv/`; they are not committed.

## Known limitations

### Interaction and interface

- **Load is not a general importer.** It always fetches the bundled house example.
- **The timeline is linear.** Replaced futures are deleted instead of retained as branches.
- **No persistent storage.** Refreshing resets unsaved browser state.
- **No candidate comparison.** The UI requests one continuation at a time.
- **No dedicated rejection control.** Users reject through undo or rewind.
- **No keyboard/accessibility audit.** Keyboard, screen-reader, contrast, and touch support need evaluation.
- **Eraser history is not semantic model input.** V4/v4.1 exclude eraser paths from context; the model sees only the remaining pen geometry and cannot reason about what was removed.
- **Pressure is recorded but unused.** It does not affect rendering, generation, or robot output.
- **Category is fixed for new sessions.** Clear creates a `cat` session for the cat-only checkpoint.

### Model and data

- **Single-category checkpoint.** There is no learned category token.
- **Limited context.** V4.1 retains at most 24 stroke units and at most six recent AI strokes.
- **Train–interaction mismatch.** Quick, Draw! differs from slow, iterative co-creation.
- **No explicit semantic planning.** Raster input improves validation likelihood, but there is no cat-part vocabulary, missing-part objective, or composition plan.
- **Long rollout degradation.** Repeated ear/arc motifs and crossing lines still appear when the model repeatedly conditions on its own output.
- **Geometric reranking only.** Sixteen candidates are scored for connection, overlap, isolation, and boundary collapse, but not recognizability, novelty, recent-stroke similarity, or user intent.
- **Compressed raster context.** The CNN projects its spatial feature map into one context vector rather than exposing spatial tokens through cross-attention.
- **No formal output evaluation.** Validation NLL exists, but recognizability, diversity, and human preference metrics do not.
- **In-memory dataset.** Large datasets can require substantial RAM.
- **No artifact installer.** Data and checkpoints are excluded from Git without automated download/setup.

### Research

- no completed formative or controlled study;
- no preregistered hypotheses;
- no participant event-logging system;
- no validated questionnaire integration;
- no qualitative coding protocol;
- no published findings yet.

These are markers of the prototype stage rather than claims that the system is study-ready.

## Research opportunities

The system can support questions such as:

- When should an AI intervene in a drawing process?
- Does one-stroke-at-a-time generation preserve more control than one-shot completion?
- How do visible provenance and replay affect authorship judgments?
- What balance of predictability and surprise supports exploration?
- Does protected human context change trust or willingness to collaborate?
- How do users appropriate rewind and undo as negotiation mechanisms?
- Should users choose among candidates or edit one suggestion?
- How does automatic AI initiative compare with user-requested initiative?
- Does robot execution change perceived ownership of a joint work?

A possible controlled study could compare:

| Condition | Description |
| --- | --- |
| Human-only | Drawing without model assistance |
| One-shot AI | AI generates a relatively complete continuation |
| Stroke-level co-creation | Human and AI alternate through interruptible timeline contributions |

Possible behavioral measures include completion time, AI requests, undo/rewind count, replaced and retained AI strokes, turn length, and alternative-history exploration. Experiential measures could include control, authorship, agency, creative support, satisfaction, trust, and surprise. Interviews could examine collaboration strategies, conflict, interpretation of AI intent, and how users decide whether a contribution belongs in “their” drawing.

Before a formal study, the prototype should add event logging, participant/session IDs, configurable experimental conditions, consent/privacy handling, reproducible prompts, and a study-safe export pipeline.

## Roadmap

### Near term

1. Add arbitrary JSON import with schema validation and error reporting.
2. Add multiple continuation candidates with accept, reject, and regenerate actions.
3. Preserve alternate futures as named branches.
4. Add session/event logging for pilot studies.
5. Add browser and end-to-end service tests.
6. Add recognizability evaluation alongside validation NLL.

### Model development

1. Add recent-AI shape-similarity and raster-occupancy penalties to suppress repeated motifs.
2. Preserve CNN feature-map cells as spatial tokens and fuse them through cross-attention.
3. Add recognizability-delta or missing-part supervision before expanding beyond cats.
4. Train on corrupted/model-generated contexts to reduce long-rollout distribution shift.
5. Expose multiple candidates for human selection and evaluate diversity, compatibility, and recognizability.
6. Add category conditioning before training multiple classes.

### HCI study development

1. Conduct formative interviews and think-aloud sessions.
2. Refine the interaction around observed strategies.
3. Define hypotheses, conditions, measures, and analysis.
4. Run a pilot before a larger controlled evaluation.
5. Report behavioral evidence and qualitative findings.

### Physical output

Robot export should remain a separate calibrated converter:

```text
normalized x/y -> calibrated paper coordinates
stroke start    -> pen down
stroke end      -> pen up
style/pressure  -> optional speed or force mapping
```

Separating interchange data from device motion keeps the browser, model, and physical executor loosely coupled.

## Project origin

co-Stroke began from the observation that the most interesting part of AI-assisted drawing may not be whether an AI can produce a polished image. It may be whether a person can see, interrupt, redirect, reinterpret, and share responsibility for the process by which that image comes into being.

The project is intentionally both technical and interaction-centered: a data format, a temporal interface, an autoregressive model, and a platform for studying how creative agency moves between a human, an AI, and eventually a machine.
