# co-Stroke v4: relational stroke completion

V4 changes the model's unit of reasoning from individual point actions to
complete strokes. It is an additive model family: v3 checkpoints and the
browser API remain compatible.

## Architecture

```text
visible co-Stroke history
        |
        v
arc-length resampling (fixed points per stroke)
        |
        +--> relative stroke shape encoder
        +--> absolute spatial feature encoder
        +--> author + original-order embeddings
                         |
                         v
             bidirectional relational Transformer
                         |
                 shared context vector
                    /             \
                   v               v
       next-start bivariate GMM   stop head
                   |
                   v
       conditional relative-shape GMM
                   |
                   v
          several complete-stroke candidates
                   |
                   v
      confidence + overlap/repetition reranking
                   |
                   v
            one interruptible AI stroke
```

The model predicts the next stroke in two stages:

1. a multimodal distribution for its absolute starting position;
2. a multimodal distribution for its fixed-width shape relative to that start.

This removes the v3 requirement that a new stroke must emerge from local
point-by-point motion. It also makes the location and shape decisions explicit,
which is useful for later human control and candidate visualization.

## Training tasks

Each Quick, Draw! drawing produces two types of completion examples:

- ordered prefix -> next stroke;
- arbitrary subset of existing strokes -> one missing stroke.

The second task reduces dependence on the dataset's canonical drawing order and
better approximates an interactive user who may begin with any object part.
Training also applies small whole-drawing rotations, scale changes, and
translations. Context author labels are sometimes randomized between dataset,
human, and AI so the browser-time source embeddings do not remain untrained.

The objective is:

```text
start-position GMM NLL
+ relative-shape GMM NLL
+ 0.25 * drawing-stop BCE
```

## Human-priority context and repetition guard

The v4 service packs context in stroke units rather than point-action units. It
keeps up to 24 strokes by default, protects human/non-AI strokes, and retains at
most the six most recent AI strokes. Eraser paths are excluded because the v4
training data contains only additive pen strokes.

At inference the service samples sixteen candidate strokes. It penalizes:

- candidates that trace existing geometry;
- candidates that especially overlap the latest human stroke;
- degenerate very-short strokes;
- candidates collapsed against the canvas boundary.

The current reranker is geometric. A raster/category encoder can be added in a
later v4 phase without changing the checkpoint API.

### v4.0.1 canvas and connection policy

The browser canvas is `960 x 640`, while Quick, Draw! training coordinates use a
square space. Directly treating browser-normalized x/y as model coordinates
turns a visually circular gesture into a vertical ellipse. V4.0.1 therefore
letterboxes browser coordinates into a square before encoding and reverses the
mapping after decoding:

```text
browser x/y -> square letterbox -> v4 model -> inverse letterbox -> browser x/y
```

Candidate scoring also distinguishes a useful connection from tracing:

- contact at one or two candidate points is allowed and mildly rewarded;
- overlap becomes a penalty only after a substantial fraction of the stroke;
- candidates whose nearest point is far from all visible geometry receive an
  isolation penalty;
- candidates whose center is far from the existing composition receive an
  additional composition penalty.

The service reports selected endpoint distance, nearest distance, overlap, and
the `letterbox-square-v1` mapping in its response metadata.

## Smoke training

```powershell
.\.venv\Scripts\python.exe -u scripts\train_stroke_relational_v4.py `
  --data data\quickdraw\cat.ndjson `
  --out-dir runs\stroke-relational-v4-smoke `
  --max-drawings 512 `
  --epochs 1 `
  --batch-size 64 `
  --d-model 64 `
  --layers 2 `
  --heads 4 `
  --mixtures 4 `
  --max-context-strokes 12 `
  --points-per-stroke 12 `
  --device cuda
```

## Full cat training

```powershell
.\.venv\Scripts\python.exe -u scripts\train_stroke_relational_v4.py `
  --data data\quickdraw\cat.ndjson `
  --out-dir runs\stroke-relational-v4-cat `
  --max-drawings 70000 `
  --epochs 16 `
  --batch-size 192 `
  --d-model 256 `
  --layers 6 `
  --heads 8 `
  --mixtures 10 `
  --max-context-strokes 24 `
  --points-per-stroke 16 `
  --device cuda
```

Training writes `latest.pt`, the best `checkpoint.pt`, `config.json`, and a
one-stroke `sample.json`.

## Serving a v4 checkpoint

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py `
  --checkpoint runs\stroke-relational-v4-cat\checkpoint.pt `
  --device cuda `
  --host 127.0.0.1 `
  --port 8787
```

The service auto-detects `model_type: stroke-relational-v4`. The existing
`GET /health` and `POST /continue` routes and browser response format remain
unchanged.

## Current boundary of v4.0

V4.0 is vector-only. It learns stroke-shape and stroke-relation embeddings, but
does not yet contain a raster image encoder or an explicit cat-part vocabulary.
The next measured upgrade should add a raster recognizability encoder and use it
to rerank candidates, after the stroke-level baseline is trained and evaluated.

## Current local checkpoint

The first full cat checkpoint was trained on 2026-08-21:

| Item | Result |
| --- | ---: |
| recognized records requested | 70,000 |
| usable drawings | 69,872 |
| train / validation | 67,078 / 2,794 |
| epochs | 16 |
| parameters | 5,385,887 |
| best epoch | 16 |
| validation total | -3.6114 |
| validation start NLL | -1.6145 |
| validation shape NLL | -2.0465 |
| validation stop BCE | 0.1986 |
| checkpoint size | about 21.6 MB |

The default server checkpoint now points to
`runs/stroke-relational-v4-cat/checkpoint.pt`. The checkpoint remains local and
is ignored by Git.
