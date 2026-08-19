# co-Stroke

co-Stroke is a process-based human-AI drawing experiment. Instead of treating drawing as a finished image, it treats drawing as a temporal sequence of strokes.

The project grows out of AI Drawing Studio, a Code Your Way project at NYU ITP. The new direction focuses on stroke sequence modeling, timeline interaction, human-AI continuation, and eventually robot-arm execution.

## Core Idea

A drawing is represented as an ordered stroke sequence. Each stroke can be played back, edited, continued by a human, continued by an AI model, or converted into a physical drawing path.

The current model is autoregressive stroke modeling:

```text
previous stroke-5 actions -> next continuous (dx, dy, pen) action
```

This is different from text-to-image generation. The goal is not only to produce a final sketch, but to design a system where humans, models, and machines can share the same drawing timeline.

## Current Scope

The first milestone is intentionally small:

- Define `co-stroke.json` v0.1
- Build a timeline player/editor in the visual style of the earlier AI Drawing Studio
- Load, play, scrub, draw, erase, export, and local-model AI continuation
- Train and serve a continuous stroke-5 Transformer on Quick, Draw! cats

## Project Structure

```text
co-Stroke/
  README.md
  blog/
    001-introduction.md
  docs/
    co-stroke-json-v0.1.md
    stroke-format.md
  schemas/
    co-stroke.schema.json
  public/
    index.html
  src/
    ai-adapter.js
    app.js
    drawing-input.js
    stroke-format.js
    styles.css
    timeline-player.js
  data/
    examples/
      simple-house.json
  scripts/
    convert_quickdraw.py
    train_stroke5_transformer.py
    serve_stroke_model.py
  tests/
    test_stroke5_transformer.py
```

## Run

Serve the folder with a local static server:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/public/
```

## Run With the Local AI Model

Keep the static web server running, then open a second terminal in the project
root and start the model service:

```powershell
.\.venv\Scripts\python.exe scripts\serve_stroke_model.py
```

The default checkpoint is the continuous stroke-5 v3 cat model at
`runs/stroke5-transformer-v3-cat/checkpoint.pt`. Verify the service at:

```text
http://127.0.0.1:8787/health
```

The browser sends the current co-Stroke history to `POST /continue`. If port
8787 is unavailable, the frontend falls back to a mock stroke.

The v3.1 inference policy resamples dense browser polylines before tokenization,
reserves up to 120 of the 191 available context actions for human strokes, and
uses only the six most recent AI strokes for the remaining budget. The response
includes `context` statistics, which the browser displays below **AI Continue**.
After 12 consecutive AI strokes the UI asks for another human stroke to avoid
long self-conditioned rollouts.

## Train the Continuous Stroke-5 v3 Model

The v3 tokenizer keeps every global relative movement, including the invisible
pen-up jump between one stroke endpoint and the next stroke start. The causal
Transformer consumes continuous `(dx, dy)` values plus a three-state pen
embedding. A 20-component full-covariance bivariate Gaussian mixture predicts
the next coordinate action, and a categorical head predicts draw, lift, or end.

The current checkpoint uses 70,000 recognized cat drawings:

```powershell
.\.venv\Scripts\python.exe scripts\train_stroke5_transformer.py --data data\quickdraw\cat.ndjson --out-dir runs\stroke5-transformer-v3-cat --max-drawings 70000 --epochs 12 --batch-size 64 --d-model 384 --layers 6 --heads 6 --mixtures 20 --dropout 0.2
```

Training writes the best `checkpoint.pt`, the final `latest.pt`, `config.json`,
and a full autoregressive `sample.json`. Run the tokenizer/model tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Train the Direction-Length v2 Model

The first primitive baseline represented segment length by repeating the same
direction token, which often collapsed into straight-line loops. The v2 model
uses one combined `direction x length-bin` token and applies a mild direction
repetition penalty during sampling.

The current 50,000-drawing cat checkpoint was trained with:

```powershell
.\.venv\Scripts\python.exe -u scripts\train_sketchgpt_primitives.py --data data\quickdraw\cat.ndjson --out-dir runs\sketchgpt-segments-v2-cat --max-drawings 50000 --max-len 192 --batch-size 128 --epochs 10 --d-model 256 --layers 6 --heads 8 --dropout 0.1 --lr 0.0003 --device cuda
```

This discrete v2 checkpoint remains available as a comparison baseline. Model
outputs and local datasets are intentionally ignored by Git.

## Next Steps

1. Add multiple continuation candidates and human selection.
2. Add class tokens before expanding beyond the cat-only checkpoint.
3. Add a sketch-recognizability evaluation alongside validation NLL.
4. Add robot path export as a separate converter.
