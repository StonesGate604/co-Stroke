# co-Stroke

co-Stroke is a process-based human-AI drawing experiment. Instead of treating drawing as a finished image, it treats drawing as a temporal sequence of strokes.

The project grows out of AI Drawing Studio, a Code Your Way project at NYU ITP. The new direction focuses on stroke sequence modeling, timeline interaction, human-AI continuation, and eventually robot-arm execution.

## Core Idea

A drawing is represented as an ordered stroke sequence. Each stroke can be played back, edited, continued by a human, continued by an AI model, or converted into a physical drawing path.

The long-term technical direction is autoregressive stroke modeling:

```text
previous stroke tokens -> next stroke token
```

This is different from text-to-image generation. The goal is not only to produce a final sketch, but to design a system where humans, models, and machines can share the same drawing timeline.

## Current Scope

The first milestone is intentionally small:

- Define `co-stroke.json` v0.1
- Build a timeline player/editor in the visual style of the earlier AI Drawing Studio
- Load, play, scrub, draw, erase, export, and mock AI continuation
- Reserve a clean adapter for a future custom autoregressive stroke model

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

## Next Steps

1. Convert a small Quick, Draw! sample into `co-stroke.json`.
2. Add import for local JSON files.
3. Replace the mock AI adapter with a small local inference server.
4. Add robot path export as a separate converter.
