# Codex Project Handoff: co-Stroke

Date: 2026-06-01
Project path: `C:\Users\rog\Documents\GitHub\co-Stroke`

## Purpose

This document summarizes the current Codex conversation so a future Codex project thread can continue development inside the `co-Stroke` folder without needing the full chat history.

## Project Concept

`co-Stroke` is a continuation of the earlier `AI Drawing Studio` / `AIdrawing` project from NYU ITP Code Your Way.

The project changed direction from image-based next-stroke prediction to stroke-sequence modeling. The core idea is:

```text
Drawing is treated as an autoregressive sequence of strokes.
previous stroke tokens -> next stroke token
```

The goal is not to build a generic text-to-image tool. The goal is to build an interactive human-AI drawing system where a human and an AI model can share the same drawing timeline, take turns drawing, rewind, continue, and eventually send the result to a robotic arm.

## Current Product Direction

The final visual output should remain close to SketchRNN-style simple line drawings, but with stronger interaction:

- timeline playback
- human continuation after pausing or rewinding
- AI continuation from the current stroke history
- author-aware display of strokes
- exportable stroke sequence data
- future robot-arm path conversion

The project should first focus on simple black/pink line drawing before expanding into complex image generation, coloring, or style transfer.

## Design Decisions So Far

### Data First

The first milestone is not model training. It is defining a stable `co-stroke.json` format and making a working timeline player/editor around it.

### Use Quick, Draw! Later

The previous idea of collecting custom user data through a web recorder was judged inefficient and noisy. The next data direction is to use Google Quick, Draw! stroke data as a starter dataset.

### Custom Stroke Model Still Matters

LLM + MCP can be useful as a tool-calling/control layer, but it does not replace the need for a dedicated autoregressive stroke model. The low-level model should learn stroke continuation; LLMs may later serve as high-level controllers.

### Author-Based Stroke Styling

The drawing UI has been simplified:

- human strokes are black: `#111111`
- AI strokes are pink: `#ff44aa`
- stroke width is fixed at `4px`
- user-facing color and brush-size controls were removed

This makes authorship clearer and keeps the prototype focused.

## Current Repository Structure

```text
co-Stroke/
  README.md
  blog/
    001-introduction.md
  data/
    examples/
      simple-house.json
  docs/
    co-stroke-json-v0.1.md
    stroke-format.md
    conversations/
      2026-06-01-codex-project-handoff.md
  public/
    index.html
  schemas/
    co-stroke.schema.json
  scripts/
    convert_quickdraw.py
  src/
    ai-adapter.js
    app.js
    drawing-input.js
    stroke-format.js
    styles.css
    timeline-player.js
    timeline-view.js
```

## Important Files

### `public/index.html`

Defines the page shell:

- topbar buttons: Load, Export, Undo, Clear
- left toolbar: pen, eraser, play, pause, reset
- central canvas
- right control panel
- bottom timeline

The timeline DOM was restored to match the original AI Drawing Studio design:

```html
<div class="tl-track">
  <div id="tl-icons"></div>
  <input id="timeline" type="range">
  <div id="tl-cursor"></div>
</div>
```

### `src/app.js`

Main browser controller. It coordinates state, UI controls, drawing input, timeline playback, AI continuation, and JSON export.

Current responsibilities:

- load example JSON
- export current drawing
- undo / clear
- select pen or eraser
- request mock AI continuation
- keep `state.drawing` as the current drawing
- pass drawing state to `TimelinePlayer` and `TimelineView`

This file is cleaner than the first prototype but may later be split into `drawing-state.js`, `dom-bindings.js`, or command modules.

### `src/stroke-format.js`

Pure data utilities. This is one of the most important files because it is not tied to the DOM.

Responsibilities:

- `createEmptyDrawing()`
- `validateDrawing()`
- `normalizeDrawing()`
- `normalizeStroke()`
- `createStroke()`
- `trimDrawingToStep()`
- `serializeDrawing()`

Keep this module DOM-free so it can later be reused by Quick, Draw! conversion, model training, and robot export.

### `src/timeline-player.js`

Canvas renderer and playback controller.

Responsibilities:

- render visible strokes onto canvas
- `play()` / `pause()` / `reset()` / `seek()`
- draw eraser strokes with `destination-out`
- emit state updates through `onChange`

This is separate from the visual timeline UI.

### `src/timeline-view.js`

Timeline UI module, based on the original AI Drawing Studio timeline design.

Responsibilities:

- render 33px step icons
- map stroke author/tool to timeline classes and icons
- active step highlighting
- green cursor positioning
- click-to-seek
- drag/scrub along the timeline track
- auto-scroll active step into view

Important mapping:

- AI stroke -> `sparkles` icon and pink/purple timeline step
- eraser -> `eraser` icon
- pen/human/dataset -> `pencil` icon

### `src/drawing-input.js`

Handles pointer input on the canvas.

Responsibilities:

- collect pointer points
- normalize x/y into 0..1 canvas coordinates
- store relative time `t`
- compact redundant points
- call `onPreview(points)` while drawing
- call `onCommit(points)` when stroke ends

### `src/ai-adapter.js`

Current AI model interface. It is currently a mock adapter.

This is the future connection point for the custom autoregressive stroke model.

Current public function:

```js
requestAIContinuation({ drawing, currentStep, options })
```

Expected return shape:

```js
{
  model: {
    name: "model-name",
    version: "0.1.0"
  },
  strokes: [/* generated co-stroke stroke objects */]
}
```

Future options:

- call a local Python inference server
- call a hosted endpoint
- run a browser-side ONNX or TensorFlow.js model

### `src/styles.css`

Visual style inspired by the previous AI Drawing Studio:

- dark UI
- topbar
- left vertical toolbar
- right control panel
- bottom icon timeline
- central white canvas

Current fixed colors:

```css
--human: #111111;
--ai: #ff44aa;
```

### `docs/co-stroke-json-v0.1.md`

Human-readable data format spec.

### `schemas/co-stroke.schema.json`

JSON Schema for `co-stroke.json` v0.1.

### `data/examples/simple-house.json`

Example drawing in v0.1 format.

## Current Data Format Summary

Top-level drawing:

```json
{
  "schema": "https://co-stroke.local/schema/co-stroke-v0.1.json",
  "version": "0.1.0",
  "id": "drawing_simple_house_001",
  "title": "simple-house",
  "category": "house",
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

Stroke:

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
    { "x": 0.25, "y": 0.70, "t": 0, "p": 0.5 }
  ],
  "metadata": { "insertedAtStep": 0 }
}
```

Coordinates are normalized absolute coordinates. Later model training can convert them into autoregressive deltas such as:

```text
dx, dy, pen_state
```

## Current Architecture

```text
public/index.html
    -> src/app.js
        -> src/stroke-format.js
        -> src/drawing-input.js
        -> src/ai-adapter.js
        -> src/timeline-player.js
        -> src/timeline-view.js
        -> src/styles.css
```

Runtime flow:

```text
User draws on canvas
  -> drawing-input.js returns normalized points
  -> app.js creates human stroke through stroke-format.js
  -> state.drawing.strokes is updated
  -> timeline-player.js redraws canvas
  -> timeline-view.js redraws bottom timeline
```

AI flow:

```text
User clicks Mock AI Stroke
  -> app.js calls requestAIContinuation()
  -> ai-adapter.js returns pink AI stroke
  -> app.js appends stroke to drawing
  -> canvas and timeline update
```

Export flow:

```text
User clicks Export
  -> app.js calls serializeDrawing()
  -> browser downloads co-stroke JSON
```

## Generated Architecture Diagram

An SVG architecture diagram was saved to the Desktop:

```text
C:\Users\rog\Desktop\co-stroke-architecture.svg
```

## Local Development URLs

The user may use VS Code Live Server:

```text
http://127.0.0.1:5500/public/
```

Codex also used Python http.server during development:

```text
http://localhost:8000/public/
```

Both are fine if the server root is the `co-Stroke` folder.

## Current Verification

Completed checks during the conversation:

- static page server returned `200`
- `stroke-format.js` imports successfully
- `ai-adapter.js` returns a generated AI stroke
- mock AI stroke uses pink `#ff44aa`
- Live Server URL returned `200`

Browser visual verification through the Codex in-app browser was attempted, but the local Codex/Electron browser runtime failed with a sandbox spawn error. This appeared to be a Codex app/browser runtime issue, not a project code issue.

## Obsidian Context

The user has an Obsidian vault at:

```text
C:\Users\rog\Documents\Obsidian Vault
```

A project cockpit was created earlier at:

```text
Projects/Co-Stroke Project Cockpit.md
```

The cockpit defines:

- one-sentence project description
- current phase: stroke sequence format + timeline player
- near-term tasks
- open problems
- Codex prompts
- output targets

## Recommended Next Steps

1. **Open a new Codex project thread directly in:**

```text
C:\Users\rog\Documents\GitHub\co-Stroke
```

2. Ask the new thread to read:

```text
README.md
docs/co-stroke-json-v0.1.md
docs/conversations/2026-06-01-codex-project-handoff.md
src/app.js
src/timeline-view.js
src/ai-adapter.js
```

3. Then continue with one of these tasks:

```text
Add local JSON import so I can load any co-stroke.json file from my computer.
```

```text
Write the first Quick, Draw! converter that outputs co-stroke.json v0.1.
```

```text
Split app.js into a cleaner drawing-state module and dom-bindings module without changing behavior.
```

```text
Replace the mock AI adapter with a local Python inference server interface, but keep the same request/response contract.
```

## Important Caution

Do not over-engineer too early. The current architecture is reasonable for Phase 1. The most important principle is:

```text
Core data logic should remain DOM-free.
```

Keep `stroke-format.js` reusable across browser UI, Quick Draw conversion, training scripts, and robot path export.
