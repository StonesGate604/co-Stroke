# co-stroke.json v0.1

`co-stroke.json` is the interchange format for the first Co-Stroke prototype. It is designed for four connected uses:

1. Timeline playback in the browser.
2. Human continuation after rewinding or pausing.
3. Autoregressive AI continuation from an existing stroke history.
4. Future conversion into robot-arm paths.

The interface format is stroke-level and human-readable. Later model training can derive point-level or delta-token sequences from the same source.

## Top-Level Drawing

```json
{
  "schema": "https://co-stroke.local/schema/co-stroke-v0.1.json",
  "version": "0.1.0",
  "id": "drawing_simple_house_001",
  "title": "simple-house",
  "category": "house",
  "createdAt": "2026-06-01T00:00:00.000Z",
  "updatedAt": "2026-06-01T00:00:00.000Z",
  "source": { "type": "example", "name": "hand-authored sample" },
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

## Stroke

Each stroke is one timeline unit. A stroke can come from a dataset, a human, an AI model, or a robot replay pass.

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

## Required Fields

Drawing: `version`, `id`, `title`, `canvas`, `strokes`.

Stroke: `id`, `author.type`, `tool`, `style`, `points`.

## Coordinate Policy

v0.1 stores points in absolute normalized coordinates:

```text
x: 0..1
y: 0..1
```

This is easiest for UI playback, resizing, export, and robot mapping. For training, a converter can transform absolute points into autoregressive deltas:

```text
dx, dy, pen_state
```

## Timeline Policy

v0.1 uses a simple linear timeline. If the user rewinds to an earlier step and draws a new stroke, future strokes are removed.

```text
branchPolicy: truncate-future-on-edit
```

Branching can be added later without changing the meaning of stroke objects.

## AI Continuation Contract

The browser should not depend on one specific model implementation. Any future model should satisfy this interface:

```ts
type StrokeContinuationRequest = {
  drawing: CoStrokeDrawing;
  currentStep: number;
  options: {
    maxStrokes: number;
    temperature?: number;
    categoryHint?: string;
  };
};

type StrokeContinuationResponse = {
  strokes: CoStrokeStroke[];
  model: {
    name: string;
    version?: string;
  };
};
```

The first frontend prototype uses a mock adapter, but the same interface can later call a local Python server, a hosted model endpoint, or a browser-side ONNX model.

## Robot Bridge Notes

Robot export should be a separate converter:

```text
normalized x/y -> calibrated paper x/y
stroke start -> pen down
stroke end -> pen up
style.width -> optional speed/pressure mapping
```
