# Stroke Format

co-Stroke represents each drawing as a timeline of strokes.

## Drawing Object

```json
{
  "version": "0.1.0",
  "title": "simple-house",
  "canvas": {
    "width": 800,
    "height": 600,
    "coordinateSystem": "normalized"
  },
  "strokes": []
}
```

## Stroke Object

```json
{
  "id": "stroke_001",
  "author": "human",
  "tool": "pen",
  "color": "#111111",
  "size": 4,
  "points": [
    { "x": 0.2, "y": 0.7, "t": 0 },
    { "x": 0.4, "y": 0.5, "t": 80 }
  ]
}
```

## Fields

- `id`: Stable stroke identifier.
- `author`: `human`, `ai`, `dataset`, or `robot`.
- `tool`: Drawing tool. The initial prototype uses `pen`.
- `color`: Hex color.
- `size`: Stroke width in pixels.
- `points`: Ordered points in normalized canvas coordinates.
- `x`, `y`: Normalized coordinate values from `0` to `1`.
- `t`: Relative time in milliseconds within the stroke.

## Why Normalized Coordinates?

Normalized coordinates make the same drawing playable on different canvas sizes and easier to convert into robot-arm workspace coordinates later.

## Future Tokenization

The training format may convert points into token-like values such as:

```text
dx, dy, pen_state
```

or into discrete bins:

```text
x_bin, y_bin, pen_down, pen_up, end_drawing
```

The JSON format is the readable interchange format. The token format will be used for model training.
