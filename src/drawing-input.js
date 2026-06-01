export class DrawingInputController {
  constructor(canvas, callbacks) {
    this.canvas = canvas;
    this.callbacks = callbacks;
    this.isDrawing = false;
    this.points = [];
    this.startedAt = 0;

    canvas.addEventListener("pointerdown", (event) => this.start(event));
    canvas.addEventListener("pointermove", (event) => this.move(event));
    window.addEventListener("pointerup", (event) => this.end(event));
    canvas.addEventListener("pointerleave", (event) => this.end(event));
  }

  start(event) {
    if (event.button !== 0) return;
    this.isDrawing = true;
    this.startedAt = performance.now();
    this.points = [this.eventToPoint(event)];
    this.canvas.setPointerCapture?.(event.pointerId);
    this.callbacks.onPreview?.(this.points);
  }

  move(event) {
    if (!this.isDrawing) return;
    this.points.push(this.eventToPoint(event));
    this.callbacks.onPreview?.(this.points);
  }

  end(event) {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    if (event?.clientX !== undefined) this.points.push(this.eventToPoint(event));

    const compactPoints = compactStrokePoints(this.points);
    this.points = [];

    if (compactPoints.length >= 2) this.callbacks.onCommit?.(compactPoints);
  }

  eventToPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);

    return {
      x,
      y,
      t: Math.max(0, Math.round(performance.now() - this.startedAt)),
      p: typeof event.pressure === "number" && event.pressure > 0 ? event.pressure : 0.5
    };
  }
}

function compactStrokePoints(points) {
  return points.filter((point, index) => {
    if (index === 0 || index === points.length - 1) return true;
    const previous = points[index - 1];
    return Math.hypot(point.x - previous.x, point.y - previous.y) > 0.002;
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
