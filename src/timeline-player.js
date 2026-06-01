export class TimelinePlayer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.drawing = null;
    this.step = 0;
    this.timer = null;
    this.onChange = () => {};
    this.clear();
  }

  load(drawing) {
    this.pause();
    this.drawing = drawing;
    this.step = drawing.timeline?.currentStep ?? 0;
    this.render();
    this.emitChange();
  }

  setDrawing(drawing, step = drawing.strokes.length) {
    this.pause();
    this.drawing = drawing;
    this.step = Math.max(0, Math.min(step, drawing.strokes.length));
    this.render();
    this.emitChange();
  }

  play() {
    if (!this.drawing || this.timer) return;
    this.timer = window.setInterval(() => {
      if (this.step >= this.drawing.strokes.length) {
        this.pause();
        return;
      }
      this.step += 1;
      this.render();
      this.emitChange();
    }, 420);
  }

  pause() {
    if (!this.timer) return;
    window.clearInterval(this.timer);
    this.timer = null;
  }

  reset() {
    this.pause();
    this.step = 0;
    this.render();
    this.emitChange();
  }

  seek(step) {
    if (!this.drawing) return;
    this.step = Math.max(0, Math.min(step, this.drawing.strokes.length));
    this.render();
    this.emitChange();
  }

  render(extraStroke = null) {
    this.clear();
    if (!this.drawing) return;
    this.drawing.strokes.slice(0, this.step).forEach((stroke) => this.drawStroke(stroke));
    if (extraStroke) this.drawStroke(extraStroke);
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillStyle = this.drawing?.canvas?.background ?? "#ffffff";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  drawStroke(stroke) {
    const points = stroke.points;
    if (!points || points.length < 2) return;

    this.ctx.save();
    this.ctx.globalAlpha = stroke.style.opacity ?? 1;
    this.ctx.globalCompositeOperation = stroke.tool === "eraser" ? "destination-out" : "source-over";
    this.ctx.strokeStyle = stroke.style.color;
    this.ctx.lineWidth = stroke.style.width;
    this.ctx.lineCap = stroke.style.lineCap ?? "round";
    this.ctx.lineJoin = stroke.style.lineJoin ?? "round";
    this.ctx.beginPath();

    const firstPoint = points[0];
    this.ctx.moveTo(firstPoint.x * this.canvas.width, firstPoint.y * this.canvas.height);
    points.slice(1).forEach((point) => this.ctx.lineTo(point.x * this.canvas.width, point.y * this.canvas.height));
    this.ctx.stroke();
    this.ctx.restore();
  }

  emitChange() {
    const currentStroke = this.drawing?.strokes[this.step - 1] ?? null;
    this.onChange({
      drawing: this.drawing,
      step: this.step,
      totalSteps: this.drawing?.strokes.length ?? 0,
      currentStroke,
      isPlaying: Boolean(this.timer)
    });
  }
}
