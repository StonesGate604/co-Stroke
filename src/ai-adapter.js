import { createStroke } from "./stroke-format.js";

export class MockStrokeModelAdapter {
  constructor() {
    this.name = "mock-autoregressive-stroke-model";
    this.version = "0.1.0";
  }

  async continue({ drawing, currentStep, options = {} }) {
    const visibleStrokes = drawing.strokes.slice(0, currentStep);
    const seedStroke = visibleStrokes[visibleStrokes.length - 1];
    const seedPoint = seedStroke?.points?.[seedStroke.points.length - 1] ?? { x: 0.48, y: 0.48 };
    const offset = Math.min(0.08, 0.03 + (options.temperature ?? 0.4) * 0.05);

    return {
      model: { name: this.name, version: this.version },
      strokes: [
        createStroke({
          id: `stroke_ai_${Date.now()}`,
          authorType: "ai",
          tool: "pen",
          color: options.style?.color ?? "#ff44aa",
          width: options.style?.width ?? 4,
          insertedAtStep: currentStep,
          points: [
            { x: clamp(seedPoint.x, 0.08, 0.92), y: clamp(seedPoint.y, 0.08, 0.92), t: 0, p: 0.5 },
            { x: clamp(seedPoint.x + offset, 0.08, 0.92), y: clamp(seedPoint.y - offset, 0.08, 0.92), t: 120, p: 0.5 },
            { x: clamp(seedPoint.x + offset * 1.8, 0.08, 0.92), y: clamp(seedPoint.y, 0.08, 0.92), t: 240, p: 0.5 }
          ]
        })
      ]
    };
  }
}

export async function requestAIContinuation(request, adapter = new MockStrokeModelAdapter()) {
  return adapter.continue(request);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

