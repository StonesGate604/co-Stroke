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

export class LocalStrokeModelAdapter {
  constructor({ endpoint = "http://127.0.0.1:8787/continue", fallback = new MockStrokeModelAdapter() } = {}) {
    this.endpoint = endpoint;
    this.fallback = fallback;
  }

  async continue(request) {
    try {
      const response = await fetch(this.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`Local model server returned ${response.status}`);
      }

      const result = await response.json();
      if (!Array.isArray(result.strokes)) {
        throw new Error("Local model response did not include strokes.");
      }
      return result;
    } catch (error) {
      console.warn("Local stroke model unavailable; using mock continuation.", error);
      return this.fallback.continue(request);
    }
  }
}

export async function requestAIContinuation(request, adapter = new LocalStrokeModelAdapter()) {
  return adapter.continue(request);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

