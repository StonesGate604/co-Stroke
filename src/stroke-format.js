export const FORMAT_VERSION = "0.1.0";

export function createEmptyDrawing() {
  const now = new Date().toISOString();

  return {
    schema: "https://co-stroke.local/schema/co-stroke-v0.1.json",
    version: FORMAT_VERSION,
    id: `drawing_${Date.now()}`,
    title: "untitled-session",
    category: "unknown",
    createdAt: now,
    updatedAt: now,
    source: { type: "human-session", name: "local browser session" },
    canvas: { width: 960, height: 640, coordinateSystem: "normalized", background: "#ffffff" },
    timeline: { unit: "stroke", currentStep: 0, branchPolicy: "truncate-future-on-edit" },
    strokes: []
  };
}

export function validateDrawing(data) {
  if (!data || typeof data !== "object") throw new Error("Drawing data must be an object.");
  if (!Array.isArray(data.strokes)) throw new Error("Drawing data must include a strokes array.");

  data.strokes.forEach((stroke, strokeIndex) => {
    if (!Array.isArray(stroke.points) || stroke.points.length < 2) {
      throw new Error(`Stroke ${strokeIndex} must include at least two points.`);
    }

    stroke.points.forEach((point, pointIndex) => {
      if (typeof point.x !== "number" || typeof point.y !== "number") {
        throw new Error(`Point ${pointIndex} in stroke ${strokeIndex} needs numeric x and y values.`);
      }
    });
  });

  return data;
}

export function normalizeDrawing(data) {
  const drawing = validateDrawing(data);
  const base = createEmptyDrawing();

  return {
    ...base,
    ...drawing,
    version: FORMAT_VERSION,
    canvas: { ...base.canvas, ...(drawing.canvas ?? {}), coordinateSystem: "normalized" },
    timeline: { ...base.timeline, ...(drawing.timeline ?? {}) },
    source: { ...base.source, ...(drawing.source ?? {}) },
    strokes: drawing.strokes.map((stroke, index) => normalizeStroke(stroke, index))
  };
}

export function normalizeStroke(stroke, index = 0) {
  const legacyAuthor = typeof stroke.author === "string" ? { type: stroke.author } : stroke.author;
  const legacyStyle = stroke.style ?? { color: stroke.color ?? "#111111", width: stroke.size ?? 4 };

  return {
    id: stroke.id ?? `stroke_${String(index + 1).padStart(3, "0")}`,
    author: {
      type: legacyAuthor?.type ?? "unknown",
      ...(legacyAuthor?.id ? { id: legacyAuthor.id } : {}),
      ...(legacyAuthor?.model ? { model: legacyAuthor.model } : {})
    },
    tool: stroke.tool ?? "pen",
    style: {
      color: legacyStyle.color ?? "#111111",
      width: legacyStyle.width ?? 4,
      opacity: legacyStyle.opacity ?? 1,
      lineCap: legacyStyle.lineCap ?? "round",
      lineJoin: legacyStyle.lineJoin ?? "round"
    },
    timing: {
      startMs: stroke.timing?.startMs ?? 0,
      durationMs: stroke.timing?.durationMs ?? estimateDuration(stroke.points)
    },
    points: stroke.points,
    metadata: stroke.metadata ?? {}
  };
}

export function createStroke({ id, authorType, tool, color, width, points, insertedAtStep }) {
  return normalizeStroke({
    id,
    author: { type: authorType, id: authorType === "human" ? "local-user" : undefined },
    tool,
    style: { color, width, opacity: 1, lineCap: "round", lineJoin: "round" },
    timing: { startMs: 0, durationMs: estimateDuration(points) },
    points,
    metadata: { insertedAtStep }
  });
}

export function trimDrawingToStep(drawing, step) {
  return {
    ...drawing,
    updatedAt: new Date().toISOString(),
    timeline: { ...drawing.timeline, currentStep: step },
    strokes: drawing.strokes.slice(0, step)
  };
}

export function serializeDrawing(drawing) {
  return JSON.stringify({
    ...drawing,
    updatedAt: new Date().toISOString(),
    timeline: { ...drawing.timeline, currentStep: drawing.strokes.length }
  }, null, 2);
}

function estimateDuration(points) {
  const lastPoint = points?.[points.length - 1];
  return typeof lastPoint?.t === "number" ? lastPoint.t : Math.max(120, (points?.length ?? 2) * 40);
}
