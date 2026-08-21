import { requestAIContinuation } from "./ai-adapter.js";
import { DrawingInputController } from "./drawing-input.js";
import { createEmptyDrawing, createStroke, normalizeDrawing, serializeDrawing, trimDrawingToStep } from "./stroke-format.js";
import { TimelinePlayer } from "./timeline-player.js";
import { TimelineView } from "./timeline-view.js";

const STROKE_STYLE = {
  humanColor: "#111111",
  aiColor: "#ff44aa",
  width: 4,
  eraserWidth: 10
};

const MAX_CONSECUTIVE_AI_STROKES = 12;

const dom = {
  canvas: document.getElementById("drawingCanvas"),
  loadExampleButton: document.getElementById("loadExampleButton"),
  exportButton: document.getElementById("exportButton"),
  undoButton: document.getElementById("undoButton"),
  clearButton: document.getElementById("clearButton"),
  penToolButton: document.getElementById("penToolButton"),
  eraserToolButton: document.getElementById("eraserToolButton"),
  playButton: document.getElementById("playButton"),
  pauseButton: document.getElementById("pauseButton"),
  resetButton: document.getElementById("resetButton"),
  aiStepButton: document.getElementById("aiStepButton"),
  contextStat: document.getElementById("contextStat"),
  timelineTrack: document.querySelector(".tl-track"),
  timelineIcons: document.getElementById("tl-icons"),
  timelineRange: document.getElementById("timeline"),
  timelineLabel: document.getElementById("tl-label"),
  timelineCursor: document.getElementById("tl-cursor"),
  titleStat: document.getElementById("titleStat"),
  categoryStat: document.getElementById("categoryStat"),
  stepStat: document.getElementById("stepStat"),
  authorStat: document.getElementById("authorStat")
};

const state = {
  currentTool: "pen",
  drawing: createCatSession()
};

const player = new TimelinePlayer(dom.canvas);
const timelineView = new TimelineView({
  track: dom.timelineTrack,
  icons: dom.timelineIcons,
  range: dom.timelineRange,
  label: dom.timelineLabel,
  cursor: dom.timelineCursor,
  onSeek: (step) => player.seek(step)
});

player.onChange = ({ drawing, step, totalSteps, currentStroke }) => {
  updateSequenceStats({ drawing, step, totalSteps, currentStroke });
  timelineView.update(drawing, step);
};

new DrawingInputController(dom.canvas, {
  onPreview(points) {
    player.render(buildHumanStroke(points));
  },
  onCommit(points) {
    player.pause();
    state.drawing = trimDrawingToStep(state.drawing, player.step);
    state.drawing.strokes.push(buildHumanStroke(points));
    setContextMessage("Human stroke added — ready to continue", false);
    player.setDrawing(state.drawing, state.drawing.strokes.length);
  }
});

bindControls();
player.load(state.drawing);

function bindControls() {
  dom.loadExampleButton.addEventListener("click", loadExample);
  dom.exportButton.addEventListener("click", exportDrawing);
  dom.undoButton.addEventListener("click", undoStroke);
  dom.clearButton.addEventListener("click", clearDrawing);
  dom.penToolButton.addEventListener("click", () => selectTool("pen"));
  dom.eraserToolButton.addEventListener("click", () => selectTool("eraser"));
  dom.playButton.addEventListener("click", () => player.play());
  dom.pauseButton.addEventListener("click", () => player.pause());
  dom.resetButton.addEventListener("click", () => player.reset());
  dom.aiStepButton.addEventListener("click", addAIStroke);
}

async function loadExample() {
  const response = await fetch("../data/examples/simple-house.json");
  const data = await response.json();
  state.drawing = normalizeDrawing(data);
  setContextMessage("Example loaded — context not packed yet", false);
  player.load(state.drawing);
}

async function addAIStroke() {
  player.pause();
  state.drawing = trimDrawingToStep(state.drawing, player.step);

  if (countTrailingAIStrokes(state.drawing.strokes) >= MAX_CONSECUTIVE_AI_STROKES) {
    setContextMessage("AI paused after 12 strokes — add a human stroke", true);
    return;
  }

  dom.aiStepButton.disabled = true;
  setContextMessage("Packing human-priority context…", false);

  try {
    const response = await requestAIContinuation({
      drawing: state.drawing,
      currentStep: player.step,
      options: {
        maxStrokes: 1,
        maxPointsPerStroke: 32,
        steps: 64,
        temperature: 0.55,
        categoryHint: state.drawing.category,
        style: {
          color: STROKE_STYLE.aiColor,
          width: STROKE_STYLE.width
        }
      }
    });

    updateContextMessage(response.context, response.model);
    if (!response.strokes.length) return;

    state.drawing.strokes.push(...response.strokes);
    state.drawing.updatedAt = new Date().toISOString();
    player.setDrawing(state.drawing, state.drawing.strokes.length);
  } finally {
    dom.aiStepButton.disabled = false;
  }
}

function updateContextMessage(context, model) {
  if (!context) {
    setContextMessage(`Context unavailable — ${model?.name ?? "mock model"}`, true);
    return;
  }

  const unit = context.unit === "strokes" ? "strokes" : "actions";
  const message = `Context: ${context.humanActionsUsed} human + ${context.aiActionsUsed} AI / ${context.maxActions} ${unit}`;
  setContextMessage(message, context.compacted);
  dom.contextStat.title = [
    `Policy: ${context.policy}`,
    `Human: ${context.humanActionsBefore} → ${context.humanActionsUsed} ${unit}`,
    `AI: ${context.aiActionsBefore} → ${context.aiActionsUsed} ${unit}`,
    `Dropped AI strokes: ${context.droppedAIStrokes}`
  ].join("\n");
}

function setContextMessage(message, compacted) {
  dom.contextStat.textContent = message;
  dom.contextStat.classList.toggle("compacted", Boolean(compacted));
  if (!compacted) dom.contextStat.removeAttribute("title");
}

function countTrailingAIStrokes(strokes) {
  let count = 0;
  for (let index = strokes.length - 1; index >= 0; index -= 1) {
    if (strokes[index]?.author?.type !== "ai") break;
    count += 1;
  }
  return count;
}

function buildHumanStroke(points) {
  const isEraser = state.currentTool === "eraser";
  return createStroke({
    id: `stroke_${String(state.drawing.strokes.length + 1).padStart(3, "0")}_${Date.now()}`,
    authorType: "human",
    tool: state.currentTool,
    color: isEraser ? "#ffffff" : STROKE_STYLE.humanColor,
    width: isEraser ? STROKE_STYLE.eraserWidth : STROKE_STYLE.width,
    points,
    insertedAtStep: player.step
  });
}

function undoStroke() {
  player.pause();
  state.drawing = trimDrawingToStep(state.drawing, Math.max(0, state.drawing.strokes.length - 1));
  player.setDrawing(state.drawing, state.drawing.strokes.length);
}

function clearDrawing() {
  player.pause();
  state.drawing = createCatSession();
  setContextMessage("Context: waiting for a drawing", false);
  player.load(state.drawing);
}

function createCatSession() {
  return {
    ...createEmptyDrawing(),
    title: "cat-session",
    category: "cat"
  };
}

function exportDrawing() {
  const blob = new Blob([serializeDrawing(state.drawing)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.drawing.title || "co-stroke"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function selectTool(tool) {
  state.currentTool = tool;
  dom.penToolButton.classList.toggle("selected", tool === "pen");
  dom.eraserToolButton.classList.toggle("selected", tool === "eraser");
}

function updateSequenceStats({ drawing, step, totalSteps, currentStroke }) {
  dom.titleStat.textContent = drawing?.title ?? "none";
  dom.categoryStat.textContent = drawing?.category ?? "none";
  dom.stepStat.textContent = `${step} / ${totalSteps}`;
  dom.authorStat.textContent = currentStroke?.author?.type ?? "none";
}

if (window.lucide) {
  window.lucide.createIcons();
}
