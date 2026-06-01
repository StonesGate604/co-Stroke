const STEP_SLOT_WIDTH = 39;

const STEP_ICON = {
  pen: "pencil",
  eraser: "eraser",
  ai: "sparkles",
  robot: "bot"
};

export class TimelineView {
  constructor({ track, icons, range, label, cursor, onSeek }) {
    this.track = track;
    this.icons = icons;
    this.range = range;
    this.label = label;
    this.cursor = cursor;
    this.onSeek = onSeek;
    this.drawing = null;
    this.activeIndex = -1;
    this.isBound = false;
    this.bindInteractions();
  }

  update(drawing, visibleStep) {
    this.drawing = drawing;
    const total = drawing?.strokes.length ?? 0;
    this.activeIndex = visibleStep > 0 ? Math.min(visibleStep - 1, total - 1) : -1;

    this.range.max = String(Math.max(0, total - 1));
    this.range.value = String(Math.max(0, this.activeIndex));
    this.label.textContent = total === 0 ? "step 0" : `step ${this.activeIndex + 1}/${total}`;

    this.renderIcons();
    this.updateCursor();
    this.ensureActiveStepVisible(false);
  }

  renderIcons() {
    const strokes = this.drawing?.strokes ?? [];
    this.icons.innerHTML = "";
    this.icons.style.width = `${Math.max(1, strokes.length) * STEP_SLOT_WIDTH + 28}px`;

    strokes.forEach((stroke, index) => {
      const step = document.createElement("button");
      const stepClass = getStepClass(stroke);
      const iconName = getStepIcon(stroke);

      step.className = `tl-step ${stepClass}${index === this.activeIndex ? " active" : ""}`;
      step.title = `Step ${index + 1}: ${stroke.author.type} ${stroke.tool}`;
      step.dataset.index = String(index);
      step.innerHTML = `<i data-lucide="${iconName}">${fallbackIcon(iconName)}</i>`;
      step.addEventListener("click", () => this.onSeek(index + 1));
      this.icons.appendChild(step);
    });

    if (window.lucide) window.lucide.createIcons();
  }

  updateActiveStep(activeIndex) {
    this.activeIndex = activeIndex;
    this.icons.querySelectorAll(".tl-step").forEach((step) => {
      const index = Number(step.dataset.index ?? -1);
      step.classList.toggle("active", index === this.activeIndex);
    });
    this.updateCursor();
  }

  updateCursor() {
    if (!this.cursor || this.activeIndex < 0) {
      this.cursor.style.opacity = "0";
      return;
    }

    const active = this.icons.querySelector(`.tl-step[data-index="${this.activeIndex}"]`);
    if (!active) {
      this.cursor.style.opacity = "0";
      return;
    }

    const trackRect = this.track.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    const x = activeRect.left - trackRect.left + activeRect.width / 2;
    this.cursor.style.left = `${x}px`;
    this.cursor.style.opacity = "0.75";
  }

  ensureActiveStepVisible(center) {
    const active = this.icons.querySelector(`.tl-step[data-index="${this.activeIndex}"]`);
    if (!active) return;

    const centerX = active.offsetLeft + active.offsetWidth / 2;
    const viewLeft = this.track.scrollLeft;
    const viewRight = viewLeft + this.track.clientWidth;
    const padding = STEP_SLOT_WIDTH * 2;
    const isOutside = centerX < viewLeft + padding || centerX > viewRight - padding;
    if (!isOutside && !center) return;

    const target = Math.max(0, centerX - this.track.clientWidth / 2);
    this.track.scrollTo({ left: target, behavior: "auto" });
  }

  bindInteractions() {
    if (this.isBound) return;
    let isScrubbing = false;

    this.track.addEventListener("scroll", () => this.updateCursor());

    this.track.addEventListener("mousedown", (event) => {
      if (event.button !== 0) return;
      if (event.target.closest(".tl-step")) return;
      isScrubbing = true;
      this.seekNearestStepByClientX(event.clientX);
    });

    window.addEventListener("mousemove", (event) => {
      if (!isScrubbing) return;
      this.seekNearestStepByClientX(event.clientX);
    });

    window.addEventListener("mouseup", () => {
      isScrubbing = false;
    });

    this.track.addEventListener("click", (event) => {
      if (event.target.closest(".tl-step")) return;
      this.seekNearestStepByClientX(event.clientX);
    });

    this.range.addEventListener("input", () => {
      const index = Number(this.range.value) || 0;
      this.onSeek(index + 1);
    });

    this.isBound = true;
  }

  seekNearestStepByClientX(clientX) {
    const strokes = this.drawing?.strokes ?? [];
    if (strokes.length === 0) return;

    const steps = this.icons.querySelectorAll(".tl-step");
    if (steps.length === 0) return;

    const rect = this.track.getBoundingClientRect();
    const localX = clientX - rect.left + this.track.scrollLeft;
    let nearest = 0;
    let minDist = Number.POSITIVE_INFINITY;

    steps.forEach((step, index) => {
      const center = step.offsetLeft + step.offsetWidth / 2;
      const dist = Math.abs(center - localX);
      if (dist < minDist) {
        minDist = dist;
        nearest = index;
      }
    });

    this.onSeek(nearest + 1);
  }
}

function getStepClass(stroke) {
  if (stroke.author?.type === "ai") return "ai";
  if (stroke.tool === "eraser") return "eraser";
  return "brush";
}

function getStepIcon(stroke) {
  if (stroke.author?.type === "ai") return STEP_ICON.ai;
  if (stroke.author?.type === "robot") return STEP_ICON.robot;
  return STEP_ICON[stroke.tool] ?? STEP_ICON.pen;
}

function fallbackIcon(iconName) {
  if (iconName === "eraser") return "E";
  if (iconName === "sparkles") return "AI";
  if (iconName === "bot") return "R";
  return "P";
}
