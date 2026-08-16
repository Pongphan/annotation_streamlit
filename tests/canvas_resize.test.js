const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");


class MockTarget {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  dispatch(type, event = {}) {
    for (const listener of this.listeners.get(type) || []) {
      listener(event);
    }
  }
}


function loadCanvasComponent() {
  const messages = [];
  const context = {
    beginPath() {},
    clearRect() {},
    closePath() {},
    drawImage() {},
    fill() {},
    fillRect() {},
    lineTo() {},
    moveTo() {},
    rect() {},
    restore() {},
    save() {},
    setLineDash() {},
    stroke() {},
    arc() {},
  };

  const canvas = Object.assign(new MockTarget(), {
    width: 0,
    height: 0,
    style: {},
    getContext: () => context,
    getBoundingClientRect() {
      return { left: 0, top: 0, width: this.width, height: this.height };
    },
    setPointerCapture() {},
    releasePointerCapture() {},
  });
  const toolbar = Object.assign(new MockTarget(), { hidden: false, offsetHeight: 42 });
  const status = { textContent: "" };
  const buttons = {
    undoButton: Object.assign(new MockTarget(), { disabled: false }),
    deleteButton: Object.assign(new MockTarget(), { disabled: false }),
    finishButton: Object.assign(new MockTarget(), { disabled: false }),
    clearButton: Object.assign(new MockTarget(), { disabled: false }),
  };
  const elements = {
    annotationCanvas: canvas,
    toolbar,
    status,
    ...buttons,
  };
  const document = {
    documentElement: { style: {} },
    getElementById: (id) => elements[id],
  };
  const window = Object.assign(new MockTarget(), {
    parent: {
      postMessage(message) {
        messages.push(message);
      },
    },
  });
  const getComputedStyle = () => ({
    getPropertyValue(name) {
      return name === "--canvas-bg" ? "#ffffff" : "#ffe8b0";
    },
  });
  class MockImage {}

  const htmlPath = path.join(__dirname, "..", "annotation_canvas_component", "index.html");
  const html = fs.readFileSync(htmlPath, "utf8");
  const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/u);
  assert.ok(scriptMatch, "canvas component script should exist");

  const execute = new Function("window", "document", "getComputedStyle", "Image", scriptMatch[1]);
  execute(window, document, getComputedStyle, MockImage);

  return { canvas, messages, status, window };
}


function pointer(x, y) {
  return {
    button: 0,
    clientX: x,
    clientY: y,
    pointerId: 1,
    preventDefault() {},
  };
}


function latestComponentValue(messages) {
  const values = messages.filter(
    (message) => message.type === "streamlit:setComponentValue"
  );
  return values.at(-1).value;
}


test("dragging the latest rectangle handle resizes it and preserves its ID", () => {
  const { canvas, messages, status, window } = loadCanvasComponent();

  window.dispatch("message", {
    data: {
      type: "streamlit:render",
      args: {
        canvasWidth: 300,
        canvasHeight: 200,
        drawingMode: "rect",
        displayToolbar: true,
        initialDrawing: { objects: [] },
      },
    },
  });

  canvas.dispatch("pointerdown", pointer(10, 10));
  canvas.dispatch("pointermove", pointer(100, 100));
  canvas.dispatch("pointerup", pointer(100, 100));

  const beforeResize = latestComponentValue(messages).objects[0];
  assert.equal(beforeResize.width, 90);
  assert.equal(beforeResize.height, 90);

  canvas.dispatch("pointerdown", pointer(100, 100));
  canvas.dispatch("pointermove", pointer(150, 130));
  canvas.dispatch("pointerup", pointer(150, 130));

  const afterResize = latestComponentValue(messages).objects[0];
  assert.equal(afterResize.width, 140);
  assert.equal(afterResize.height, 120);
  assert.equal(afterResize._annotationId, beforeResize._annotationId);
  assert.match(status.textContent, /140 x 120px/u);

  canvas.dispatch("pointerdown", pointer(180, 20));
  canvas.dispatch("pointermove", pointer(230, 70));
  canvas.dispatch("pointerup", pointer(230, 70));
  canvas.dispatch("pointerdown", pointer(230, 70));
  canvas.dispatch("pointermove", pointer(260, 100));
  canvas.dispatch("pointerup", pointer(260, 100));

  const finalObjects = latestComponentValue(messages).objects;
  assert.equal(finalObjects.length, 2, "resizing must not draw another rectangle");
  assert.equal(finalObjects[0].width, 140, "the older rectangle should be unchanged");
  assert.equal(finalObjects[1].width, 80, "the latest rectangle should be resized");
  assert.equal(finalObjects[1].height, 80);
});
