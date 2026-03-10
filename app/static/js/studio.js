// studio.js — Style Lab: SVG canvas drawing + style CRUD

(function () {
  "use strict";

  // ── Canvas state ──────────────────────────────────────────────────────────
  const canvas = document.getElementById("draw-canvas");
  const ctx = canvas.getContext("2d");
  let isDrawing = false;
  let lastX = 0, lastY = 0;
  let strokes = [];       // [{x,y,eos}] accumulated during this draw session
  let currentStroke = []; // points in the stroke being drawn right now

  // Canvas DPI fix
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // ── Pointer helpers (mouse + touch) ───────────────────────────────────────
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    if (e.touches) {
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top,
      };
    }
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getPos(e);
    lastX = x; lastY = y;
    currentStroke = [{ x, y, eos: 0 }];
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    currentStroke.push({ x, y, eos: 0 });
    lastX = x; lastY = y;
  }

  function endDraw(e) {
    if (!isDrawing) return;
    isDrawing = false;
    if (currentStroke.length) {
      currentStroke[currentStroke.length - 1].eos = 1; // mark pen-up
      strokes.push(...currentStroke);
    }
    currentStroke = [];
    ctx.beginPath();
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", endDraw);
  canvas.addEventListener("mouseleave", endDraw);
  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", endDraw);

  // ── Clear button ──────────────────────────────────────────────────────────
  document.getElementById("btn-clear").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    strokes = [];
    currentStroke = [];
  });

  // ── Convert absolute points → offsets [eos, dx, dy] ──────────────────────
  function pointsToOffsets(pts) {
    if (!pts.length) return [];
    const result = [[pts[0].eos, 0, 0]];
    for (let i = 1; i < pts.length; i++) {
      result.push([
        pts[i].eos,
        pts[i].x - pts[i - 1].x,
        pts[i].y - pts[i - 1].y,
      ]);
    }
    return result;
  }

  // ── Save Style ────────────────────────────────────────────────────────────
  document.getElementById("btn-save").addEventListener("click", async () => {
    if (!strokes.length) {
      flashMessage("Draw something first!", "error");
      return;
    }
    const offsets = pointsToOffsets(strokes);
    // Build priming text from canvas (use empty string; server uses default)
    const primedText = "";

    const btn = document.getElementById("btn-save");
    btn.disabled = true;
    btn.textContent = "SAVING…";

    try {
      const resp = await fetch("/api/styles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stroke_data: offsets, priming_text: primedText }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Save failed");
      }
      const data = await resp.json();
      flashMessage(`Style "${data.name}" saved!`, "success");
      // Clear canvas after save
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      strokes = [];
      await loadStyles();
    } catch (err) {
      flashMessage(err.message, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "[ SAVE STYLE ]";
    }
  });

  // ── Style list ────────────────────────────────────────────────────────────
  let stylesList = [];   // [{id, name, created_at}]

  async function loadStyles() {
    try {
      const resp = await fetch("/api/styles");
      if (!resp.ok) throw new Error("Failed to load styles");
      stylesList = await resp.json();
    } catch (e) {
      stylesList = [];
    }
    renderStyleList();
    // Notify generate.js that the style list changed
    document.dispatchEvent(new CustomEvent("stylesUpdated", { detail: stylesList }));
  }

  function renderStyleList() {
    const container = document.getElementById("style-list");
    if (!stylesList.length) {
      container.innerHTML = '<p class="no-styles">No styles yet.<br>Draw &amp; save one!</p>';
      return;
    }
    container.innerHTML = stylesList.map((s) => styleCard(s)).join("");
    // Attach events
    container.querySelectorAll(".style-name").forEach((el) => {
      el.addEventListener("click", () => startRename(el.dataset.id, el));
    });
    container.querySelectorAll(".btn-delete-style").forEach((el) => {
      el.addEventListener("click", () => deleteStyle(el.dataset.id));
    });
  }

  function styleCard(s) {
    return `
      <div class="style-card" data-id="${s.id}">
        <span class="style-name" data-id="${s.id}" title="Click to rename">${escHtml(s.name)}</span>
        <button class="btn-delete-style" data-id="${s.id}" title="Delete style">✕</button>
      </div>`;
  }

  // ── Inline rename ─────────────────────────────────────────────────────────
  function startRename(id, el) {
    const currentName = el.textContent;
    const input = document.createElement("input");
    input.className = "rename-input";
    input.value = currentName;
    el.replaceWith(input);
    input.focus();
    input.select();

    async function commitRename() {
      const newName = input.value.trim();
      if (!newName || newName === currentName) {
        loadStyles();
        return;
      }
      try {
        const resp = await fetch(`/api/styles/${id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: newName }),
        });
        if (!resp.ok) throw new Error("Rename failed");
        flashMessage("Renamed!", "success");
      } catch (e) {
        flashMessage(e.message, "error");
      }
      loadStyles();
    }

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") commitRename();
      if (e.key === "Escape") loadStyles();
    });
    input.addEventListener("blur", commitRename);
  }

  // ── Delete style ──────────────────────────────────────────────────────────
  async function deleteStyle(id) {
    try {
      const resp = await fetch(`/api/styles/${id}`, { method: "DELETE" });
      if (!resp.ok) throw new Error("Delete failed");
      flashMessage("Style deleted", "success");
      await loadStyles();
    } catch (e) {
      flashMessage(e.message, "error");
    }
  }

  // ── Flash message ─────────────────────────────────────────────────────────
  function flashMessage(msg, type) {
    let el = document.getElementById("flash-msg");
    if (!el) {
      el = document.createElement("div");
      el.id = "flash-msg";
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.className = "flash-msg " + (type || "info");
    el.style.display = "block";
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.style.display = "none"; }, 3000);
  }

  function escHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  loadStyles();

  // Expose for generate.js
  window.studioGetStyles = () => stylesList;
})();
