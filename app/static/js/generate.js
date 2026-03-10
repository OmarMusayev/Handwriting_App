// generate.js — Generation form, job polling, progressive sample rendering

(function () {
  "use strict";

  const POLL_MS = 1500;

  // ── Elements ──────────────────────────────────────────────────────────────
  const form = document.getElementById("generate-form");
  const textInput = document.getElementById("gen-text");
  const stylePills = document.getElementById("style-pills");
  const biasSlider = document.getElementById("bias-slider");
  const biasDisplay = document.getElementById("bias-display");
  const synthBtn = document.getElementById("btn-synthesize");
  const progressBar = document.getElementById("gen-progress-bar");
  const progressFill = document.getElementById("gen-progress-fill");
  const progressLabel = document.getElementById("gen-progress-label");
  const resultsGrid = document.getElementById("results-grid");
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");

  // ── Bias slider ───────────────────────────────────────────────────────────
  biasSlider.addEventListener("input", () => {
    biasDisplay.textContent = parseFloat(biasSlider.value).toFixed(1);
  });

  // ── Style pills ───────────────────────────────────────────────────────────
  let selectedStyleId = null;  // null = default style

  function renderStylePills(styles) {
    stylePills.innerHTML = "";

    // Default pill
    const def = document.createElement("button");
    def.type = "button";
    def.className = "style-pill" + (selectedStyleId === null ? " active" : "");
    def.textContent = "Default";
    def.dataset.id = "";
    def.addEventListener("click", () => selectPill(null, def));
    stylePills.appendChild(def);

    styles.forEach((s) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "style-pill" + (selectedStyleId === s.id ? " active" : "");
      pill.textContent = s.name;
      pill.dataset.id = s.id;
      pill.addEventListener("click", () => selectPill(s.id, pill));
      stylePills.appendChild(pill);
    });
  }

  function selectPill(id, el) {
    selectedStyleId = id;
    stylePills.querySelectorAll(".style-pill").forEach((p) => p.classList.remove("active"));
    el.classList.add("active");
  }

  // Listen for style list updates from studio.js
  document.addEventListener("stylesUpdated", (e) => {
    renderStylePills(e.detail);
  });

  // Initial render with empty list (studio.js will fire the event shortly)
  renderStylePills([]);

  // ── Generate form submit ───────────────────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = textInput.value.trim();
    if (!text) {
      flashError("Please enter some text.");
      return;
    }

    const bias = parseFloat(biasSlider.value);
    const payload = { text, bias };
    if (selectedStyleId) payload.style_id = selectedStyleId;

    // UI: disable form, clear results, show progress
    synthBtn.disabled = true;
    synthBtn.textContent = "⏳ GENERATING…";
    resultsGrid.innerHTML = "";
    progressBar.style.display = "block";
    progressFill.style.width = "0%";
    progressLabel.textContent = "Starting…";

    try {
      const resp = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Generate request failed");
      }
      const { job_id } = await resp.json();
      pollJob(job_id, 0);
    } catch (err) {
      flashError(err.message);
      resetUI();
    }
  });

  // ── Job polling ───────────────────────────────────────────────────────────
  let renderedSamples = 0;

  async function pollJob(jobId, previousDone) {
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      if (!resp.ok) throw new Error("Job fetch failed");
      const job = await resp.json();

      const total = job.total || 5;
      const done = job.done || 0;

      // Render any newly completed samples
      for (let n = previousDone; n < done; n++) {
        await renderSample(jobId, n);
      }

      // Update progress bar
      const pct = Math.round((done / total) * 100);
      progressFill.style.width = pct + "%";
      progressLabel.textContent = `${done} / ${total} samples`;

      if (job.status === "done") {
        progressLabel.textContent = "Done!";
        setTimeout(() => { progressBar.style.display = "none"; }, 1500);
        resetUI();
        return;
      }
      if (job.status === "error") {
        flashError("Generation error: " + (job.error || "unknown"));
        progressBar.style.display = "none";
        resetUI();
        return;
      }

      // Still running — poll again
      setTimeout(() => pollJob(jobId, done), POLL_MS);
    } catch (err) {
      flashError("Polling error: " + err.message);
      resetUI();
    }
  }

  async function renderSample(jobId, n) {
    try {
      const resp = await fetch(`/api/jobs/${jobId}/sample/${n}`);
      if (!resp.ok) return;
      const { data_url } = await resp.json();

      const wrapper = document.createElement("div");
      wrapper.className = "result-item";

      const img = document.createElement("img");
      img.src = data_url;
      img.alt = `Sample ${n + 1}`;
      img.addEventListener("click", () => openLightbox(data_url));

      const dl = document.createElement("a");
      dl.className = "result-download";
      dl.href = data_url;
      dl.download = `handwriting_${n + 1}.png`;
      dl.textContent = "↓";
      dl.title = "Download";

      wrapper.appendChild(img);
      wrapper.appendChild(dl);
      resultsGrid.appendChild(wrapper);

      // Animate in
      requestAnimationFrame(() => wrapper.classList.add("visible"));
    } catch (_) {}
  }

  // ── Lightbox ──────────────────────────────────────────────────────────────
  function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.style.display = "flex";
  }

  if (lightbox) {
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox) lightbox.style.display = "none";
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") lightbox.style.display = "none";
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function resetUI() {
    synthBtn.disabled = false;
    synthBtn.textContent = "[ ▶ SYNTHESIZE ]";
  }

  function flashError(msg) {
    let el = document.getElementById("flash-msg");
    if (!el) {
      el = document.createElement("div");
      el.id = "flash-msg";
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.className = "flash-msg error";
    el.style.display = "block";
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.style.display = "none"; }, 4000);
  }
})();
