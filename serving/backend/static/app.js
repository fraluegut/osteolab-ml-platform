document.getElementById("year").textContent = new Date().getFullYear();

// ---------- nav móvil ----------
const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navLinks");
navToggle.addEventListener("click", () => {
  const isOpen = navLinks.classList.toggle("is-open");
  navToggle.setAttribute("aria-expanded", String(isOpen));
});
navLinks.querySelectorAll("a").forEach((a) =>
  a.addEventListener("click", () => navLinks.classList.remove("is-open"))
);

// ---------- lightbox (zoom de imágenes de ejemplo) ----------
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxClose = document.getElementById("lightboxClose");

function openLightbox(src, alt) {
  lightboxImg.src = src;
  lightboxImg.alt = alt;
  lightbox.hidden = false;
}
function closeLightbox() {
  lightbox.hidden = true;
  lightboxImg.src = "";
}
// delegado (no solo forEach) porque hay imágenes de resultado que se crean
// dinámicamente después de analizar una foto — no existen todavía al cargar la página.
document.addEventListener("click", (e) => {
  const img = e.target.closest("[data-zoomable]");
  if (img) openLightbox(img.src, img.alt);
});
lightbox.addEventListener("click", closeLightbox);
lightboxClose.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
});

// ---------- demo ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const demoStatus = document.getElementById("demoStatus");
const resultsGrid = document.getElementById("resultsGrid");
const cardGeometric = document.querySelector("#cardGeometric .result-body");
const cardHybrid = document.querySelector("#cardHybrid .result-body");
const cardQwen = document.querySelector("#cardQwen .result-body");

const flow = document.getElementById("flow");
const flowSteps = {
  clip: document.getElementById("flowClip"),
  geometric: document.getElementById("flowGeometric"),
  hybrid: document.getElementById("flowHybrid"),
  qwen: document.getElementById("flowQwen"),
};

function setFlowStatus(key, status, statusText = "") {
  const step = flowSteps[key];
  step.dataset.status = status;
  step.querySelector(".flow-step-status").textContent = statusText;
}
function resetFlow() {
  Object.keys(flowSteps).forEach((key) => setFlowStatus(key, "pending"));
}

let selectedFile = null;

function openFileDialog() {
  fileInput.click();
}
dropzone.addEventListener("click", openFileDialog);
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    openFileDialog();
  }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setSelectedFile(fileInput.files[0]);
});

function setSelectedFile(file) {
  if (!file.type.startsWith("image/")) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.hidden = false;
  dropzoneEmpty.hidden = true;
  analyzeBtn.disabled = false;
  hideStatus();
  resultsGrid.hidden = true;
  flow.hidden = true;
}

function resetDemo() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.hidden = true;
  dropzoneEmpty.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  resultsGrid.hidden = true;
  flow.hidden = true;
  resetFlow();
  hideStatus();
}
resetBtn.addEventListener("click", resetDemo);

function showStatus(message, isError = false) {
  demoStatus.textContent = message;
  demoStatus.hidden = false;
  demoStatus.classList.toggle("is-error", isError);
}
function hideStatus() {
  demoStatus.hidden = true;
  demoStatus.classList.remove("is-error");
}

async function classify(endpoint, file, extraFields = {}) {
  const formData = new FormData();
  formData.append("file", file);
  Object.entries(extraFields).forEach(([key, value]) => formData.append(key, value));
  const response = await fetch(endpoint, { method: "POST", body: formData });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_) {}
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

function appendImageBlock(container, label, base64, altText) {
  if (!base64) return;
  const wrap = document.createElement("div");
  wrap.className = "result-viz";
  wrap.innerHTML = `
    <p class="result-viz-label">${label}</p>
    <img class="result-viz-img" src="data:image/png;base64,${base64}" alt="${altText}" data-zoomable />
  `;
  container.appendChild(wrap);
}

function renderResult(container, result) {
  container.innerHTML = "";

  if (!result.found) {
    const p = document.createElement("p");
    p.className = "result-empty";
    p.textContent = "No se pudo localizar un hueso claro en la imagen.";
    container.appendChild(p);
    return;
  }

  const sorted = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  const [topLabel, topValue] = sorted[0];

  const top = document.createElement("div");
  top.className = "result-top";
  top.innerHTML = `
    <span class="result-top-label">${topLabel.replace(/_/g, " ")}</span>
    <span class="result-top-value">${(topValue * 100).toFixed(1)}%</span>
  `;
  container.appendChild(top);

  sorted.slice(0, 5).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label">${label.replace(/_/g, " ")}</span>
      <span class="prob-bar-track"><span class="prob-bar-fill" style="width:${(value * 100).toFixed(1)}%"></span></span>
      <span class="prob-value">${(value * 100).toFixed(1)}%</span>
    `;
    container.appendChild(row);
  });
}

function renderError(container, message) {
  container.innerHTML = "";
  const p = document.createElement("p");
  p.className = "result-empty";
  p.textContent = message;
  container.appendChild(p);
}

const GROUP_LABELS = {
  no_hueso: "no es un hueso",
  no_seguro: "no está seguro",
};

function renderQwenResult(container, result) {
  container.innerHTML = "";

  const desc = document.createElement("p");
  desc.className = "result-text";
  desc.textContent = result.description || "(sin descripción)";
  container.appendChild(desc);

  const top = document.createElement("div");
  top.className = "result-top";
  const groupLabel = result.suggested_group
    ? GROUP_LABELS[result.suggested_group] || result.suggested_group.replace(/_/g, " ")
    : "sin sugerencia clara";
  top.innerHTML = `
    <span class="result-group">${groupLabel}</span>
    <span class="result-top-value">${result.confidence_label ? "confianza " + result.confidence_label : ""}</span>
  `;
  container.appendChild(top);

  if (result.raw_response) {
    const details = document.createElement("details");
    details.className = "result-raw";
    details.innerHTML = `<summary>Ver respuesta completa</summary><pre>${result.raw_response.replace(/</g, "&lt;")}</pre>`;
    container.appendChild(details);
  }
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const file = selectedFile;
  analyzeBtn.disabled = true;
  resultsGrid.hidden = true;
  resetFlow();
  flow.hidden = false;
  showStatus("Paso 1/2 — filtro CLIP: ¿es un hueso?");

  setFlowStatus("clip", "running", "analizando…");
  let clipResult;
  try {
    clipResult = await classify("/filter/clip", file);
  } catch (err) {
    setFlowStatus("clip", "error", "error");
    showStatus(`Error en el filtro CLIP: ${err.message}`, true);
    analyzeBtn.disabled = false;
    return;
  }

  const clipPct = (clipResult.confidence * 100).toFixed(1);
  if (!clipResult.is_bone) {
    setFlowStatus("clip", "warn", `no es un hueso (${clipPct}%)`);
    ["geometric", "hybrid", "qwen"].forEach((key) => setFlowStatus(key, "skipped", "omitido"));
    showStatus("CLIP no detecta un hueso en la imagen — flujo detenido, no se llama a los otros modelos.", true);
    analyzeBtn.disabled = false;
    resetBtn.hidden = false;
    return;
  }
  const suggestion = clipResult.suggestion ? ` — sugiere ${clipResult.suggestion.type.replace(/_/g, " ")}` : "";
  setFlowStatus("clip", "done", `es un hueso (${clipPct}%)${suggestion}`);
  showStatus("Paso 2/2 — clasificando con los 3 modelos en paralelo…");

  setFlowStatus("geometric", "running", "clasificando…");
  setFlowStatus("hybrid", "running", "clasificando + Grad-CAM…");
  setFlowStatus("qwen", "running", "puede tardar más (VLM 7B)…");
  resultsGrid.hidden = false;
  resetBtn.hidden = false;

  const [geometricResult, hybridResult, qwenResult, gradcamResult] = await Promise.allSettled([
    classify("/classify", file),
    classify("/hybrid/classify", file),
    classify("/describe/qwen", file),
    classify("/explain/gradcam", file, { model_type: "hybrid" }),
  ]);

  hideStatus();
  analyzeBtn.disabled = false;

  if (geometricResult.status === "fulfilled") {
    renderResult(cardGeometric, geometricResult.value);
    appendImageBlock(
      cardGeometric,
      "Máscara de segmentación — zona que midió OpenCV",
      geometricResult.value.cv_features?.mask_overlay_base64,
      "Máscara de segmentación detectada por OpenCV"
    );
    setFlowStatus("geometric", "done", "listo");
  } else {
    renderError(cardGeometric, `Error al clasificar: ${geometricResult.reason.message}`);
    setFlowStatus("geometric", "error", "error");
  }

  if (hybridResult.status === "fulfilled") {
    renderResult(cardHybrid, hybridResult.value);
  } else {
    renderError(cardHybrid, `Error al clasificar: ${hybridResult.reason.message}`);
  }
  if (gradcamResult.status === "fulfilled" && gradcamResult.value.found) {
    appendImageBlock(
      cardHybrid,
      "Grad-CAM — zona que más pesó en la predicción",
      gradcamResult.value.heatmap_base64,
      "Mapa de calor Grad-CAM sobre el recorte usado por la CNN"
    );
  }
  if (hybridResult.status === "fulfilled" && gradcamResult.status === "fulfilled") {
    setFlowStatus("hybrid", "done", "listo + Grad-CAM");
  } else if (hybridResult.status === "fulfilled") {
    setFlowStatus("hybrid", "done", "listo (sin Grad-CAM)");
  } else {
    setFlowStatus("hybrid", "error", "error");
  }

  if (qwenResult.status === "fulfilled") {
    renderQwenResult(cardQwen, qwenResult.value);
    setFlowStatus("qwen", "done", "listo");
  } else {
    renderError(cardQwen, `Error al describir: ${qwenResult.reason.message}`);
    setFlowStatus("qwen", "error", "error");
  }
});

const BONE_GROUPS = [
  "cranio", "mandibula_maxilar", "hueso_largo", "hueso_plano",
  "pelvis", "sacro", "vertebra", "costilla", "hueso_pequeno",
];

async function postJson(endpoint, body) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_) {}
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

// ---------- ampliar base de datos ----------
const ingestUrlInput = document.getElementById("ingestUrlInput");
const ingestSubmitBtn = document.getElementById("ingestSubmitBtn");
const ingestStatusEl = document.getElementById("ingestStatus");
const ingestJobsEl = document.getElementById("ingestJobs");

const JOB_STATUS_LABELS = {
  queued: "en cola…",
  fetching_info: "consultando Sketchfab…",
  downloading: "descargando modelo…",
  splitting: "separando y renderizando huesos… (puede tardar unos minutos)",
  done: "listo",
  error: "error",
};

function ingestShowStatus(message, isError = false) {
  ingestStatusEl.textContent = message;
  ingestStatusEl.hidden = false;
  ingestStatusEl.classList.toggle("is-error", isError);
}
function ingestHideStatus() {
  ingestStatusEl.hidden = true;
  ingestStatusEl.classList.remove("is-error");
}

function renderIngestItem(item) {
  const div = document.createElement("div");
  div.className = "ingest-item";
  if (item.status === "accepted" || item.status === "rejected") div.classList.add("is-accepted");
  if (item.status === "needs_review") div.classList.add("is-review");

  const groupLabel = (item.final_bone_group || item.suggested_bone_group || "?").replace(/_/g, " ");
  const statusNote = {
    suggested: "",
    needs_review: "→ Identificación Manual",
    accepted: "✓ aceptado",
    rejected: "✗ descartado",
  }[item.status] || "";

  div.innerHTML = `
    <img src="${item.thumbnail_path}" alt="${item.object_name}" data-zoomable />
    <span class="ingest-item-name" title="${item.object_name}">${item.object_name}</span>
    <span class="ingest-item-group">${groupLabel}${statusNote ? " " + statusNote : ""}</span>
  `;
  if (item.status === "suggested") {
    const btn = document.createElement("button");
    btn.className = "ingest-item-flag";
    btn.textContent = "no es esto, revisar";
    btn.addEventListener("click", async () => {
      await postJson(`/ingest/items/${item.item_id}/flag-review`, {}).catch(() => {});
      const job = await fetch(`/ingest/jobs/${item.job_id}`).then((r) => r.json());
      renderIngestJob(job);
    });
    div.appendChild(btn);
  }
  return div;
}
// flag-review es POST sin body pero postJson siempre manda JSON — el backend lo ignora, no pasa nada.

function renderIngestJob(job) {
  let el = document.getElementById(`ingest-job-${job.job_id}`);
  if (!el) {
    el = document.createElement("div");
    el.className = "ingest-job";
    el.id = `ingest-job-${job.job_id}`;
    ingestJobsEl.prepend(el);
  }

  const suggestedCount = (job.items || []).filter((i) => i.status === "suggested").length;
  const reviewCount = (job.items || []).filter((i) => i.status === "needs_review").length;
  const acceptedCount = (job.items || []).filter((i) => i.status === "accepted").length;

  el.innerHTML = `
    <div class="ingest-job-header">
      <span class="ingest-job-title">${job.model_name || job.source_url}</span>
      <span class="ingest-job-status">${JOB_STATUS_LABELS[job.status] || job.status}</span>
    </div>
    <div class="ingest-job-meta">
      ${job.author ? `${job.author} · ` : ""}${job.license || ""} · <a href="${job.source_url}" target="_blank" rel="noopener">ver en Sketchfab ↗</a>
    </div>
    ${job.status === "error" ? `<p class="demo-status is-error" style="margin-top:1rem">${job.error}</p>` : ""}
    ${job.status === "done" ? `
      <div class="ingest-job-actions">
        <span class="ingest-count">${job.items.length} huesos · ${suggestedCount} sugeridos · ${reviewCount} en revisión manual · ${acceptedCount} aceptados</span>
        ${suggestedCount > 0 ? `<button class="btn btn-primary" id="bulk-${job.job_id}">Aceptar todos los sugeridos (${suggestedCount})</button>` : ""}
      </div>
      <div class="ingest-grid" id="grid-${job.job_id}"></div>
    ` : ""}
  `;

  if (job.status === "done") {
    const grid = el.querySelector(`#grid-${job.job_id}`);
    job.items.forEach((item) => grid.appendChild(renderIngestItem(item)));

    const bulkBtn = el.querySelector(`#bulk-${job.job_id}`);
    if (bulkBtn) {
      bulkBtn.addEventListener("click", async () => {
        bulkBtn.disabled = true;
        await postJson(`/ingest/jobs/${job.job_id}/bulk-accept`, {}).catch(() => {});
        const fresh = await fetch(`/ingest/jobs/${job.job_id}`).then((r) => r.json());
        renderIngestJob(fresh);
        refreshTinderQueue();
      });
    }
  }
}

async function pollIngestJob(jobId) {
  const job = await fetch(`/ingest/jobs/${jobId}`).then((r) => r.json());
  renderIngestJob(job);
  if (job.status === "done" || job.status === "error") {
    if (job.status === "done") refreshTinderQueue();
    return;
  }
  setTimeout(() => pollIngestJob(jobId), 3000);
}

ingestSubmitBtn.addEventListener("click", async () => {
  const url = ingestUrlInput.value.trim();
  if (!url) return;
  ingestSubmitBtn.disabled = true;
  ingestShowStatus("Enviando…");
  try {
    const { job_id } = await postJson("/ingest/submit", { url });
    ingestHideStatus();
    ingestUrlInput.value = "";
    pollIngestJob(job_id);
  } catch (err) {
    ingestShowStatus(`Error: ${err.message}`, true);
  } finally {
    ingestSubmitBtn.disabled = false;
  }
});

// al cargar, recupera jobs ya existentes (persisten en disco entre recargas)
fetch("/ingest/jobs")
  .then((r) => r.json())
  .then((jobs) => {
    Object.values(jobs)
      .sort((a, b) => b.submitted_at - a.submitted_at)
      .forEach((job) => {
        renderIngestJob(job);
        if (job.status !== "done" && job.status !== "error") pollIngestJob(job.job_id);
      });
  })
  .catch(() => {});

// ---------- identificación manual (tinder) ----------
const tinderEmpty = document.getElementById("tinderEmpty");
const tinderStack = document.getElementById("tinderStack");
const tinderProgress = document.getElementById("tinderProgress");
const tinderImg = document.getElementById("tinderImg");
const tinderName = document.getElementById("tinderName");
const tinderDims = document.getElementById("tinderDims");
const tinderGroups = document.getElementById("tinderGroups");
const tinderDiscard = document.getElementById("tinderDiscard");

let tinderQueue = [];
let tinderTotal = 0;

function renderTinderCard() {
  if (tinderQueue.length === 0) {
    tinderStack.hidden = true;
    tinderEmpty.hidden = false;
    return;
  }
  tinderEmpty.hidden = true;
  tinderStack.hidden = false;
  const item = tinderQueue[0];
  tinderProgress.textContent = `${tinderTotal - tinderQueue.length + 1} de ${tinderTotal}`;
  tinderImg.src = item.thumbnail_path;
  tinderImg.alt = item.object_name;
  tinderName.textContent = `${item.object_name}${item.model_name ? " — " + item.model_name : ""}`;
  tinderDims.textContent = item.dimensions_m
    ? `${(item.dimensions_m[0] * 100).toFixed(1)} × ${(item.dimensions_m[1] * 100).toFixed(1)} × ${(item.dimensions_m[2] * 100).toFixed(1)} cm`
    : "";
}

async function tinderDecide(accept, boneGroup) {
  const item = tinderQueue[0];
  if (!item) return;
  tinderQueue.shift();
  renderTinderCard();
  try {
    await postJson("/ingest/decide", { item_id: item.item_id, accept, bone_group: boneGroup || null });
  } catch (_) {
    // si falla, se pierde esta decisión pero no rompemos el flujo — reaparecerá
    // en la cola la próxima vez que se recargue (sigue en needs_review en el server).
  }
}

tinderGroups.innerHTML = BONE_GROUPS.map(
  (g) => `<button class="tinder-group-btn" data-group="${g}">${g.replace(/_/g, " ")}</button>`
).join("");
tinderGroups.addEventListener("click", (e) => {
  const btn = e.target.closest(".tinder-group-btn");
  if (btn) tinderDecide(true, btn.dataset.group);
});
tinderDiscard.addEventListener("click", () => tinderDecide(false, null));

async function refreshTinderQueue() {
  const { items } = await fetch("/ingest/review-queue").then((r) => r.json());
  tinderQueue = items;
  tinderTotal = items.length;
  renderTinderCard();
}
refreshTinderQueue().catch(() => {});
