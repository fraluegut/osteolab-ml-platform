// ---------- demo: árbol de 3 etapas (página /probar.html) ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const demoStatus = document.getElementById("demoStatus");
const leakBanner = document.getElementById("leakBanner");

const stageTree = document.getElementById("stageTree");
const stageBone = document.getElementById("stageBone");
const stageBoneBadge = document.getElementById("stageBoneBadge");
const stageBoneBody = document.getElementById("stageBoneBody");
const stageGroup = document.getElementById("stageGroup");
const stageGroupBadge = document.getElementById("stageGroupBadge");
const stageExpert = document.getElementById("stageExpert");
const stageExpertTitle = document.getElementById("stageExpertTitle");
const stageExpertBadge = document.getElementById("stageExpertBadge");
const stageExpertBody = document.getElementById("stageExpertBody");

const cardGeometric = document.querySelector("#cardGeometric .result-body");
const cardHybrid = document.querySelector("#cardHybrid .result-body");
const cardQwen = document.querySelector("#cardQwen .result-body");

function setBadge(el, text, variant) {
  el.textContent = text;
  el.className = "stage-badge" + (variant ? ` is-${variant}` : "");
}

function resetStages() {
  stageTree.hidden = true;
  stageGroup.hidden = true;
  stageExpert.hidden = true;
  setBadge(stageBoneBadge, "esperando…");
  setBadge(stageGroupBadge, "esperando…");
  setBadge(stageExpertBadge, "esperando…");
  stageBoneBody.innerHTML = `<p class="stage-card-note">Filtro CLIP zero-shot — si dice que no, el resto de la cascada ni se llama.</p>`;
  cardGeometric.innerHTML = "";
  cardHybrid.innerHTML = "";
  cardQwen.innerHTML = "";
  stageExpertBody.innerHTML = "";
  leakBanner.hidden = true;
  leakBanner.className = "leak-banner";
}

const SPLIT_LABELS = {
  train: "entrenamiento",
  test: "examen (test reservado)",
  selection: "selección del banco externo",
  held_out: "examen del banco externo (held-out)",
};

function renderLeakBanner(result) {
  leakBanner.hidden = false;
  if (!result || !result.seen_before) {
    leakBanner.className = "leak-banner is-unseen";
    leakBanner.textContent = "Esta foto no coincide (por hash) con ninguna foto de nuestros sets conocidos — parece nueva de verdad.";
    return;
  }
  leakBanner.className = "leak-banner is-seen";
  const splitLabel = SPLIT_LABELS[result.split] || result.split || "split desconocido";
  leakBanner.textContent = `⚠️ Esta foto YA la conocemos: está en "${result.dataset}" (${result.bone_group}, split "${splitLabel}"). El resultado de abajo no es una prueba ciega.`;
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
  resetStages();
}

function resetDemo() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.hidden = true;
  dropzoneEmpty.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  resetStages();
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

function topEntry(probabilities) {
  return Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0];
}

function renderProbBars(container, probabilities, limit = 5) {
  Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .forEach(([label, value]) => {
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

function renderResult(container, result) {
  container.innerHTML = "";

  if (!result.found) {
    const p = document.createElement("p");
    p.className = "result-empty";
    p.textContent = "No se pudo localizar un hueso claro en la imagen.";
    container.appendChild(p);
    return;
  }

  const [topLabel, topValue] = topEntry(result.probabilities);
  const top = document.createElement("div");
  top.className = "result-top";
  top.innerHTML = `
    <span class="result-top-label">${topLabel.replace(/_/g, " ")}</span>
    <span class="result-top-value">${(topValue * 100).toFixed(1)}%</span>
  `;
  container.appendChild(top);
  renderProbBars(container, result.probabilities);
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

// Etapa 3: taxón (cráneo/mandíbula) o hueso concreto (hueso_largo/pequeño/vértebra) cuando el
// grupo tiene experto entrenado; "pendiente" en el resto (costilla/hueso_plano/pelvis/sacro).
function renderExpertStage(hybridResult) {
  stageExpert.hidden = false;
  if (!hybridResult || !hybridResult.found) {
    setBadge(stageExpertBadge, "n/a", "pending");
    stageExpertTitle.textContent = "Afinando dentro del grupo";
    stageExpertBody.innerHTML = `<p class="expert-empty">No se pudo determinar el grupo en la etapa 2, así que no hay nada que afinar.</p>`;
    return;
  }

  const group = hybridResult.prediction.replace(/_/g, " ");

  if (hybridResult.taxon) {
    stageExpertTitle.textContent = `Taxón dentro de "${group}"`;
    const [label, value] = topEntry(hybridResult.taxon.probabilities);
    setBadge(stageExpertBadge, `${label.replace(/_/g, " ")} (${(value * 100).toFixed(1)}%)`, "good");
    stageExpertBody.innerHTML = "";
    renderProbBars(stageExpertBody, hybridResult.taxon.probabilities, 5);
    return;
  }

  if (hybridResult.fine_bone) {
    stageExpertTitle.textContent = `Hueso concreto dentro de "${group}"`;
    const [label, value] = topEntry(hybridResult.fine_bone.probabilities);
    setBadge(stageExpertBadge, `${label.replace(/_/g, " ")} (${(value * 100).toFixed(1)}%)`, "good");
    stageExpertBody.innerHTML = "";
    renderProbBars(stageExpertBody, hybridResult.fine_bone.probabilities, 5);
    return;
  }

  stageExpertTitle.textContent = `Afinando dentro de "${group}"`;
  setBadge(stageExpertBadge, "no disponible", "pending");
  stageExpertBody.innerHTML = `<p class="expert-empty">Todavía no hay un experto de etapa 3 entrenado para el grupo "${group}". El grupo en sí (etapa 2) ya es el resultado final para esta foto.</p>`;
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const file = selectedFile;
  analyzeBtn.disabled = true;
  resetStages();
  stageTree.hidden = false;
  showStatus("Etapa 1/3 — filtro CLIP: ¿es un hueso?");

  setBadge(stageBoneBadge, "analizando…", "pending");
  let clipResult;
  try {
    clipResult = await classify("/filter/clip", file);
  } catch (err) {
    setBadge(stageBoneBadge, "error", "bad");
    showStatus(`Error en el filtro CLIP: ${err.message}`, true);
    analyzeBtn.disabled = false;
    return;
  }

  const clipPct = (clipResult.confidence * 100).toFixed(1);
  if (!clipResult.is_bone) {
    setBadge(stageBoneBadge, `no (${clipPct}%)`, "bad");
    stageBoneBody.innerHTML = `<p class="stage-card-note">CLIP no detecta un hueso en la imagen — el flujo se detiene aquí, no se llama al resto de modelos.</p>`;
    showStatus("", false);
    hideStatus();
    analyzeBtn.disabled = false;
    resetBtn.hidden = false;
    return;
  }
  const suggestion = clipResult.suggestion ? ` — sugiere ${clipResult.suggestion.type.replace(/_/g, " ")}` : "";
  setBadge(stageBoneBadge, `sí (${clipPct}%)`, "good");
  stageBoneBody.innerHTML = `<p class="stage-card-note">Es un hueso${suggestion}. Pasa a la etapa 2.</p>`;
  showStatus("Etapa 2/3 — clasificando grupo osteológico (3 modelos en paralelo)…");

  stageGroup.hidden = false;
  setBadge(stageGroupBadge, "clasificando…", "pending");
  resetBtn.hidden = false;

  const [geometricResult, hybridResult, qwenResult, gradcamResult, leakResult] = await Promise.allSettled([
    classify("/classify", file),
    classify("/hybrid/classify", file),
    classify("/describe/qwen", file),
    classify("/explain/gradcam", file, { model_type: "hybrid" }),
    classify("/leak-check", file),
  ]);

  hideStatus();
  analyzeBtn.disabled = false;

  renderLeakBanner(leakResult.status === "fulfilled" ? leakResult.value : null);

  if (geometricResult.status === "fulfilled") {
    renderResult(cardGeometric, geometricResult.value);
    appendImageBlock(
      cardGeometric,
      "Máscara de segmentación — zona que midió OpenCV",
      geometricResult.value.cv_features?.mask_overlay_base64,
      "Máscara de segmentación detectada por OpenCV"
    );
  } else {
    renderError(cardGeometric, `Error al clasificar: ${geometricResult.reason.message}`);
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

  if (qwenResult.status === "fulfilled") {
    renderQwenResult(cardQwen, qwenResult.value);
  } else {
    renderError(cardQwen, `Error al describir: ${qwenResult.reason.message}`);
  }

  // El híbrido es el modelo "oficial" que decide a qué grupo pasa la etapa 3 (ver
  // shared/inference/predict_two_stage.py) — el badge de la etapa 2 refleja su veredicto,
  // no el de los otros dos (que son comparación, no el camino que sigue la cascada real).
  if (hybridResult.status === "fulfilled" && hybridResult.value.found) {
    const [label, value] = topEntry(hybridResult.value.probabilities);
    setBadge(stageGroupBadge, `${label.replace(/_/g, " ")} (${(value * 100).toFixed(1)}%)`, "good");
  } else {
    setBadge(stageGroupBadge, "sin grupo claro", "bad");
  }

  renderExpertStage(hybridResult.status === "fulfilled" ? hybridResult.value : null);
});
