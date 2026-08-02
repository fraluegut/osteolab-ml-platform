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
}

function resetDemo() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.hidden = true;
  dropzoneEmpty.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  resultsGrid.hidden = true;
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

async function classify(endpoint, file) {
  const formData = new FormData();
  formData.append("file", file);
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

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  resultsGrid.hidden = true;
  showStatus("Analizando imagen con ambos modelos…");

  const [geometricResult, hybridResult] = await Promise.allSettled([
    classify("/classify", selectedFile),
    classify("/hybrid/classify", selectedFile),
  ]);

  hideStatus();
  resultsGrid.hidden = false;
  resetBtn.hidden = false;
  analyzeBtn.disabled = false;

  if (geometricResult.status === "fulfilled") {
    renderResult(cardGeometric, geometricResult.value);
  } else {
    renderError(cardGeometric, `Error al clasificar: ${geometricResult.reason.message}`);
  }

  if (hybridResult.status === "fulfilled") {
    renderResult(cardHybrid, hybridResult.value);
  } else {
    renderError(cardHybrid, `Error al clasificar: ${hybridResult.reason.message}`);
  }
});
