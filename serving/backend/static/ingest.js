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
      });
    }
  }
}

async function pollIngestJob(jobId) {
  const job = await fetch(`/ingest/jobs/${jobId}`).then((r) => r.json());
  renderIngestJob(job);
  if (job.status === "done" || job.status === "error") {
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
