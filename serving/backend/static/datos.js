// ---------- catálogo 3D ----------
const catalogFiltersEl = document.getElementById("catalogFilters");
const catalogBodyEl = document.getElementById("catalogTableBody");
let catalogSpecimens = [];
let catalogFilter = "todos";

function renderCatalogTable() {
  const rows = catalogFilter === "todos"
    ? catalogSpecimens
    : catalogSpecimens.filter((s) => s.bone_group === catalogFilter);

  catalogBodyEl.innerHTML = rows.map((s) => {
    const thumb = s.render_path
      ? `<img src="/catalog-renders/${s.render_path}" alt="${s.species} ${s.bone_canonical}" data-zoomable />`
      : `<span class="no-thumb">s/render</span>`;
    return `
      <tr>
        <td>${thumb}</td>
        <td>${s.bone_group.replace(/_/g, " ")}</td>
        <td><em>${s.species}</em></td>
        <td>${s.bone_canonical.replace(/_/g, " ")}</td>
        <td>${s.source}</td>
        <td>${s.license || "—"}</td>
        <td><a href="${s.external_url}" target="_blank" rel="noopener">ver ↗</a></td>
      </tr>
    `;
  }).join("") || `<tr><td colspan="7" class="table-note">Sin resultados para este filtro.</td></tr>`;
}

function renderCatalogFilters() {
  const groups = ["todos", ...new Set(catalogSpecimens.map((s) => s.bone_group))].sort((a, b) =>
    a === "todos" ? -1 : b === "todos" ? 1 : a.localeCompare(b)
  );
  catalogFiltersEl.innerHTML = groups.map((g) => {
    const count = g === "todos" ? catalogSpecimens.length : catalogSpecimens.filter((s) => s.bone_group === g).length;
    return `<button class="catalog-filter-btn${g === catalogFilter ? " is-active" : ""}" data-group="${g}">${g.replace(/_/g, " ")} (${count})</button>`;
  }).join("");
}

catalogFiltersEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".catalog-filter-btn");
  if (!btn) return;
  catalogFilter = btn.dataset.group;
  renderCatalogFilters();
  renderCatalogTable();
});

fetch("/catalog/specimens")
  .then((r) => r.json())
  .then(({ specimens }) => {
    catalogSpecimens = specimens;
    renderCatalogFilters();
    renderCatalogTable();
  })
  .catch(() => {
    catalogBodyEl.innerHTML = `<tr><td colspan="7" class="table-note">No se pudo cargar el catálogo.</td></tr>`;
  });

// ---------- balance de datos (fotos reales / especímenes 3D vs objetivo) ----------
const realPhotosBalanceBodyEl = document.getElementById("realPhotosBalanceBody");
const specimens3dBalanceBodyEl = document.getElementById("specimens3dBalanceBody");

function renderRealPhotosBalance(groups) {
  const rows = [...groups].sort((a, b) => a.real_photos_training - b.real_photos_training);
  realPhotosBalanceBodyEl.innerHTML = rows.map((g) => `
    <tr${g.real_photos_missing > 0 ? ' class="row-warn"' : ""}>
      <td>${g.bone_group.replace(/_/g, " ")}</td>
      <td${g.real_photos_missing > 0 ? ' class="cell-warn"' : ""}>${g.real_photos_training}</td>
      <td>${g.real_photos_benchmark}</td>
      <td>${g.real_photos_target}</td>
      <td class="${g.real_photos_missing > 0 ? "cell-warn" : "cell-best"}">${g.real_photos_missing}</td>
    </tr>
  `).join("");
}

function renderSpecimens3dBalance(groups) {
  const rows = [...groups].sort((a, b) => a.specimens_3d - b.specimens_3d);
  specimens3dBalanceBodyEl.innerHTML = rows.map((g) => `
    <tr${g.specimens_3d_missing > 0 ? ' class="row-warn"' : ""}>
      <td>${g.bone_group.replace(/_/g, " ")}</td>
      <td${g.specimens_3d_missing > 0 ? ' class="cell-warn"' : ""}>${g.specimens_3d}</td>
      <td>${g.species_3d}</td>
      <td>${g.specimens_3d_target}</td>
      <td class="${g.specimens_3d_missing > 0 ? "cell-warn" : "cell-best"}">${g.specimens_3d_missing}</td>
    </tr>
  `).join("");
}

fetch("/catalog/balance")
  .then((r) => r.json())
  .then(({ groups }) => {
    renderRealPhotosBalance(groups);
    renderSpecimens3dBalance(groups);
  })
  .catch(() => {
    const errCell = `<tr><td colspan="5" class="table-note">No se pudo cargar el balance de datos.</td></tr>`;
    realPhotosBalanceBodyEl.innerHTML = errCell;
    specimens3dBalanceBodyEl.innerHTML = errCell;
  });
