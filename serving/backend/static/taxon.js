const taxonGalleryStatus = document.getElementById("taxonGalleryStatus");
const taxonGalleryEl = document.getElementById("taxonGallery");

const SPLIT_LABELS = { train: "Entrenamiento", test: "Test real (nunca entrenado)" };
const MAX_THUMBS = 10;

function renderTaxonGallery(groups) {
  const keys = Object.keys(groups).sort((a, b) => {
    const [splitA, taxonA] = a.split("/");
    const [splitB, taxonB] = b.split("/");
    if (splitA !== splitB) return splitA === "train" ? -1 : 1;
    return taxonA.localeCompare(taxonB);
  });

  taxonGalleryEl.innerHTML = keys.map((key) => {
    const g = groups[key];
    const shown = g.files.slice(0, MAX_THUMBS);
    const thumbs = shown.map((filename) => {
      const src = `/taxon-cranio-images/${g.split}/${g.taxon}/${encodeURIComponent(filename)}`;
      const isReal = filename.startsWith("real__");
      return `<img src="${src}" alt="${g.taxon}" class="taxon-thumb${isReal ? " is-real" : ""}" data-zoomable loading="lazy" />`;
    }).join("");
    return `
      <div class="taxon-gallery-card">
        <div class="taxon-gallery-header">
          <h4>${g.taxon.replace(/_/g, " ")}</h4>
          <span class="tag">${SPLIT_LABELS[g.split] || g.split}</span>
          <span class="taxon-gallery-count">${g.count} fotos</span>
        </div>
        <div class="taxon-gallery-thumbs">${thumbs}</div>
        ${g.count > MAX_THUMBS ? `<p class="table-note">mostrando ${MAX_THUMBS} de ${g.count}</p>` : ""}
      </div>
    `;
  }).join("");
}

fetch("/catalog/taxon-cranio-images")
  .then((r) => r.json())
  .then(({ groups }) => {
    taxonGalleryStatus.hidden = true;
    renderTaxonGallery(groups);
  })
  .catch(() => {
    taxonGalleryStatus.textContent = "No se pudo cargar la galería de imágenes.";
    taxonGalleryStatus.classList.add("is-error");
  });
