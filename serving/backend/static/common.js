document.getElementById("year").textContent = new Date().getFullYear();

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
// No todas las páginas tienen el div #lightbox (solo las que muestran
// miniaturas con data-zoomable) — todo este bloque se salta si no existe.
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxClose = document.getElementById("lightboxClose");

if (lightbox) {
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
}
