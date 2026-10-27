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
