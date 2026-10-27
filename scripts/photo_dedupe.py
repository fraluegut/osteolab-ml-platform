"""Perceptual-hash (phash) para detectar fotos reales repetidas entre
`data/real_photos_review/` y `data/real_photos_accepted/` -- mismo stock photo
bajado dos veces con otro id de Openverse, o la foto aceptada colándose otra
vez en un split distinto (train Y test = fuga, invalida cualquier métrica).

phash (no un hash criptográfico) porque sobrevive a recompresión/resize/recorte
ligero -- justo el caso real: Openverse sirve la misma foto de stock en varias
resoluciones con distinto `id`. Usa `imagehash` (ya en requirements, lo usa
ninguna otra parte del repo todavía).

Uso como librería:
    from scripts.photo_dedupe import compute_phash, find_near_duplicates
Uso como script (auditoría de lo ya aceptado, busca fuga train/test):
    python scripts/photo_dedupe.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import imagehash
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
ACCEPTED_DIR = BASE_DIR / "data/real_photos_accepted"
ACCEPTED_MANIFEST = ACCEPTED_DIR / "manifest.csv"
REVIEW_DIR = BASE_DIR / "data/real_photos_review"
REVIEW_MANIFEST = REVIEW_DIR / "manifest.csv"

# Hamming distance máxima entre dos phash de 64 bits para considerarlas la
# misma foto. 8 es el umbral habitual recomendado por la librería imagehash
# para "near duplicate" tolerando recompresión/resize -- por debajo de esto
# empiezan a colarse fotos distintas con composición muy parecida (dos
# cráneos en fondo negro, por ejemplo), que SÍ queremos conservar.
DEFAULT_THRESHOLD = 8


def compute_phash(path: Path) -> imagehash.ImageHash | None:
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"))
    except Exception as exc:
        print(f"  AVISO: no se pudo hashear {path}: {exc}")
        return None


def _load_manifest_hashes(manifest_path: Path, root_dir: Path) -> list[dict]:
    """Devuelve una fila por foto con su phash, o la omite si no se pudo leer
    el archivo (p.ej. manifest desincronizado con disco)."""
    if not manifest_path.exists():
        return []
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            img_path = root_dir / row["bone_group"] / row["filename"]
            if not img_path.exists():
                continue
            h = compute_phash(img_path)
            if h is None:
                continue
            rows.append({**row, "path": img_path, "phash": h})
    return rows


def load_accepted_hashes() -> list[dict]:
    return _load_manifest_hashes(ACCEPTED_MANIFEST, ACCEPTED_DIR)


def load_review_pending_hashes() -> list[dict]:
    """Solo las que siguen `pending` -- las ya decididas (accepted/rejected/
    flagged) no hace falta re-chequear en cada descarga nueva."""
    rows = _load_manifest_hashes(REVIEW_MANIFEST, REVIEW_DIR)
    return [r for r in rows if r.get("status") == "pending"]


def find_near_duplicates(rows: list[dict], threshold: int = DEFAULT_THRESHOLD) -> list[list[dict]]:
    """O(n²) comparando hamming distance -- a los pocos cientos/miles de fotos
    reales que maneja este proyecto es instantáneo, no hace falta un índice
    aproximado (LSH/FAISS). Devuelve clusters (listas) de 2+ filas que se
    consideran la misma foto."""
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if rows[i]["phash"] - rows[j]["phash"] <= threshold:
                union(i, j)

    clusters: dict[int, list[dict]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(rows[i])
    return [c for c in clusters.values() if len(c) > 1]


def is_duplicate_of_known(candidate_path: Path, known_hashes: list[imagehash.ImageHash],
                           threshold: int = DEFAULT_THRESHOLD) -> bool:
    h = compute_phash(candidate_path)
    if h is None:
        return False
    return any(h - known <= threshold for known in known_hashes)


def _audit():
    accepted = load_accepted_hashes()
    print(f"{len(accepted)} fotos aceptadas hasheadas")
    clusters = find_near_duplicates(accepted)
    if not clusters:
        print("Sin duplicados cerca entre las fotos aceptadas.")
    for cluster in clusters:
        splits = {r["split"] for r in cluster}
        bone_groups = {r["bone_group"] for r in cluster}
        tag = "FUGA TRAIN/TEST" if len(splits) > 1 else "duplicado (mismo split)"
        print(f"\n[{tag}] {len(cluster)} fotos casi idénticas:")
        for r in cluster:
            print(f"  {r['bone_group']}/{r['filename']}  split={r['split']}")


if __name__ == "__main__":
    _audit()
