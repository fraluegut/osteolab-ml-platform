"""Extrae una paleta de color a partir de las fotos reales de hueso ya
segmentadas (data/segmented_crops/real_photos/), para usarla de forma
procedural y determinista en el material de los renders sintéticos — en vez
de los 8 colores fijos a mano que había en BONE_COLOR_PALETTE
(render_bone_bproc.py).

Puro Python + numpy + scikit-learn (KMeans con random_state fijo). Nada de
IA generativa ni de red: dado el mismo banco de fotos, siempre da la misma
paleta — repetible y versionable junto al resto del repo.

Las fotos de `data/segmented_crops/real_photos/` ya tienen el fondo pintado
de gris neutro exacto (128,128,128 en los 3 canales, ver
shared/cv_extractor/extract.py::CROP_BG_GRAY) fuera de la máscara del hueso
— se descartan esos píxeles por coincidencia exacta antes de clusterizar,
así el fondo no contamina la paleta.

Uso:
    python3 base_datos_osea/blender/extract_bone_palette.py
    # -> base_datos_osea/blender/bone_palette.json

Salida (bone_palette.json):
    {
      "n_photos": int,
      "n_pixels": int,
      "clusters": [{"rgb": [r,g,b] en 0-1, "weight": float}, ...]  # desc por peso
      "luminance_std": float,  # dispersión de luminosidad real -> amplitud de
                                 # la variación macro procedural
      "saturation_std": float,
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PHOTOS_DIR = REPO_ROOT / "data" / "segmented_crops" / "real_photos"
OUT_PATH = Path(__file__).resolve().parent / "bone_palette.json"

BG_GRAY = 128  # mismo valor que shared/cv_extractor/extract.py::CROP_BG_GRAY
N_CLUSTERS = 8
RANDOM_STATE = 42


def load_bone_pixels() -> np.ndarray:
    """RGB (0-1) de todos los píxeles de hueso (sin fondo) de todas las
    fotos reales segmentadas, como un solo array (N, 3)."""
    paths = sorted(REAL_PHOTOS_DIR.rglob("*.png")) + sorted(REAL_PHOTOS_DIR.rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No hay fotos reales segmentadas en {REAL_PHOTOS_DIR}")

    all_pixels = []
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        flat = im.reshape(-1, 3)
        # Fondo = gris neutro exacto en los 3 canales (relleno sin ruido de
        # compresión, PNG sin pérdida), se descarta por coincidencia exacta.
        is_bg = np.all(flat == BG_GRAY, axis=1)
        bone_pixels = flat[~is_bg]
        if bone_pixels.size:
            all_pixels.append(bone_pixels)

    pixels = np.concatenate(all_pixels, axis=0).astype(np.float64) / 255.0
    return pixels, len(paths)


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """HSV vectorizado (evita traer colorsys por 1 solo uso) — misma fórmula
    estándar, solo para medir dispersión de saturación."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    v = maxc
    delta = maxc - minc
    s = np.divide(delta, maxc, out=np.zeros_like(maxc), where=maxc != 0)
    return np.stack([s, v], axis=1)  # solo saturación y valor, que es lo que usamos


def main():
    pixels, n_photos = load_bone_pixels()
    print(f"{n_photos} fotos reales, {len(pixels)} píxeles de hueso tras descartar fondo")

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(pixels)
    counts = np.bincount(labels, minlength=N_CLUSTERS)
    weights = counts / counts.sum()

    order = np.argsort(-weights)
    clusters = [
        {"rgb": [round(c, 4) for c in km.cluster_centers_[i].tolist()], "weight": round(float(weights[i]), 4)}
        for i in order
    ]

    sv = rgb_to_hsv_np(pixels)
    luminance = pixels.mean(axis=1)  # proxy simple de luminosidad perceptual no hace falta aquí

    out = {
        "n_photos": n_photos,
        "n_pixels": int(len(pixels)),
        "clusters": clusters,
        "luminance_std": round(float(luminance.std()), 4),
        "saturation_std": round(float(sv[:, 0].std()), 4),
    }

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Paleta escrita en {OUT_PATH}")
    print(f"luminance_std={out['luminance_std']}  saturation_std={out['saturation_std']}")
    for c in clusters:
        print(f"  rgb={c['rgb']}  peso={c['weight']}")


if __name__ == "__main__":
    main()
