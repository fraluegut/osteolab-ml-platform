"""Construye la tabla de entrenamiento a partir de las fotos reales YA
ACEPTADAS (`data/real_photos_accepted/manifest.csv`, revisadas a mano —
ver HISTORIAL.md 2026-07-26). Hasta ahora el entrenamiento era 100%
sintético; esto es lo que lo cambia.

Mismo formato de fila que `build_dataset.py`/`build_dataset_bproc.py`
(especie/hueso/especimen/vista/source_path + features), para poder
concatenar directamente en `bone_geometric_features_combined.csv`. No hay
"especie"/"hueso" fino real aquí (las fotos vienen ya clasificadas por
`bone_group`, no por hueso canónico) — se usa "real" como especie y el
propio `bone_group` como "hueso", y el nombre de fichero (sin extensión)
como "especimen" (cada foto es de un espécimen físico distinto, salvo que
el nombre lo diga explícitamente).

Uso:
    python -m src.cv_extractor.build_dataset_real_photos
"""
from pathlib import Path

import pandas as pd

from src.cv_extractor.build_dataset import flatten_features
from src.cv_extractor.extract import extract_features

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCEPTED_DIR = REPO_ROOT / "data/real_photos_accepted"
MANIFEST_PATH = ACCEPTED_DIR / "manifest.csv"
OUT = REPO_ROOT / "data/processed/bone_geometric_features_real_photos.csv"
CROPS_OUT_DIR = REPO_ROOT / "data/segmented_crops/real_photos"


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"No existe {MANIFEST_PATH}.")

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"{len(manifest)} fotos aceptadas en el manifest")

    rows, skipped = [], []
    for _, m in manifest.iterrows():
        img_path = ACCEPTED_DIR / m["bone_group"] / m["filename"]
        if not img_path.exists():
            skipped.append((m["filename"], "fichero no existe"))
            continue

        with open(img_path, "rb") as fh:
            raw = extract_features(fh, backend="grounded_sam", include_crop=True)

        if not raw.get("found"):
            skipped.append((m["filename"], "sin contorno (found=False)"))
            continue

        specimen = Path(m["filename"]).stem
        crop_out = CROPS_OUT_DIR / m["bone_group"] / f"{specimen}.png"
        crop_out.parent.mkdir(parents=True, exist_ok=True)
        raw["crop_image"].save(crop_out)

        row = {
            "species": "real", "bone": m["bone_group"], "bone_group": m["bone_group"],
            "specimen": specimen, "view": 0,
            "source_path": str(img_path.relative_to(REPO_ROOT)),
            "segmented_path": str(crop_out.relative_to(REPO_ROOT)),
        }
        row.update(flatten_features(raw))
        row.pop("crop_image", None)
        rows.append(row)

    if skipped:
        print(f"AVISO: {len(skipped)} fotos descartadas:")
        for filename, reason in skipped:
            print(f"  - {filename}: {reason}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nTabla escrita en: {OUT}")
    print(f"Filas: {len(df)}  Columnas: {len(df.columns)}")
    if not df.empty:
        print(df.groupby("bone_group").size().to_string())


if __name__ == "__main__":
    main()
