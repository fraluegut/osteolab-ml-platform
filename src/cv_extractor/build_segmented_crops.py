"""Precalcula recortes segmentados (bounding box + fondo a gris neutro fuera
de la máscara, ver `segment_and_crop`/`extract_features(..., include_crop=True)`
en `extract.py`) para las 4 tablas de features geométricas ya construidas.

Decisión del 2026-07-26 (ver HISTORIAL.md): la CNN (cnn_only e híbrida)
entrenaba y predecía siempre sobre la imagen completa sin segmentar — solo
el vector de features geométricas pasaba por la máscara. Para que el fondo
deje de influir del todo, la CNN pasa a entrenar sobre estos recortes.

No vuelve a decidir qué imágenes "se encuentran" — reutiliza que cada fila
de estas tablas ya tiene `found=True` con el backend `grounded_sam` (por
construcción, vienen de `build_dataset.py`), así que solo genera el recorte
correspondiente. Si al recomputar el recorte no aparece contorno (raro, sin
aleatoriedad de por medio no debería cambiar respecto a cuando se construyó
la tabla), esa fila se descarta de la tabla igual que ya se hace cuando
`found=False` en `build_dataset.py`.

Añade una columna `segmented_path` (ruta relativa al repo) a cada tabla y
las sobrescribe in place.

Uso:
    python -m src.cv_extractor.build_segmented_crops
"""
from pathlib import Path

import pandas as pd

from src.cv_extractor.extract import extract_features

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data/segmented_crops"

TABLES = [
    "data/processed/bone_geometric_features.csv",
    "data/processed/bone_geometric_features_noisy.csv",
    "data/processed/bone_geometric_benchmark.csv",
    "data/processed/bone_geometric_benchmark_noisy_excluded.csv",
]


def resolve_source_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for rel_table in TABLES:
        table_path = REPO_ROOT / rel_table
        df = pd.read_csv(table_path)
        print(f"\n=== {rel_table}: {len(df)} filas ===")

        segmented_paths = [None] * len(df)
        n_failed = 0
        for i, row in df.iterrows():
            src = resolve_source_path(row["source_path"])
            rel_out = src.relative_to(REPO_ROOT).with_suffix(".png")
            out_path = OUT_DIR / rel_out

            with open(src, "rb") as fh:
                result = extract_features(fh, backend="grounded_sam", include_crop=True)

            if not result.get("found"):
                n_failed += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            result["crop_image"].save(out_path)
            segmented_paths[i] = str(out_path.relative_to(REPO_ROOT))

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(df)}")

        df["segmented_path"] = segmented_paths
        before = len(df)
        df = df[df["segmented_path"].notna()].reset_index(drop=True)
        df.to_csv(table_path, index=False)
        print(f"  {before - len(df)} filas descartadas (sin contorno al recortar), "
              f"{n_failed} fallos. Tabla final: {len(df)} filas -> {table_path}")

    print("\nSEGMENTED_CROPS_DONE")


if __name__ == "__main__":
    main()
