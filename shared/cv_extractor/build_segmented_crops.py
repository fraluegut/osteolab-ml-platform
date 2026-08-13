"""Precalcula recortes segmentados (bounding box + fondo a gris neutro fuera
de la máscara, ver `segment_and_crop`/`extract_features(..., include_crop=True)`
en `extract.py`) para las tablas de features geométricas ya construidas.

Decisión del 2026-07-26 (ver HISTORIAL.md): la CNN (cnn_only e híbrida)
entrenaba y predecía siempre sobre la imagen completa sin segmentar — solo
el vector de features geométricas pasaba por la máscara. Para que el fondo
deje de influir del todo, la CNN pasa a entrenar sobre estos recortes.

Dos métodos según la tabla (ver `TABLES`): `"grounded_sam"` para renders sin
máscara ground-truth (hay que adivinar el contorno, igual que en
producción); `"mask"` para renders de BlenderProc (2026-07-26, migración a
BlenderProc), que traen su propia máscara exacta exportada desde la
geometría 3D — más barato y más fiel que volver a segmentar por CV.

No vuelve a decidir qué imágenes "se encuentran" desde cero — reutiliza que
cada fila de estas tablas ya tiene `found=True` (por construcción, vienen de
`build_dataset.py`/`build_dataset_bproc.py`), así que solo genera el
recorte correspondiente. Si al recomputar el recorte no aparece contorno
(raro, sin aleatoriedad de por medio no debería cambiar respecto a cuando se
construyó la tabla), esa fila se descarta igual que ya se hace cuando
`found=False` en el script que generó la tabla.

Añade una columna `segmented_path` (ruta relativa al repo) a cada tabla y
las sobrescribe in place.

Uso:
    python -m shared.cv_extractor.build_segmented_crops
"""
from pathlib import Path

import pandas as pd

from shared.cv_extractor.extract import extract_features, extract_features_from_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data/segmented_crops"

# "grounded_sam": no hay máscara ground-truth (renders limpios de bpy sin ruido,
# banco fijo) -- hay que adivinar el contorno igual que en producción.
# "mask": renders de BlenderProc, cada vista trae su `<...>_mask.png` hermano
# (ver render_bone_bproc.py) -- se usa esa máscara exacta, más barato y más
# fiel que volver a segmentar con grounded_sam sobre el color ya renderizado.
#
# 2026-08-03: `bone_geometric_features_noisy.csv` (ruido clásico compuesto)
# faltaba de esta lista pese a que el CSV commiteado SÍ traía `segmented_path`
# — se había añadido en algún momento anterior sin dejar el cambio en
# `TABLES` (o corrido con una versión modificada del script). Al regenerar
# `bone_geometric_features*.csv` desde cero (nuevas features de forma,
# 2026-08-03) esa tabla se quedó sin la columna y rompió el entrenamiento
# (`cnn_dataset.py` la necesita para cargar el recorte). Reincorporada aquí.
TABLES = [
    ("data/processed/bone_geometric_features.csv", "grounded_sam"),
    ("data/processed/bone_geometric_features_noisy.csv", "grounded_sam"),
    ("data/processed/bone_geometric_features_noisy_bproc.csv", "mask"),
    ("data/processed/bone_geometric_features_lab.csv", "mask"),
    ("data/processed/bone_geometric_benchmark.csv", "grounded_sam"),
    ("data/processed/bone_geometric_benchmark_noisy_bproc_excluded.csv", "mask"),
    ("data/processed/bone_geometric_benchmark_lab_excluded.csv", "mask"),
]


def resolve_source_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)


def extract_crop(src: Path, method: str):
    if method == "mask":
        mask_path = src.with_name(src.stem + "_mask.png")
        with open(src, "rb") as f_img, open(mask_path, "rb") as f_mask:
            return extract_features_from_mask(f_img, f_mask, include_crop=True)
    with open(src, "rb") as fh:
        return extract_features(fh, backend="grounded_sam", include_crop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for rel_table, method in TABLES:
        table_path = REPO_ROOT / rel_table
        df = pd.read_csv(table_path)
        print(f"\n=== {rel_table} ({method}): {len(df)} filas ===")

        segmented_paths = [None] * len(df)
        n_failed = 0
        for i, row in df.iterrows():
            src = resolve_source_path(row["source_path"])
            rel_out = src.relative_to(REPO_ROOT).with_suffix(".png")
            out_path = OUT_DIR / rel_out

            result = extract_crop(src, method)

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
