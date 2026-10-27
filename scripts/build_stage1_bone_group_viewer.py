"""Materializa en disco (hardlinks) las imagenes que usa `hybrid_lab`
(etapa 1, grupo osteologico) para entrenar/medir, organizadas por
split/grupo/origen -- mismo patron que build_stage2_taxon_cranio_viewer.py,
pero para la etapa 1 completa en vez de un experto de etapa 2/3. Petician
del usuario (2026-10-27): "quiero ver arriba del todo los expertos por
fase... Fase 1 es grupos osteologicos y querria ver dentro de ello carpetas
por grupo osteologico y dentro poder filtrar por sintticas o no sinteticas".
El filtro sintetico/real es, aqui, directamente la carpeta: no hace falta
nada en el frontend del explorador.

Uso:
    python scripts/build_stage1_bone_group_viewer.py
"""
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO_ROOT / "data/processed/bone_geometric_features_combined_lab.csv"
TEST_CSV = REPO_ROOT / "data/processed/bone_geometric_features_real_photos_test.csv"
OUT_DIR = REPO_ROOT / "data/stage1_bone_group_viewer"


def link_rows(df: pd.DataFrame, split: str):
    n = 0
    for _, row in df.iterrows():
        origin = "real" if row["species"] == "real" else "sintetico"
        group_dir = OUT_DIR / split / row["bone_group"] / origin
        group_dir.mkdir(parents=True, exist_ok=True)

        view = row.get("view", 0)
        stem = f"{row['species']}__{row['specimen']}__view{view}"

        for suffix, col in [("original", "source_path"), ("segmentado", "segmented_path")]:
            if col not in row or pd.isna(row[col]):
                continue
            src = REPO_ROOT / row[col]
            if not src.exists():
                continue
            dest = group_dir / f"{stem}_{suffix}{src.suffix}"
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            try:
                dest.hardlink_to(src.resolve())
            except OSError:
                shutil.copy(src.resolve(), dest)
            n += 1
    return n


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    train_df = pd.read_csv(TRAIN_CSV)
    n_train = link_rows(train_df, "train")
    print(f"train: {n_train} archivos")
    print(train_df.groupby("bone_group")["species"].apply(lambda s: (s == "real").sum()).rename("fotos_reales").to_string())

    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV)
        n_test = link_rows(test_df, "test")
        print(f"\ntest (solo fotos reales, nunca entrenan): {n_test} archivos")

    print(f"\nVisor en: {OUT_DIR}")


if __name__ == "__main__":
    main()
