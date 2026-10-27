"""Materializa en disco (hardlinks) las imagenes que usa `stage2_taxon_cranio`
para entrenar/medir, igual que build_stage2_familia_cranio_viewer.py pero por
taxon en vez de familia. Se usa para el apartado "Taxon > Craneo" del
frontend publico (ver /catalog/taxon-cranio-images en main.py).

Uso:
    python scripts/build_stage2_taxon_cranio_viewer.py
"""
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO_ROOT / "data/processed/bone_geometric_features_stage2_taxon_cranio.csv"
TEST_CSV = REPO_ROOT / "data/processed/bone_geometric_features_stage2_taxon_cranio_test.csv"
OUT_DIR = REPO_ROOT / "data/stage2_taxon_cranio_viewer"


def link_rows(df: pd.DataFrame, split: str):
    n = 0
    for _, row in df.iterrows():
        taxon_dir = OUT_DIR / split / row["taxon"]
        taxon_dir.mkdir(parents=True, exist_ok=True)

        origin = "sintetico" if row["species"] != "real" else "real"
        view = row.get("view", 0)
        stem = f"{origin}__{row['species']}__{row['specimen']}__view{view}"

        for suffix, col in [("original", "source_path"), ("segmentado", "segmented_path")]:
            if col not in row or pd.isna(row[col]):
                continue
            src = REPO_ROOT / row[col]
            if not src.exists():
                continue
            dest = taxon_dir / f"{stem}_{suffix}{src.suffix}"
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
    test_df = pd.read_csv(TEST_CSV)

    n_train = link_rows(train_df, "train")
    n_test = link_rows(test_df, "test")

    print(f"train: {n_train} archivos ({train_df['taxon'].value_counts().to_dict()})")
    print(f"test: {n_test} archivos ({test_df['taxon'].value_counts().to_dict()})")
    print(f"\nVisor en: {OUT_DIR}")


if __name__ == "__main__":
    main()
