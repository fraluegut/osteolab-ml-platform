"""Materializa en disco (hardlinks, no copias) exactamente las imágenes que usa
`stage2_familia_cranio` para entrenar/medir -- tanto el original como el
recorte segmentado que de verdad ve la CNN (`segmented_path`) -- para poder
inspeccionarlas en `scripts/explorador_datos.py` sin tener que leer los CSV a
mano. Petición del usuario (2026-10-26): "ver qué imágenes está usando" este
experto en concreto.

Reconstruye la carpeta desde cero cada vez que se corre (no acumula enlaces
viejos si el dataset cambia) -- barato, son hardlinks (mismo inodo, 0 bytes
extra), no copias. Hardlink en vez de symlink a propósito: `explorador_datos.py`
resuelve symlinks a su ruta real para comprobar que no se sale de la raíz
permitida (`resolve()`/`is_relative_to`), y esa ruta real cae FUERA de esta
carpeta -- 403 silencioso, miniatura en negro. Un hardlink es una entrada de
directorio más para el mismo archivo, sin indirección que resolver, así que
pasa ese chequeo sin tocarlo.

Uso:
    python scripts/build_stage2_familia_cranio_viewer.py
"""
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO_ROOT / "data/processed/bone_geometric_features_stage2_familia_cranio.csv"
TEST_CSV = REPO_ROOT / "data/processed/bone_geometric_features_stage2_familia_cranio_test.csv"
OUT_DIR = REPO_ROOT / "data/stage2_familia_cranio_viewer"


def link_rows(df: pd.DataFrame, split: str):
    n = 0
    for _, row in df.iterrows():
        familia_dir = OUT_DIR / split / row["familia"]
        familia_dir.mkdir(parents=True, exist_ok=True)

        origin = "sintetico" if row["species"] != "real" else "real"
        view = row.get("view", 0)
        stem = f"{origin}__{row['species']}__{row['specimen']}__view{view}"

        for suffix, col in [("original", "source_path"), ("segmentado", "segmented_path")]:
            src = REPO_ROOT / row[col]
            if not src.exists():
                continue
            dest = familia_dir / f"{stem}_{suffix}{src.suffix}"
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            try:
                dest.hardlink_to(src.resolve())
            except OSError:
                shutil.copy(src.resolve(), dest)  # distinto filesystem -- fallback a copia
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

    print(f"train: {n_train} symlinks ({train_df['familia'].value_counts().to_dict()})")
    print(f"test: {n_test} symlinks ({test_df['familia'].value_counts().to_dict()})")
    print(f"\nVisor en: {OUT_DIR}")


if __name__ == "__main__":
    main()
