"""Construye bone_geometric_features_combined_lab.csv: la misma receta que
bone_geometric_features_combined.csv (renders + noisy_composited +
noisy_bproc + real_photos train) pero sustituyendo el componente `renders/`
(silueta plana, sin sombreado) por `renders_lab/` (escena de laboratorio
nueva) -- para comparar el híbrido entrenado con cada uno sin tocar el
resto de la receta.

Uso:
    python scripts/build_combined_lab_variant.py
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data/processed"

SOURCES = [
    PROCESSED / "bone_geometric_features_lab.csv",
    PROCESSED / "bone_geometric_features_noisy.csv",
    PROCESSED / "bone_geometric_features_noisy_bproc.csv",
    PROCESSED / "bone_geometric_features_real_photos.csv",
]
OUT = PROCESSED / "bone_geometric_features_combined_lab.csv"


def main():
    frames = []
    for path in SOURCES:
        if not path.exists():
            raise SystemExit(f"Falta {path} -- ¿se generó ya la tabla?")
        df = pd.read_csv(path)
        print(f"{path.name}: {len(df)} filas")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT, index=False)
    print(f"\nEscrito: {OUT} -- {len(combined)} filas, {len(combined.columns)} columnas")
    print(combined.groupby("bone_group").size().to_string())


if __name__ == "__main__":
    main()
