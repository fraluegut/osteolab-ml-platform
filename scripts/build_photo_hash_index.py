"""Índice hash (sha256) de todas las fotos reales conocidas, con su split
(train/test/selection/held_out), para poder responder "¿esta foto que
acabas de subir ya la usamos en algún set?" sin depender del nombre de
fichero (que puede cambiar) -- sesión 2026-10-27, pedido del usuario al
desplegar el frontend de demo.

Cubre las dos fuentes con manifest de split conocido:
  - data/real_photos_accepted/manifest.csv (pool de entrenamiento, columna
    `split` train/test)
  - data/real_world_benchmark/manifest.csv (banco externo original, columna
    `split` selection/held_out)

Uso:
    python scripts/build_photo_hash_index.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data/processed/photo_hash_index.json"

SOURCES = [
    ("real_photos_accepted", REPO_ROOT / "data/real_photos_accepted"),
    ("real_world_benchmark", REPO_ROOT / "data/real_world_benchmark"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    index = {}
    for dataset_name, base_dir in SOURCES:
        manifest_path = base_dir / "manifest.csv"
        if not manifest_path.exists():
            print(f"AVISO: no existe {manifest_path}, saltado")
            continue
        manifest = pd.read_csv(manifest_path)
        n_ok, n_missing = 0, 0
        for _, row in manifest.iterrows():
            img_path = base_dir / row["bone_group"] / row["filename"]
            if not img_path.exists():
                n_missing += 1
                continue
            digest = sha256_of(img_path)
            index[digest] = {
                "dataset": dataset_name,
                "bone_group": row["bone_group"],
                "filename": row["filename"],
                "split": row.get("split", None),
            }
            n_ok += 1
        print(f"{dataset_name}: {n_ok} fotos indexadas, {n_missing} en manifest pero sin fichero")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"\nÍndice escrito en {OUT}: {len(index)} hashes únicos")


if __name__ == "__main__":
    main()
