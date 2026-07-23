"""Recorre los renders de `base_datos_osea` y construye una tabla (CSV) con
las features geométricas de `extract.extract_features()` de cada imagen: una
fila por vista, columnas = las métricas numéricas (Hu moments, ratios, perfil
de anchura...), más la clase (hueso canónico) y metadatos de procedencia
(especie, espécimen, nº de vista).

No copia ni modifica las imágenes: `base_datos_osea` sigue siendo la única
fuente de verdad para las mallas/renders, este script solo lee de ahí.

Uso:
    python -m src.cv_extractor.build_dataset \\
        --renders-dir /root/dev/base_datos_osea/renders \\
        --out data/processed/bone_geometric_features.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from src.cv_extractor.extract import extract_features

DEFAULT_RENDERS_DIR = Path("/root/dev/base_datos_osea/renders")
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data/processed/bone_geometric_features.csv"

# Grupo morfológico general de cada hueso canónico — debe mantenerse igual
# que `BONE_GROUPS` en base_datos_osea/scripts/bones.py (fuente de verdad de
# la taxonomía; se duplica aquí en vez de importar entre repos/venvs
# distintos). Ver ese fichero para el razonamiento de cada caso dudoso.
BONE_GROUPS = {
    "cranium": "cranio",
    "mandible": "mandibula_maxilar",
    "maxilla": "mandibula_maxilar",
    "scapula": "hueso_plano",
    "clavicle": "hueso_largo",
    "humerus": "hueso_largo",
    "radius": "hueso_largo",
    "ulna": "hueso_largo",
    "femur": "hueso_largo",
    "tibia": "hueso_largo",
    "fibula": "hueso_largo",
    "patella": "hueso_pequeno",
    "pelvis": "pelvis",
    "sacrum": "sacro",
    "vertebra": "vertebra",
    "rib": "costilla",
    "metacarpal_1": "hueso_pequeno",
    "metatarsal_1": "hueso_pequeno",
    "proximal_phalanx_1": "hueso_pequeno",
    "trapezium": "hueso_pequeno",
    "sesamoid": "hueso_pequeno",
}

VIEW_RE = re.compile(r"_view(\d+)\.png$")

# Claves que van dentro de un dict anidado en el resultado de extract_features
# y se aplanan a columnas con este prefijo (p.ej. bbox_x, bbox_y...).
NESTED_PREFIXES = {
    "image_size": "img",
    "bounding_box_px": "bbox",
    "rotated_box_px": "rbox",
}
# Claves que se descartan de la tabla: no son features numéricas de
# entrenamiento, son solo para mostrar la imagen anotada en la UI/API.
DROP_KEYS = {"annotated_image_base64", "mask_overlay_base64"}


def flatten_features(raw: dict) -> dict:
    flat = {}
    for key, value in raw.items():
        if key in DROP_KEYS:
            continue
        if key == "hu_moments":
            flat.update(value)
        elif key == "width_profile":
            flat.update({f"wp{i:02d}": v for i, v in enumerate(value)})
        elif key in NESTED_PREFIXES:
            prefix = NESTED_PREFIXES[key]
            flat.update({f"{prefix}_{k}": v for k, v in value.items()})
        else:
            flat[key] = value
    return flat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--renders-dir", type=Path, default=DEFAULT_RENDERS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    renders_dir = args.renders_dir
    if not renders_dir.is_dir():
        raise SystemExit(f"No existe el directorio de renders: {renders_dir}")

    rows = []
    skipped_not_found = []
    image_paths = sorted(renders_dir.glob("*/*/*/*.png"))
    print(f"{len(image_paths)} imágenes encontradas bajo {renders_dir}")

    for img_path in image_paths:
        species, bone, specimen = img_path.relative_to(renders_dir).parts[:3]
        view_match = VIEW_RE.search(img_path.name)
        view = int(view_match.group(1)) if view_match else None

        with img_path.open("rb") as f:
            raw = extract_features(f)

        if not raw.get("found"):
            skipped_not_found.append(str(img_path))
            continue

        row = {
            "species": species,
            "bone": bone,
            "bone_group": BONE_GROUPS.get(bone, "sin_grupo"),
            "specimen": specimen,
            "view": view,
            "source_path": str(img_path),
        }
        row.update(flatten_features(raw))
        rows.append(row)

    if skipped_not_found:
        print(f"AVISO: {len(skipped_not_found)} imágenes sin contorno detectado (descartadas):")
        for p in skipped_not_found[:10]:
            print(f"  - {p}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nTabla escrita en: {args.out}")
    print(f"Filas: {len(df)}  Columnas: {len(df.columns)}")
    print("\nImágenes por hueso:")
    print(df.groupby("bone").size().sort_values(ascending=False).to_string())
    print("\nImágenes y especímenes por grupo morfológico:")
    grp = df.groupby("bone_group").agg(imagenes=("bone", "size"), especimenes=("specimen", "nunique"))
    print(grp.sort_values("especimenes").to_string())


if __name__ == "__main__":
    main()
