"""Recorre los renders de `base_datos_osea/` (subcarpeta de este mismo repo)
y construye una tabla (CSV) con las features geométricas de
`extract.extract_features()` de cada imagen: una fila por vista, columnas =
las métricas numéricas (Hu moments, ratios, perfil de anchura...), más la
clase (hueso canónico) y metadatos de procedencia (especie, espécimen, nº de
vista).

No copia ni modifica las imágenes: `base_datos_osea/` sigue siendo la única
fuente de verdad para las mallas/renders, este script solo lee de ahí.

Uso:
    python -m src.cv_extractor.build_dataset \\
        --renders-dir base_datos_osea/renders \\
        --out data/processed/bone_geometric_features.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from src.cv_extractor.extract import extract_features

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDERS_DIR = REPO_ROOT / "base_datos_osea" / "renders"
DEFAULT_OUT = REPO_ROOT / "data/processed/bone_geometric_features.csv"

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

# Especímenes reservados como banco de evaluación fijo: nunca entran en el
# CSV de entrenamiento, van a un CSV aparte (`--out-benchmark`) que no cambia
# de un entrenamiento a otro — así se puede comparar el mismo examen contra
# distintas versiones del modelo a lo largo del tiempo (logueado en MLflow,
# ver `src/training/evaluate.py`).
#
# Revisión 2026-07-24 (tras la ampliación grande de especímenes vía Sketchfab):
# el criterio original de "reservar el naledi donde lo haya" (ver git history
# para la versión anterior) resultó ser injusto en la práctica — para
# costilla/sacro/vertebra/mandíbula/hueso_pequeno el naledi reservado era el
# ÚNICO espécimen de esa especie para ese hueso, así que el banco medía
# "especie nunca vista" en vez de "espécimen nunca visto" (0-30% recall,
# confirmado con evaluate.py). Con muchos más especímenes sapiens ya
# disponibles por grupo, ahora se reserva un espécimen sapiens normal (banco =
# generalización a espécimen nuevo, la pregunta que sí queríamos responder) y
# el naledi liberado pasa a entrenamiento — añade la única diversidad
# interespecífica real que tenemos para esos huesos. Cráneo sigue reservando
# macaca (única especie no-sapiens con volumen suficiente ahí) y hueso_plano
# sigue con el sapiens de peor calidad, sin cambios. `pelvis` entra ahora al
# banco (antes fuera a propósito con solo 2 especímenes, ahora hay 4).
BENCHMARK_SPECIMENS = {
    ("macaca_radiata", "cranium", "nmnhvz_7049315"),          # cranio
    ("homo_sapiens", "mandible", "sketchfab_acbe657d10164192b85a5737c41064d4"),  # mandibula_maxilar
    ("homo_sapiens", "clavicle", "sketchfab_6e0e3695676e442c90efcdf71e491682"),  # hueso_largo
    ("homo_sapiens", "scapula", "sketchfab_da38f448a06749daa09859d767971088"),  # hueso_plano
    ("homo_sapiens", "sacrum", "sketchfab_eecdb1ffbeca440b87cb3ca9760e7dc8"),  # sacro
    ("homo_sapiens", "vertebra", "sketchfab_f70c894e0d9440b9a77861d278e1c409"),  # vertebra
    ("homo_sapiens", "rib", "sketchfab_cf67ae5ec5df432883667c34f29f7d6e"),  # costilla
    ("homo_sapiens", "metatarsal_1", "sketchfab_ea217e8618ff4ee89b6a684ead3cec1d"),  # hueso_pequeno
    ("homo_sapiens", "pelvis", "sketchfab_10f950afa4284019a1d744ac14b6cf90"),  # pelvis
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


DEFAULT_OUT_BENCHMARK = REPO_ROOT / "data/processed/bone_geometric_benchmark.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--renders-dir", type=Path, default=DEFAULT_RENDERS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-benchmark", type=Path, default=DEFAULT_OUT_BENCHMARK)
    return p.parse_args()


def main():
    args = parse_args()
    renders_dir = args.renders_dir
    if not renders_dir.is_dir():
        raise SystemExit(f"No existe el directorio de renders: {renders_dir}")

    rows = []
    benchmark_rows = []
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

        if (species, bone, specimen) in BENCHMARK_SPECIMENS:
            benchmark_rows.append(row)
        else:
            rows.append(row)

    if skipped_not_found:
        print(f"AVISO: {len(skipped_not_found)} imágenes sin contorno detectado (descartadas):")
        for p in skipped_not_found[:10]:
            print(f"  - {p}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nTabla de entrenamiento escrita en: {args.out}")
    print(f"Filas: {len(df)}  Columnas: {len(df.columns)}")
    print("\nImágenes por hueso:")
    print(df.groupby("bone").size().sort_values(ascending=False).to_string())
    print("\nImágenes y especímenes por grupo morfológico:")
    grp = df.groupby("bone_group").agg(imagenes=("bone", "size"), especimenes=("specimen", "nunique"))
    print(grp.sort_values("especimenes").to_string())

    bench_df = pd.DataFrame(benchmark_rows)
    args.out_benchmark.parent.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(args.out_benchmark, index=False)
    print(f"\nBanco de evaluación fijo escrito en: {args.out_benchmark}")
    print(f"Filas: {len(bench_df)} ({len(BENCHMARK_SPECIMENS)} especímenes × 24 vistas, salvo descartes)")
    if not bench_df.empty:
        print(bench_df.groupby(["bone_group", "species", "bone", "specimen"]).size().to_string())

    missing_benchmark = BENCHMARK_SPECIMENS - {
        (r["species"], r["bone"], r["specimen"]) for r in benchmark_rows
    }
    if missing_benchmark:
        print(f"\nAVISO: {len(missing_benchmark)} espécimen(es) del banco no se encontraron en "
              f"{renders_dir} (¿renombrados o borrados?): {sorted(missing_benchmark)}")


if __name__ == "__main__":
    main()
