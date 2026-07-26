"""Igual que `build_dataset.py` pero para renders de BlenderProc
(`base_datos_osea/renders_noisy_bproc/`, ver `render_bone_bproc.py`): cada
vista trae su propia máscara ground-truth (`<...>_mask.png`) exportada
directamente desde la geometría 3D, así que se usa
`extract.extract_features_from_mask()` en vez de adivinar el contorno con
un backend de segmentación sobre los píxeles ya renderizados.

Uso:
    python -m src.cv_extractor.build_dataset_bproc \\
        --renders-dir base_datos_osea/renders_noisy_bproc \\
        --out data/processed/bone_geometric_features_noisy_bproc.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from src.cv_extractor.build_dataset import BENCHMARK_SPECIMENS, BONE_GROUPS, DROP_KEYS, NESTED_PREFIXES, flatten_features
from src.cv_extractor.extract import extract_features_from_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDERS_DIR = REPO_ROOT / "base_datos_osea" / "renders_noisy_bproc"
DEFAULT_OUT = REPO_ROOT / "data/processed/bone_geometric_features_noisy_bproc.csv"
DEFAULT_OUT_BENCHMARK = REPO_ROOT / "data/processed/bone_geometric_benchmark_noisy_bproc_excluded.csv"
VIEW_RE = re.compile(r"_view(\d+)\.\w+$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--renders-dir", type=Path, default=DEFAULT_RENDERS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out-benchmark", type=Path, default=DEFAULT_OUT_BENCHMARK)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.renders_dir.is_dir():
        raise SystemExit(f"No existe el directorio de renders: {args.renders_dir}")

    image_paths = sorted(p for p in args.renders_dir.glob("*/*/*/*.png") if "_mask" not in p.stem)
    print(f"{len(image_paths)} imágenes encontradas bajo {args.renders_dir}")

    rows, benchmark_rows, skipped = [], [], []
    for img_path in image_paths:
        mask_path = img_path.with_name(img_path.stem + "_mask.png")
        if not mask_path.exists():
            skipped.append(str(img_path))
            continue
        species, bone, specimen = img_path.relative_to(args.renders_dir).parts[:3]
        view_match = VIEW_RE.search(img_path.name)
        view = int(view_match.group(1)) if view_match else None

        with open(img_path, "rb") as f_img, open(mask_path, "rb") as f_mask:
            raw = extract_features_from_mask(f_img, f_mask)

        if not raw.get("found"):
            skipped.append(str(img_path))
            continue

        row = {
            "species": species, "bone": bone, "bone_group": BONE_GROUPS.get(bone, "sin_grupo"),
            "specimen": specimen, "view": view, "source_path": str(img_path),
        }
        row.update(flatten_features(raw))
        # Mismo criterio que build_dataset.py: especímenes reservados como banco de
        # evaluación fijo nunca entran en la tabla de entrenamiento — bug encontrado
        # el 2026-07-26, esta variante para BlenderProc no lo comprobaba y los 9
        # especímenes del banco se colaban en el CSV de entrenamiento.
        if (species, bone, specimen) in BENCHMARK_SPECIMENS:
            benchmark_rows.append(row)
        else:
            rows.append(row)

    if skipped:
        print(f"AVISO: {len(skipped)} imágenes sin máscara o sin contorno (descartadas)")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nTabla escrita en: {args.out}")
    print(f"Filas: {len(df)}  Columnas: {len(df.columns)}")
    if not df.empty:
        print(df.groupby("bone_group").agg(imagenes=("bone", "size"), especimenes=("specimen", "nunique")).to_string())

    bench_df = pd.DataFrame(benchmark_rows)
    args.out_benchmark.parent.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(args.out_benchmark, index=False)
    print(f"\nBanco de evaluación (especímenes reservados) escrito en: {args.out_benchmark}")
    print(f"Filas: {len(bench_df)}")


if __name__ == "__main__":
    main()
