"""Pipeline PARALELO al de siempre (no lo toca ni lo sustituye): fondo NEGRO
en vez del gris neutro de `CROP_BG_GRAY` (128,128,128) en todos los recortes
segmentados, para comparar si un fondo más "cero información" ayuda o
estorba a la CNN.

Idea del usuario (2026-08-03): replicar TODO el dataset (sintético limpio,
ruido clásico, ruido BlenderProc, fotos reales de entrenamiento y examen,
banco sintético limpio) con esta única diferencia. Nombrado con sufijo
`_blackbg` en todas partes para que nunca se mezcle con los ficheros del
pipeline gris — mismas features geométricas (no dependen del color del
fondo, se calculan sobre el contorno/máscara antes de pintarlo), solo
cambia el recorte de imagen que ve la CNN.

Atajo barato: `_crop_masked` usa máscara DURA (sin difuminado en el borde,
ver `extract.py`), así que todo píxel de fondo en un recorte ya generado es
EXACTAMENTE (128,128,128) — no hace falta volver a segmentar nada, basta con
sustituir ese color por (0,0,0) en los PNG ya existentes. Resultado idéntico
a haber corrido `extract_features(..., bg_gray=0)` desde cero, sin pagar el
coste de Grounding DINO+SAM otra vez.

Uso:
    python -m shared.cv_extractor.build_blackbg_variant
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_CROPS_PREFIX = "data/segmented_crops/"
NEW_CROPS_PREFIX = "data/segmented_crops_blackbg/"

TABLES = [
    "bone_geometric_features.csv",
    "bone_geometric_features_noisy.csv",
    "bone_geometric_features_noisy_bproc.csv",
    "bone_geometric_features_real_photos.csv",
    "bone_geometric_features_real_photos_test.csv",
    "bone_geometric_benchmark.csv",
]

OLD_BG = np.array([128, 128, 128], dtype=np.uint8)
NEW_BG = np.array([0, 0, 0], dtype=np.uint8)


def _recolor(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    arr = np.array(img)
    mask = np.all(arr == OLD_BG, axis=-1)
    arr[mask] = NEW_BG
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dst)


def main():
    for table_name in TABLES:
        table_path = REPO_ROOT / "data/processed" / table_name
        if not table_path.exists():
            print(f"AVISO: no existe {table_path}, se salta.")
            continue

        df = pd.read_csv(table_path)
        new_paths = []
        n_missing = 0
        for rel_path in df["segmented_path"]:
            if not isinstance(rel_path, str) or not rel_path.startswith(OLD_CROPS_PREFIX):
                new_paths.append(None)
                n_missing += 1
                continue
            src = REPO_ROOT / rel_path
            new_rel = NEW_CROPS_PREFIX + rel_path[len(OLD_CROPS_PREFIX):]
            dst = REPO_ROOT / new_rel
            if not src.exists():
                new_paths.append(None)
                n_missing += 1
                continue
            _recolor(src, dst)
            new_paths.append(new_rel)

        df["segmented_path"] = new_paths
        before = len(df)
        df = df[df["segmented_path"].notna()].reset_index(drop=True)

        out_name = table_name.replace(".csv", "_blackbg.csv")
        out_path = REPO_ROOT / "data/processed" / out_name
        df.to_csv(out_path, index=False)
        print(f"{table_name}: {before - len(df)} filas sin recorte origen (saltadas), "
              f"{len(df)} filas -> {out_path}")

    print("\nBLACKBG_VARIANT_DONE")


if __name__ == "__main__":
    main()
