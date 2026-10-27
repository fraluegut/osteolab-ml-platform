"""Construye el dataset de entrenamiento/test para el modelo de etapa 3
`stage2_familia_cranio`: dentro de cráneo, afinar a nivel de familia
(cánido/suido/équido/humano) -- un nivel más fino que el taxón amplio ya
existente (`stage2_taxon_cranio`, humano/carnivoro/ungulado/roedor/
otro_primate, ver predict_two_stage.py).

Combina dos fuentes con el mismo criterio que las demás tablas de
entrenamiento del proyecto (ver build_dataset_real_photos.py):
  - Sintético: `data/processed/bone_geometric_features_lab.csv` (renders de
    `render_bone_lab.py`, ya con `segmented_path` tras `build_segmented_crops.py`),
    filtrado a bone_group=cranio y a las 4 especies mapeadas a su familia.
  - Real: `data/real_photos_accepted/manifest.csv`, filtrado a
    bone_group=cranio y con `familia` ya confirmada (columna añadida en la
    sesión 2026-10-26, revisión visual una a una) -- split `train` entra en
    el CSV de entrenamiento, split `test` se queda aparte, NUNCA mezclado,
    para medir de verdad (mismo criterio que build_dataset_real_photos.py).

Uso:
    python scripts/build_stage2_familia_cranio_dataset.py
"""
from pathlib import Path

import pandas as pd

from shared.cv_extractor.build_dataset import flatten_features
from shared.cv_extractor.extract import extract_features

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_FEATURES = REPO_ROOT / "data/processed/bone_geometric_features_lab.csv"
ACCEPTED_DIR = REPO_ROOT / "data/real_photos_accepted"
MANIFEST_PATH = ACCEPTED_DIR / "manifest.csv"
CROPS_OUT_DIR = REPO_ROOT / "data/segmented_crops/real_photos_familia_cranio"

OUT_TRAIN = REPO_ROOT / "data/processed/bone_geometric_features_stage2_familia_cranio.csv"
OUT_TEST = REPO_ROOT / "data/processed/bone_geometric_features_stage2_familia_cranio_test.csv"

# species (carpeta en data/meshes/) -> familia -- las 4 que pidió el usuario.
# Deliberadamente NO incluye felis_catus/bos_taurus/etc: esas no se pidieron
# y meterlas aquí diluiría las 4 clases objetivo (mismo motivo que hueso_fino
# vs taxon ya documentado en predict_two_stage.py: una clase de más interfiere
# con las que sí importan).
SPECIES_TO_FAMILIA = {
    "canis_familiaris": "cánido",
    "sus_domesticus": "suido",
    "equus_caballus": "équido",
    "homo_sapiens": "humano",
}


def load_synthetic() -> pd.DataFrame:
    if not LAB_FEATURES.exists():
        raise SystemExit(f"Falta {LAB_FEATURES} -- correr antes build_dataset_lab.py + build_segmented_crops.py")
    df = pd.read_csv(LAB_FEATURES)
    df = df[(df["bone_group"] == "cranio") & (df["species"].isin(SPECIES_TO_FAMILIA))].copy()
    df["familia"] = df["species"].map(SPECIES_TO_FAMILIA)
    print(f"Sintético: {len(df)} vistas, {df.groupby('familia')['specimen'].nunique().to_dict()} especímenes por familia")
    return df


def load_real(split: str) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[
        (manifest["bone_group"] == "cranio")
        & manifest["familia"].notna()
        & (manifest["familia"] != "")
        & (manifest["split"] == split)
    ]
    print(f"Real ({split}): {len(manifest)} fotos en manifest")

    rows, skipped = [], []
    for _, m in manifest.iterrows():
        img_path = ACCEPTED_DIR / m["bone_group"] / m["filename"]
        if not img_path.exists():
            skipped.append((m["filename"], "fichero no existe"))
            continue

        with open(img_path, "rb") as fh:
            raw = extract_features(fh, backend="grounded_sam", include_crop=True)
        if not raw.get("found"):
            skipped.append((m["filename"], "sin contorno (found=False)"))
            continue

        specimen = Path(m["filename"]).stem
        crop_out = CROPS_OUT_DIR / split / f"{specimen}.png"
        crop_out.parent.mkdir(parents=True, exist_ok=True)
        raw["crop_image"].save(crop_out)

        row = {
            "species": "real", "bone": "cranium", "bone_group": "cranio",
            "specimen": specimen, "view": 0,
            "source_path": str(img_path.relative_to(REPO_ROOT)),
            "segmented_path": str(crop_out.relative_to(REPO_ROOT)),
            "familia": m["familia"],
        }
        row.update(flatten_features(raw))
        row.pop("crop_image", None)
        rows.append(row)

    if skipped:
        print(f"  AVISO: {len(skipped)} descartadas:")
        for filename, reason in skipped:
            print(f"    - {filename}: {reason}")

    return pd.DataFrame(rows)


def main():
    synthetic = load_synthetic()
    real_train = load_real("train")
    real_test = load_real("test")

    train_df = pd.concat([synthetic, real_train], ignore_index=True)
    train_df.to_csv(OUT_TRAIN, index=False)
    print(f"\nTrain combinado -> {OUT_TRAIN}: {len(train_df)} filas")
    print(train_df.groupby(["familia", "species"]).size().to_string())

    real_test.to_csv(OUT_TEST, index=False)
    print(f"\nTest real reservado -> {OUT_TEST}: {len(real_test)} filas")
    if not real_test.empty:
        print(real_test.groupby("familia").size().to_string())


if __name__ == "__main__":
    main()
