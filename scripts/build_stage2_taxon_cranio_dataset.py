"""Construye el dataset de entrenamiento/test para `stage2_taxon_cranio`
(humano/carnivoro/ungulado/roedor/otro_primate), con los renders nuevos de
`render_bone_lab.py` (ver commit 92396bd para el origen del modelo).

Mismo patrón que `build_stage2_familia_cranio_dataset.py`:
  - Sintético: `data/processed/bone_geometric_features_lab.csv`, filtrado a
    bone_group=cranio, especie mapeada a taxón vía SPECIES_TO_TAXON.
  - Real: `data/real_photos_accepted/manifest.csv`, columna `taxon` ya
    anotada a mano, filtrado a bone_group=cranio -- split train entra al
    CSV de entrenamiento, split test se queda aparte (examen honesto).

Uso:
    python scripts/build_stage2_taxon_cranio_dataset.py
"""
from pathlib import Path

import pandas as pd

from shared.cv_extractor.build_dataset import flatten_features
from shared.cv_extractor.extract import extract_features

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_FEATURES = REPO_ROOT / "data/processed/bone_geometric_features_lab.csv"
ACCEPTED_DIR = REPO_ROOT / "data/real_photos_accepted"
MANIFEST_PATH = ACCEPTED_DIR / "manifest.csv"
CROPS_OUT_DIR = REPO_ROOT / "data/segmented_crops/real_photos_taxon_cranio"

OUT_TRAIN = REPO_ROOT / "data/processed/bone_geometric_features_stage2_taxon_cranio.csv"
OUT_TEST = REPO_ROOT / "data/processed/bone_geometric_features_stage2_taxon_cranio_test.csv"

# especie (carpeta data/meshes/) -> taxon, según el mapeo ya usado en el
# commit 92396bd (verificado contra el CSV viejo, ver sesión 2026-10-27).
SPECIES_TO_TAXON = {
    "homo_sapiens": "humano",
    "canis_familiaris": "carnivoro",
    "felis_catus": "carnivoro",
    "cynomys_sp.": "roedor",
    "bos_taurus": "ungulado",
    "sus_domesticus": "ungulado",
    "equus_caballus": "ungulado",
    "odocoileus_sp.": "ungulado",
    "odocoileus_virginianus": "ungulado",
    "alouatta_palliata": "otro_primate",
    "gorilla_gorilla_gorilla": "otro_primate",
    "macaca_sylvanus": "otro_primate",
    "pan_troglodytes_verus": "otro_primate",
    "pongo_sp.": "otro_primate",
    "symphalangus_syndactylus": "otro_primate",
}


def load_synthetic() -> pd.DataFrame:
    if not LAB_FEATURES.exists():
        raise SystemExit(f"Falta {LAB_FEATURES} -- correr antes build_dataset_lab.py + build_segmented_crops.py")
    df = pd.read_csv(LAB_FEATURES)
    df = df[(df["bone_group"] == "cranio") & (df["species"].isin(SPECIES_TO_TAXON))].copy()
    df["taxon"] = df["species"].map(SPECIES_TO_TAXON)
    print(f"Sintético: {len(df)} vistas, {df.groupby('taxon')['specimen'].nunique().to_dict()} especímenes por taxón")
    unmapped = sorted(set(pd.read_csv(LAB_FEATURES).query("bone_group == 'cranio'")["species"]) - set(SPECIES_TO_TAXON))
    if unmapped:
        print(f"  AVISO: especies de cráneo sin mapear a taxón (ignoradas): {unmapped}")
    return df


def load_real(split: str) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[
        (manifest["bone_group"] == "cranio")
        & manifest["taxon"].notna()
        & (manifest["taxon"] != "")
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
            "taxon": m["taxon"],
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
    print(train_df.groupby(["taxon", "species"]).size().to_string())

    real_test.to_csv(OUT_TEST, index=False)
    print(f"\nTest real reservado -> {OUT_TEST}: {len(real_test)} filas")
    if not real_test.empty:
        print(real_test.groupby("taxon").size().to_string())


if __name__ == "__main__":
    main()
