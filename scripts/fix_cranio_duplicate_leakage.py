"""Corrige la fuga train/test encontrada por `photo_dedupe.py` el 2026-10-26:
8 clusters de fotos casi idénticas (mismo stock photo aceptado dos veces con
nombre distinto) repartidas entre train Y test en pelvis/sacro/vertebra/
hueso_largo/costilla. Deja UNA sola copia por cluster (en train, nunca se
quita del todo un ejemplo: se mueve a train en vez de descartarlo) y mueve el
resto a `data/real_photos_accepted/_duplicados_removidos/<grupo>/` (no se
borra nada, por si hiciera falta revisar la decisión).

Ejecutar una vez, no es idempotente de forma segura si se vuelve a correr
tras mover los archivos (el manifest ya no tendría la fila del 'keep' con
status viejo) -- de ahí que no sea un script de uso recurrente, al contrario
que `photo_dedupe.py`.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ACCEPTED_DIR = BASE_DIR / "data/real_photos_accepted"
MANIFEST_PATH = ACCEPTED_DIR / "manifest.csv"
QUARANTINE_DIR = ACCEPTED_DIR / "_duplicados_removidos"

# (bone_group, filename a mantener en train, [filenames a mover a cuarentena])
CLUSTERS = [
    ("pelvis", "pelvis_110", ["pelvis_108.jpg"]),
    ("sacro", "sacro_102", ["sacro_111.jpg"]),
    ("sacro", "sacro_122", ["sacro_106.jpg"]),
    ("vertebra", "vertebra_113", ["vertebra_114.jpg"]),
    ("vertebra", "vertebra_117", ["vertebra_116.jpg"]),
    ("hueso_largo", "hueso_largo_066", ["hueso_largo_062.jpg"]),
    ("costilla", "costilla_123", ["costilla_134.jpg", "costilla_136.jpg"]),
    ("costilla", "costilla_139", ["costilla_126.jpg"]),
]


def main():
    df = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    moved = 0

    for bone_group, keep_stem, remove_filenames in CLUSTERS:
        keep_mask = (df["bone_group"] == bone_group) & (df["filename"].str.startswith(keep_stem))
        if keep_mask.sum() != 1:
            print(f"AVISO: esperaba 1 fila para {bone_group}/{keep_stem}*, encontré {keep_mask.sum()} -- salto")
            continue
        df.loc[keep_mask, "split"] = "train"

        for fname in remove_filenames:
            row_mask = (df["bone_group"] == bone_group) & (df["filename"] == fname)
            if row_mask.sum() != 1:
                print(f"AVISO: esperaba 1 fila para {bone_group}/{fname}, encontré {row_mask.sum()} -- salto")
                continue
            src = ACCEPTED_DIR / bone_group / fname
            if src.exists():
                dest_dir = QUARANTINE_DIR / bone_group
                dest_dir.mkdir(parents=True, exist_ok=True)
                src.rename(dest_dir / fname)
                moved += 1
            df = df[~row_mask]
            print(f"  {bone_group}/{fname} -> cuarentena (duplicado de {keep_stem}, ahora train)")

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\n{moved} archivos movidos a {QUARANTINE_DIR}, manifest actualizado")


if __name__ == "__main__":
    main()
