"""Arbitra la selección de descarga ENTRE catálogos de distintas especies.

`catalog.py` elige el mejor ejemplar de cada hueso dentro de UNA especie.
Este script mira todos los `catalog/*_media.csv` a la vez y, para cada hueso
canónico, deja seleccionado un único ejemplar (el más completo disponible en
cualquier especie), desmarcando el resto — así `download.py` nunca baja dos
especies distintas para tapar el mismo hueso. Si un hueso ya tiene una
descarga `done` en alguna especie, esa se respeta y no se toca.

Uso:
    python scripts/gap_check.py
    python scripts/gap_check.py catalog/homo_sapiens_media.csv catalog/homo_naledi_media.csv
"""

import csv
import sys
from pathlib import Path

from bones import TARGET_BONES

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO_ROOT / "catalog"


def load_catalog(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_catalog(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(CATALOG_DIR.glob("*_media.csv"))
    if len(paths) < 2:
        sys.exit("Necesito al menos 2 catálogos (ejecuta catalog.py para varias especies antes).")

    catalogs = {p: load_catalog(p) for p in paths}

    print("=== Cobertura combinada por hueso ===")
    for bone in sorted(TARGET_BONES):
        candidates = []
        for path, rows in catalogs.items():
            for row in rows:
                if row["bone_canonical"] == bone and row["media_type"] == "Mesh":
                    candidates.append((path, row))

        if not candidates:
            print(f"  {bone:<20} sin candidatos en ninguna especie catalogada")
            continue

        done = [c for c in candidates if c[1]["download_status"] == "done"]
        if done:
            winner_path, winner = done[0]
        else:
            winner_path, winner = sorted(
                candidates,
                key=lambda c: (c[1]["is_partial"] == "True", -int(c[1]["file_size_all"] or 0), c[1]["media_id"]),
            )[0]

        for path, row in candidates:
            row["selected_for_download"] = "True" if (path, row) == (winner_path, winner) else "False"

        flag = " (PARCIAL)" if winner["is_partial"] == "True" else ""
        status = " [ya descargado]" if winner["download_status"] == "done" else ""
        print(f"  {bone:<20} {winner['species']:<24} {winner['media_id']}{flag}{status}  ({winner_path.name})")

    for path, rows in catalogs.items():
        write_catalog(path, rows)
    print(f"\n{len(catalogs)} catálogo(s) actualizados. Ejecuta download.py --catalog <archivo> por cada uno para bajar lo pendiente.")


if __name__ == "__main__":
    main()
