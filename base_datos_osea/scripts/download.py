"""Descarga los ejemplares marcados `selected_for_download` en el catálogo.

Solo descarga media con visibility "open": son las únicas que la API permite
bajar sin que un curador de MorphoSource apruebe antes una solicitud manual
desde la web (ver README).

Requiere una API key de MorphoSource (Dashboard > Profile > View API Key), como variable de
entorno o en un fichero `.env` en la raíz del repo (no versionado, ver .gitignore):
    MORPHOSOURCE_API_KEY=...

Uso:
    python scripts/download.py
    python scripts/download.py --catalog catalog/homo_sapiens_media.csv
"""

import argparse
import csv
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from morphosource import DownloadConfig, get_media

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USE_STATEMENT = (
    "Dataset de entrenamiento para clasificador ML de tipos de huesos "
    "(osteolab-ml-platform) y generación de renders sintéticos en Blender."
)


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh)), fh


def write_rows(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_one(row, download_config, meshes_dir):
    media_id = row["media_id"]
    bone_dir = meshes_dir / row["species"].lower().replace(" ", "_") / row["bone_canonical"] / media_id
    bone_dir.mkdir(parents=True, exist_ok=True)
    zip_path = bone_dir / f"{media_id}.zip"

    media = get_media(media_id)
    media.download_bundle(str(zip_path), download_config)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(bone_dir)

    row["download_status"] = "done"
    row["local_path"] = str(bone_dir.relative_to(REPO_ROOT))
    row["downloaded_at"] = datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(REPO_ROOT / "catalog" / "homo_sapiens_media.csv"))
    parser.add_argument("--meshes-dir", default=str(REPO_ROOT / "data" / "meshes"))
    parser.add_argument("--use-statement", default=DEFAULT_USE_STATEMENT)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("MORPHOSOURCE_API_KEY")
    if not api_key:
        sys.exit("Falta MORPHOSOURCE_API_KEY en el entorno o en .env. Consigue tu key en "
                 "morphosource.org > Profile > View API Key.")

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        sys.exit(f"No existe {catalog_path}. Ejecuta antes scripts/catalog.py.")

    rows, fh = load_rows(catalog_path)
    fh.close()
    fieldnames = list(rows[0].keys())

    download_config = DownloadConfig(
        api_key=api_key, use_statement=args.use_statement, use_categories=["Research"]
    )
    meshes_dir = Path(args.meshes_dir)

    pending = [
        r for r in rows
        if r["selected_for_download"] == "True"
        and r["visibility"] == "open"
        and r["download_status"] != "done"
    ]

    print(f"{len(pending)} elemento(s) por descargar de {catalog_path.name}")
    for row in pending:
        print(f"  -> {row['bone_canonical']} ({row['media_id']}) ... ", end="", flush=True)
        try:
            download_one(row, download_config, meshes_dir)
            print("OK")
        except Exception as exc:
            row["download_status"] = f"error: {exc}"
            print(f"ERROR: {exc}")
        write_rows(catalog_path, rows, fieldnames)  # persistir tras cada ítem: una interrupción no pierde progreso

    print(f"\nCatálogo actualizado: {catalog_path}")


if __name__ == "__main__":
    main()
