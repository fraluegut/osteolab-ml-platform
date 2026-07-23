"""Cataloga y descarga mallas 3D de la API de Open Access de Smithsonian (api.si.edu).

A diferencia de MorphoSource, aquí no buscamos "un ejemplo por hueso": la colección
3D de primates de Smithsonian es casi enteramente cráneo/mandíbula (confirmado a
mano el 2026-07-22 — no tiene fémur/húmero/etc.), así que este conector se usa para
sumar DIVERSIDAD de especies en esas dos clases, no para tapar huecos de otros huesos.
Por eso no pasa por `gap_check.py`: cada especie de la lista se descarga, no compite
por "ganar" el hueso.

Todo lo que devuelve esta API para contenido 3D está en CC0 (dominio público) y los
ficheros son de descarga directa, sin key ni aprobación — la API key de api.data.gov
solo hace falta para el paso de búsqueda (rate limit más alto que la DEMO_KEY pública).

Uso:
    python scripts/smithsonian.py catalog   # requiere SI_API_KEY (o .env)
    python scripts/smithsonian.py download  # descarga lo cataloged, sin key
"""

import argparse
import csv
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bones import classify_part

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "catalog" / "smithsonian_primates_media.csv"
MESHES_DIR = REPO_ROOT / "data" / "meshes"
SEARCH_URL = "https://api.si.edu/openaccess/api/v1.0/search"

# Especies curadas para dar diversidad taxonómica de cráneo/mandíbula: gran simio,
# simio menor, mono del Viejo Mundo, mono del Nuevo Mundo, prosimio.
TARGET_SPECIES = [
    "Pan troglodytes verus",
    "Gorilla gorilla gorilla",
    "Symphalangus syndactylus",
    "Macaca radiata",
    "Alouatta palliata",
    "Lemur catta",
]
TARGET_BONES_HERE = {"cranium", "mandible"}

FIELDS = [
    "species", "bone_canonical", "record_id", "title", "license",
    "resource_url", "resource_format", "usnm_number", "record_link",
    "download_status", "local_path", "downloaded_at",
]


def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def first(items, default=""):
    return items[0]["content"] if items else default


def resource_attributes(resource):
    # `attributes` a veces es un único dict con todas las claves, a veces una lista
    # de dicts de una clave cada uno — se normaliza fusionándolos.
    merged = {}
    for attrs in resource.get("attributes", []):
        merged.update(attrs)
    return merged


def pick_resource(resources):
    """Prefiere obj/ply de resolución completa; si no hay, cualquier malla 3D disponible."""
    def score(r):
        full = r.get("category") == "Full resolution"
        obj_like = resource_attributes(r).get("MODEL_FILE_TYPE") in ("obj", "ply")
        return (full, obj_like)
    return sorted(resources, key=score, reverse=True)[0]


def bone_from_title(title):
    # "Pan troglodytes verus: Cranium (Full resolution 3D mesh, obj, scale in mm)" -> "cranium"
    part = title.split("(")[0].strip().split(":")[-1].strip()
    _, bone, is_target, _ = classify_part(part)
    return bone if is_target else None


def search_species(species, api_key):
    params = {"q": species, "api_key": api_key, "rows": 100}
    resp = requests.get(SEARCH_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["response"]["rows"]


def build_catalog(api_key):
    existing = {}
    if CATALOG_PATH.exists():
        with CATALOG_PATH.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[(row["species"], row["bone_canonical"])] = row

    picked = {}  # (species, bone) -> row, primer match estable por record_ID
    for species in TARGET_SPECIES:
        for row in sorted(search_species(species, api_key), key=lambda r: r["content"]["descriptiveNonRepeating"]["record_ID"]):
            if not row["title"].lower().startswith(species.lower()):
                continue  # descarta coincidencias de texto libre que no son este taxón
            desc = row["content"]["descriptiveNonRepeating"]
            om = desc.get("online_media")
            if not om:
                continue
            for media in om.get("media", []):
                if media.get("type") != "3d_voyager" or not media.get("resources"):
                    continue
                bone = bone_from_title(media["resources"][0]["title"])
                if bone not in TARGET_BONES_HERE:
                    continue
                key = (species, bone)
                if key in picked:
                    continue  # ya tenemos un ejemplar de esta especie+hueso
                resource = pick_resource(media["resources"])
                prev = existing.get(key, {})
                picked[key] = {
                    "species": species,
                    "bone_canonical": bone,
                    "record_id": desc["record_ID"],
                    "title": media["resources"][0]["title"],
                    "license": media.get("usage", {}).get("access", "CC0"),
                    "resource_url": resource["url"],
                    "resource_format": resource_attributes(resource).get("MODEL_FILE_TYPE", ""),
                    "usnm_number": first(row["content"]["freetext"].get("identifier", [])),
                    "record_link": desc.get("record_link", ""),
                    "download_status": prev.get("download_status", ""),
                    "local_path": prev.get("local_path", ""),
                    "downloaded_at": prev.get("downloaded_at", ""),
                }

    rows = sorted(picked.values(), key=lambda r: (r["bone_canonical"], r["species"]))
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== Catálogo Smithsonian (cráneo/mandíbula, {len(TARGET_SPECIES)} especies) ===")
    for row in rows:
        print(f"  {row['bone_canonical']:<10} {row['species']:<28} {row['record_id']} ({row['resource_format']})")
    found = {(r["species"], r["bone_canonical"]) for r in rows}
    for species in TARGET_SPECIES:
        for bone in sorted(TARGET_BONES_HERE):
            if (species, bone) not in found:
                print(f"  {bone:<10} {species:<28} sin escaneo 3D disponible")
    print(f"\nCSV escrito en: {CATALOG_PATH}")


def download_catalog():
    if not CATALOG_PATH.exists():
        sys.exit(f"No existe {CATALOG_PATH}. Ejecuta antes: python scripts/smithsonian.py catalog")

    with CATALOG_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    pending = [r for r in rows if r["download_status"] != "done"]
    print(f"{len(pending)} elemento(s) por descargar")
    for row in pending:
        species_slug = re.sub(r"[^a-z0-9]+", "_", row["species"].lower()).strip("_")
        bone_dir = MESHES_DIR / species_slug / row["bone_canonical"] / row["record_id"]
        print(f"  -> {row['bone_canonical']} {row['species']} ({row['record_id']}) ... ", end="", flush=True)
        try:
            bone_dir.mkdir(parents=True, exist_ok=True)
            filename = row["resource_url"].rsplit("/", 1)[-1]
            dest = bone_dir / filename
            with requests.get(row["resource_url"], stream=True) as resp:
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
            if dest.suffix == ".zip":
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(bone_dir)
            row["download_status"] = "done"
            row["local_path"] = str(bone_dir.relative_to(REPO_ROOT))
            row["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            print("OK")
        except Exception as exc:
            row["download_status"] = f"error: {exc}"
            print(f"ERROR: {exc}")

        with CATALOG_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nCatálogo actualizado: {CATALOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "download"])
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if args.command == "catalog":
        api_key = os.environ.get("SI_API_KEY")
        if not api_key:
            sys.exit("Falta SI_API_KEY en el entorno o en .env (consíguela gratis en https://api.data.gov/signup/).")
        build_catalog(api_key)
    else:
        download_catalog()


if __name__ == "__main__":
    main()
