"""Cataloga y descarga mallas 3D de Sketchfab, curadas a mano.

A diferencia de MorphoSource/Smithsonian, aquí NO hay búsqueda+selección
automática: el fémur `000031310` de MorphoSource resultó ser un fragmento
(le faltaban los cóndilos distales) etiquetado como "femur" sin ninguna
palabra como "proximal"/"fragmento" que un clasificador de texto pudiera
pillar. Sketchfab permite ver una miniatura de cada modelo ANTES de
descargarlo — así que cada candidato de este fichero fue revisado
visualmente (por Claude, mirando la miniatura) antes de añadirlo aquí.
Añadir un candidato nuevo: buscar en https://sketchfab.com/developers/data-api/v3
(`/v3/search?type=models&q=...&downloadable=true`), mirar su miniatura, y
solo si se ve la pieza completa añadirlo a CURATED_CANDIDATES.

La búsqueda y las miniaturas son públicas (sin token). La descarga requiere
un token de cuenta gratuita de Sketchfab (Settings > Password & API).

Uso:
    python scripts/sketchfab.py catalog
    python scripts/sketchfab.py download
"""

import argparse
import csv
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "catalog" / "sketchfab_media.csv"
MESHES_DIR = REPO_ROOT / "data" / "meshes"
API_URL = "https://api.sketchfab.com/v3"

# Cada entrada fue revisada visualmente (miniatura) antes de añadirse — ver
# docstring. `species` por defecto Homo sapiens salvo que se indique otra.
CURATED_CANDIDATES = [
    {
        "bone_canonical": "femur",
        "uid": "a9c1f1a88b104c3fbfe975fa10b31b31",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cabeza + diáfisis + cóndilos distales). Elon University anatomy lab.",
    },
    {
        "bone_canonical": "humerus",
        "uid": "12744d7c36be47d8ad94f620a22d649c",
        "species": "Homo sapiens",
        "notes": "Verificado completo (ambas epífisis visibles). CC0.",
    },
    {
        "bone_canonical": "cranium",
        "uid": "c39855833ff74943a2847972c9ff8700",
        "species": "Homo sapiens",
        "notes": "Verificado completo con mandíbula visible. Fotogrametría documentada, UNC Greensboro, permiso citado.",
    },
    {
        "bone_canonical": "mandible",
        "uid": "7a3c3b99916e481197e6f18fda107c15",
        "species": "Homo sapiens",
        "notes": "Verificado completo (ambas ramas con cóndilos, dientes y textura). University of Dundee, creadores documentados.",
    },
    {
        "bone_canonical": "scapula",
        "uid": "24d2b770d5e049e08b4239dac463de39",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cuerpo, glenoides, acromion). UNCG Imaging Lab, escaneo NextEngine de espécimen de colección con permiso citado.",
    },
    {
        "bone_canonical": "clavicle",
        "uid": "b58db46ed8564fedade966310a3e95ae",
        "species": "Homo sapiens",
        "notes": "Verificado completo (ambos extremos, curva en S). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "radius",
        "uid": "68c3f1c56b1b400e905072293e0a878a",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cabeza radial proximal + apófisis estiloides distal). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "ulna",
        "uid": "4b849ef3f5c142099b0a244b00011e47",
        "species": "Homo sapiens",
        "notes": "Verificado completo (olécranon proximal + apófisis estiloides distal). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "tibia",
        "uid": "7c1979d6127749bc80a9d9276d24edcd",
        "species": "Homo sapiens",
        "notes": "Verificado completo (mesetas tibiales proximales + maléolo medial distal). Eric Bauer, alta resolución.",
    },
    {
        "bone_canonical": "fibula",
        "uid": "5f239617db8d4d2eaaf13b8616acf70e",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cabeza proximal + maléolo lateral distal). Eric Bauer.",
    },
    {
        "bone_canonical": "patella",
        "uid": "8459a5c2baee48c8905631322cdf8edf",
        "species": "Homo sapiens",
        "notes": "Verificado completo (forma triangular característica). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "sacrum",
        "uid": "3f7cfb377bb74b49aaf3a98e8933e94b",
        "species": "Homo sapiens",
        "notes": "Verificado completo (forma triangular, forámenes sacros visibles). UNCG Imaging Lab.",
    },
    {
        "bone_canonical": "vertebra",
        "uid": "da43de500e614feea3458da2e6bc8e5f",
        "species": "Homo sapiens",
        "notes": "Verificado completo (cuerpo, apófisis espinosa y transversas, pedículos). UNCG Imaging Lab, lumbar.",
    },
    {
        "bone_canonical": "rib",
        "uid": "6ca3afdf9a084b0db8b6141f50a94f49",
        "species": "Homo sapiens",
        "notes": "Verificado completo (costilla central típica, cabeza a un extremo). UNCG Imaging Lab.",
    },
    # --- 2º espécimen para los 2 grupos morfológicos que solo tenían 1 (ver
    # BONE_GROUPS en bones.py): sin esto, el clasificador de grupo no puede
    # distinguir "forma del hueso" de "forma de ESTE hueso concreto" para
    # hueso_plano/pelvis.
    {
        "bone_canonical": "scapula",
        "uid": "0745bbb368b4401db89e73babe440ee8",
        "species": "Homo sapiens",
        "notes": "2º espécimen (grupo hueso_plano tenía solo 1). Verificado completo (glenoides, acromion, coracoides). Eric Bauer, alta resolución (9.8M caras).",
    },
    {
        "bone_canonical": "pelvis",
        "uid": "35586f343d9c4c6eb813f9006f036595",
        "species": "Homo sapiens",
        "notes": "2º espécimen (grupo pelvis tenía solo 1). Verificado completo (ambos ilion + sacro, forámenes visibles). Oregon State University.",
    },
]

# Maxilar: 2 candidatos revisados (uid 47125f083e464d87b23ec748b2679983 y
# e59a4be88e8545268dda24c72eed2a0c), ambos fragmentos unilaterales sin ambos
# lados de la arcada — rechazados. Sigue cubierto por H. naledi.

FIELDS = [
    "bone_canonical", "species", "uid", "name", "license", "verified_complete", "verification_notes",
    "face_count", "creator", "description", "thumbnail_url", "viewer_url",
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


def load_existing(path):
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[row["uid"]] = row
    return existing


def build_catalog():
    existing = load_existing(CATALOG_PATH)
    rows = []
    for candidate in CURATED_CANDIDATES:
        resp = requests.get(f"{API_URL}/models/{candidate['uid']}")
        resp.raise_for_status()
        data = resp.json()
        prev = existing.get(candidate["uid"], {})
        thumb = max(data["thumbnails"]["images"], key=lambda i: i["width"])
        rows.append({
            "bone_canonical": candidate["bone_canonical"],
            "species": candidate["species"],
            "uid": candidate["uid"],
            "name": data["name"],
            "license": data["license"]["label"],
            "verified_complete": True,
            "verification_notes": candidate["notes"],
            "face_count": data.get("faceCount", ""),
            "creator": data["user"]["displayName"],
            "description": (data.get("description") or "").replace("\n", " ").strip(),
            "thumbnail_url": thumb["url"],
            "viewer_url": data["viewerUrl"],
            "download_status": prev.get("download_status", ""),
            "local_path": prev.get("local_path", ""),
            "downloaded_at": prev.get("downloaded_at", ""),
        })

    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== Catálogo Sketchfab ({len(rows)} candidatos curados a mano) ===")
    for row in rows:
        print(f"  {row['bone_canonical']:<10} {row['name']:<30} {row['license']:<12} {row['uid']}")
    print(f"\nCSV escrito en: {CATALOG_PATH}")


def download_catalog():
    if not CATALOG_PATH.exists():
        sys.exit(f"No existe {CATALOG_PATH}. Ejecuta antes: python scripts/sketchfab.py catalog")

    token = os.environ.get("SKETCHFAB_API_TOKEN")
    if not token:
        sys.exit("Falta SKETCHFAB_API_TOKEN en el entorno o en .env.")
    headers = {"Authorization": f"Token {token}"}

    with CATALOG_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    pending = [r for r in rows if r["download_status"] != "done"]
    print(f"{len(pending)} elemento(s) por descargar")
    for row in pending:
        print(f"  -> {row['bone_canonical']} ({row['uid']}) ... ", end="", flush=True)
        try:
            resp = requests.get(f"{API_URL}/models/{row['uid']}/download", headers=headers)
            resp.raise_for_status()
            archives = resp.json()
            archive = archives.get("glb") or archives.get("gltf") or archives.get("source")
            if not archive:
                raise RuntimeError(f"sin archivo descargable en {list(archives.keys())}")

            species_slug = row["species"].lower().replace(" ", "_")
            bone_dir = MESHES_DIR / species_slug / row["bone_canonical"] / f"sketchfab_{row['uid']}"
            bone_dir.mkdir(parents=True, exist_ok=True)
            filename = archive["url"].split("?")[0].rsplit("/", 1)[-1]
            dest = bone_dir / filename

            with requests.get(archive["url"], stream=True) as file_resp:
                file_resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in file_resp.iter_content(chunk_size=1024 * 1024):
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
        build_catalog()
    else:
        download_catalog()


if __name__ == "__main__":
    main()
