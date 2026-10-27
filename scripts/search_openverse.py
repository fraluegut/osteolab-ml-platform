"""Busca en Openverse (openverse.org — buscador dedicado a contenido con
licencia abierta, agrega Flickr, Wikimedia, museos...) candidatos de fotos
reales de especímenes óseos, los descarga a `data/real_photos_review/<grupo>/`
y anota cada uno en `data/real_photos_review/manifest.csv` con `status=pending`
para revisión visual manual — NADA se acepta automáticamente, hay que
revisar cada foto (fragmentos, varios huesos en la misma imagen, primeros
planos no representativos... ver HISTORIAL.md 2026-07-26 para los criterios
ya usados) y mover a mano lo bueno a `data/real_photos_accepted/<grupo>/`
con su fila correspondiente en `data/real_photos_accepted/manifest.csv`.

No hace falta API key. `--license-type all-cc` (por defecto) incluye CC
BY-NC — igual que ya se aceptó para archeozoo.org, vale para este proyecto
personal/no comercial; usa `--license-type modification` si en algún
momento hiciera falta evitar restricciones de "no derivados".

Uso:
    python scripts/search_openverse.py --bone-group hueso_largo \\
        --query "human femur bone specimen photo" --n 20
    python scripts/search_openverse.py --bone-group cranio \\
        --query "human skull specimen photo museum" --n 20
"""
import argparse
import csv
import sys
import tempfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from photo_dedupe import (  # noqa: E402
    compute_phash,
    is_duplicate_of_known,
    load_accepted_hashes,
    load_review_pending_hashes,
)

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEW_DIR = BASE_DIR / "data/real_photos_review"
MANIFEST_PATH = REVIEW_DIR / "manifest.csv"
API_URL = "https://api.openverse.org/v1/images/"
FIELDS = ["bone_group", "filename", "source_url", "foreign_landing_url", "license", "creator", "status", "notes"]


def load_existing_filenames() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    with open(MANIFEST_PATH, newline="") as f:
        return {row["filename"] for row in csv.DictReader(f)}


def search(query: str, license_type: str, page_size: int) -> list[dict]:
    resp = requests.get(
        API_URL,
        params={"q": query, "license_type": license_type, "page_size": page_size, "mature": "false"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bone-group", required=True,
                    choices=["cranio", "mandibula_maxilar", "hueso_plano", "hueso_largo", "hueso_pequeno",
                             "pelvis", "sacro", "vertebra", "costilla"])
    p.add_argument("--query", required=True, help="término de búsqueda, en inglés suele dar mejores resultados")
    p.add_argument("--n", type=int, default=20, help="máximo de candidatos a traer en esta búsqueda")
    p.add_argument("--license-type", default="all-cc", choices=["all-cc", "commercial", "modification"])
    args = p.parse_args()

    out_dir = REVIEW_DIR / args.bone_group
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_filenames()
    results = search(args.query, args.license_type, args.n)
    print(f"{len(results)} resultados de Openverse para {args.query!r}")

    # phash de todo lo ya aceptado + lo que sigue pendiente de revisar (no de
    # lo ya rechazado/dudoso -- descartar aposta no debería bloquear volver a
    # verlo si aparece en otra búsqueda) -- evita bajar dos veces el mismo
    # stock photo con otro id de Openverse, o repetir una que ya se aceptó.
    known_hashes = [r["phash"] for r in load_accepted_hashes()]
    known_hashes += [r["phash"] for r in load_review_pending_hashes()]
    print(f"{len(known_hashes)} fotos ya conocidas (aceptadas + pendientes) para chequeo anti-duplicado")

    new_rows = []
    for r in results:
        img_url = r.get("url")
        if not img_url:
            continue
        ext = (Path(img_url.split("?")[0]).suffix or ".jpg").lower()
        filename = f"openverse_{r['id']}{ext}"
        if filename in existing:
            continue

        try:
            resp = requests.get(img_url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  AVISO: fallo descargando {img_url}: {exc}")
            continue

        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(resp.content)
            tmp.flush()
            tmp_path = Path(tmp.name)
            if is_duplicate_of_known(tmp_path, known_hashes):
                print(f"  SALTADO (duplicado/casi igual a una ya conocida): {filename}  {r.get('title', '')}")
                continue
            new_hash = compute_phash(tmp_path)

        (out_dir / filename).write_bytes(resp.content)
        if new_hash is not None:
            known_hashes.append(new_hash)  # también deduplica DENTRO de esta misma búsqueda
        license_str = f"{r.get('license', '')} {r.get('license_version', '')}".strip()
        new_rows.append({
            "bone_group": args.bone_group,
            "filename": filename,
            "source_url": img_url,
            "foreign_landing_url": r.get("foreign_landing_url", ""),
            "license": license_str,
            "creator": r.get("creator", ""),
            "status": "pending",
            "notes": r.get("title", ""),
        })
        print(f"  descargado {filename}  [{license_str}]  {r.get('title', '')}")

    if new_rows:
        write_header = not MANIFEST_PATH.exists()
        with open(MANIFEST_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)

    print(f"\n{len(new_rows)} candidatos nuevos en {out_dir}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
