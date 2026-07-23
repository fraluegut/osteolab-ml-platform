"""Construye/actualiza el catálogo (tabla CSV) de media de MorphoSource.

Consulta la API pública de búsqueda de MorphoSource (no requiere API key:
solo la descarga de bytes la requiere), clasifica cada resultado por hueso
canónico (`bones.py`) y elige un ejemplar por hueso para descarga automática.

No descarga nada. Eso lo hace `download.py`, leyendo este mismo CSV.

Uso:
    python scripts/catalog.py
    python scripts/catalog.py --taxon sapiens --species-name "Homo sapiens"
"""

import argparse
import csv
import re
import unicodedata
from pathlib import Path

from morphosource import search_media
from morphosource.download import DownloadVisibility

from bones import classify_part

FIELDS = [
    "media_id", "bone_canonical", "category", "is_target", "is_partial", "selected_for_download",
    "title", "part_raw", "side", "media_type", "modality",
    "species", "physical_object_id", "physical_object_title", "sex",
    "visibility", "license", "rights_statement", "ip_holder", "use_agreement_type",
    "permits_commercial_use", "permits_3d_use",
    "file_size_all", "date_uploaded", "creator",
    "website_url", "thumbnail_url", "external_media_url", "doi", "ark",
    "download_status", "local_path", "downloaded_at",
]

# Estas columnas las rellena/gestiona download.py; catalog.py nunca las pisa
# si ya existían en el CSV previo, para no perder el estado de descargas hechas.
DOWNLOAD_MANAGED_FIELDS = ["download_status", "local_path", "downloaded_at"]


def slugify(name):
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def first(data, key, default=""):
    value = data.get(key)
    if value:
        return value[0]
    return default


def load_existing(path):
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[row["media_id"]] = row
    return existing


def build_rows(species_name, taxon, existing, exact_taxonomy_name=None):
    results = search_media(taxonomy_gbif=taxon, visibility=DownloadVisibility.OPEN, per_page=100)
    rows = []
    for media in results.items:
        data = media.data
        if exact_taxonomy_name and first(data, "physical_object_taxonomy_name") != exact_taxonomy_name:
            continue
        part_raw = first(data, "part")
        category, bone_canonical, is_target, is_partial = classify_part(part_raw)

        prev = existing.get(media.id, {})
        row = {
            "media_id": media.id,
            "bone_canonical": bone_canonical or "",
            "category": category,
            "is_target": is_target,
            "is_partial": is_partial,
            "selected_for_download": False,  # se decide más abajo
            "title": first(data, "title"),
            "part_raw": part_raw,
            "side": first(data, "side"),
            "media_type": first(data, "media_type"),
            "modality": first(data, "modality"),
            "species": species_name,
            "physical_object_id": first(data, "physical_object_id"),
            "physical_object_title": first(data, "physical_object_title"),
            "sex": first(data, "sex"),
            "visibility": first(data, "visibility"),
            "license": first(data, "license"),
            "rights_statement": first(data, "copyright_statement"),
            "ip_holder": first(data, "ip_holder"),
            "use_agreement_type": first(data, "morphosource_use_agreement_type"),
            "permits_commercial_use": first(data, "permits_commercial_use"),
            "permits_3d_use": first(data, "permits_3d_use"),
            "file_size_all": first(data, "file_size_all", 0),
            "date_uploaded": first(data, "date_uploaded"),
            "creator": "; ".join(data.get("creator", [])),
            "website_url": media.get_website_url(),
            "thumbnail_url": media.get_thumbnail_url() or "",
            "external_media_url": first(data, "external_media_url"),
            "doi": first(data, "doi"),
            "ark": first(data, "ark"),
            "download_status": prev.get("download_status", ""),
            "local_path": prev.get("local_path", ""),
            "downloaded_at": prev.get("downloaded_at", ""),
        }
        rows.append(row)
    return rows


def select_for_download(rows):
    """Marca un ejemplar por hueso canónico. Si ya hay uno descargado (done)
    para ese hueso, se mantiene su selección aunque aparezca un candidato
    'mejor' (evita huérfanos en disco al re-ejecutar el catálogo)."""
    by_bone = {}
    for row in rows:
        if row["is_target"] and row["media_type"] == "Mesh":
            by_bone.setdefault(row["bone_canonical"], []).append(row)

    for bone, candidates in by_bone.items():
        already_done = next((r for r in candidates if r["download_status"] == "done"), None)
        chosen = already_done or sorted(
            candidates, key=lambda r: (r["is_partial"], -int(r["file_size_all"] or 0), r["media_id"])
        )[0]
        chosen["selected_for_download"] = True


def write_csv(path, rows):
    rows_sorted = sorted(rows, key=lambda r: (r["category"], r["bone_canonical"], r["media_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_sorted)


def print_summary(rows, species_name):
    from bones import TARGET_BONES

    print(f"\n=== Catálogo de {species_name} (Open Download) ===")
    print(f"Total de registros: {len(rows)}")
    for bone in sorted(TARGET_BONES):
        candidates = [r for r in rows if r["bone_canonical"] == bone and r["media_type"] == "Mesh"]
        selected = next((r for r in candidates if r["selected_for_download"] is True), None)
        if selected:
            flag = " (PARCIAL, no es la pieza completa)" if selected["is_partial"] else ""
            print(f"  {bone:<20} {len(candidates)} candidato(s) -> seleccionado {selected['media_id']}{flag}")
        else:
            print(f"  {bone:<20} 0 candidatos en abierto -> requeriría solicitud 'Restricted Download'")

    by_category = {}
    for r in rows:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
    print("\nOtras categorías detectadas en los datos (no seleccionadas para descarga):")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        if cat != "target_bone":
            print(f"  {cat:<24} {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxon", default="sapiens", help="Valor GBIF a filtrar, p.ej. 'sapiens' o el género 'Homo'")
    parser.add_argument("--species-name", default="Homo sapiens", help="Nombre de especie para la tabla/carpetas")
    parser.add_argument("--exact-taxonomy-name", default=None,
                         help="Filtra client-side por physical_object_taxonomy_name exacto "
                              "(útil con --taxon Homo para especies sin epíteto GBIF propio, p.ej. 'Homo sp.')")
    parser.add_argument("--out", default=None, help="Ruta del CSV de salida (por defecto catalog/<especie>_media.csv)")
    args = parser.parse_args()

    slug = slugify(args.species_name)
    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "catalog" / f"{slug}_media.csv"

    existing = load_existing(out_path)
    rows = build_rows(args.species_name, args.taxon, existing, args.exact_taxonomy_name)
    select_for_download(rows)
    write_csv(out_path, rows)
    print_summary(rows, args.species_name)
    print(f"\nCSV escrito en: {out_path}")


if __name__ == "__main__":
    main()
