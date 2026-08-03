"""Unifica los 3 catálogos de especímenes 3D curados en `base_datos_osea/catalog/`
(Sketchfab, MorphoSource — `homo_*_media.csv` —, Smithsonian) en una sola
lista, con un render de ejemplo por espécimen. Pensado para la sección
"Catálogo 3D" de la web (`GET /catalog/specimens`) — no toca nada del
pipeline de entrenamiento, es solo lectura para mostrar de dónde sale cada
fila del dataset.

Cada catálogo tiene su propio esquema de columnas (ver los CSVs), así que
cada `_rows_from_*` sabe leer el suyo y todos convergen a un dict con las
mismas claves.
"""
import csv
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = REPO_ROOT / "base_datos_osea" / "catalog"
RENDERS_DIR = REPO_ROOT / "base_datos_osea" / "renders"
REAL_PHOTOS_ACCEPTED_MANIFEST = REPO_ROOT / "data" / "real_photos_accepted" / "manifest.csv"
REAL_WORLD_BENCHMARK_MANIFEST = REPO_ROOT / "data" / "real_world_benchmark" / "manifest.csv"

# Objetivos de trabajo (ver base_datos_osea/HISTORIAL.md 2026-08-03): 15 especímenes
# 3D/grupo ya es la referencia usada en la web desde antes; 40 fotos reales de
# entrenamiento/grupo es la cifra nueva discutida esa sesión (punto medio del rango
# 30-50 recomendado para poder reservar un held-out de examen con más de 1
# foto/clase sin dejar casi nada para entrenar).
SPECIMENS_3D_TARGET = 15
REAL_PHOTOS_TARGET = 40

# Mismo destino que shared/cv_extractor/build_dataset.py::BONE_GROUPS — se
# duplica aquí (catálogo aparte, sin acoplar módulos) igual que el resto del
# proyecto ya hace entre shared/*.
BONE_GROUPS = {
    "cranium": "cranio",
    "mandible": "mandibula_maxilar",
    "maxilla": "mandibula_maxilar",
    "scapula": "hueso_plano",
    "clavicle": "hueso_largo",
    "humerus": "hueso_largo",
    "radius": "hueso_largo",
    "ulna": "hueso_largo",
    "femur": "hueso_largo",
    "tibia": "hueso_largo",
    "fibula": "hueso_largo",
    "patella": "hueso_pequeno",
    "pelvis": "pelvis",
    "sacrum": "sacro",
    "vertebra": "vertebra",
    "rib": "costilla",
    "metacarpal_1": "hueso_pequeno",
    "metatarsal_1": "hueso_pequeno",
    "proximal_phalanx_1": "hueso_pequeno",
    "trapezium": "hueso_pequeno",
    "sesamoid": "hueso_pequeno",
}


def _species_slug(species: str) -> str:
    return species.strip().lower().replace(" ", "_")


def _find_render(species_slug: str, bone_canonical: str, specimen: str) -> str | None:
    """Primera vista limpia disponible para este espécimen, o None si no se
    renderizó (algunos especímenes descartados/fragmentados no llegaron a
    esa etapa)."""
    specimen_dir = RENDERS_DIR / species_slug / bone_canonical / specimen
    if not specimen_dir.is_dir():
        return None
    views = sorted(specimen_dir.glob("*_view*.png"))
    if not views:
        return None
    # relativo a RENDERS_DIR: el backend monta esa carpeta en /catalog-renders/
    return str(views[0].relative_to(RENDERS_DIR))


def _rows_from_sketchfab() -> list[dict]:
    path = CATALOG_DIR / "sketchfab_media.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["download_status"] == "done"]

    out = []
    for r in rows:
        species_slug = _species_slug(r["species"])
        out.append({
            "source": "Sketchfab",
            "species": r["species"],
            "bone_canonical": r["bone_canonical"],
            "bone_group": BONE_GROUPS.get(r["bone_canonical"], "sin_grupo"),
            "specimen": f"sketchfab_{r['uid']}",
            "external_url": r["viewer_url"],
            "license": r["license"],
            "creator": r["creator"],
            "render_path": _find_render(species_slug, r["bone_canonical"], f"sketchfab_{r['uid']}"),
        })
    return out


def _rows_from_morphosource(filename: str) -> list[dict]:
    path = CATALOG_DIR / filename
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["download_status"] == "done"]

    out = []
    for r in rows:
        species_slug = _species_slug(r["species"])
        out.append({
            "source": "MorphoSource",
            "species": r["species"],
            "bone_canonical": r["bone_canonical"],
            "bone_group": BONE_GROUPS.get(r["bone_canonical"], "sin_grupo"),
            "specimen": r["media_id"],
            "external_url": r["website_url"],
            "license": r["license"],
            "creator": r.get("ip_holder") or r.get("creator") or "",
            "render_path": _find_render(species_slug, r["bone_canonical"], r["media_id"]),
        })
    return out


def _rows_from_smithsonian() -> list[dict]:
    path = CATALOG_DIR / "smithsonian_primates_media.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["download_status"] == "done"]

    out = []
    for r in rows:
        species_slug = _species_slug(r["species"])
        out.append({
            "source": "Smithsonian",
            "species": r["species"],
            "bone_canonical": r["bone_canonical"],
            "bone_group": BONE_GROUPS.get(r["bone_canonical"], "sin_grupo"),
            "specimen": r["record_id"],
            "external_url": r["record_link"],
            "license": r["license"],
            "creator": "Smithsonian NMNH",
            "render_path": _find_render(species_slug, r["bone_canonical"], r["record_id"]),
        })
    return out


@lru_cache(maxsize=1)
def list_specimens() -> list[dict]:
    """Los 122 especímenes 3D curados y descargados (`download_status=done`
    en algún catálogo) — no incluye las fotos reales 2D, esas viven en
    `data/real_photos_accepted/manifest.csv` (otro origen, sin malla 3D)."""
    specimens = (
        _rows_from_sketchfab()
        + _rows_from_morphosource("homo_sapiens_media.csv")
        + _rows_from_morphosource("homo_naledi_media.csv")
        + _rows_from_morphosource("homo_neanderthalensis_media.csv")
        + _rows_from_morphosource("homo_sp_media.csv")
        + _rows_from_smithsonian()
    )
    specimens.sort(key=lambda s: (s["bone_group"], s["species"], s["specimen"]))
    return specimens


def _count_by_group(manifest_path: Path) -> dict[str, int]:
    """Cuenta filas por `bone_group` en un manifest.csv de fotos reales
    (`data/real_photos_accepted/` o `data/real_world_benchmark/`) — ambos usan
    ya el mismo vocabulario de grupo que `BONE_GROUPS`, así que no hace falta
    traducir nada, solo contar."""
    if not manifest_path.exists():
        return {}
    counts: dict[str, int] = {}
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            group = row["bone_group"]
            counts[group] = counts.get(group, 0) + 1
    return counts


@lru_cache(maxsize=1)
def list_group_balance() -> list[dict]:
    """Para la sección "Balance de datos" de la web: por cada uno de los 9
    grupos, especímenes 3D (+especies distintas) y fotos reales
    (entrenamiento/banco de examen), con el objetivo de trabajo y cuántos
    faltan — ver `base_datos_osea/HISTORIAL.md` 2026-08-03 (por qué 40 fotos
    reales de entrenamiento por grupo, no una cifra arbitraria)."""
    specimens_by_group: dict[str, list[dict]] = {}
    for s in list_specimens():
        specimens_by_group.setdefault(s["bone_group"], []).append(s)

    training_counts = _count_by_group(REAL_PHOTOS_ACCEPTED_MANIFEST)
    benchmark_counts = _count_by_group(REAL_WORLD_BENCHMARK_MANIFEST)

    all_groups = sorted(set(BONE_GROUPS.values()))
    rows = []
    for group in all_groups:
        rows_3d = specimens_by_group.get(group, [])
        n_3d = len(rows_3d)
        n_species = len({r["species"] for r in rows_3d})
        n_training = training_counts.get(group, 0)
        n_benchmark = benchmark_counts.get(group, 0)
        rows.append({
            "bone_group": group,
            "specimens_3d": n_3d,
            "species_3d": n_species,
            "specimens_3d_target": SPECIMENS_3D_TARGET,
            "specimens_3d_missing": max(0, SPECIMENS_3D_TARGET - n_3d),
            "real_photos_training": n_training,
            "real_photos_benchmark": n_benchmark,
            "real_photos_target": REAL_PHOTOS_TARGET,
            "real_photos_missing": max(0, REAL_PHOTOS_TARGET - n_training),
        })
    rows.sort(key=lambda r: r["real_photos_training"])
    return rows
