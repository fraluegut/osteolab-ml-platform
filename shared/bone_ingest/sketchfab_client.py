"""Descarga un modelo de Sketchfab a partir de su URL — generaliza
`base_datos_osea/scripts/sketchfab.py::download_catalog()` (que solo
descarga uids ya curados a mano en un CSV) para aceptar cualquier URL que
pegue el usuario en "Ampliar Base de Datos".

Usa el mismo `SKETCHFAB_API_TOKEN` que el resto del proyecto (Settings >
Password & API en una cuenta gratuita de Sketchfab). Prioriza el archivo
`source` (el original subido por el autor — más probable que conserve la
escena con objetos separados) sobre `gltf`/`glb` (Sketchfab suele fusionar
todo en una malla al optimizar para el visor).
"""
import os
import re
import zipfile
from pathlib import Path

import requests

API_URL = "https://api.sketchfab.com/v3"
_UID_RE = re.compile(r"([0-9a-f]{32})")
_ENV_PATH = Path(__file__).resolve().parents[2] / "base_datos_osea" / ".env"


class SketchfabError(Exception):
    pass


def _load_env_file():
    """`base_datos_osea/.env` trae el token de Sketchfab que ya usa el
    downloader curado — sin dotenv como dependencia nueva, solo KEY=VALUE."""
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def extract_uid(url_or_uid: str) -> str:
    match = _UID_RE.search(url_or_uid.strip())
    if not match:
        raise SketchfabError(f"No se encontró un uid de Sketchfab (32 hex) en: {url_or_uid!r}")
    return match.group(1)


def _headers() -> dict:
    _load_env_file()
    token = os.environ.get("SKETCHFAB_API_TOKEN")
    if not token:
        raise SketchfabError("Falta SKETCHFAB_API_TOKEN en el entorno (ver base_datos_osea/.env).")
    return {"Authorization": f"Token {token}"}


def fetch_model_info(uid: str) -> dict:
    resp = requests.get(f"{API_URL}/models/{uid}", timeout=30)
    if resp.status_code == 404:
        raise SketchfabError(f"Modelo {uid} no existe o no es público.")
    resp.raise_for_status()
    return resp.json()


def download_and_extract(uid: str, dest_dir: Path) -> Path:
    """Descarga el mejor archivo disponible (source > gltf > glb) y lo
    descomprime en `dest_dir`. Devuelve `dest_dir`."""
    resp = requests.get(f"{API_URL}/models/{uid}/download", headers=_headers(), timeout=30)
    if resp.status_code == 403:
        raise SketchfabError(
            f"Modelo {uid} no descargable con este token (el autor no lo marcó como "
            "descargable, o requiere estar logueado como el propio autor)."
        )
    resp.raise_for_status()
    archives = resp.json()
    archive = archives.get("source") or archives.get("gltf") or archives.get("glb")
    if not archive:
        raise SketchfabError(f"Sin archivo descargable para {uid} (opciones: {list(archives.keys())})")

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = archive["url"].split("?")[0].rsplit("/", 1)[-1]
    zip_path = dest_dir / filename

    with requests.get(archive["url"], stream=True, timeout=120) as file_resp:
        file_resp.raise_for_status()
        with zip_path.open("wb") as fh:
            for chunk in file_resp.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)

    if zip_path.suffix == ".zip":
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)

    return dest_dir


_SCENE_EXTS = (".fbx", ".glb", ".gltf", ".obj", ".blend")


def find_scene_file(dest_dir: Path) -> Path:
    """Busca el archivo de escena principal entre lo descomprimido —
    prioriza los formatos que preservan mejor la jerarquía de objetos."""
    candidates = [p for p in dest_dir.rglob("*") if p.suffix.lower() in _SCENE_EXTS]
    if not candidates:
        raise SketchfabError(f"No se encontró ningún archivo de escena en {dest_dir}")
    priority = {ext: i for i, ext in enumerate(_SCENE_EXTS)}
    candidates.sort(key=lambda p: priority.get(p.suffix.lower(), 99))
    return candidates[0]
