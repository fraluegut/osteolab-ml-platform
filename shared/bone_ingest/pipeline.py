"""Orquesta el ingest completo: URL de Sketchfab -> descarga -> separación
por objeto (Blender, `blender_split_render.py`) -> sugerencia de grupo por
nombre (`taxonomy.py`) -> cola de revisión en `data/bone_ingest/queue.json`.

Corre en un hilo aparte (lanzado desde `serving/backend/main.py`) porque
tarda minutos — descarga + decenas/cientos de renders — y no es apto para
bloquear una petición HTTP. Cola persistida en disco como JSON plano: es una
herramienta interna de un solo usuario, no hace falta una base de datos para
esto.

Cada item pasa por uno de dos caminos:
- nombre de objeto reconocido (`taxonomy.guess_bone_group`) -> status
  "suggested", visible en "Ampliar Base de Datos" con sugerencia + botón de
  aceptar en bloque.
- nombre ambiguo/desconocido -> status "needs_review", entra en la cola de
  "Identificación Manual" (estilo Tinder).
"""
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

from shared.bone_ingest.sketchfab_client import (
    SketchfabError,
    download_and_extract,
    extract_uid,
    fetch_model_info,
    find_scene_file,
)
from shared.bone_ingest.taxonomy import BONE_GROUPS, guess_bone_group

REPO_ROOT = Path(__file__).resolve().parents[2]
INGEST_DIR = REPO_ROOT / "data" / "bone_ingest"
QUEUE_PATH = INGEST_DIR / "queue.json"
BLENDER_BIN = "/opt/blender/blender"
SPLIT_SCRIPT = Path(__file__).resolve().parent / "blender_split_render.py"

_lock = threading.Lock()


def _load_queue() -> dict:
    if not QUEUE_PATH.exists():
        return {"jobs": {}}
    return json.loads(QUEUE_PATH.read_text())


def _save_queue(data: dict):
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _update_job(job_id: str, **fields):
    with _lock:
        data = _load_queue()
        data["jobs"][job_id].update(fields)
        _save_queue(data)


def submit_job(url_or_uid: str) -> str:
    uid = extract_uid(url_or_uid)
    job_id = uuid.uuid4().hex[:12]

    with _lock:
        data = _load_queue()
        data["jobs"][job_id] = {
            "job_id": job_id,
            "source_url": url_or_uid,
            "uid": uid,
            "status": "queued",
            "submitted_at": time.time(),
            "items": [],
        }
        _save_queue(data)

    threading.Thread(target=_run_job, args=(job_id, uid), daemon=True).start()
    return job_id


def _run_job(job_id: str, uid: str):
    job_dir = INGEST_DIR / job_id
    try:
        _update_job(job_id, status="fetching_info")
        info = fetch_model_info(uid)
        images = (info.get("thumbnails") or {}).get("images") or []
        _update_job(
            job_id,
            status="downloading",
            model_name=info.get("name"),
            license=(info.get("license") or {}).get("label"),
            author=(info.get("user") or {}).get("displayName"),
            thumbnail_url=images[-1]["url"] if images else None,
        )

        source_dir = job_dir / "source"
        download_and_extract(uid, source_dir)
        scene_path = find_scene_file(source_dir)

        _update_job(job_id, status="splitting")
        thumbs_dir = job_dir / "thumbnails"
        result = subprocess.run(
            [
                BLENDER_BIN, "--background", "--python", str(SPLIT_SCRIPT), "--",
                "--scene", str(scene_path), "--out-dir", str(thumbs_dir), "--resolution", "256",
            ],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Blender falló (código {result.returncode}): {result.stderr[-2000:]}")

        parts = json.loads((thumbs_dir / "parts.json").read_text())
        items = []
        for i, part in enumerate(parts):
            if "error" in part:
                continue
            suggested = guess_bone_group(part["object_name"])
            items.append({
                "item_id": f"{job_id}_{i}",
                "object_name": part["object_name"],
                "vertex_count": part["vertex_count"],
                "dimensions_m": part.get("dimensions_m"),
                "thumbnail_path": f"/ingest/thumbnail/{job_id}/{part['thumbnail']}",
                "suggested_bone_group": suggested,
                "species": "Homo sapiens",
                "status": "suggested" if suggested else "needs_review",
                "final_bone_group": None,
            })

        with _lock:
            data = _load_queue()
            data["jobs"][job_id]["items"] = items
            data["jobs"][job_id]["status"] = "done"
            data["jobs"][job_id]["completed_at"] = time.time()
            _save_queue(data)
    except SketchfabError as exc:
        _update_job(job_id, status="error", error=str(exc))
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))


def get_queue() -> dict:
    return _load_queue()


def get_review_items() -> list[dict]:
    """Items con status "needs_review" de todos los jobs — la cola que
    consume "Identificación Manual"."""
    data = _load_queue()
    items = []
    for job in data["jobs"].values():
        for item in job.get("items", []):
            if item["status"] == "needs_review":
                items.append({**item, "job_id": job["job_id"], "model_name": job.get("model_name")})
    return items


def _find_item(data: dict, item_id: str) -> tuple[dict, dict]:
    job_id = item_id.rsplit("_", 1)[0]
    job = data["jobs"].get(job_id)
    if not job:
        raise ValueError(f"job {job_id} no existe (item {item_id})")
    item = next((it for it in job["items"] if it["item_id"] == item_id), None)
    if not item:
        raise ValueError(f"item {item_id} no existe")
    return job, item


def decide_item(item_id: str, accept: bool, bone_group: str | None, species: str | None) -> dict:
    if accept and bone_group not in BONE_GROUPS:
        raise ValueError(f"bone_group inválido: {bone_group!r} (debe ser uno de {BONE_GROUPS})")
    with _lock:
        data = _load_queue()
        _job, item = _find_item(data, item_id)
        item["status"] = "accepted" if accept else "rejected"
        item["final_bone_group"] = bone_group if accept else None
        if species:
            item["species"] = species
        item["reviewed_at"] = time.time()
        _save_queue(data)
        return item


def flag_for_review(item_id: str) -> dict:
    """Manda un item "suggested" a la cola de Identificación Manual en vez de
    aceptar la sugerencia automática a ciegas."""
    with _lock:
        data = _load_queue()
        _job, item = _find_item(data, item_id)
        item["status"] = "needs_review"
        _save_queue(data)
        return item


def bulk_accept_suggested(job_id: str) -> int:
    with _lock:
        data = _load_queue()
        job = data["jobs"].get(job_id)
        if not job:
            raise ValueError(f"job {job_id} no existe")
        n = 0
        for item in job["items"]:
            if item["status"] == "suggested":
                item["status"] = "accepted"
                item["final_bone_group"] = item["suggested_bone_group"]
                item["reviewed_at"] = time.time()
                n += 1
        _save_queue(data)
        return n
