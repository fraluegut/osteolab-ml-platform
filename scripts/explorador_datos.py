"""Explorador web de todos los datos visuales del proyecto: mallas 3D, renders
de cada generación, fotos reales y recortes segmentados. Se navega como
carpetas, con miniaturas, visor de imágenes y visor 3D (GLB/OBJ).

Uso:
    python scripts/explorador_datos.py            # http://localhost:8510
    python scripts/explorador_datos.py --port 8511

Solo lectura: no mueve ni borra nada. Las miniaturas se cachean en
~/.cache/osteolab-explorador para que las carpetas grandes (5k recortes)
carguen rápido la segunda vez.
"""
import argparse
import hashlib
import io
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Mapea cada raiz "Expertos stage2" al run_name (param logueado en train_cnn.py)
# que entrena ese experto en concreto -- para poder listar sus reentrenamientos
# en MLflow (nombre, fecha, metricas) desde el propio explorador.
ROOT_MLFLOW_RUN_NAME = {
    "stage2_familia_cranio": "stage2_familia_cranio",
    "stage2_taxon_cranio": "stage2_taxon_cranio",
}
RUN_INFO_METRICS = ["best_val_acc", "val_acc", "val_worst_class_recall"]
RUN_INFO_PARAMS = ["backbone", "n_train_rows", "model_type"]

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "explorador_static"
THUMB_DIR = Path.home() / ".cache" / "osteolab-explorador"
THUMB_SIZE = 320

# Raíces visibles, primero las imágenes resultantes y al final el material
# auxiliar. Las 4 generaciones de render son las mismas que en la charla.
# (clave, sección, nombre, etiqueta, ruta relativa al repo, descripción)
# etiqueta: "actual" = lo que usa el modelo hoy, "descartado", o "".
ROOTS = [
    ("stage2_familia_cranio", "Expertos stage2", "Cráneo · familia (versión vieja)", "descartado",
     "data/stage2_familia_cranio_viewer",
     "Iteracion anterior del experto de craneo (cánido/suido/équido/humano) -- NO esta en "
     "produccion, predict_two_stage.py no lo referencia. Sustituido por 'Cráneo · taxón' "
     "(stage2_taxon_cranio), el unico experto de etapa 3 para craneo que corre hoy."),
    ("stage2_taxon_cranio", "Expertos stage2", "Cráneo · taxón", "actual",
     "data/stage2_taxon_cranio_viewer",
     "Exactamente lo que entrena/mide este experto -- original + recorte segmentado "
     "(lo que de verdad ve la CNN), separado train/test y por taxón. Regenerar con "
     "scripts/build_stage2_taxon_cranio_viewer.py tras cambiar el dataset."),
    ("render_limpio", "Renders sintéticos · 4 generaciones", "1 · Limpio", "",
     "base_datos_osea/renders",
     "Generación 1: silueta sobre fondo blanco, material mate."),
    ("render_compuesto", "Renders sintéticos · 4 generaciones", "2 · Ruido + fondo real", "",
     "base_datos_osea/renders_noisy_composited",
     "Generación 2: material de hueso aleatorio y sombra, compuesto sobre fondos reales."),
    ("render_bproc", "Renders sintéticos · 4 generaciones", "3 · BlenderProc", "",
     "base_datos_osea/renders_noisy_bproc",
     "Generación 3: HDRI real, suciedad procedural y máscara exacta."),
    ("render_lab", "Renders sintéticos · 4 generaciones", "4 · Laboratorio", "actual",
     "base_datos_osea/renders_lab",
     "Generación 4: mesa, hueso que cae con física, luz de estudio. El sintético que usa el modelo hoy."),
    ("fotos_aceptadas", "Fotos reales", "Aceptadas (entrenan)", "actual",
     "data/real_photos_accepted",
     "Fotos curadas a mano que entran en entrenamiento."),
    ("fotos_examen", "Fotos reales", "Examen fijo", "",
     "data/real_world_benchmark",
     "Banco de evaluación que nunca entrena."),
    ("fotos_revision", "Fotos reales", "Pendientes de revisar", "",
     "data/real_photos_review",
     "Candidatas del backlog de la app de revisión. Todavía no entrenan."),
    ("fotos_entrenamiento", "Fotos reales", "Primer lote", "",
     "base_datos_osea/Fotos entrenamiento",
     "Primer lote de fotos reales del proyecto."),
    ("recortes", "Lo que ve el modelo", "Recortes · fondo gris", "actual",
     "data/segmented_crops",
     "El hueso recortado por Grounded-SAM sobre gris: la entrada real de la CNN."),
    ("recortes_negro", "Lo que ve el modelo", "Recortes · fondo negro", "descartado",
     "data/segmented_crops_blackbg",
     "Variante con fondo negro. Empate técnico con el gris, no se adoptó."),
    ("mallas", "Mallas 3D", "Mallas por especie y hueso", "",
     "base_datos_osea/data/meshes",
     "Mallas de MorphoSource, Smithsonian y Sketchfab: el origen de todos los renders."),
    ("ingesta", "Material auxiliar", "Descargas en bruto de Sketchfab", "",
     "data/bone_ingest",
     "Descargas tal cual llegan, antes de normalizarlas en «Mallas 3D»."),
    ("render_ruido", "Material auxiliar", "Gen. 2 sin fondo (paso intermedio)", "",
     "base_datos_osea/renders_noisy",
     "Renders de la generación 2 antes de componerlos sobre fondo real."),
    ("render_lab_test", "Material auxiliar", "Pruebas de la escena de laboratorio", "",
     "base_datos_osea/renders_lab_test",
     "Lotes de prueba mientras se diseñaba la generación 4."),
    ("render_lab_luces", "Material auxiliar", "Galería de presets de luz", "",
     "base_datos_osea/renders_lab_lightgallery",
     "Un mismo hueso con cada preset de luz de la generación 4."),
    ("hdris", "Material auxiliar", "HDRIs de iluminación", "",
     "base_datos_osea/hdris",
     "Entornos de Poly Haven que usan las generaciones 3 y 4."),
]
AUX_SECTION = "Material auxiliar"
ROOTS_BY_KEY = {key: (BASE_DIR / rel).resolve() for key, _, _, _, rel, _ in ROOTS}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
HDR_EXT = {".hdr", ".exr"}
MODEL_EXT = {".glb", ".gltf", ".obj", ".ply", ".stl"}
TEXT_EXT = {".csv", ".json", ".txt", ".md", ".yaml", ".yml", ".mtl", ".log"}


def file_kind(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in HDR_EXT:
        return "hdr"
    if ext in MODEL_EXT:
        return "model"
    if ext in TEXT_EXT:
        return "text"
    return "other"


def resolve(root: str, rel: str) -> Path:
    """Ruta dentro de una raíz permitida. Rechaza `..` y enlaces que salgan de ella."""
    if root not in ROOTS_BY_KEY:
        raise HTTPException(404, f"Raíz desconocida: {root}")
    base = ROOTS_BY_KEY[root]
    target = (base / rel).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(403, "Ruta fuera de la raíz")
    if not target.exists():
        raise HTTPException(404, f"No existe: {rel}")
    return target


@lru_cache(maxsize=None)
def count_files(path: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(path))


app = FastAPI(title="OsteoLab · explorador de datos")


@app.get("/api/roots")
def list_roots():
    return [
        {"key": key, "section": section, "auxiliary": section == AUX_SECTION, "name": name,
         "tag": tag, "path": rel, "description": desc,
         "exists": ROOTS_BY_KEY[key].exists(),
         "files": count_files(ROOTS_BY_KEY[key]) if ROOTS_BY_KEY[key].exists() else 0,
         "mlflow_run_name": ROOT_MLFLOW_RUN_NAME.get(key)}
        for key, section, name, tag, rel, desc in ROOTS
    ]


@app.get("/api/runs")
def list_runs(root: str):
    """Reentrenamientos en MLflow del experto detras de esta raiz -- nombre,
    fecha y metricas clave, para saber que version del dataset esta viendo el
    visor (siempre la mas reciente en disco, no una foto congelada por run)."""
    run_name = ROOT_MLFLOW_RUN_NAME.get(root)
    if not run_name:
        return {"run_name": None, "runs": []}
    try:
        exps = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
                             params={"max_results": 50}, timeout=5).json().get("experiments", [])
        exp_ids = [e["experiment_id"] for e in exps]
        if not exp_ids:
            return {"run_name": run_name, "runs": []}
        body = {
            "experiment_ids": exp_ids,
            "filter": f"params.run_name = '{run_name}'",
            "max_results": 50,
            "order_by": ["attribute.start_time DESC"],
        }
        runs = requests.post(f"{MLFLOW_URL}/api/2.0/mlflow/runs/search", json=body, timeout=5).json().get("runs", [])
    except requests.RequestException:
        return {"run_name": run_name, "runs": [], "error": "MLflow no responde"}

    out = []
    for r in runs:
        info = r["info"]
        metrics = {m["key"]: m["value"] for m in r.get("data", {}).get("metrics", [])}
        params = {p["key"]: p["value"] for p in r.get("data", {}).get("params", [])}
        out.append({
            "run_id": info["run_id"],
            "mlflow_name": info.get("run_name", info["run_id"][:8]),
            "date": datetime.fromtimestamp(info["start_time"] / 1000).strftime("%Y-%m-%d %H:%M"),
            "metrics": {k: round(metrics[k], 4) for k in RUN_INFO_METRICS if k in metrics},
            "params": {k: params[k] for k in RUN_INFO_PARAMS if k in params},
        })
    return {"run_name": run_name, "runs": out}


@app.get("/api/ls")
def list_dir(root: str, path: str = ""):
    target = resolve(root, path)
    if not target.is_dir():
        raise HTTPException(400, "No es una carpeta")
    dirs, files = [], []
    for entry in sorted(os.scandir(target), key=lambda e: e.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            dirs.append({"name": entry.name, "files": count_files(Path(entry.path))})
        else:
            files.append({"name": entry.name, "kind": file_kind(entry.name), "size": entry.stat().st_size})
    return {"root": root, "path": path, "dirs": dirs, "files": files}


@app.get("/raw")
def raw_file(root: str, path: str):
    target = resolve(root, path)
    if not target.is_file():
        raise HTTPException(400, "No es un fichero")
    return FileResponse(target)


def _load_preview(target: Path) -> Image.Image:
    if target.suffix.lower() in HDR_EXT:
        # HDR lineal → tonemap simple para poder verlo en el navegador
        hdr = cv2.imread(str(target), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdr is None:
            raise HTTPException(415, "No se pudo leer el HDR")
        ldr = np.clip((hdr / (1.0 + hdr)) ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB))
    return Image.open(target)


@app.get("/thumb")
def thumbnail(root: str, path: str, size: int = Query(THUMB_SIZE, ge=64, le=1600)):
    target = resolve(root, path)
    if file_kind(target.name) not in ("image", "hdr"):
        raise HTTPException(400, "Sin miniatura para este tipo")
    stat = target.stat()
    key = hashlib.sha1(f"{target}|{stat.st_mtime_ns}|{size}".encode()).hexdigest()
    cached = THUMB_DIR / f"{key}.jpg"
    if not cached.exists():
        img = _load_preview(target)
        img.thumbnail((size, size))
        if img.mode in ("RGBA", "LA", "P"):
            # transparencia sobre gris medio, el mismo fondo que usa la CNN
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (128, 128, 128))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=82)
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(buf.getvalue())
    return FileResponse(cached, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def index():
    # sin caché: si no, el navegador sigue usando la versión anterior de la página
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8510)
    args = parser.parse_args()
    print(f"Explorador en http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
