from pathlib import Path

import csv
import hashlib
import io
import json

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shared.inference.predict_cnn import explain_gradcam
from shared.inference.predict_two_stage import classify_image_two_stage
from shared.inference.predict_geometric import classify_image, get_reference_points
from shared.bone_filter.predict import is_bone
from shared.clip_filter.predict import classify_image as clip_classify
from shared.cv_extractor.extract import extract_features
from shared.vlm_describe.predict import describe_image as qwen_describe_image
from shared.bone_ingest import pipeline as bone_ingest
from shared.catalog import specimens as catalog_specimens

app = FastAPI(title="OsteoLab", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
PHOTO_HASH_INDEX_PATH = REPO_ROOT / "data/processed/photo_hash_index.json"
_photo_hash_index_cache = None


def _load_photo_hash_index() -> dict:
    """Carga data/processed/photo_hash_index.json (sha256 -> dataset/split),
    construido con scripts/build_photo_hash_index.py. Cacheado en memoria --
    el índice solo cambia cuando se reconstruye a mano, no en cada request.
    """
    global _photo_hash_index_cache
    if _photo_hash_index_cache is None:
        if PHOTO_HASH_INDEX_PATH.exists():
            with open(PHOTO_HASH_INDEX_PATH) as f:
                _photo_hash_index_cache = json.load(f)
        else:
            _photo_hash_index_cache = {}
    return _photo_hash_index_cache


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "api_version": "0.2.0",
        "model_source": classify_image.__module__,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Mide la imagen con OpenCV y clasifica el grupo morfológico de hueso
    (cráneo, hueso largo, vértebra...) con el modelo entrenado sobre esas
    medidas — ver `training/train_geometric.py`. Sustituye al antiguo
    `/predict` (regresión logística sobre píxeles, 3 clases, retirado por no
    tener relación con el catálogo de huesos real).

    Devuelve también el punto en el espacio PCA de referencia (2D, ajustado
    sobre las mismas features) para poder dibujar la imagen subida junto a
    las 1300+ vistas ya conocidas — ver `/pca/reference`.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return classify_image(file.file)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hybrid/classify")
async def hybrid_classification(file: UploadFile = File(...)):
    """Clasificación híbrida usando el modelo CNN híbrido entrenado.

    Este endpoint usa la segmentación de la imagen, extrae las features
    geométricas y las combina con el embedding del backbone CNN para obtener
    una predicción más robusta basada en el mejor modelo disponible.

    Cascada de 2 etapas (ver HISTORIAL.md 2026-08-14 y
    `shared/inference/predict_two_stage.py`): además del grupo morfológico
    (`prediction`), la respuesta trae `fine_bone` — el hueso concreto DENTRO
    del grupo (ej. fémur/húmero dentro de hueso_largo, astrágalo dentro de
    hueso_pequeño) cuando ese grupo tiene un clasificador de etapa 2
    entrenado; `None` en el resto de grupos (aún no hay etapa 2 para ellos).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return classify_image_two_stage(file.file)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/leak-check")
async def leak_check(file: UploadFile = File(...)):
    """¿Esta foto ya la usamos en entrenamiento o examen? Hash sha256 del
    fichero subido contra `data/processed/photo_hash_index.json` (indexa
    `real_photos_accepted` + `real_world_benchmark`, ver
    scripts/build_photo_hash_index.py) -- identifica la foto aunque cambie
    de nombre, a diferencia de comparar por filename.

    Pedido del usuario al desplegar el frontend de demo (2026-10-27): si
    alguien sube una foto del propio banco de entrenamiento/examen sin darse
    cuenta, que la UI lo diga en vez de aparentar una prueba "nueva".
    """
    raw = await file.read()
    digest = hashlib.sha256(raw).hexdigest()
    match = _load_photo_hash_index().get(digest)
    if match is None:
        return {"seen_before": False}
    return {"seen_before": True, **match}


@app.post("/explain/gradcam")
async def explain_gradcam_endpoint(file: UploadFile = File(...), model_type: str = Form("hybrid")):
    """Grad-CAM del backbone ResNet18 (CNN/híbrido): qué región de la imagen
    pesó más en la predicción. Ver shared/inference/predict_cnn.py::explain_gradcam.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    if model_type not in ("cnn_only", "hybrid"):
        raise HTTPException(status_code=400, detail="model_type debe ser 'cnn_only' o 'hybrid'")

    try:
        return explain_gradcam(file.file, model_type=model_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/describe/qwen")
async def describe_qwen(file: UploadFile = File(...)):
    """Descripción libre + sugerencia de grupo morfológico con Qwen2.5-VL 7B
    (VLM zero-shot, en lenguaje natural). Corre en paralelo con los modelos
    geométrico e híbrido — no es un filtro previo, es una tercera opinión
    para comparar contra ellos; ver shared/vlm_describe/.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return qwen_describe_image(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pca/reference")
def pca_reference():
    """Coordenadas PCA de las vistas usadas para entrenar (una por imagen),
    con su especie/hueso/grupo — para dibujar de fondo en el scatter y
    comparar visualmente la imagen nueva contra lo ya conocido."""
    try:
        return {"points": get_reference_points()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/filter/clip")
async def filter_clip(file: UploadFile = File(...)):
    """Filtro CLIP zero-shot: ¿es un hueso? y, si lo es, sugerencia de tipo.

    Primera capa del pipeline, antes del filtro ResNet18 (`/filter/is-bone`).
    No requiere entrenamiento ni dataset propio; ver shared/clip_filter/.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return clip_classify(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/extract")
async def features_extract(
    file: UploadFile = File(...),
    largo_cm: str | None = Form(None),
    ancho_cm: str | None = Form(None),
    profundo_cm: str | None = Form(None),
    tipo_hueso: str | None = Form(None),
    peso_g: str | None = Form(None),
    color: str | None = Form(None),
    estado_conservacion: str | None = Form(None),
    procedencia: str | None = Form(None),
    notas: str | None = Form(None),
):
    """Localiza el hueso en la imagen con OpenCV y extrae medidas relativas
    (ratio de aspecto, proporción de píxeles ocupados, etc.), ver
    shared/cv_extractor/. Se llama tras confirmar con el filtro CLIP que la
    imagen contiene un hueso.

    El contexto opcional que aporte el usuario (medidas reales, tipo, peso...)
    se devuelve junto a las features pero sin mezclarse en el cálculo: los
    campos vacíos se descartan y no se usan para nada.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    context = {
        "largo_cm": largo_cm,
        "ancho_cm": ancho_cm,
        "profundo_cm": profundo_cm,
        "tipo_hueso": tipo_hueso,
        "peso_g": peso_g,
        "color": color,
        "estado_conservacion": estado_conservacion,
        "procedencia": procedencia,
        "notas": notas,
    }
    context = {k: v for k, v in context.items() if v}

    try:
        cv_features = extract_features(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"cv_features": cv_features, "context": context}


@app.post("/filter/is-bone")
async def filter_is_bone(file: UploadFile = File(...)):
    """Filtro aislado (ResNet18): ¿la imagen es un hueso o no?

    Independiente del clasificador multi-clase de /predict; ver shared/bone_filter/.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return is_bone(file.file)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class IngestSubmitRequest(BaseModel):
    url: str


class IngestDecideRequest(BaseModel):
    item_id: str
    accept: bool
    bone_group: str | None = None
    species: str | None = None


@app.post("/ingest/submit")
def ingest_submit(body: IngestSubmitRequest):
    """Recibe una URL (o uid) de Sketchfab y lanza en un hilo aparte:
    descarga -> separar por objeto (Blender) -> sugerir grupo por nombre.
    Ver shared/bone_ingest/pipeline.py::submit_job.
    """
    try:
        job_id = bone_ingest.submit_job(body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"job_id": job_id}


@app.get("/ingest/jobs")
def ingest_jobs():
    """Todos los jobs de ingest (para "Ampliar Base de Datos"), con sus items."""
    return bone_ingest.get_queue()["jobs"]


@app.get("/ingest/jobs/{job_id}")
def ingest_job_detail(job_id: str):
    jobs = bone_ingest.get_queue()["jobs"]
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} no existe")
    return job


@app.post("/ingest/jobs/{job_id}/bulk-accept")
def ingest_bulk_accept(job_id: str):
    """Acepta en bloque todos los items "suggested" (nombre reconocido con
    confianza) de un job — no toca los que ya están en "needs_review"."""
    try:
        n = bone_ingest.bulk_accept_suggested(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"accepted": n}


@app.get("/ingest/thumbnail/{job_id}/{filename}")
def ingest_thumbnail(job_id: str, filename: str):
    path = bone_ingest.INGEST_DIR / job_id / "thumbnails" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Miniatura no encontrada")
    return FileResponse(path)


@app.get("/ingest/review-queue")
def ingest_review_queue():
    """Items con nombre ambiguo/desconocido — la cola de "Identificación
    Manual" (Tinder)."""
    return {"items": bone_ingest.get_review_items()}


@app.post("/ingest/decide")
def ingest_decide(body: IngestDecideRequest):
    """Decisión manual (Tinder o lista de "Ampliar Base de Datos"): aceptar
    con un grupo morfológico, o rechazar."""
    try:
        item = bone_ingest.decide_item(body.item_id, body.accept, body.bone_group, body.species)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return item


@app.post("/ingest/items/{item_id}/flag-review")
def ingest_flag_review(item_id: str):
    """Manda un item "suggested" a revisión manual en vez de aceptar la
    sugerencia automática a ciegas."""
    try:
        item = bone_ingest.flag_for_review(item_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return item


@app.get("/catalog/taxon-cranio-images")
def catalog_taxon_cranio_images():
    """Imagenes que entrenan/miden stage2_taxon_cranio (ver
    data/stage2_taxon_cranio_viewer/, scripts/build_stage2_taxon_cranio_viewer.py),
    agrupadas por split/taxon -- para el apartado "Taxon > Craneo" del frontend."""
    viewer_dir = REPO_ROOT / "data" / "stage2_taxon_cranio_viewer"
    groups = {}
    if viewer_dir.is_dir():
        for split_dir in sorted(viewer_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            for taxon_dir in sorted(split_dir.iterdir()):
                if not taxon_dir.is_dir():
                    continue
                key = f"{split_dir.name}/{taxon_dir.name}"
                files = sorted(
                    f.name for f in taxon_dir.iterdir()
                    if f.is_file() and "_original" in f.name
                )
                groups[key] = {
                    "split": split_dir.name,
                    "taxon": taxon_dir.name,
                    "count": len(files),
                    "files": files,
                }
    return {"groups": groups}


@app.get("/catalog/specimens")
def catalog_list():
    """Los 122 especímenes 3D curados (Sketchfab + MorphoSource + Smithsonian),
    con su render de ejemplo — ver shared/catalog/specimens.py."""
    return {"specimens": catalog_specimens.list_specimens()}


@app.get("/catalog/balance")
def catalog_balance():
    """Especímenes 3D + fotos reales (entrenamiento/banco) por grupo, con
    objetivo y déficit — ver shared/catalog/specimens.py::list_group_balance
    y base_datos_osea/HISTORIAL.md 2026-08-03."""
    return {"groups": catalog_specimens.list_group_balance()}


@app.get("/catalog/specimens.csv")
def catalog_csv():
    """Mismo listado que /catalog/specimens, como CSV descargable."""
    rows = catalog_specimens.list_specimens()
    buf = io.StringIO()
    fieldnames = ["source", "species", "bone_canonical", "bone_group", "specimen",
                  "external_url", "license", "creator", "render_path"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=osteolab_catalogo_3d.csv"},
    )


# Sirve los renders de base_datos_osea/ para las miniaturas del catálogo 3D
# (/catalog/specimens da rutas relativas al repo, esto las hace accesibles por HTTP).
# Debe montarse ANTES del "/" de abajo — Starlette resuelve mounts en orden de
# registro, y "/" como prefijo se comería cualquier ruta registrada después.
_renders_dir = REPO_ROOT / "base_datos_osea" / "renders"
if _renders_dir.is_dir():
    app.mount("/catalog-renders", StaticFiles(directory=_renders_dir), name="catalog_renders")

_taxon_cranio_viewer_dir = REPO_ROOT / "data" / "stage2_taxon_cranio_viewer"
if _taxon_cranio_viewer_dir.is_dir():
    app.mount("/taxon-cranio-images", StaticFiles(directory=_taxon_cranio_viewer_dir), name="taxon_cranio_images")

# Montado al final: las rutas de arriba (API) se resuelven primero, esto solo
# atrapa lo que no matchea ninguna — sirve la web de producto (serving/backend/static/)
# en la misma URL que la API, sin necesitar CORS ni un segundo servicio.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")