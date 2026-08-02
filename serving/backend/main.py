from pathlib import Path

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from shared.inference.predict_cnn import classify_image as hybrid_classify_image
from shared.inference.predict_cnn import explain_gradcam
from shared.inference.predict_geometric import classify_image, get_reference_points
from shared.bone_filter.predict import is_bone
from shared.clip_filter.predict import classify_image as clip_classify
from shared.cv_extractor.extract import extract_features
from shared.vlm_describe.predict import describe_image as qwen_describe_image

app = FastAPI(title="OsteoLab", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"


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
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return hybrid_classify_image(file.file, model_type="hybrid")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# Montado al final: las rutas de arriba (API) se resuelven primero, esto solo
# atrapa lo que no matchea ninguna — sirve la web de producto (serving/backend/static/)
# en la misma URL que la API, sin necesitar CORS ni un segundo servicio.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")