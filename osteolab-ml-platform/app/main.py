from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from src.inference.predict_geometric import classify_image, get_reference_points
from src.bone_filter.predict import is_bone
from src.clip_filter.predict import classify_image as clip_classify
from src.cv_extractor.extract import extract_features

app = FastAPI(title="OsteoLab", version="0.2.0")


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
    medidas — ver `src/training/train_geometric.py`. Sustituye al antiguo
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
    No requiere entrenamiento ni dataset propio; ver src/clip_filter/.
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
    src/cv_extractor/. Se llama tras confirmar con el filtro CLIP que la
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

    Independiente del clasificador multi-clase de /predict; ver src/bone_filter/.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        return is_bone(file.file)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))