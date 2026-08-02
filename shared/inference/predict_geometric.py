"""Clasificación de una imagen subida usando el modelo geométrico entrenado
en `training/train_geometric.py` sobre las features de
`shared/cv_extractor/extract.py` — reemplaza al clasificador de píxeles crudos
(`shared/inference/predict.py`, retirado: 3 clases, dataset no relacionado con
el catálogo de huesos real).

Reutiliza literalmente la misma función de aplanado de features
(`flatten_features`) que usó `build_dataset.py` para construir la tabla de
entrenamiento — así la imagen subida se mide exactamente igual que los datos
con los que se entrenó el modelo, columna por columna en el mismo orden
(`geometric_feature_columns_bone_group.joblib`).
"""
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

from shared.cv_extractor.extract import extract_features
from shared.cv_extractor.build_dataset import flatten_features

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
TARGET = "bone_group"

MODEL_PATH = MODEL_DIR / f"geometric_model_{TARGET}.joblib"
ENCODER_PATH = MODEL_DIR / f"geometric_encoder_{TARGET}.joblib"
COLUMNS_PATH = MODEL_DIR / f"geometric_feature_columns_{TARGET}.joblib"
PCA_PATH = MODEL_DIR / "geometric_pca_pipeline.joblib"
REFERENCE_PATH = BASE_DIR / "data/processed/bone_geometric_pca_reference.csv"
TRAINING_TABLE_PATH = BASE_DIR / "data/processed/bone_geometric_features.csv"

_MISSING_MODEL_MSG = (
    f"No se encontró el modelo geométrico en {MODEL_PATH}. "
    "Entrénalo primero con `python -m training.train_geometric`."
)


def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(_MISSING_MODEL_MSG)
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_cols = joblib.load(COLUMNS_PATH)
    pca_pipeline = joblib.load(PCA_PATH)
    return model, encoder, feature_cols, pca_pipeline


def _load_fallback_means(feature_cols) -> dict:
    """Media por columna en la tabla de entrenamiento, para imputar features
    que a veces salen `None` en fotos reales (contornos degenerados: muy
    pocos puntos para `cv2.fitEllipse`, lado corto ~0, etc. — nunca pasa en
    los renders limpios de entrenamiento, donde el contorno siempre es
    grande y bien formado)."""
    df = pd.read_csv(TRAINING_TABLE_PATH, usecols=feature_cols)
    return df.mean(numeric_only=True).to_dict()


_model, _encoder, _feature_cols, _pca_pipeline = _load()
_fallback_means = _load_fallback_means(_feature_cols)


@lru_cache(maxsize=1)
def get_reference_points() -> list[dict]:
    """Coordenadas PCA de las 1320 vistas ya conocidas, para dibujar de fondo
    en el scatter y comparar visualmente contra la vista nueva. Se cachea en
    memoria (no cambia entre peticiones — solo cambiaría si se re-entrena)."""
    df = pd.read_csv(REFERENCE_PATH)
    return df.to_dict(orient="records")


def classify_image(image_file, backend: str = "grounded_sam") -> dict:
    """Mide la imagen con OpenCV, clasifica por grupo morfológico y proyecta
    la misma medida en el espacio PCA de referencia.

    Devuelve:
        {
          "found": bool,                        # si OpenCV pudo aislar un contorno
          "prediction": str,                     # grupo con mayor probabilidad
          "probabilities": {grupo: float, ...},  # 9 clases, suman 1
          "pca_point": {"x": float, "y": float},
          "cv_features": {...},                  # el dict completo de extract_features, para mostrar en la UI
        }
    """
    raw_features = extract_features(image_file, backend=backend)
    if not raw_features.get("found"):
        return {"found": False, "cv_features": raw_features}

    flat = flatten_features(raw_features)
    row = [[
        flat[col] if flat[col] is not None else _fallback_means[col]
        for col in _feature_cols
    ]]

    probs = _model.predict_proba(row)[0]
    class_probabilities = {
        _encoder.inverse_transform([i])[0]: round(float(p), 6)
        for i, p in enumerate(probs)
    }
    prediction = max(class_probabilities, key=class_probabilities.get)

    pca_xy = _pca_pipeline.transform(row)[0]

    return {
        "found": True,
        "prediction": prediction,
        "probabilities": class_probabilities,
        "pca_point": {"x": round(float(pca_xy[0]), 4), "y": round(float(pca_xy[1]), 4)},
        "cv_features": raw_features,
    }
