"""Clasificación de una imagen con las variantes CNN/híbrida entrenadas en
`training/train_cnn.py` — análogo a `predict_geometric.py` pero cargando
un backbone de `timm` en vez de un `RandomForestClassifier`.

Desde el 2026-07-26, AMBAS variantes segmentan primero (ver HISTORIAL.md):
la CNN nunca ve la imagen completa, solo el recorte al bounding box del
hueso con el fondo pintado de gris (`extract_features(..., include_crop=True)`
→ `"crop_image"`, mismo backend de segmentación que entrenamiento). La
`hybrid`, además, usa el vector de features geométricas de esa misma
llamada (con el mismo fallback de medias de entrenamiento para features que
salgan `None`) — así que ambas comparten una única pasada de segmentación
por imagen, no dos.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image

from shared.cv_extractor.build_dataset import flatten_features
from shared.cv_extractor.extract import extract_features
from training.cnn_dataset import build_transform

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
TARGET = "bone_group"
TRAINING_TABLE_PATH = BASE_DIR / "data/processed/bone_geometric_features_combined.csv"

_TRANSFORM = build_transform(train=False)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoneClassifier(nn.Module):
    """Copia mínima de la clase de `train_cnn.py` (mismo nombre y forma)
    para poder reconstruir el modelo a partir del `state_dict` guardado sin
    importar el script de entrenamiento completo (que trae dependencias de
    MLflow/argparse innecesarias para servir el modelo)."""

    def __init__(self, backbone_name: str, num_classes: int, n_geom_features: int, model_type: str):
        super().__init__()
        self.model_type = model_type
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        embed_dim = self.backbone.num_features
        head_in = embed_dim + (n_geom_features if model_type == "hybrid" else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, geom):
        embedding = self.backbone(image)
        if self.model_type == "hybrid":
            embedding = torch.cat([embedding, geom], dim=1)
        return self.head(embedding)


def _missing_model_msg(model_type: str) -> str:
    return (
        f"No se encontró el modelo CNN ({model_type}) en {MODEL_DIR}. "
        f"Entrénalo primero con `python -m training.train_cnn --model-type {model_type}`."
    )


@lru_cache(maxsize=2)
def _load(model_type: str):
    model_path = MODEL_DIR / f"cnn_model_{model_type}_{TARGET}.pt"
    encoder_path = MODEL_DIR / f"cnn_encoder_{model_type}_{TARGET}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(_missing_model_msg(model_type))

    checkpoint = torch.load(model_path, map_location=_DEVICE, weights_only=False)
    encoder = joblib.load(encoder_path)

    model = BoneClassifier(
        checkpoint["backbone"], checkpoint["num_classes"], checkpoint["n_geom_features"], model_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(_DEVICE)
    model.eval()

    feature_cols, scaler, fallback_means = None, None, None
    if model_type == "hybrid":
        feature_cols = joblib.load(MODEL_DIR / f"cnn_feature_columns_{model_type}_{TARGET}.joblib")
        scaler = joblib.load(MODEL_DIR / f"cnn_scaler_{model_type}_{TARGET}.joblib")
        fallback_means = pd.read_csv(TRAINING_TABLE_PATH, usecols=feature_cols).mean(numeric_only=True).to_dict()

    return model, encoder, feature_cols, scaler, fallback_means


def classify_image(image_file, model_type: str = "hybrid", backend: str = "grounded_sam") -> dict:
    """Clasifica una imagen (path o file-like) con la CNN.

    Devuelve:
        {
          "found": bool,               # False si la segmentación no encuentra contorno
          "prediction": str,
          "probabilities": {grupo: float, ...},
          "cv_features": {...} | None,  # solo si model_type="hybrid"
        }
    """
    model, encoder, feature_cols, scaler, fallback_means = _load(model_type)

    raw_features = extract_features(image_file, backend=backend, include_crop=True)
    if not raw_features.get("found"):
        return {"found": False, "cv_features": raw_features if model_type == "hybrid" else None}

    geom_tensor = torch.zeros((1, 0), dtype=torch.float32)
    if model_type == "hybrid":
        flat = flatten_features(raw_features)
        row = [[
            flat[col] if flat[col] is not None else fallback_means[col]
            for col in feature_cols
        ]]
        scaled = scaler.transform(row)
        geom_tensor = torch.tensor(scaled, dtype=torch.float32)

    image = raw_features.pop("crop_image")  # nunca serializable/JSON-able, no debe salir en la respuesta
    image_tensor = _TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor.to(_DEVICE), geom_tensor.to(_DEVICE))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    class_probabilities = {
        encoder.inverse_transform([i])[0]: round(float(p), 6)
        for i, p in enumerate(probs)
    }
    prediction = max(class_probabilities, key=class_probabilities.get)

    return {
        "found": True,
        "prediction": prediction,
        "probabilities": class_probabilities,
        "cv_features": raw_features if model_type == "hybrid" else None,
    }
