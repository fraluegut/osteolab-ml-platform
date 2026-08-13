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
import base64
import io
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
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
GEOMETRIC_MODEL_TYPES = {"hybrid", "gated_hybrid"}


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
        if model_type == "gated_hybrid":
            self.cnn_head = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes),
            )
            self.geom_head = nn.Sequential(
                nn.Linear(n_geom_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes),
            )
            self.gate = nn.Sequential(
                nn.Linear(embed_dim + n_geom_features, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1),
            )
        else:
            head_in = embed_dim + (n_geom_features if model_type == "hybrid" else 0)
            self.head = nn.Sequential(
                nn.Linear(head_in, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes),
            )

    def forward(self, image, geom):
        embedding = self.backbone(image)
        if self.model_type == "gated_hybrid":
            return self._gated_outputs(embedding, geom)[0]
        if self.model_type == "hybrid":
            embedding = torch.cat([embedding, geom], dim=1)
        return self.head(embedding)

    def _gated_outputs(self, embedding, geom):
        cnn_logits = self.cnn_head(embedding)
        geom_logits = self.geom_head(geom)
        cnn_weight = torch.sigmoid(self.gate(torch.cat([embedding, geom], dim=1)))
        logits = cnn_weight * cnn_logits + (1.0 - cnn_weight) * geom_logits
        return logits, cnn_logits, geom_logits, cnn_weight

    def forward_with_components(self, image, geom):
        return self._gated_outputs(self.backbone(image), geom)

    def modality_weights(self, image, geom):
        embedding = self.backbone(image)
        return torch.sigmoid(self.gate(torch.cat([embedding, geom], dim=1)))


def _missing_model_msg(model_type: str) -> str:
    return (
        f"No se encontró el modelo CNN ({model_type}) en {MODEL_DIR}. "
        f"Entrénalo primero con `python -m training.train_cnn --model-type {model_type}`."
    )


@lru_cache(maxsize=4)
def _load(model_type: str, run_name: str | None = None):
    """`run_name` (default `None` -> usa `model_type`) es el sufijo de fichero
    real — permite cargar variantes de pipeline paralelas (ej. 'hybrid_blackbg',
    ver `train_cnn.py --run-name`) sin que `model_type` deje de controlar la
    arquitectura (hybrid/cnn_only)."""
    suffix = run_name or model_type
    model_path = MODEL_DIR / f"cnn_model_{suffix}_{TARGET}.pt"
    encoder_path = MODEL_DIR / f"cnn_encoder_{suffix}_{TARGET}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(_missing_model_msg(suffix))

    checkpoint = torch.load(model_path, map_location=_DEVICE, weights_only=False)
    encoder = joblib.load(encoder_path)

    model = BoneClassifier(
        checkpoint["backbone"], checkpoint["num_classes"], checkpoint["n_geom_features"], model_type,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(_DEVICE)
    model.eval()

    feature_cols, scaler, fallback_means = None, None, None
    if model_type in GEOMETRIC_MODEL_TYPES:
        feature_cols = joblib.load(MODEL_DIR / f"cnn_feature_columns_{suffix}_{TARGET}.joblib")
        scaler = joblib.load(MODEL_DIR / f"cnn_scaler_{suffix}_{TARGET}.joblib")
        fallback_means = pd.read_csv(TRAINING_TABLE_PATH, usecols=feature_cols).mean(numeric_only=True).to_dict()

    return model, encoder, feature_cols, scaler, fallback_means


def classify_image(image_file, model_type: str = "hybrid", backend: str = "grounded_sam",
                    run_name: str | None = None, bg_gray: int = 128) -> dict:
    """Clasifica una imagen (path o file-like) con la CNN.

    `run_name`/`bg_gray`: para evaluar variantes de pipeline paralelas (ver
    `_load`) — con el pipeline normal, dejar los defaults.

    Devuelve:
        {
          "found": bool,               # False si la segmentación no encuentra contorno
          "prediction": str,
          "probabilities": {grupo: float, ...},
          "cv_features": {...} | None,  # solo si model_type="hybrid"
        }
    """
    model, encoder, feature_cols, scaler, fallback_means = _load(model_type, run_name)

    raw_features = extract_features(image_file, backend=backend, include_crop=True, bg_gray=bg_gray)
    if not raw_features.get("found"):
        return {"found": False, "cv_features": raw_features if model_type in GEOMETRIC_MODEL_TYPES else None}

    geom_tensor = torch.zeros((1, 0), dtype=torch.float32)
    if model_type in GEOMETRIC_MODEL_TYPES:
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
        if model_type == "gated_hybrid":
            logits, _, _, weight_tensor = model.forward_with_components(
                image_tensor.to(_DEVICE), geom_tensor.to(_DEVICE),
            )
            cnn_weight = float(weight_tensor[0, 0].cpu())
        else:
            logits = model(image_tensor.to(_DEVICE), geom_tensor.to(_DEVICE))
            cnn_weight = None
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
        "cv_features": raw_features if model_type in GEOMETRIC_MODEL_TYPES else None,
        "modality_weights": ({"cnn": round(cnn_weight, 6), "geometric": round(1.0 - cnn_weight, 6)}
                             if cnn_weight is not None else None),
    }


def _encode_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def explain_gradcam(image_file, model_type: str = "hybrid", backend: str = "grounded_sam") -> dict:
    """Grad-CAM sobre la última etapa convolucional del backbone ResNet18
    (`backbone.layer4`), calculado respecto a la clase predicha — qué región
    del recorte que ve la CNN pesó más en esa predicción. Pensado para
    evaluar si el modelo mira el hueso o se apoya en fondo/ruido/artefactos.

    No aplica al modelo geométrico (Random Forest sobre medidas OpenCV, sin
    capas convolucionales) — ahí se usa `mask_overlay_base64` de
    `extract_features()` en su lugar.

    Devuelve:
        {
          "found": bool,
          "prediction": str,
          "heatmap_base64": str,   # PNG: el recorte con el mapa de calor superpuesto
        }
    """
    model, encoder, feature_cols, scaler, fallback_means = _load(model_type)

    raw_features = extract_features(image_file, backend=backend, include_crop=True)
    if not raw_features.get("found"):
        return {"found": False}

    geom_tensor = torch.zeros((1, 0), dtype=torch.float32)
    if model_type in GEOMETRIC_MODEL_TYPES:
        flat = flatten_features(raw_features)
        row = [[
            flat[col] if flat[col] is not None else fallback_means[col]
            for col in feature_cols
        ]]
        scaled = scaler.transform(row)
        geom_tensor = torch.tensor(scaled, dtype=torch.float32)

    crop_image = raw_features.pop("crop_image")
    image_tensor = _TRANSFORM(crop_image).unsqueeze(0).to(_DEVICE)
    geom_tensor = geom_tensor.to(_DEVICE)

    activations, gradients = {}, {}

    def save_activation(_module, _input, output):
        activations["value"] = output

    def save_gradient(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.backbone.layer4
    fwd_handle = target_layer.register_forward_hook(save_activation)
    bwd_handle = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor, geom_tensor)
        class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        acts = activations["value"][0]  # [C, H, W]
        grads = gradients["value"][0]  # [C, H, W]
        weights = grads.mean(dim=(1, 2))  # [C], global average pool de los gradientes
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # [H, W]
    finally:
        fwd_handle.remove()
        bwd_handle.remove()
        model.zero_grad(set_to_none=True)

    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    cam_max = cam.max()
    if cam_max > 1e-8:
        cam /= cam_max

    crop_rgb = np.array(crop_image.convert("RGB"))
    h, w = crop_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * crop_rgb + 0.45 * heatmap).astype(np.uint8)

    return {
        "found": True,
        "prediction": encoder.inverse_transform([class_idx])[0],
        "heatmap_base64": _encode_png(Image.fromarray(overlay)),
    }
