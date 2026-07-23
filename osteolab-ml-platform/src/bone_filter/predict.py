"""Inferencia del filtro binario "es hueso / no es hueso".

Módulo aislado: no comparte estado, modelo ni lógica con
`src/inference/predict_geometric.py` (clasificador de grupo morfológico).
Se invoca de forma explícita desde el endpoint dedicado en `app/main.py`.
"""
import json

import torch
from PIL import Image
from torchvision import transforms

from src.bone_filter.model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    LABELS_PATH,
    WEIGHTS_PATH,
    build_model,
    get_device,
)

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_device = get_device()
_model = None
_idx_to_class = None


def _load():
    global _model, _idx_to_class
    if _model is not None:
        return

    if not WEIGHTS_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo del filtro en {WEIGHTS_PATH}. "
            "Entrénalo primero con `python src/bone_filter/train.py`."
        )

    with open(LABELS_PATH) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(pretrained=False)
    state_dict = torch.load(WEIGHTS_PATH, map_location=_device)
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()

    _model = model
    _idx_to_class = idx_to_class


def is_bone(image_file) -> dict:
    """Predice si `image_file` es una foto de un hueso.

    Devuelve: {"is_bone": bool, "label": str, "confidence": float, "probabilities": dict}
    """
    _load()

    img = Image.open(image_file).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_probabilities = {
        _idx_to_class[i]: round(float(p), 6) for i, p in enumerate(probs.tolist())
    }
    pred_idx = int(torch.argmax(probs).item())
    pred_label = _idx_to_class[pred_idx]

    return {
        "is_bone": pred_label == "bone",
        "label": pred_label,
        "confidence": class_probabilities[pred_label],
        "probabilities": class_probabilities,
    }
