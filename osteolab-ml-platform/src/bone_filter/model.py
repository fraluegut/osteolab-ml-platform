"""Definición del modelo del filtro binario hueso / no-hueso (ResNet18).

Módulo totalmente aislado del pipeline de clasificación multi-clase
(`src/training`, `src/inference`, `models/model.joblib`). No importa nada de
esos módulos y no debe ser importado por ellos: solo se expone hacia afuera
a través de `src/bone_filter/predict.py`.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "bone_filter"
WEIGHTS_PATH = MODEL_DIR / "bone_filter_resnet18.pt"
LABELS_PATH = MODEL_DIR / "labels.json"

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(pretrained: bool = True) -> nn.Module:
    """Crea un ResNet18 con la última capa adaptada a 2 clases (bone / not_bone)."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
