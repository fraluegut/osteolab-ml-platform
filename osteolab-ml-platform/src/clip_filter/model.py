"""Definición del filtro CLIP zero-shot (hueso / no-hueso + sugerencia de tipo).

Módulo aislado: no depende de `src/bone_filter` ni de `src/inference` / `src/training`,
y no requiere entrenamiento ni dataset propio (usa el modelo CLIP preentrenado
de Hugging Face en modo zero-shot). Se ejecuta como primera capa, antes del
filtro ResNet18 (`src/bone_filter/`).
"""
import torch

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Plantillas de prompt sobre las que se promedia el embedding de texto de cada
# descripción (ensembling estándar de CLIP para mejorar la precisión zero-shot).
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a close-up photo of {}",
    "an X-ray of {}",
]

# Descripciones para decidir si la imagen es o no un hueso.
BONE_DESCRIPTIONS = [
    "a human bone or skeletal remains",
    "a bone specimen",
]
NOT_BONE_DESCRIPTIONS = [
    "a person",
    "an animal",
    "an everyday object",
    "food",
    "a landscape or a building",
    "a random photo unrelated to bones",
]

# Descripciones para la sugerencia de tipo de hueso. Las claves deben coincidir
# con las clases del clasificador multiclase (`models/encoder.joblib`):
# craneo, femur, humero.
BONE_TYPE_DESCRIPTIONS = {
    "craneo": "a human skull bone",
    "femur": "a human femur bone, the thigh bone",
    "humero": "a human humerus bone, the upper arm bone",
}

# Umbral de confianza a partir del cual se considera que CLIP "lo tiene muy
# claro" y se propone el tipo de hueso final en vez de una simple sugerencia.
SUGGESTION_CERTAIN_THRESHOLD = 0.7


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    device = get_device()
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.to(device)
    model.eval()
    return model, processor, device
