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
# con las clases del clasificador geométrico (`models/geometric_encoder_bone_group.joblib`,
# ver `src/training/train_geometric.py` y `BONE_GROUPS` en base_datos_osea/scripts/bones.py):
# los 9 grupos morfológicos, no huesos ni especies concretas.
BONE_TYPE_DESCRIPTIONS = {
    "cranio": "a human skull or cranium",
    "mandibula_maxilar": "a human jawbone or mandible",
    "hueso_largo": "a long bone such as a femur, tibia, humerus, radius or ulna",
    "hueso_plano": "a flat shoulder blade or scapula bone",
    "pelvis": "a human pelvis or hip bone",
    "sacro": "a human sacrum bone at the base of the spine",
    "vertebra": "a single human vertebra bone from the spine",
    "costilla": "a human rib bone",
    "hueso_pequeno": "a small hand or foot bone such as a finger or toe bone",
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
