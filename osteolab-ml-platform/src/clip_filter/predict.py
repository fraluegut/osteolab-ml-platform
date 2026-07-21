"""Inferencia del filtro CLIP zero-shot.

Módulo aislado: no comparte estado, modelo ni lógica con
`src/bone_filter/predict.py` (filtro ResNet18 entrenado) ni con
`src/inference/predict.py` (clasificador multiclase). Se invoca de forma
explícita desde el endpoint dedicado en `app/main.py`, como primera capa,
antes del filtro ResNet18.
"""
import torch
import torch.nn.functional as F
from PIL import Image

from src.clip_filter.model import (
    BONE_DESCRIPTIONS,
    BONE_TYPE_DESCRIPTIONS,
    NOT_BONE_DESCRIPTIONS,
    PROMPT_TEMPLATES,
    SUGGESTION_CERTAIN_THRESHOLD,
    load_clip,
)

_model = None
_processor = None
_device = None
_bone_text_embeds = None  # [n_bone_desc, dim]
_not_bone_text_embeds = None  # [n_not_bone_desc, dim]
_type_labels = None  # list[str], orden fijo de BONE_TYPE_DESCRIPTIONS
_type_text_embeds = None  # [n_types, dim]


def _embed_descriptions(descriptions: list[str]) -> torch.Tensor:
    """Embedding de texto normalizado por descripción, promediado sobre
    `PROMPT_TEMPLATES` (ensembling estándar de CLIP)."""
    prompts = [tpl.format(desc) for desc in descriptions for tpl in PROMPT_TEMPLATES]
    inputs = _processor(text=prompts, return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        # En transformers>=5, get_text_features devuelve un BaseModelOutputWithPooling
        # (no un tensor), con el embedding proyectado en .pooler_output.
        features = _model.get_text_features(**inputs).pooler_output
    features = F.normalize(features, dim=-1)
    features = features.view(len(descriptions), len(PROMPT_TEMPLATES), -1).mean(dim=1)
    return F.normalize(features, dim=-1)


def _load():
    global _model, _processor, _device
    global _bone_text_embeds, _not_bone_text_embeds, _type_labels, _type_text_embeds

    if _model is not None:
        return

    _model, _processor, _device = load_clip()
    _bone_text_embeds = _embed_descriptions(BONE_DESCRIPTIONS)
    _not_bone_text_embeds = _embed_descriptions(NOT_BONE_DESCRIPTIONS)
    _type_labels = list(BONE_TYPE_DESCRIPTIONS.keys())
    _type_text_embeds = _embed_descriptions(list(BONE_TYPE_DESCRIPTIONS.values()))


def _image_embedding(image_file) -> torch.Tensor:
    img = Image.open(image_file).convert("RGB")
    inputs = _processor(images=img, return_tensors="pt").to(_device)
    with torch.no_grad():
        features = _model.get_image_features(**inputs).pooler_output
    return F.normalize(features, dim=-1)


def _softmax_over_groups(image_embed: torch.Tensor, *groups: torch.Tensor) -> list[torch.Tensor]:
    """Softmax conjunta sobre la concatenación de varios grupos de embeddings
    de texto, devolviendo las probabilidades de cada grupo por separado."""
    with torch.no_grad():
        logit_scale = _model.logit_scale.exp()
        all_embeds = torch.cat(groups, dim=0)
        logits = logit_scale * image_embed @ all_embeds.T
        probs = F.softmax(logits, dim=-1)[0]

    result, offset = [], 0
    for group in groups:
        n = group.shape[0]
        result.append(probs[offset : offset + n])
        offset += n
    return result


def classify_image(image_file) -> dict:
    """Filtro CLIP zero-shot: ¿es un hueso? y, si lo es, sugerencia de tipo.

    Devuelve:
        {
          "is_bone": bool,
          "confidence": float,               # confianza de la decisión hueso/no-hueso
          "probabilities": {"bone": .., "not_bone": ..},
          "suggestion": {
              "type": str,                   # craneo/femur/humero más probable
              "confidence": float,
              "certain": bool,                # True si supera el umbral de "muy claro"
              "probabilities": {"craneo": .., "femur": .., "humero": ..},
          } | None,                          # None si is_bone es False
        }
    """
    _load()

    image_embed = _image_embedding(image_file)

    bone_probs, not_bone_probs = _softmax_over_groups(
        image_embed, _bone_text_embeds, _not_bone_text_embeds
    )
    bone_prob = float(bone_probs.sum())
    not_bone_prob = float(not_bone_probs.sum())
    is_bone = bone_prob > not_bone_prob

    result = {
        "is_bone": is_bone,
        "confidence": round(bone_prob if is_bone else not_bone_prob, 6),
        "probabilities": {
            "bone": round(bone_prob, 6),
            "not_bone": round(not_bone_prob, 6),
        },
        "suggestion": None,
    }

    if not is_bone:
        return result

    (type_probs,) = _softmax_over_groups(image_embed, _type_text_embeds)
    type_probabilities = {
        label: round(float(p), 6) for label, p in zip(_type_labels, type_probs)
    }
    best_type = max(type_probabilities, key=type_probabilities.get)
    best_confidence = type_probabilities[best_type]

    result["suggestion"] = {
        "type": best_type,
        "confidence": best_confidence,
        "certain": best_confidence >= SUGGESTION_CERTAIN_THRESHOLD,
        "probabilities": type_probabilities,
    }
    return result
