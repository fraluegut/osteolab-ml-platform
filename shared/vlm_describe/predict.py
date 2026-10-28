"""Inferencia del describer Qwen2.5-VL 7B.

Módulo aislado: no comparte estado, modelo ni lógica con
`shared/clip_filter/predict.py` ni con los clasificadores entrenados. Se
invoca de forma explícita desde el endpoint dedicado en
`serving/backend/main.py`, en paralelo con el modelo geométrico e híbrido —
no como filtro previo (ese rol lo cumple CLIP).
"""
import json
import re

import torch
from PIL import Image

from shared.vlm_describe.model import (
    BONE_GROUPS,
    FRAGMENTACION_VALUES,
    PRESENCE_VALUES,
    PROMPT,
    PROMPT_FEATURES,
    load_qwen,
)

_model = None
_processor = None

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_LOOSE_RE = re.compile(r"\{.*\}", re.DOTALL)

_FEATURE_PRESENCE_KEYS = (
    "diafisis",
    "epifisis",
    "condilos",
    "superficie_articular",
    "dientes",
)


def _load():
    global _model, _processor
    if _model is not None:
        return
    _model, _processor = load_qwen()


def _parse_response(text: str) -> dict:
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return {"description": text.strip(), "suggested_group": None, "confidence_label": None}

    description = text[: match.start()].strip()
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"description": text.strip(), "suggested_group": None, "confidence_label": None}

    group = payload.get("grupo")
    if group not in BONE_GROUPS and group not in ("no_hueso", "no_seguro"):
        group = None

    return {
        "description": description,
        "suggested_group": group,
        "confidence_label": payload.get("confianza"),
    }


def _parse_features_response(text: str) -> dict:
    match = _JSON_BLOCK_RE.search(text) or _JSON_LOOSE_RE.search(text)
    if not match:
        return {"error": "no_json_block", "raw_response": text.strip()}

    group = match.group(1) if match.re is _JSON_BLOCK_RE else match.group(0)
    try:
        payload = json.loads(group)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw_response": text.strip()}

    for key in _FEATURE_PRESENCE_KEYS:
        if payload.get(key) not in PRESENCE_VALUES:
            payload[key] = None

    if payload.get("fragmentacion") not in FRAGMENTACION_VALUES:
        payload["fragmentacion"] = None

    tipo = payload.get("tipo_morfologico")
    if tipo not in BONE_GROUPS and tipo not in ("no_hueso", "no_seguro"):
        payload["tipo_morfologico"] = None

    observaciones = payload.get("observaciones")
    if not isinstance(observaciones, list):
        payload["observaciones"] = []

    return payload


def _generate(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[image], padding=True, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return _processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def describe_anatomical_features(image_file) -> dict:
    """Descriptores anatómicos estructurados (diáfisis, epífisis, cóndilos,
    superficie articular, dientes, fragmentación, tipo morfológico,
    observaciones) con Qwen2.5-VL 7B, pensados como features explicables —
    no sustituye a `describe_image`, es un modo de salida distinto del mismo
    modelo ya cargado.

    Devuelve el dict parseado del JSON (ver `shared/vlm_describe/model.py::PROMPT_FEATURES`),
    con cada campo a None si el modelo no respetó los valores permitidos, más
    `raw_response` siempre incluido para auditoría.
    """
    _load()

    image = Image.open(image_file).convert("RGB")
    raw = _generate(image, PROMPT_FEATURES, max_new_tokens=300)
    result = _parse_features_response(raw)
    result["raw_response"] = raw
    return result


def describe_image(image_file) -> dict:
    """Descripción libre + sugerencia de grupo morfológico con Qwen2.5-VL 7B.

    Devuelve:
        {
          "description": str,               # descripción libre del hueso
          "suggested_group": str | None,     # uno de BONE_GROUPS, "no_hueso",
                                              # "no_seguro", o None si no se pudo parsear
          "confidence_label": str | None,    # "alta" | "media" | "baja" | None
          "raw_response": str,               # respuesta completa del modelo
        }
    """
    _load()

    image = Image.open(image_file).convert("RGB")
    raw = _generate(image, PROMPT, max_new_tokens=400)
    result = _parse_response(raw)
    result["raw_response"] = raw
    return result
