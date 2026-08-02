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

from shared.vlm_describe.model import BONE_GROUPS, PROMPT, load_qwen

_model = None
_processor = None

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[image], padding=True, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        generated_ids = _model.generate(**inputs, max_new_tokens=400)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    raw = _processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    result = _parse_response(raw)
    result["raw_response"] = raw
    return result
