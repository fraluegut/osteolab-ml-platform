"""Definición del describer Qwen2.5-VL 7B (VLM zero-shot, en lenguaje natural).

Módulo aislado: no depende de `shared/clip_filter`, `shared/bone_filter` ni
`shared/inference`, y no requiere entrenamiento ni dataset propio (usa el
modelo Qwen2.5-VL-7B-Instruct preentrenado de Hugging Face). A diferencia de
CLIP (que da una probabilidad por grupo), a este se le pide una descripción
libre de lo que ve más una sugerencia de grupo con su propio razonamiento —
pensado para comparar contra el criterio de los 3 modelos entrenados
(geométrico, CNN, híbrido), no para sustituirlos.
"""
import torch

QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Mismos 9 grupos morfológicos que el resto del pipeline (ver
# `shared/clip_filter/model.py::BONE_TYPE_DESCRIPTIONS` y
# `training/train_geometric.py`), repetidos aquí para no acoplar este módulo
# a CLIP.
BONE_GROUPS = [
    "cranio",
    "mandibula_maxilar",
    "hueso_largo",
    "hueso_plano",
    "pelvis",
    "sacro",
    "vertebra",
    "costilla",
    "hueso_pequeno",
]

PROMPT = """Eres un especialista en osteología describiendo una foto para un catálogo.

1. Describe libremente lo que ves en la imagen: forma general, textura de la \
superficie, proporciones aproximadas, y cualquier rasgo distintivo (articulaciones, \
cavidades, curvatura, simetría...). Dos o tres frases bastan.
2. Según esa descripción, sugiere a cuál de estos 9 grupos morfológicos pertenece \
el hueso — o responde "no_hueso" si la imagen no muestra un hueso, o "no_seguro" si \
no puedes determinarlo con lo que ves:
   - cranio: cráneo o cara
   - mandibula_maxilar: mandíbula o maxilar
   - hueso_largo: hueso largo de extremidad (fémur, tibia, húmero, radio, cúbito...)
   - hueso_plano: omóplato/escápula u otro hueso plano
   - pelvis: pelvis o cadera
   - sacro: sacro
   - vertebra: una vértebra suelta
   - costilla: costilla
   - hueso_pequeno: hueso pequeño de mano o pie

Responde con la descripción libre primero, y termina siempre con un bloque JSON \
en una línea aparte, en este formato exacto:

```json
{"grupo": "<uno de los códigos de arriba, o no_hueso, o no_seguro>", "confianza": "alta|media|baja"}
```
"""


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_qwen():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = get_device()
    kwargs = {}
    if device.type == "cuda":
        from transformers import BitsAndBytesConfig

        # 4-bit: ~5-6GB de VRAM en vez de ~16GB en bf16 — necesario para
        # convivir con CLIP/ResNet18/CNN en la misma GPU de 16GB.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL_NAME, **kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
    return model, processor
