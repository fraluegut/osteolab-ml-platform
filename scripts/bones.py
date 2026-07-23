"""Clasifica el campo de texto libre `part` de MorphoSource en huesos canónicos.

MorphoSource no tiene un facet de "tipo de hueso": `part` lo escribe cada
laboratorio a mano ("femur", "Proximal femur", "U.W.101-948 - Right humerus").
Este módulo mapea esas variantes a un nombre canónico estable, para poder
agrupar y elegir "un ejemplo por hueso" por especie.

Cada hueso tiene patrones "whole" (pieza completa) y "partial" (fragmento:
"proximal X", "distal X", reconstrucciones, calvaria...). Si el texto matchea
un patrón partial, se prioriza sobre whole aunque ambos calcen.
"""

import re

TARGET_BONES = {
    "cranium": {
        "whole": [r"\bcranium\b", r"\bskull\b"],
        "partial": [r"frontal part of skull", r"calvari", r"reconstruction of cranium", r"\bcalotte\b"],
    },
    "mandible": {
        "whole": [r"\bmandible\b"],
        "partial": [r"mandible fragment", r"mandibular.*fragment"],
    },
    "maxilla": {"whole": [r"\bmaxilla[e]?\b"], "partial": []},
    "scapula": {"whole": [r"\bscapula\b"], "partial": []},
    "clavicle": {"whole": [r"\bclavicle\b"], "partial": []},
    "humerus": {"whole": [r"\bhumerus\b"], "partial": [r"proximal humerus", r"distal humerus"]},
    "radius": {"whole": [r"\bradius\b"], "partial": [r"proximal radius", r"distal radius"]},
    "ulna": {"whole": [r"\bulna\b"], "partial": [r"proximal ulna", r"distal ulna"]},
    "femur": {"whole": [r"\bfemur\b"], "partial": [r"proximal femur", r"distal femur"]},
    "tibia": {"whole": [r"\btibia\b"], "partial": [r"proximal tibia", r"distal tibia"]},
    "fibula": {"whole": [r"\bfibula\b"], "partial": [r"proximal fibula", r"distal fibula"]},
    "patella": {"whole": [r"\bpatella\b"], "partial": []},
    "pelvis": {"whole": [r"^pelvis$"], "partial": []},
    "sacrum": {"whole": [r"\bsacrum\b"], "partial": [r"partial sacrum"]},
    "vertebra": {"whole": [r"\bvertebra\b"], "partial": []},
    "rib": {"whole": [r"\brib\b"], "partial": []},
    "metacarpal_1": {"whole": [r"first metacarpal", r"\bmc-?1\b"], "partial": []},
    "metatarsal_1": {"whole": [r"first met[ae]tarsal", r"\bmt-?1\b"], "partial": []},
    "proximal_phalanx_1": {"whole": [r"first proximal phalanx"], "partial": []},
    "trapezium": {"whole": [r"\btrapezium\b"], "partial": []},
    "sesamoid": {"whole": [r"sesamoid bone"], "partial": []},
}

# Categorías que aparecen en los datos pero no son "un hueso limpio": se
# catalogan igualmente (para tener trazabilidad completa) pero nunca se
# seleccionan para descarga automática.
EXCLUDED_CATEGORIES = {
    "tooth": [r"\btooth\b", r"\bmolar\b"],
    "soft_tissue": [r"\bartery\b", r"\bventricle\b", r"\bmuscles?\b", r"\bnerves?\b"],
    "pelvis_pose_variant": [r"pelvis.*\b(standing|rotate|flexion)\b"],
    "composite_multi_tissue": [r"with bones, muscles"],
    "endocast": [r"\bendocast\b"],
}


def classify_part(part_raw):
    """Devuelve (category, bone_canonical, is_target, is_partial) para un `part` crudo."""
    text = (part_raw or "").strip().lower()
    if not text or text == "?":
        return "unclassified", None, False, False

    for category, patterns in EXCLUDED_CATEGORIES.items():
        if any(re.search(p, text) for p in patterns):
            return category, None, False, False

    for bone, spec in TARGET_BONES.items():
        if any(re.search(p, text) for p in spec["partial"]):
            return "target_bone", bone, True, True
        if any(re.search(p, text) for p in spec["whole"]):
            return "target_bone", bone, True, False

    return "other", None, False, False
