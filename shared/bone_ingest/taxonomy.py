"""Mapea el nombre de un objeto/malla (tal cual lo puso el autor del modelo 3D
en Sketchfab, p.ej. `l_femur`, `Cranium`, `c3`, `r_rib7`) al grupo morfológico
de 9 clases del proyecto — mismo destino que `BONE_GROUPS` en
`base_datos_osea/scripts/bones.py` y `shared/cv_extractor/build_dataset.py`,
pero la entrada aquí es el nombre de un objeto de escena 3D, no el campo
`part` de texto libre de MorphoSource, así que las reglas son otras (prefijos
`l_`/`r_` de lateralidad, numeración `c1`-`c7`/`t1`-`t12`/`l1`-`l5` de
vértebras, huesos de mano/pie con muchas variantes de nombre).

Deliberadamente conservador: si el nombre no matchea con confianza, se
devuelve `None` (grupo desconocido) en vez de forzar una clase — mejor que un
hueso ambiguo pase por revisión manual a que se cuele mal etiquetado en el
dataset de entrenamiento.
"""
import re

# (patrón, bone_group) en orden de prioridad — el primero que matchee gana.
# Los patrones de vértebra/costilla van antes que los genéricos para no
# confundir "l_rib7" con nada de mano/pie.
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^cranium$|skull", re.I), "cranio"),
    (re.compile(r"mandible", re.I), "mandibula_maxilar"),
    (re.compile(r"maxilla", re.I), "mandibula_maxilar"),
    (re.compile(r"scapula", re.I), "hueso_plano"),
    (re.compile(r"clavicle", re.I), "hueso_largo"),
    (re.compile(r"humerus", re.I), "hueso_largo"),
    (re.compile(r"radius", re.I), "hueso_largo"),
    (re.compile(r"\bulna\b", re.I), "hueso_largo"),
    (re.compile(r"femur", re.I), "hueso_largo"),
    (re.compile(r"tibia", re.I), "hueso_largo"),
    (re.compile(r"fibula", re.I), "hueso_largo"),
    (re.compile(r"patella", re.I), "hueso_pequeno"),
    (re.compile(r"^[lr]?[\s_]*os[\s_]?coxa$|^pelvis$|hip[\s_]?bone", re.I), "pelvis"),
    (re.compile(r"sacrum", re.I), "sacro"),
    (re.compile(r"^[ctl][\s_]?\d{1,2}$", re.I), "vertebra"),  # c1-c7, t1-t12, l1-l5
    (re.compile(r"\brib[\s_]?\d*$", re.I), "costilla"),
    (re.compile(r"capitate|hamate|lunate|pisiform|scaphoid|trapezium|trapezoid|triquetral", re.I), "hueso_pequeno"),
    (re.compile(r"calcaneus|cuboid|cuneiform|navicular|\btalus\b", re.I), "hueso_pequeno"),
    (re.compile(r"metacarpal|metatarsal", re.I), "hueso_pequeno"),
    (re.compile(r"phalan(ge|x)", re.I), "hueso_pequeno"),
    (re.compile(r"sesamoid", re.I), "hueso_pequeno"),
]

BONE_GROUPS = [
    "cranio", "mandibula_maxilar", "hueso_largo", "hueso_plano",
    "pelvis", "sacro", "vertebra", "costilla", "hueso_pequeno",
]


def guess_bone_group(object_name: str) -> str | None:
    """Devuelve uno de `BONE_GROUPS` si el nombre matchea con confianza, o
    `None` si es ambiguo/desconocido (sternum, hyoid, coccyx, xiphoid... no
    están en la taxonomía de 9 grupos del proyecto — van a revisión manual)."""
    # "_" cuenta como carácter de palabra para \b, así que "l_rib1"/"l_ulna"
    # no separarían de "l" sin esto — normalizar a espacios antes de matchear.
    name = object_name.strip().replace("_", " ")
    for pattern, group in _RULES:
        if pattern.search(name):
            return group
    return None
