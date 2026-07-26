"""Clasificador en cascada: una regla geométrica barata y de alta precisión para
`hueso_largo`, CNN (`cnn_only`) para todo lo demás.

Motivación (2026-07-25, ver HISTORIAL.md): al buscar qué categorías tienen un
ratio geométrico lo bastante característico como para evitar la CNN, se probó
`elongation`/`aspect_ratio` (hueso_largo), `n_holes` (sacro/vertebra),
`circularity`/`solidity` (cranio/hueso_plano) — todas parecían igual de buenas
en el banco sintético limpio (88-92% accuracy binaria), pero SOLO
`elongation` generalizó a fotos reales (93%, 1 falso positivo en 47). El resto
se rompió (16-80%, hasta 46 falsos positivos en 49): son features que dependen
de un contorno fino y limpio (contar agujeros, circularidad exacta), y la
segmentación sobre una foto real (sombras, textura) los ensucia mucho más que
la envolvente general del objeto, que es lo que mide `elongation`. Por eso la
cascada es específica de `hueso_largo`, no un patrón que se pueda copiar sin
más a otras categorías — antes de añadir una nueva regla aquí, validarla igual
que se hizo con estas (accuracy Y falsos positivos en el banco de fotos
reales, no solo en el banco sintético).

Resultado medido en el banco de fotos reales (55 imágenes, ver HISTORIAL.md):
accuracy sube de 0.5273 (CNN sola) a 0.6000, y el recall de `hueso_largo` de
0.38 a 0.88.
"""
import io

from src.cv_extractor.extract import extract_features
from src.inference.predict_cnn import classify_image as classify_image_cnn

TARGET = "bone_group"
ELONGATION_THRESHOLD = 0.900  # umbral óptimo de un stump de profundidad 1 sobre datos de ENTRENAMIENTO
GATED_CLASS = "hueso_largo"


def classify_image(image_file, model_type: str = "cnn_only", backend: str = "grounded_sam") -> dict:
    """Igual que `predict_cnn.classify_image`, pero primero prueba la puerta
    de elongación. Si no salta, delega en la CNN (`model_type` decide cuál).
    `backend` es el backend de segmentación ("grounded_sam"/"fastsam"/
    "classic", ver `src/cv_extractor/extract.py`) usado tanto para medir
    `elongation` como, si hace falta, para las features geométricas del
    modelo híbrido.

    Devuelve el mismo formato que `predict_cnn.classify_image` más una clave
    `"gate"`: `"elongation_rule"` si decidió la regla geométrica, `"cnn"` si
    decidió el modelo de respaldo.
    """
    raw_bytes = image_file.read()
    raw_features = extract_features(io.BytesIO(raw_bytes), backend=backend)

    if raw_features.get("found") and raw_features.get("elongation", 0) > ELONGATION_THRESHOLD:
        return {
            "found": True,
            "prediction": GATED_CLASS,
            "probabilities": {GATED_CLASS: 1.0},
            "cv_features": raw_features,
            "gate": "elongation_rule",
        }

    result = classify_image_cnn(io.BytesIO(raw_bytes), model_type=model_type, backend=backend)
    result["gate"] = "cnn"
    return result
