from pathlib import Path
import os
import joblib
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models/model.joblib"
ENCODER_PATH = BASE_DIR / "models/encoder.joblib"
IMG_SIZE = (32, 32)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


def _load_local_model():
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH), "local"


def _load_mlflow_model():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/osteolab-bone-classifier/Production")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder, model_uri
    except Exception:
        return _load_local_model()


def _get_model():
    if MLFLOW_TRACKING_URI:
        return _load_mlflow_model()
    return _load_local_model()


model, encoder, _model_source = _get_model()


def predict_image(image_file):
    img = Image.open(image_file).convert("L")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32).flatten().reshape(1, -1) / 255.0

    pred = model.predict(arr)[0]
    probs = model.predict_proba(arr)[0]
    label = encoder.inverse_transform([pred])[0]

    class_probabilities = {
        encoder.inverse_transform([i])[0]: round(float(prob), 6)
        for i, prob in enumerate(probs)
    }

    return {
        "prediction": label,
        "probabilities": class_probabilities,
        "model_source": _model_source,
    }