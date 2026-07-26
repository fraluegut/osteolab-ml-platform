"""Evalúa el modelo ya entrenado contra el banco de imágenes fijo
(`data/processed/bone_geometric_benchmark.csv`, especímenes que
`build_dataset.py` reserva y NUNCA mete en el CSV de entrenamiento — ver
`BENCHMARK_SPECIMENS` ahí) y loguea el resultado como una corrida de MLflow.

Pensado para ejecutarse cada vez que cambie el modelo (nuevas features, más
especímenes de entrenamiento, otro algoritmo...) y comparar en la UI de
MLflow (http://localhost:5000, experimento `osteolab-benchmark`) si el nuevo
modelo mejora o empeora sobre el MISMO examen — a diferencia del accuracy que
imprime `train_geometric.py`, que cambia de valor cada vez que cambian los
datos de entrenamiento y no es comparable entre corridas.

Uso:
    python -m src.training.evaluate
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.training.evaluate
"""
import json
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.training.metrics_utils import extra_metrics, format_confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1].parent
MODEL_DIR = BASE_DIR / "models"
BENCHMARK_PATH = BASE_DIR / "data/processed/bone_geometric_benchmark.csv"
TARGET = "bone_group"

NON_FEATURE_COLS = {"species", "bone", "bone_group", "specimen", "view", "source_path", "found", "segmented_path"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except Exception:
        return "desconocido"


def _training_meta_params():
    """Params de qué entrenamiento produjo el modelo actual en MODEL_DIR (ver
    `models/geometric_training_meta_<target>.json`, escrito por
    `train_geometric.py`) — para poder ver en esta misma run de MLflow contra
    qué dataset se entrenó el modelo evaluado, sin tener que adivinarlo por
    fecha. Modelos entrenados antes de que esto existiera no tienen el
    fichero: se evalúa igual, solo sin estos params."""
    meta_path = MODEL_DIR / f"geometric_training_meta_{TARGET}.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    return {
        "training_run_id": meta.get("mlflow_run_id", "desconocido"),
        "training_dataset_path": meta.get("dataset_path", "desconocido"),
        "training_n_rows": meta.get("n_train_rows", -1),
    }


def main():
    if not BENCHMARK_PATH.exists():
        raise SystemExit(
            f"No existe {BENCHMARK_PATH}. Generarlo con "
            "`python -m src.cv_extractor.build_dataset` primero."
        )

    model = joblib.load(MODEL_DIR / f"geometric_model_{TARGET}.joblib")
    encoder = joblib.load(MODEL_DIR / f"geometric_encoder_{TARGET}.joblib")
    feature_cols = joblib.load(MODEL_DIR / f"geometric_feature_columns_{TARGET}.joblib")

    df = pd.read_csv(BENCHMARK_PATH)
    X = df[feature_cols].to_numpy()
    y = df[TARGET].to_numpy()
    y_encoded = encoder.transform(y)

    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)
    accuracy = accuracy_score(y_encoded, preds)
    report = classification_report(
        y_encoded, preds, labels=range(len(encoder.classes_)), target_names=encoder.classes_,
        output_dict=True, zero_division=0,
    )
    extra = extra_metrics(y_encoded, preds, y_proba=proba, n_classes=len(encoder.classes_))

    print(f"Banco de evaluación: {len(df)} imágenes, {df['specimen'].nunique()} especímenes "
          f"nunca vistos en entrenamiento")
    print(classification_report(
        y_encoded, preds, labels=range(len(encoder.classes_)), target_names=encoder.classes_,
        zero_division=0,
    ))
    print(f"Accuracy (banco fijo): {accuracy:.4f}")
    print(f"Balanced accuracy: {extra['balanced_accuracy']:.4f}")
    if "top3_accuracy" in extra:
        print(f"Top-3 accuracy: {extra['top3_accuracy']:.4f}")
    print("\nMatriz de confusión:")
    print(format_confusion_matrix(y_encoded, preds, list(encoder.classes_)))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("osteolab-benchmark")
    with mlflow.start_run():
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("n_benchmark_images", len(df))
        mlflow.log_param("n_benchmark_specimens", int(df["specimen"].nunique()))
        mlflow.log_param("n_train_features", len(feature_cols))
        mlflow.log_params(_training_meta_params())
        mlflow.log_metric("accuracy", accuracy)
        for metric_name, value in extra.items():
            mlflow.log_metric(metric_name, value)
        mlflow.log_text(format_confusion_matrix(y_encoded, preds, list(encoder.classes_)), "confusion_matrix.txt")
        for class_name in encoder.classes_:
            mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
            mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
            mlflow.log_metric(f"f1_{class_name}", report[class_name]["f1-score"])
        print(f"\nCorrida logueada en MLflow (experimento 'osteolab-benchmark'), "
              f"run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
