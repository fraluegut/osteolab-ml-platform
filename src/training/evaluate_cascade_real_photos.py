"""Evalúa la cascada (`src/inference/predict_cascade.py`: regla de elongación
para `hueso_largo` + CNN para el resto) contra el banco de fotos reales —
mismo formato que `evaluate_cnn_real_photos.py`, para comparar en la misma
tabla de MLflow (`osteolab-benchmark-real-photos`).

Uso:
    python -m src.training.evaluate_cascade_real_photos --model-type cnn_only
"""
import argparse
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.inference.predict_cascade import classify_image

BASE_DIR = Path(__file__).resolve().parents[1].parent
BENCHMARK_DIR = BASE_DIR / "data/real_world_benchmark"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.csv"
TARGET = "bone_group"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except Exception:
        return "desconocido"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["cnn_only", "hybrid"], default="cnn_only",
                   help="modelo de respaldo para lo que no capture la puerta de elongación")
    p.add_argument("--segmenter", choices=["classic", "fastsam"], default="classic")
    return p.parse_args()


def main():
    args = parse_args()
    if not BENCHMARK_DIR.exists():
        raise SystemExit(f"No existe {BENCHMARK_DIR}.")

    df = pd.read_csv(MANIFEST_PATH)
    y_true, y_pred, n_gated = [], [], 0

    for _, row in df.iterrows():
        path = BENCHMARK_DIR / row["bone_group"] / row["filename"]
        if not path.exists():
            print(f"AVISO: en manifest.csv pero no existe en disco: {path}")
            continue
        with path.open("rb") as f:
            result = classify_image(f, model_type=args.model_type, backend=args.segmenter)
        if result.get("gate") == "elongation_rule":
            n_gated += 1
        y_true.append(row["bone_group"])
        y_pred.append(result["prediction"] if result["found"] else "sin_segmentar")

    classes = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    print(f"Banco de fotos reales (cascada elongación+{args.model_type}): {len(y_true)} imágenes, "
          f"{n_gated} capturadas por la regla de elongación")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    print(f"Accuracy (fotos reales, cascada): {accuracy:.4f}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("osteolab-benchmark-real-photos")
    with mlflow.start_run():
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("model_type", f"cascade_elongation_{args.model_type}")
        mlflow.log_param("n_images", len(y_true))
        mlflow.log_param("n_gated_by_rule", n_gated)
        mlflow.log_param("segmenter", args.segmenter)
        mlflow.set_tag("model_type", f"cascade_elongation_{args.model_type}")
        mlflow.set_tag("segmenter", args.segmenter)
        mlflow.log_metric("accuracy", accuracy)
        for class_name in set(y_true):
            if class_name in report:
                mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
                mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
        print(f"\nCorrida logueada en MLflow (experimento 'osteolab-benchmark-real-photos'), "
              f"run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
