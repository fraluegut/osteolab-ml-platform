"""Evalúa un modelo CNN/híbrido ya entrenado (`train_cnn.py`) contra el
banco de evaluación fijo (`data/processed/bone_geometric_benchmark.csv`,
mismos especímenes reservados que usa `evaluate.py` para el modelo
geométrico) — mismo examen, para poder comparar directamente en la UI de
MLflow (experimento `osteolab-benchmark`, igual que `evaluate.py`, con un
param `model_type` para distinguir filas).

Uso:
    python -m training.evaluate_cnn --model-type cnn_only
    python -m training.evaluate_cnn --model-type hybrid
"""
import argparse
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, top_k_accuracy_score

from shared.inference.predict_cnn import classify_image
from training.metrics_utils import format_confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = BASE_DIR / "data/processed/bone_geometric_benchmark.csv"
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
    p.add_argument("--model-type", choices=["cnn_only", "hybrid"], default="hybrid")
    return p.parse_args()


def main():
    args = parse_args()
    if not BENCHMARK_PATH.exists():
        raise SystemExit(f"No existe {BENCHMARK_PATH}. Generarlo con `python -m shared.cv_extractor.build_dataset`.")

    df = pd.read_csv(BENCHMARK_PATH)
    y_true, y_pred = [], []
    probas, proba_true = [], []
    for _, row in df.iterrows():
        with open(row["source_path"], "rb") as f:
            result = classify_image(f, model_type=args.model_type)
        y_true.append(row[TARGET])
        y_pred.append(result["prediction"] if result["found"] else "sin_segmentar")
        if result["found"] and result.get("probabilities"):
            probas.append(result["probabilities"])
            proba_true.append(row[TARGET])

    classes = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    print(f"Banco de evaluación ({args.model_type}): {len(df)} imágenes, "
          f"{df['specimen'].nunique()} especímenes nunca vistos en entrenamiento")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    print(f"Accuracy (banco fijo, {args.model_type}): {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    top3_acc = None
    if probas:
        proba_labels = sorted({c for p in probas for c in p})
        proba_matrix = [[p.get(c, 0.0) for c in proba_labels] for p in probas]
        try:
            top3_acc = top_k_accuracy_score(proba_true, proba_matrix, k=min(3, len(proba_labels)), labels=proba_labels)
            print(f"Top-3 accuracy: {top3_acc:.4f}")
        except ValueError:
            pass
    print("\nMatriz de confusión:")
    print(format_confusion_matrix([classes.index(v) for v in y_true], [classes.index(v) for v in y_pred], classes))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("osteolab-benchmark")
    with mlflow.start_run():
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_benchmark_images", len(df))
        mlflow.log_param("n_benchmark_specimens", int(df["specimen"].nunique()))
        mlflow.set_tag("model_type", args.model_type)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("balanced_accuracy", balanced_acc)
        if top3_acc is not None:
            mlflow.log_metric("top3_accuracy", top3_acc)
        mlflow.log_text(
            format_confusion_matrix([classes.index(v) for v in y_true], [classes.index(v) for v in y_pred], classes),
            "confusion_matrix.txt",
        )
        for class_name in df[TARGET].unique():
            if class_name in report:
                mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
                mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
                mlflow.log_metric(f"f1_{class_name}", report[class_name]["f1-score"])
        print(f"\nCorrida logueada en MLflow (experimento 'osteolab-benchmark'), "
              f"run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
