"""Evalúa una variante CNN/híbrida (`train_cnn.py`) contra el banco de FOTOS
REALES (`data/real_world_benchmark/`) — análogo a `evaluate_real_photos.py`
pero con `shared.inference.predict_cnn` en vez del modelo geométrico. Es la
comparación que de verdad importa (ver HISTORIAL.md, 2026-07-25): el banco
limpio sintético es optimista para cualquier modelo entrenado sobre renders,
las fotos reales son el examen honesto.

Uso:
    python -m training.evaluate_cnn_real_photos --model-type hybrid
    python -m training.evaluate_cnn_real_photos --model-type cnn_only
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
TARGET = "bone_group"

# 2026-08-03: dos bancos de fotos reales disponibles — `benchmark` es el
# original (63 fotos externas, `selection`/`held_out`, n=9 en el examen);
# `accepted` es el pool de entrenamiento (357 fotos y creciendo) con su
# propio split `train`/`test` (~20% reservado, 72 fotos) — mucho más grande
# que el held_out del benchmark, mejor para medir con confianza. Ver
# HISTORIAL.md.
SOURCES = {
    "benchmark": BASE_DIR / "data/real_world_benchmark",
    "accepted": BASE_DIR / "data/real_photos_accepted",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except Exception:
        return "desconocido"


def _iter_images(benchmark_dir: Path, manifest_path: Path, split: str | None):
    df = pd.read_csv(manifest_path)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    for _, row in df.iterrows():
        path = benchmark_dir / row["bone_group"] / row["filename"]
        if not path.exists():
            print(f"AVISO: en manifest.csv pero no existe en disco: {path}")
            continue
        yield path, row["bone_group"], row.to_dict()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["cnn_only", "hybrid"], default="hybrid")
    p.add_argument("--segmenter", choices=["classic", "fastsam", "grounded_sam"], default="grounded_sam",
                   help="backend de segmentación (solo afecta a 'hybrid', 'cnn_only' no usa geometría)")
    p.add_argument("--source", choices=["benchmark", "accepted"], default="benchmark",
                   help="'benchmark' = data/real_world_benchmark (63 fotos externas, splits "
                        "'selection'/'held_out'). 'accepted' = data/real_photos_accepted (el pool "
                        "de entrenamiento, 357+ fotos, splits 'train'/'test' — el 'test' es mucho "
                        "más grande que el held_out del benchmark, 72 vs 9 fotos).")
    p.add_argument("--split", default="all",
                   help="'all' = sin filtrar. Con --source=benchmark: 'selection'/'held_out'. Con "
                        "--source=accepted: 'train'/'test' (usar 'test' para el examen honesto, "
                        "NUNCA 'train' — esas fotos ya las vio el modelo). 2026-08-03: si "
                        "train_cnn.py seleccionó el checkpoint maximizando real_bench_acc sobre "
                        "'selection', evaluar con --split=all o --split=selection en el benchmark "
                        "es circular — ver HISTORIAL.md.")
    p.add_argument("--run-name", default=None,
                    help="Cargar un modelo guardado con --run-name distinto en train_cnn.py (ej. "
                         "'hybrid_blackbg') en vez del nombre por defecto de --model-type.")
    p.add_argument("--bg-gray", type=int, default=128,
                    help="Color de fondo (0-255) para la segmentación en vivo de este examen — "
                         "debe coincidir con el que se usó para entrenar el modelo que se está "
                         "evaluando (0 para la variante 'blackbg', 128 = gris, el normal).")
    return p.parse_args()


def main():
    args = parse_args()
    benchmark_dir = SOURCES[args.source]
    manifest_path = benchmark_dir / "manifest.csv"
    if not benchmark_dir.exists():
        raise SystemExit(f"No existe {benchmark_dir}.")

    y_true, y_pred, rows = [], [], []
    probas, proba_true = [], []
    n_not_found = 0

    split_filter = None if args.split == "all" else args.split
    for path, true_group, meta in _iter_images(benchmark_dir, manifest_path, split_filter):
        with path.open("rb") as f:
            result = classify_image(f, model_type=args.model_type, backend=args.segmenter,
                                     run_name=args.run_name, bg_gray=args.bg_gray)
        if not result["found"]:
            n_not_found += 1
            pred = "sin_segmentar"
        else:
            pred = result["prediction"]
            if result.get("probabilities"):
                probas.append(result["probabilities"])
                proba_true.append(true_group)
        y_true.append(true_group)
        y_pred.append(pred)
        rows.append({**meta, "bone_group": true_group, "prediction": pred, "correct": pred == true_group})

    classes = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    print(f"Banco de fotos reales ({args.model_type}): {len(y_true)} imágenes, "
          f"{n_not_found} sin segmentar ({n_not_found / len(y_true) * 100:.1f}%)")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    print(f"Accuracy (fotos reales, {args.model_type}): {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    top3_acc = None
    if probas:
        proba_labels = sorted({c for p in probas for c in p})
        proba_matrix = [[p.get(c, 0.0) for c in proba_labels] for p in probas]
        try:
            top3_acc = top_k_accuracy_score(proba_true, proba_matrix, k=min(3, len(proba_labels)), labels=proba_labels)
            print(f"Top-3 accuracy (sobre {len(probas)} imágenes segmentadas): {top3_acc:.4f}")
        except ValueError:
            pass
    print("\nMatriz de confusión:")
    print(format_confusion_matrix([classes.index(v) for v in y_true], [classes.index(v) for v in y_pred], classes))

    results_df = pd.DataFrame(rows)
    if "background" in results_df.columns:
        print("\nAccuracy por tipo de fondo:")
        print(results_df.groupby("background")["correct"].mean().sort_values(ascending=False).to_string())

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("osteolab-benchmark-real-photos")
    with mlflow.start_run():
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_images", len(y_true))
        mlflow.log_param("n_not_found", n_not_found)
        mlflow.log_param("segmenter", args.segmenter)
        mlflow.log_param("source", args.source)
        mlflow.log_param("run_name", args.run_name)
        mlflow.log_param("bg_gray", args.bg_gray)
        mlflow.log_param("split", args.split)
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("segmenter", args.segmenter)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("balanced_accuracy", balanced_acc)
        if top3_acc is not None:
            mlflow.log_metric("top3_accuracy", top3_acc)
        mlflow.log_text(
            format_confusion_matrix([classes.index(v) for v in y_true], [classes.index(v) for v in y_pred], classes),
            "confusion_matrix.txt",
        )
        for class_name in set(y_true):
            if class_name in report:
                mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
                mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
        print(f"\nCorrida logueada en MLflow (experimento 'osteolab-benchmark-real-photos'), "
              f"run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
