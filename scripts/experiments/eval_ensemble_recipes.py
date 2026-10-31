"""Experimento puntual (2026-10-31): ensemble de 3 modelos híbridos, cada uno
entrenado solo con UNA receta de render + fotos reales (`hybrid_lab_dedup_seed0`,
`hybrid_bproc_real_seed0`, `hybrid_composited_real_seed0`), en vez de un único
modelo entrenado sobre la mezcla de las tres (`combined_lab`, producción).
Promedia probabilidades (ensemble blando) y compara contra producción sobre
`benchmark/selection`. Solo evaluación -- no entrena nada, reusa los 3 checkpoints.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from shared.inference.predict_cnn import classify_image
from training.evaluate_cnn_real_photos import SOURCES

RUN_NAMES = ["hybrid_lab_dedup_seed0", "hybrid_bproc_real_seed0", "hybrid_composited_real_seed0"]
TARGET = "bone_group"


def main():
    benchmark_dir = SOURCES["benchmark"]
    df = pd.read_csv(benchmark_dir / "manifest.csv")
    df = df[df["split"] == "selection"]

    y_true, y_pred = [], []
    n_not_found = 0
    for _, row in df.iterrows():
        path = benchmark_dir / row["bone_group"] / row["filename"]
        if not path.exists():
            continue
        results = []
        for rn in RUN_NAMES:
            with path.open("rb") as f:
                results.append(classify_image(f, model_type="hybrid", run_name=rn, target=TARGET))

        found = [r for r in results if r["found"]]
        if not found:
            n_not_found += 1
            pred = "sin_segmentar"
        else:
            classes = sorted(set().union(*[r["probabilities"].keys() for r in found]))
            avg = {c: sum(r["probabilities"].get(c, 0.0) for r in found) / len(found) for c in classes}
            pred = max(avg, key=avg.get)

        y_true.append(row[TARGET])
        y_pred.append(pred)

    classes = sorted(set(y_true) | set(y_pred))
    print(f"Ensemble 3 recetas: {len(y_true)} imágenes, {n_not_found} sin segmentar")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()
