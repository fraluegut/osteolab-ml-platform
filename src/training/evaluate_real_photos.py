"""Evalúa el pipeline completo (segmentación OpenCV + clasificador geométrico)
contra un banco de FOTOS REALES (no renders 3D) — a diferencia de
`evaluate.py`, que mide el modelo sobre features ya extraídas de renders
limpios, este script vuelve a correr `extract_features` desde la imagen
cruda, así que también prueba si la segmentación (`_segment()` en
`src/cv_extractor/extract.py`) aguanta fondos/iluminación/texto reales, no
solo si el clasificador acierta el grupo.

Las imágenes viven en `data/real_world_benchmark/<bone_group>/*` (no
versionadas, ver .gitignore — son fotos de terceros descargadas de la web
solo para evaluación interna) con un `manifest.csv` que además de la
etiqueta trae metadatos (con/sin texto superpuesto, tipo de fondo, ángulo)
para poder desglosar dónde falla más.

Uso:
    python -m src.training.evaluate_real_photos
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.training.evaluate_real_photos
"""
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.inference.predict_geometric import classify_image

BASE_DIR = Path(__file__).resolve().parents[1].parent
BENCHMARK_DIR = BASE_DIR / "data/real_world_benchmark"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.csv"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except Exception:
        return "desconocido"


def _iter_images():
    """Recorre el manifest si existe (trae metadatos); si no, cae a listar
    las carpetas directamente (solo con la etiqueta de grupo)."""
    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)
        for _, row in df.iterrows():
            path = BENCHMARK_DIR / row["bone_group"] / row["filename"]
            if not path.exists():
                print(f"AVISO: en manifest.csv pero no existe en disco: {path}")
                continue
            yield path, row["bone_group"], row.to_dict()
        return

    for group_dir in sorted(BENCHMARK_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        for path in sorted(group_dir.iterdir()):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                yield path, group_dir.name, {}


def main():
    if not BENCHMARK_DIR.exists():
        raise SystemExit(f"No existe {BENCHMARK_DIR}.")

    y_true, y_pred, rows = [], [], []
    n_not_found = 0

    for path, true_group, meta in _iter_images():
        with open(path, "rb") as fh:
            try:
                result = classify_image(fh)
            except Exception as exc:
                print(f"AVISO: error clasificando {path.name}: {exc}")
                result = {"found": False}

        found = result.get("found", False)
        pred = result.get("prediction") if found else "sin_segmentar"
        if not found:
            n_not_found += 1

        y_true.append(true_group)
        y_pred.append(pred)
        rows.append({
            "path": str(path.relative_to(BENCHMARK_DIR)),
            "true": true_group,
            "pred": pred,
            "correct": pred == true_group,
            "found": found,
            **{k: v for k, v in meta.items() if k not in ("bone_group", "filename")},
        })

    total = len(y_true)
    if total == 0:
        raise SystemExit(f"No se encontraron imágenes bajo {BENCHMARK_DIR}.")

    accuracy = accuracy_score(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    report_txt = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    print(f"Banco de fotos reales: {total} imágenes, {n_not_found} sin segmentar "
          f"({100*n_not_found/total:.1f}%)")
    print(report_txt)
    print(f"Accuracy (fotos reales): {accuracy:.4f}")

    df_rows = pd.DataFrame(rows)
    if "has_text_overlay" in df_rows.columns:
        print("\nAccuracy por presencia de texto/marca de agua:")
        print(df_rows.groupby("has_text_overlay")["correct"].mean().to_string())
    if "background" in df_rows.columns:
        print("\nAccuracy por tipo de fondo:")
        print(df_rows.groupby("background")["correct"].agg(["mean", "count"]).to_string())

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("osteolab-benchmark-real-photos")
    with mlflow.start_run():
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("n_images", total)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("not_found_rate", n_not_found / total)
        for class_name in labels:
            if class_name in report:
                mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
                mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
        print(f"\nCorrida logueada en MLflow (experimento 'osteolab-benchmark-real-photos'), "
              f"run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
