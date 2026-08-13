"""Genera curvas ROC multiclase sobre el banco fijo de sujetos no vistos.

Uso:
    python -m training.plot_roc_curves
    python -m training.plot_roc_curves --output-dir reports/roc_curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from shared.inference.predict_cnn import _DEVICE, _load
from training.cnn_dataset import BoneImageDataset


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_DIR = BASE_DIR / "reports/roc_curves"
TARGET = "bone_group"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _geometric_predictions():
    benchmark = BASE_DIR / "data/processed/bone_geometric_benchmark.csv"
    df = pd.read_csv(benchmark)
    model = joblib.load(MODEL_DIR / "geometric_model_bone_group.joblib")
    encoder = joblib.load(MODEL_DIR / "geometric_encoder_bone_group.joblib")
    feature_cols = joblib.load(MODEL_DIR / "geometric_feature_columns_bone_group.joblib")
    y_true = encoder.transform(df[TARGET])
    return df, encoder.classes_, y_true, model.predict_proba(df[feature_cols].to_numpy())


def _cnn_predictions(model_type, run_name=None, blackbg=False, batch_size=64, num_workers=0):
    suffix = "_blackbg" if blackbg else ""
    benchmark = BASE_DIR / f"data/processed/bone_geometric_benchmark{suffix}.csv"
    df = pd.read_csv(benchmark)
    model, encoder, feature_cols, scaler, _ = _load(model_type, run_name)
    dataset = BoneImageDataset(df, TARGET, encoder, feature_cols, scaler, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probabilities, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, geom, batch_labels in loader:
            logits = model(images.to(_DEVICE), geom.to(_DEVICE))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    return df, encoder.classes_, np.concatenate(labels), np.concatenate(probabilities)


def calculate_roc(y_true, probabilities, classes):
    y_binary = label_binarize(y_true, classes=np.arange(len(classes)))
    curves = {}
    for index, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_binary[:, index], probabilities[:, index])
        curves[str(class_name)] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

    micro_fpr, micro_tpr, _ = roc_curve(y_binary.ravel(), probabilities.ravel())
    all_fpr = np.unique(np.concatenate([curve["fpr"] for curve in curves.values()]))
    mean_tpr = np.zeros_like(all_fpr)
    for curve in curves.values():
        mean_tpr += np.interp(all_fpr, curve["fpr"], curve["tpr"])
    mean_tpr /= len(classes)
    curves["micro"] = {"fpr": micro_fpr, "tpr": micro_tpr, "auc": auc(micro_fpr, micro_tpr)}
    curves["macro"] = {"fpr": all_fpr, "tpr": mean_tpr, "auc": auc(all_fpr, mean_tpr)}
    curves["macro_auc_ovr"] = float(
        roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr")
    )
    return curves


def plot_model(name, curves, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves) - 3))
    class_names = [key for key in curves if key not in {"micro", "macro", "macro_auc_ovr"}]
    for color, class_name in zip(colors, class_names):
        curve = curves[class_name]
        ax.plot(curve["fpr"], curve["tpr"], color=color, lw=1.5,
                label=f"{class_name} (AUC={curve['auc']:.3f})")
    for average, style in (("micro", "--"), ("macro", "-.")):
        curve = curves[average]
        ax.plot(curve["fpr"], curve["tpr"], style, lw=3,
                label=f"{average} (AUC={curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", label="Azar (AUC=0.500)")
    ax.set(title=f"Curvas ROC multiclase — {name}", xlabel="Tasa de falsos positivos",
           ylabel="Tasa de verdaderos positivos", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _serializable(curves):
    result = {}
    for key, value in curves.items():
        if key == "macro_auc_ovr":
            result[key] = value
        else:
            result[key] = {
                "fpr": value["fpr"].tolist(),
                "tpr": value["tpr"].tolist(),
                "auc": float(value["auc"]),
            }
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictors = {
        "geometric": lambda: _geometric_predictions(),
        "cnn_only": lambda: _cnn_predictions("cnn_only", batch_size=args.batch_size,
                                                num_workers=args.num_workers),
        "hybrid": lambda: _cnn_predictions("hybrid", batch_size=args.batch_size,
                                              num_workers=args.num_workers),
        "hybrid_blackbg": lambda: _cnn_predictions(
            "hybrid", run_name="hybrid_blackbg", blackbg=True,
            batch_size=args.batch_size, num_workers=args.num_workers,
        ),
    }
    all_curves, summary_rows = {}, []
    for name, predict in predictors.items():
        print(f"Calculando ROC: {name}...")
        df, classes, y_true, probabilities = predict()
        curves = calculate_roc(y_true, probabilities, classes)
        all_curves[name] = curves
        plot_model(name, curves, args.output_dir / f"roc_{name}.png")
        for class_name in classes:
            summary_rows.append({"model": name, "scope": class_name,
                                 "auc": curves[str(class_name)]["auc"],
                                 "n_images": len(df), "n_subjects": df["specimen"].nunique()})
        for average in ("micro", "macro"):
            summary_rows.append({"model": name, "scope": average,
                                 "auc": curves[average]["auc"],
                                 "n_images": len(df), "n_subjects": df["specimen"].nunique()})

    fig, ax = plt.subplots(figsize=(9, 7))
    for name, curves in all_curves.items():
        curve = curves["macro"]
        ax.plot(curve["fpr"], curve["tpr"], lw=2.5, label=f"{name} (AUC={curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", label="Azar")
    ax.set(title="Comparativa ROC macro — sujetos no vistos", xlabel="Tasa de falsos positivos",
           ylabel="Tasa de verdaderos positivos", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "roc_models_comparison.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "roc_auc_summary.csv", index=False)
    payload = {name: _serializable(curves) for name, curves in all_curves.items()}
    (args.output_dir / "roc_curve_points.json").write_text(json.dumps(payload, indent=2))
    print("\nAUC macro por modelo:")
    print(summary[summary["scope"] == "macro"][["model", "auc"]].to_string(index=False))
    print(f"\nResultados guardados en {args.output_dir}")


if __name__ == "__main__":
    main()
