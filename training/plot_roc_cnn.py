"""Curvas ROC (one-vs-rest, multiclase) para una variante CNN/híbrida ya
entrenada, evaluada sobre un banco de fotos reales. Complementa a
`evaluate_cnn_real_photos.py` (que ya calcula accuracy/balanced accuracy)
con la visualización de sensibilidad/especificidad por clase.

Uso:
    python -m training.plot_roc_cnn --model-type hybrid --run-name hybrid_blackbg \
        --bg-gray 0 --source accepted --split test
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from shared.inference.predict_cnn import classify_image

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCES = {
    "benchmark": BASE_DIR / "data/real_world_benchmark",
    "accepted": BASE_DIR / "data/real_photos_accepted",
}


def _iter_images(benchmark_dir: Path, manifest_path: Path, split: str | None):
    df = pd.read_csv(manifest_path)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    for _, row in df.iterrows():
        path = benchmark_dir / row["bone_group"] / row["filename"]
        if path.exists():
            yield path, row["bone_group"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["cnn_only", "hybrid", "gated_hybrid"], default="hybrid")
    p.add_argument("--segmenter", choices=["classic", "fastsam", "grounded_sam"], default="grounded_sam")
    p.add_argument("--source", choices=["benchmark", "accepted"], default="accepted")
    p.add_argument("--split", default="test")
    p.add_argument("--run-name", default=None)
    p.add_argument("--bg-gray", type=int, default=128)
    p.add_argument("--out", default=None, help="Ruta del PNG de salida (default: roc_<run_name o model_type>.png)")
    return p.parse_args()


def main():
    args = parse_args()
    benchmark_dir = SOURCES[args.source]
    manifest_path = benchmark_dir / "manifest.csv"
    if not benchmark_dir.exists():
        raise SystemExit(f"No existe {benchmark_dir}.")

    split_filter = None if args.split == "all" else args.split
    y_true, probas = [], []
    n_not_found = 0
    for path, true_group in _iter_images(benchmark_dir, manifest_path, split_filter):
        with path.open("rb") as f:
            result = classify_image(f, model_type=args.model_type, backend=args.segmenter,
                                     run_name=args.run_name, bg_gray=args.bg_gray)
        if not result["found"] or not result.get("probabilities"):
            n_not_found += 1
            continue
        y_true.append(true_group)
        probas.append(result["probabilities"])

    if not probas:
        raise SystemExit("Ninguna imagen segmentada con probabilidades — no se puede graficar ROC.")

    classes = sorted({c for p in probas for c in p})
    proba_matrix = np.array([[p.get(c, 0.0) for c in classes] for p in probas])
    y_bin = label_binarize(y_true, classes=classes)

    tag = args.run_name or args.model_type
    out_path = Path(args.out) if args.out else BASE_DIR / f"roc_{tag}_{args.source}_{args.split}.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    aucs = {}
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba_matrix[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls] = roc_auc
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f}, n={int(y_bin[:, i].sum())})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="azar (AUC=0.500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"ROC one-vs-rest — {tag} ({args.source}/{args.split}, n={len(y_true)})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Guardado: {out_path}")

    macro_auc = float(np.mean(list(aucs.values())))
    print(f"\n{n_not_found} imágenes sin segmentar, excluidas ({n_not_found}/{len(y_true) + n_not_found})")
    print("AUC por clase:")
    for cls, a in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {a:.4f}")
    print(f"Macro AUC: {macro_auc:.4f}")


if __name__ == "__main__":
    main()
