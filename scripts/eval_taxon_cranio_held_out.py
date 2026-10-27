"""Examen honesto de stage2_taxon_cranio contra el test real reservado
(bone_geometric_features_stage2_taxon_cranio_test.csv, nunca visto en
entrenamiento). Ad-hoc, sesión 2026-10-27.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.cnn_dataset import BoneImageDataset  # noqa: E402
from training.train_cnn import BoneClassifier, evaluate_external  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = "stage2_taxon_cranio"
TARGET = "taxon"

encoder = joblib.load(REPO_ROOT / f"models/cnn_encoder_{RUN}_{TARGET}.joblib")
scaler = joblib.load(REPO_ROOT / f"models/cnn_scaler_{RUN}_{TARGET}.joblib")
feature_cols = joblib.load(REPO_ROOT / f"models/cnn_feature_columns_{RUN}_{TARGET}.joblib")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BoneClassifier(backbone_name="resnet18", num_classes=len(encoder.classes_),
                        n_geom_features=len(feature_cols), model_type="hybrid")
checkpoint = torch.load(REPO_ROOT / f"models/cnn_model_{RUN}_{TARGET}.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)

test_df = pd.read_csv(REPO_ROOT / f"data/processed/bone_geometric_features_{RUN}_test.csv")
print(f"Test: {len(test_df)} filas")
print(test_df["taxon"].value_counts().to_string())

accuracy, report = evaluate_external(model, test_df, TARGET, encoder, feature_cols, scaler, device,
                                      batch_size=16, num_workers=0)
print(f"\nAccuracy examen honesto (test real reservado): {accuracy:.4f}")
for cls, m in report.items():
    if cls in encoder.classes_:
        print(f"  {cls:15s} recall={m['recall']:.3f}  n={int(m['support'])}")
