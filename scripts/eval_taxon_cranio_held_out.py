import sys
sys.path.insert(0, ".")
import json
import joblib
import torch
import pandas as pd
from shared.inference.predict_cnn import BoneClassifier
from training.train_cnn import evaluate_external

PREFIX = "models/cnn_{}_stage2_taxon_cranio_taxon"
TEST_CSV = "data/processed/bone_geometric_features_stage2_taxon_cranio_test.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(PREFIX.format("model") + ".pt", map_location=device, weights_only=False)
encoder = joblib.load(PREFIX.format("encoder") + ".joblib")
feature_cols = joblib.load(PREFIX.format("feature_columns") + ".joblib")
scaler = joblib.load(PREFIX.format("scaler") + ".joblib")

model = BoneClassifier(backbone_name="resnet18", num_classes=len(encoder.classes_), n_geom_features=len(feature_cols), model_type="hybrid")
model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
model.to(device)

df = pd.read_csv(TEST_CSV)
accuracy, report = evaluate_external(model, df, "taxon", encoder, feature_cols, scaler, device, batch_size=16, num_workers=0)
print("accuracy", accuracy)
for cls in encoder.classes_:
    r = report[cls]
    print(f"{cls:15s} recall={r['recall']:.3f} precision={r['precision']:.3f} n={int(r['support'])}")
