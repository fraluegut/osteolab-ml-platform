"""Entrena un clasificador de `bone_group` a partir de PÍXELES (backbone
preentrenado vía `timm`), en vez de las features geométricas de OpenCV que
usa `train_geometric.py` — ver el pendiente del 2026-07-25 en
`base_datos_osea/HISTORIAL.md` ("¿ayuda una CNN?"): el extractor geométrico
descarta toda la textura/sombreado, y el hueco entre el banco limpio
sintético (0.77) y las fotos reales (0.27) sugiere que hay señal que el
pipeline actual no usa.

Dos variantes, elegidas con `--model-type`:
  - `cnn_only`: backbone + cabeza lineal, aprende solo de la imagen.
  - `hybrid`: backbone (como extractor de embeddings) + el mismo vector de
    features geométricas que usa el modelo RandomForest, concatenados antes
    de una cabeza MLP pequeña — la apuesta es que la forma (geométrico) y la
    textura/sombreado (CNN) son señales complementarias.

Entrenamiento en 2 fases (transfer learning estándar): fase 1 con el
backbone congelado (solo se entrena la cabeza, para no destrozar los pesos
preentrenados con gradientes grandes desde una cabeza sin entrenar); fase 2
con el backbone descongelado a LR bajo + cabeza a LR normal (grupos de
parámetros distintos en el mismo optimizador).

Reutiliza las tablas ya construidas por `build_dataset.py`
(`bone_geometric_features_combined.csv` como train/val,
`bone_geometric_benchmark.csv` como banco externo limpio) en vez de recorrer
`renders/` de nuevo — ver `cnn_dataset.py`.

Uso:
    python -m src.training.train_cnn --model-type cnn_only
    python -m src.training.train_cnn --model-type hybrid --epochs-frozen 5 --epochs-finetune 10
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.training.cnn_dataset import (
    BoneImageDataset,
    geometric_feature_columns,
    resolve_source_path,
    specimen_split,
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
TARGET = "bone_group"
DEFAULT_DATASET = BASE_DIR / "data/processed/bone_geometric_features_combined.csv"
DEFAULT_BENCHMARK = BASE_DIR / "data/processed/bone_geometric_benchmark.csv"


class BoneClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, n_geom_features: int, model_type: str):
        super().__init__()
        self.model_type = model_type
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features
        head_in = embed_dim + (n_geom_features if model_type == "hybrid" else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, geom):
        embedding = self.backbone(image)
        if self.model_type == "hybrid":
            embedding = torch.cat([embedding, geom], dim=1)
        return self.head(embedding)

    def backbone_params(self):
        return self.backbone.parameters()

    def head_params(self):
        return self.head.parameters()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["cnn_only", "hybrid"], default="cnn_only")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--epochs-frozen", type=int, default=5)
    p.add_argument("--epochs-finetune", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for images, geom, labels in loader:
            images, geom, labels = images.to(device), geom.to(device), labels.to(device)
            logits = model(images, geom)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), accuracy_score(labels, preds), preds, labels


def evaluate_external(model, df, target_col, label_encoder, feature_cols, scaler, device, batch_size, num_workers):
    dataset = BoneImageDataset(df, target_col, label_encoder, feature_cols, scaler, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, geom, labels in loader:
            images, geom = images.to(device), geom.to(device)
            logits = model(images, geom)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    accuracy = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    return accuracy, report


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.dataset_path)
    bench_df = pd.read_csv(args.benchmark_path) if args.benchmark_path.exists() else None

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET])
    num_classes = len(label_encoder.classes_)

    feature_cols = geometric_feature_columns(df) if args.model_type == "hybrid" else None
    n_geom_features = len(feature_cols) if feature_cols else 0

    if feature_cols is not None:
        # Un puñado de filas se quedan sin `ellipse_ratio` cuando el contorno
        # tiene <5 puntos (ver `_ellipse_ratio` en extract.py). A diferencia
        # de RandomForestClassifier (tolera NaN nativamente), el NaN llega tal
        # cual al tensor de la CNN vía StandardScaler y colapsa el
        # entrenamiento entero a loss=nan desde el primer batch que lo
        # contenga. Imputar con la media de columna, igual que
        # `_fallback_means` en inferencia.
        col_means = df[feature_cols].mean(numeric_only=True)
        df[feature_cols] = df[feature_cols].fillna(col_means)
        if bench_df is not None:
            bench_df[feature_cols] = bench_df[feature_cols].fillna(col_means)

    train_df, val_df = specimen_split(df, test_size=args.val_size, random_state=args.random_state)
    print(f"Train: {len(train_df)} filas / {train_df['specimen'].nunique()} especímenes  "
          f"Val: {len(val_df)} filas / {val_df['specimen'].nunique()} especímenes")
    leak = set(train_df["specimen"]) & set(val_df["specimen"])
    if leak:
        raise RuntimeError(f"Fuga de espécimen entre train/val: {leak}")

    scaler = None
    if feature_cols is not None:
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    train_dataset = BoneImageDataset(train_df, TARGET, label_encoder, feature_cols, scaler, train=True)
    val_dataset = BoneImageDataset(val_df, TARGET, label_encoder, feature_cols, scaler, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    y_train_encoded = label_encoder.transform(train_df[TARGET])
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train_encoded)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    model = BoneClassifier(args.backbone, num_classes, n_geom_features, args.model_type).to(device)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(f"osteolab-bone-classification-cnn-{TARGET}")

    try:
        dataset_path_display = str(args.dataset_path.relative_to(BASE_DIR))
    except ValueError:
        dataset_path_display = str(args.dataset_path)

    with mlflow.start_run():
        mlflow.log_params({
            "model_type": args.model_type,
            "backbone": args.backbone,
            "n_geom_features": n_geom_features,
            "epochs_frozen": args.epochs_frozen,
            "epochs_finetune": args.epochs_finetune,
            "batch_size": args.batch_size,
            "lr_head": args.lr_head,
            "lr_backbone": args.lr_backbone,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "dataset_path": dataset_path_display,
            "n_train_rows": len(train_df),
            "n_train_specimens": int(train_df["specimen"].nunique()),
            "n_val_rows": len(val_df),
            "n_val_specimens": int(val_df["specimen"].nunique()),
        })
        mlflow.set_tag("model_type", args.model_type)
        for tag in args.tag:
            if "=" not in tag:
                raise SystemExit(f"--tag mal formado (esperaba KEY=VALUE): {tag!r}")
            key, value = tag.split("=", 1)
            mlflow.set_tag(key, value)

        # Fase 1: backbone congelado, solo se entrena la cabeza.
        for p in model.backbone_params():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(model.head_params(), lr=args.lr_head)

        best_val_accuracy = -1.0
        best_state = None
        epoch = 0
        total_epochs = args.epochs_frozen + args.epochs_finetune
        for phase_epoch in range(args.epochs_frozen):
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc, _, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)
            print(f"[fase1 congelada {phase_epoch + 1}/{args.epochs_frozen}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc,
                                 "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epoch += 1

        # Fase 2: backbone descongelado a LR bajo, cabeza sigue a LR normal.
        for p in model.backbone_params():
            p.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": model.backbone_params(), "lr": args.lr_backbone},
            {"params": model.head_params(), "lr": args.lr_head},
        ])
        for phase_epoch in range(args.epochs_finetune):
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc, _, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)
            print(f"[fase2 fine-tune {phase_epoch + 1}/{args.epochs_finetune}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc,
                                 "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epoch += 1

        model.load_state_dict(best_state)
        print(f"\nMejor accuracy en validación (espécimen fuera): {best_val_accuracy:.4f}")
        mlflow.log_metric("best_val_acc", best_val_accuracy)

        _, val_report = evaluate_external(model, val_df, TARGET, label_encoder, feature_cols, scaler,
                                           device, args.batch_size, args.num_workers)
        print("\nRecall por grupo en validación (espécimen fuera):")
        for class_name in label_encoder.classes_:
            recall = val_report[class_name]["recall"]
            print(f"  {class_name:<20} {recall:.3f}")
            mlflow.log_metric(f"val_recall_{class_name}", recall)

        bench_accuracy = None
        if bench_df is not None:
            bench_accuracy, bench_report = evaluate_external(
                model, bench_df, TARGET, label_encoder, feature_cols, scaler,
                device, args.batch_size, args.num_workers,
            )
            print(f"\nAccuracy banco fijo (renders limpios, especímenes nunca vistos): {bench_accuracy:.4f}")
            mlflow.log_metric("accuracy_clean_benchmark", bench_accuracy)
            for class_name in label_encoder.classes_:
                mlflow.log_metric(f"bench_recall_{class_name}", bench_report[class_name]["recall"])

        run_id = mlflow.active_run().info.run_id
        suffix = args.model_type
        model_path = MODEL_DIR / f"cnn_model_{suffix}_{TARGET}.pt"
        encoder_path = MODEL_DIR / f"cnn_encoder_{suffix}_{TARGET}.joblib"
        torch.save({
            "state_dict": model.state_dict(),
            "backbone": args.backbone,
            "model_type": args.model_type,
            "num_classes": num_classes,
            "n_geom_features": n_geom_features,
        }, model_path)
        joblib.dump(label_encoder, encoder_path)
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(encoder_path), artifact_path="artifacts")

        if feature_cols is not None:
            columns_path = MODEL_DIR / f"cnn_feature_columns_{suffix}_{TARGET}.joblib"
            scaler_path = MODEL_DIR / f"cnn_scaler_{suffix}_{TARGET}.joblib"
            joblib.dump(feature_cols, columns_path)
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(str(columns_path), artifact_path="artifacts")
            mlflow.log_artifact(str(scaler_path), artifact_path="artifacts")

        meta = {
            "target": TARGET,
            "model_type": args.model_type,
            "backbone": args.backbone,
            "dataset_path": dataset_path_display,
            "n_train_rows": len(train_df),
            "n_train_specimens": int(train_df["specimen"].nunique()),
            "mlflow_run_id": run_id,
            "mlflow_experiment": mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            "best_val_accuracy": best_val_accuracy,
            "accuracy_clean_benchmark": bench_accuracy,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "tags": {t.split("=", 1)[0]: t.split("=", 1)[1] for t in args.tag},
        }
        meta_path = MODEL_DIR / f"cnn_training_meta_{suffix}_{TARGET}.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\nModelo guardado en: {model_path}")
        print(f"Metadatos guardados en: {meta_path}")
        print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
