"""Entrenamiento del filtro binario "es hueso / no es hueso" (ResNet18).

Módulo aislado: no depende de `src/training` ni `src/inference`, ni comparte
datos, modelos o parámetros con el clasificador multi-clase (craneo/femur/
humero). Pensado para lanzarse a mano en tu máquina con GPU.

Dataset esperado (una carpeta por clase, formato ImageFolder de torchvision):

    data/raw/bone_filter/
        bone/         -> fotos de huesos (cualquier tipo)
        not_bone/     -> fotos de cosas que NO son huesos

Uso:
    python src/bone_filter/train.py
    python src/bone_filter/train.py --epochs 15 --batch-size 64 --lr 1e-4
    python src/bone_filter/train.py --freeze-backbone   # solo entrena la última capa

Guarda:
    models/bone_filter/bone_filter_resnet18.pt   (state_dict del mejor modelo)
    models/bone_filter/labels.json                (mapeo clase -> índice)
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.bone_filter.model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    MODEL_DIR,
    build_model,
    get_device,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "bone_filter"
REQUIRED_CLASSES = {"bone", "not_bone"}


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def load_datasets(data_dir: Path, val_split: float, seed: int):
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe {data_dir}")

    found_classes = {p.name for p in data_dir.iterdir() if p.is_dir()}
    missing = REQUIRED_CLASSES - found_classes
    if missing:
        raise ValueError(
            f"Faltan carpetas de clase en {data_dir}: {sorted(missing)}. "
            f"Se esperan exactamente: {sorted(REQUIRED_CLASSES)}"
        )

    train_tf, eval_tf = build_transforms()

    base_train = datasets.ImageFolder(data_dir, transform=train_tf)
    base_eval = datasets.ImageFolder(data_dir, transform=eval_tf)

    if base_train.classes != sorted(REQUIRED_CLASSES):
        raise ValueError(f"Clases inesperadas: {base_train.classes}")

    n = len(base_train)
    if n == 0:
        raise ValueError(f"No se encontraron imágenes en {DATA_DIR}")

    generator = torch.Generator().manual_seed(seed)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val
    perm = torch.randperm(n, generator=generator).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = Subset(base_train, train_idx)
    val_ds = Subset(base_eval, val_idx)
    return train_ds, val_ds, base_train.class_to_idx


def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    weights_path = output_dir / "bone_filter_resnet18.pt"
    labels_path = output_dir / "labels.json"

    device = get_device()
    print(f"Usando device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, val_ds, class_to_idx = load_datasets(data_dir, args.val_split, args.seed)
    print(f"Train: {len(train_ds)} imágenes | Val: {len(val_ds)} imágenes | Clases: {class_to_idx}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(pretrained=True).to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        print("Backbone congelado: solo se entrena la última capa (fc).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            with open(labels_path, "w") as f:
                json.dump(class_to_idx, f, indent=2)
            print(f"  -> Nuevo mejor modelo guardado (val_acc={val_acc:.4f})")

    print(f"Entrenamiento terminado. Mejor val_acc={best_val_acc:.4f}")
    print(f"Pesos: {weights_path}")
    print(f"Labels: {labels_path}")

    _log_to_mlflow_if_available(args, best_val_acc, weights_path, labels_path)


def _mlflow_server_reachable(tracking_uri: str, timeout: float = 2.0) -> bool:
    """Comprobación rápida por socket para no bloquear el entrenamiento
    si el servidor de MLflow no está levantado (el cliente de mlflow por
    defecto reintenta durante minutos antes de fallar)."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(tracking_uri)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _log_to_mlflow_if_available(args, best_val_acc, weights_path, labels_path):
    """Registro best-effort en MLflow, en un experimento propio y aislado.

    Si no hay servidor de MLflow disponible, no rompe el entrenamiento.
    """
    import os

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not _mlflow_server_reachable(tracking_uri):
        print(f"Aviso: MLflow ({tracking_uri}) no está disponible, se omite el registro.")
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("osteolab-bone-filter")
        with mlflow.start_run():
            mlflow.log_params({
                "model_name": "resnet18",
                "img_size": IMG_SIZE,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "val_split": args.val_split,
                "freeze_backbone": args.freeze_backbone,
            })
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.log_artifact(str(weights_path), artifact_path="artifacts")
            mlflow.log_artifact(str(labels_path), artifact_path="artifacts")
        print("Métricas registradas en MLflow (experimento: osteolab-bone-filter)")
    except Exception as e:
        print(f"Aviso: no se pudo registrar en MLflow ({e}). El modelo se guardó igualmente en local.")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el filtro binario hueso / no-hueso")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Congela el backbone de ResNet18 y solo entrena la capa final (más rápido, dataset pequeño).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
