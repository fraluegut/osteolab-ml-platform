"""Experimento puntual (2026-10-31): mixup estándar (Zhang et al.) sobre batches de
entrenamiento -- mezcla pares de imágenes (y su geometría) con lambda~Beta(0.4,0.4),
pérdida interpolada entre las dos etiquetas. No limitado a pares sintético/real (eso
requeriría exponer `is_real` por fila en el Dataset/DataLoader, fuera de alcance de
un parche aislado) -- mixup estándar entre cualquier par del batch. Solo se aplica en
fase de entrenamiento, nunca en validación. Parche vive solo en este script.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import training.train_cnn as train_cnn  # noqa: E402

MIXUP_ALPHA = 0.4
MIXUP_PROB = 0.5


def run_epoch_mixup(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for images, geom, labels in loader:
            images, geom, labels = images.to(device), geom.to(device), labels.to(device)
            use_mixup = train and np.random.rand() < MIXUP_PROB
            if use_mixup:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(images.size(0), device=device)
                images = lam * images + (1 - lam) * images[perm]
                geom = lam * geom + (1 - lam) * geom[perm]
                labels_b = labels[perm]

            logits = model(images, geom)
            if use_mixup:
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    labels_out = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), accuracy_score(labels_out, preds), preds, labels_out


train_cnn.run_epoch = run_epoch_mixup

if __name__ == "__main__":
    train_cnn.main()
