"""Experimento puntual (2026-10-30): ¿ayuda subir la intensidad de augmentation
en vez de cambiar la receta del dataset? Parchea build_transform con magnitudes
más agresivas y entrena sobre el dataset de PRODUCCIÓN (combined_lab, 2938 filas,
mejor receta conocida) para aislar el efecto de la augmentation sola, sin mezclar
con un cambio de dataset. No modifica training/cnn_dataset.py -- el parche vive
solo en este script, descartable tras el experimento.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torchvision.transforms import v2

from training.cnn_dataset import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


def build_transform_strong(train: bool, img_size: int = IMAGE_SIZE) -> v2.Compose:
    if train:
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.70, 1.0), ratio=(0.8, 1.25)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(25),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.3, 3.0))], p=0.4),
            v2.RandomApply([v2.JPEG(quality=(15, 90))], p=0.4),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomApply([v2.GaussianNoise(sigma=0.08)], p=0.4),
            v2.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


import training.cnn_dataset as cnn_dataset  # noqa: E402
import training.train_cnn as train_cnn  # noqa: E402

# BoneImageDataset (definida en cnn_dataset.py) llama a build_transform resuelto en el
# namespace de su PROPIO módulo -- parchear train_cnn.build_transform (el alias importado)
# no tiene ningún efecto ahí. Hay que parchear el símbolo en cnn_dataset, que es donde
# realmente se usa.
cnn_dataset.build_transform = build_transform_strong
train_cnn.build_transform = build_transform_strong

if __name__ == "__main__":
    train_cnn.main()
