"""Dataset de PyTorch para las variantes CNN/híbrida del clasificador de
`bone_group`, construido sobre las MISMAS tablas que ya usa el modelo
geométrico (`data/processed/bone_geometric_features*.csv`,
`bone_geometric_benchmark.csv`) en vez de volver a recorrer `renders/` por
separado: esas tablas ya traen `source_path`/`bone_group`/`specimen` por
fila (una fila = una vista ya segmentada con éxito), así que sirven de
manifest gratis y garantizan que CNN y modelo geométrico se miden sobre
exactamente el mismo conjunto de imágenes.

La variante CNN pura NO usa las columnas de features geométricas (aprende
directamente de los píxeles); la híbrida sí, concatenando el embedding del
backbone con ese vector ya escalado. Por eso este módulo siempre expone
ambas cosas (imagen + vector geométrico) por fila y cada script de
entrenamiento decide si usa la segunda.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from training.group_splits import subject_holdout

BASE_DIR = Path(__file__).resolve().parents[1]

# Mismo criterio que train_geometric.py: img_width/img_height son constantes (512x512) en
# entrenamiento y explotan en z-score sobre fotos reales de otra resolución.
NON_FEATURE_COLS = {"species", "bone", "bone_group", "specimen", "view", "source_path", "found",
                     "img_width", "img_height", "segmented_path"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def geometric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def resolve_source_path(raw_path: str) -> Path:
    """Las tablas mezclan paths absolutos (dataset limpio, escritos por una
    sesión anterior) y relativos al repo (dataset con ruido, más reciente) —
    normalizar aquí en vez de en cada script que lea estas tablas."""
    p = Path(raw_path)
    return p if p.is_absolute() else (BASE_DIR / p)


def build_transform(train: bool) -> v2.Compose:
    """Augmentations reforzadas (2026-07-26, propuesta del usuario): antes
    solo había flip+rotación+color jitter — nada que simule cámara/compresión
    real, así que la CNN nunca veía en entrenamiento el tipo de degradación
    que sí tiene una foto de usuario real. Magnitudes deliberadamente
    moderadas (no agresivas): esto sigue siendo una tarea de forma/silueta
    sobre un recorte ya segmentado, un crop o rotación fuerte puede cortar
    la parte anatómica que distingue un grupo de otro.
    """
    if train:
        return v2.Compose([
            v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.3, 2.0))], p=0.25),
            v2.RandomApply([v2.JPEG(quality=(25, 90))], p=0.3),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomApply([v2.GaussianNoise(sigma=0.04)], p=0.3),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BoneImageDataset(Dataset):
    """Cada item: (imagen_tensor, geom_tensor, label_int). `geom_tensor` es
    el vector de features geométricas ya escalado con un `StandardScaler`
    de sklearn ajustado sobre el set de entrenamiento — vacío (0 columnas)
    si `feature_cols` es None, para que la variante CNN pura no pague el
    coste de leer/escalar algo que no usa."""

    def __init__(self, df: pd.DataFrame, target_col: str, label_encoder,
                 feature_cols: list[str] | None, scaler, train: bool):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.label_encoder = label_encoder
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.transform = build_transform(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Recorte segmentado (fondo a gris, ver extract.py/build_segmented_crops.py) en vez de
        # la imagen completa: decisión del 2026-07-26 para que el fondo no influya en la CNN.
        path = resolve_source_path(row["segmented_path"])
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)

        if self.feature_cols is not None:
            raw = row[self.feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
            scaled = self.scaler.transform(raw)[0]
            geom_tensor = torch.tensor(scaled, dtype=torch.float32)
        else:
            geom_tensor = torch.zeros(0, dtype=torch.float32)

        label = int(self.label_encoder.transform([row[self.target_col]])[0])
        return image_tensor, geom_tensor, label


def specimen_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Reparte especímenes completos (no vistas sueltas) entre train/val —
    misma lógica de fondo que el GroupKFold de `train_geometric.py`, pero un
    único split fijo en vez de 10 repeticiones promediadas: entrenar una CNN
    es demasiado caro para repetir la validación cruzada 10 veces por
    semilla, así que aquí se acepta leer el resultado con más cautela (una
    sola partición puede tener algo de suerte/mala suerte) a cambio de que
    el entrenamiento sea viable en tiempo razonable."""
    return subject_holdout(
        df,
        target_col="bone_group",
        test_size=test_size,
        random_state=random_state,
    )
