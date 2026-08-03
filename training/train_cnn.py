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
    python -m training.train_cnn --model-type cnn_only
    python -m training.train_cnn --model-type hybrid --epochs-frozen 5 --epochs-finetune 10
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
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from shared.cv_extractor.build_dataset import flatten_features
from shared.cv_extractor.extract import extract_features
from training.cnn_dataset import (
    BoneImageDataset,
    build_transform,
    geometric_feature_columns,
    resolve_source_path,
    specimen_split,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
TARGET = "bone_group"
DEFAULT_DATASET = BASE_DIR / "data/processed/bone_geometric_features_combined.csv"
REAL_BENCH_DIR = BASE_DIR / "data/real_world_benchmark"
REAL_BENCH_MANIFEST = REAL_BENCH_DIR / "manifest.csv"
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
    p.add_argument("--bg-gray", type=int, default=128,
                    help="Color de fondo (0-255, gris neutro por defecto) usado SOLO para "
                         "precomputar el banco real externo de monitoreo (real_bench_acc) — el "
                         "dataset de entrenamiento en sí ya trae su propio fondo horneado en los "
                         "recortes de --dataset-path, esto no lo cambia. Pasar 0 al entrenar la "
                         "variante 'blackbg' para que este monitoreo sea consistente.")
    p.add_argument("--run-name", default=None,
                    help="Sufijo de los ficheros de salida (modelo/encoder/scaler/meta), en vez de "
                         "--model-type. Para variantes de pipeline paralelas (ej. 'hybrid_blackbg') "
                         "que no deben pisar los ficheros del pipeline normal — --model-type sigue "
                         "controlando la arquitectura (hybrid/cnn_only), esto solo el nombre de "
                         "fichero.")
    p.add_argument("--epochs-frozen", type=int, default=5)
    p.add_argument("--epochs-finetune", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-head-finetune", type=float, default=None,
                    help="LR de la cabeza en fase 2 (descongelada); por defecto igual a lr_head. "
                         "Se probó bajarlo a lr_head/10 (hipótesis: menos descalibración de la "
                         "cabeza al descongelar el backbone) pero perdió en 3/3 semillas de modelo "
                         "contra el banco real de 63 fotos (media 0.661 vs 0.684) — ver HISTORIAL.md "
                         "2026-08-03. Flag disponible para reabrir la comparación si cambia el "
                         "dataset o el backbone.")
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42,
                    help="Semilla del split train/val (specimen_split) — cambiarla cambia QUÉ "
                         "especímenes caen en val, no solo la inicialización del modelo.")
    p.add_argument("--model-seed", type=int, default=None,
                    help="Semilla de inicialización del modelo y orden de batches "
                         "(torch.manual_seed/np.random.seed); por defecto igual a --random-state. "
                         "Separado de --random-state para poder repetir el mismo split con distinta "
                         "inicialización (ver HISTORIAL.md 2026-08-03: sin esto, comparar 2 configs "
                         "de entrenamiento confunde el efecto de la config con el ruido de init).")
    p.add_argument("--select-by-real-bench", action="store_true",
                    help="Seleccionar checkpoint maximizando real_bench_acc (subconjunto "
                         "'selection' de data/real_world_benchmark) en vez de val_acc sintético. "
                         "Default False: probado el 2026-08-03 y, medido de forma honesta contra "
                         "el subconjunto held_out (nunca visto en selección), dio 0.4444 — peor que "
                         "el criterio de val_acc sintético (0.684-0.702 documentado con 3 semillas). "
                         "El banco real se sigue precomputando y logueando (`real_bench_acc`) "
                         "aunque este flag esté apagado, solo para verlo en MLflow sin que decida.")
    p.add_argument("--real-photo-boost", type=float, default=4.0,
                    help="Peso relativo de las filas de foto real frente a sintética en el "
                         "muestreo de entrenamiento (WeightedRandomSampler) — compensa que las "
                         "reales son minoría dentro de cada grupo (ej. costilla: 5%% de sus filas "
                         "son reales). No afecta el balance ENTRE grupos, eso ya lo hace "
                         "class_weight='balanced' en la pérdida. 1.0 = sin sobremuestreo.")
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


def precompute_real_bench(label_encoder, feature_cols, scaler, fallback_means, backend="grounded_sam",
                           bg_gray: int = 128):
    """Segmenta UNA VEZ (Grounding DINO + SAM, ~0.5s/imagen en GPU) el subconjunto
    de SELECCIÓN del banco de fotos reales externo (`data/real_world_benchmark/`)
    y devuelve tensores ya listos. La segmentación no depende de los pesos del
    modelo, así que hacerla una vez antes de entrenar y reevaluar cada epoch con
    un forward pass barato sobre estos tensores cacheados sale gratis comparado
    con resegmentar en cada una de las 15 epochs.

    2026-08-03: usar el banco real para seleccionar checkpoint Y para examinar
    con `evaluate_cnn_real_photos.py` es circular (el examen deja de ser
    honesto). `manifest.csv` tiene una columna `split` (`selection`/`held_out`,
    1 foto por bone_group reservada como held_out, ver HISTORIAL.md) — aquí
    solo se usa `selection`; `held_out` queda sin tocar para el examen final.
    """
    if not REAL_BENCH_MANIFEST.exists():
        return None
    manifest = pd.read_csv(REAL_BENCH_MANIFEST)
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"] == "selection"]
    transform = build_transform(train=False)
    images, geoms, labels = [], [], []
    n_not_found = 0
    for _, row in manifest.iterrows():
        path = REAL_BENCH_DIR / row["bone_group"] / row["filename"]
        if not path.exists():
            continue
        with path.open("rb") as fh:
            raw = extract_features(fh, backend=backend, include_crop=True, bg_gray=bg_gray)
        if not raw.get("found"):
            n_not_found += 1
            continue
        crop = raw.pop("crop_image")
        images.append(transform(crop))
        if feature_cols is not None:
            flat = flatten_features(raw)
            geoms.append([flat[c] if flat[c] is not None else fallback_means[c] for c in feature_cols])
        labels.append(int(label_encoder.transform([row["bone_group"]])[0]))
    if not images:
        return None
    print(f"Banco real precomputado (subconjunto de selección): {len(images)} imágenes "
          f"({n_not_found} sin segmentar, backend={backend}) — se reevalúa cada epoch con un "
          f"forward pass, sin resegmentar. El held_out queda intacto para el examen final.")
    image_tensor = torch.stack(images)
    if feature_cols is not None:
        geom_tensor = torch.tensor(scaler.transform(geoms), dtype=torch.float32)
    else:
        geom_tensor = torch.zeros((len(images), 0), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return image_tensor, geom_tensor, label_tensor


def eval_real_bench(model, real_bench, device):
    image_tensor, geom_tensor, label_tensor = real_bench
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device), geom_tensor.to(device))
        preds = logits.argmax(dim=1).cpu()
    return accuracy_score(label_tensor.numpy(), preds.numpy())


def main():
    args = parse_args()
    if args.lr_head_finetune is None:
        # 2026-08-03: la hipótesis era que lr_head/10 en fase 2 ayudaría (menos
        # descalibración de la cabeza al descongelar el backbone) — probado con
        # 3 semillas de modelo (split fijo) contra el banco real de 63 fotos:
        # PIERDE en las 3 (media 0.661 vs 0.684 sin bajarlo). Default revertido
        # a lr_head (el flag sigue existiendo por si se quiere reabrir la
        # comparación con más semillas/otro dataset).
        args.lr_head_finetune = args.lr_head
    if args.model_seed is None:
        args.model_seed = args.random_state
    # 2026-08-03: `random_state` solo fijaba el split train/val — la
    # inicialización de la cabeza y el orden/augmentation de los batches no
    # tenían semilla. Dos corridas con checkpoints muy distintos (fase 1 con
    # backbone congelado vs fase 2 ya afinado) dieron el mismo accuracy en el
    # banco real (0.6491) con la misma matriz de confusión — señal de que la
    # varianza entre corridas (no el cambio que se estaba probando) domina el
    # resultado. `--model-seed` (separado de `--random-state`, que sigue
    # controlando solo el split) fija esa parte para que comparar dos configs
    # sea comparar la config, no el azar de la inicialización.
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
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

    # 2026-08-03: primero se probó un proxy de 12 fotos reales DENTRO de val
    # (las de `real_photos_accepted` que caían ahí por el split de espécimen)
    # como criterio de selección — descartado con evidencia de 3 semillas de
    # modelo (empataba constantemente, terminaba eligiendo épocas de fase 1
    # con el backbone congelado). Segunda vuelta: usar el banco EXTERNO de 63
    # fotos (`data/real_world_benchmark/`, el mismo que usa
    # `evaluate_cnn_real_photos.py`), segmentado una sola vez antes de
    # entrenar — más señal (57-63 filas vs 12) y no depende del split.
    real_bench = precompute_real_bench(
        label_encoder, feature_cols, scaler,
        col_means.to_dict() if feature_cols is not None else None,
        bg_gray=args.bg_gray,
    )
    if real_bench is not None and args.select_by_real_bench:
        print("Selección de checkpoint por val_acc en el banco real externo (con guardarraíl "
              "anti clase-cero sobre el val sintético). AVISO: esta vía dio 0.4444 en el held_out "
              "el 2026-08-03, peor que val_acc sintético — usar --select-by-real-bench a sabiendas.")
    elif real_bench is not None:
        print("Banco real externo precomputado y logueado (real_bench_acc) solo para visibilidad — "
              "selección de checkpoint por val_acc sintético (criterio validado con 3 semillas).")
    else:
        print("Sin banco real externo disponible — selección de checkpoint por val_acc sintético.")

    train_dataset = BoneImageDataset(train_df, TARGET, label_encoder, feature_cols, scaler, train=True)
    val_dataset = BoneImageDataset(val_df, TARGET, label_encoder, feature_cols, scaler, train=False)

    # 2026-08-03: `class_weight="balanced"` en la pérdida (más abajo) ya compensa el
    # desequilibrio ENTRE grupos (hueso_largo con 952 filas vs pelvis con 132) — eso
    # ya existía. Lo que no compensaba nada es que, DENTRO de cada grupo, las fotos
    # reales son minoría frente a las sintéticas (costilla: 18 reales de 360 filas,
    # 5%) — cada ronda de fotos reales añadidas esta sesión se diluye entre muchas
    # más filas sintéticas del mismo grupo. `WeightedRandomSampler` sobremuestrea las
    # filas reales (`--real-photo-boost`, default 4x) para que su señal no se pierda
    # en el ruido de lo sintético, sin tocar el balance entre grupos (ese sigue en la
    # pérdida, no aquí).
    is_real = train_df["source_path"].str.startswith("data/real_photos_accepted/").to_numpy()
    sample_weights = np.where(is_real, args.real_photo_boost, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_dataset), replacement=True,
    )
    print(f"WeightedRandomSampler: {int(is_real.sum())} filas reales de {len(train_df)} "
          f"({is_real.mean() * 100:.1f}%) sobremuestreadas x{args.real_photo_boost}.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
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
            "lr_head_finetune": args.lr_head_finetune,
            "lr_backbone": args.lr_backbone,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "model_seed": args.model_seed,
            "select_by_real_bench": args.select_by_real_bench,
            "real_photo_boost": args.real_photo_boost,
            "bg_gray": args.bg_gray,
            "run_name": args.run_name,
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

        # 2026-08-03 (cont.): un primer intento de seleccionar el checkpoint solo
        # por val_acc_real_photos disparó el accuracy real (0.83->0.92) pero se lo
        # llevó por delante regalando mandibula_maxilar entero (recall 0.000) — esa
        # clase no tiene NINGUNA foto real en el proxy de 12 filas, así que
        # maximizarlo a ciegas no la protege. Guardarraíl: nunca aceptar como "mejor"
        # una epoch que deja alguna clase en recall 0 en el val sintético completo
        # (que sí cubre las 9 clases), usando los mismos preds/labels que ya calcula
        # `run_epoch` para el val sintético (sin pasada extra). Si NINGUNA epoch
        # cumple el guardarraíl (caso borde, no debería pasar tras fase1), se cae de
        # vuelta al criterio antiguo (max val_acc sintético) como red de seguridad.
        best = {"metric": (-1.0, -1.0), "val_acc": -1.0, "real": None, "state": None}
        fallback = {"val_acc": -1.0, "state": None}

        def consider_checkpoint(val_acc, real_bench_acc, val_preds, val_labels):
            # `labels=range(num_classes)` trataría una clase SIN filas en val (p.ej.
            # mandibula_maxilar con este split, ver HISTORIAL.md 2026-08-03) como
            # recall=0 con zero_division=0 — clase fantasma que nunca puede pasar el
            # guardarraíl. Limitar a las clases con al menos una fila real en val.
            present_labels = np.unique(val_labels)
            worst_recall = recall_score(val_labels, val_preds, average=None,
                                         labels=present_labels, zero_division=0).min()
            if val_acc > fallback["val_acc"]:
                fallback["val_acc"] = val_acc
                fallback["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # 2026-08-03: con el proxy de 12 filas DENTRO de val esto perdía (empataba,
            # elegía fase 1). Con el banco externo de 48-57 filas de 'selection' se
            # midió honesto contra el 'held_out' (9 fotos nunca vistas en selección):
            # 0.4444 — peor que val_acc sintético. `--select-by-real-bench` deja
            # reabrir esta vía a propósito; por defecto usa val_acc sintético (con
            # `real_bench_acc` ya calculado arriba, solo para loguear).
            use_real_bench = args.select_by_real_bench and real_bench_acc is not None
            selection_metric = (real_bench_acc, val_acc) if use_real_bench else (val_acc, val_acc)
            if worst_recall > 0 and selection_metric > best["metric"]:
                best["metric"] = selection_metric
                best["val_acc"] = val_acc
                best["real"] = real_bench_acc
                best["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return worst_recall

        epoch = 0
        total_epochs = args.epochs_frozen + args.epochs_finetune
        for phase_epoch in range(args.epochs_frozen):
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, device, criterion, optimizer=None)
            real_bench_acc = eval_real_bench(model, real_bench, device) if real_bench is not None else None
            worst_recall = consider_checkpoint(val_acc, real_bench_acc, val_preds, val_labels)
            print(f"[fase1 congelada {phase_epoch + 1}/{args.epochs_frozen}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_worst_class_recall={worst_recall:.4f}"
                  + (f" real_bench_acc={real_bench_acc:.4f}" if real_bench_acc is not None else ""))
            metrics = {"train_loss": train_loss, "train_acc": train_acc,
                       "val_loss": val_loss, "val_acc": val_acc, "val_worst_class_recall": worst_recall}
            if real_bench_acc is not None:
                metrics["real_bench_acc"] = real_bench_acc
            mlflow.log_metrics(metrics, step=epoch)
            epoch += 1

        # Fase 2: backbone descongelado a LR bajo, cabeza sigue a LR normal.
        for p in model.backbone_params():
            p.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": model.backbone_params(), "lr": args.lr_backbone},
            {"params": model.head_params(), "lr": args.lr_head_finetune},
        ])
        for phase_epoch in range(args.epochs_finetune):
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, device, criterion, optimizer=None)
            real_bench_acc = eval_real_bench(model, real_bench, device) if real_bench is not None else None
            worst_recall = consider_checkpoint(val_acc, real_bench_acc, val_preds, val_labels)
            print(f"[fase2 fine-tune {phase_epoch + 1}/{args.epochs_finetune}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_worst_class_recall={worst_recall:.4f}"
                  + (f" real_bench_acc={real_bench_acc:.4f}" if real_bench_acc is not None else ""))
            metrics = {"train_loss": train_loss, "train_acc": train_acc,
                       "val_loss": val_loss, "val_acc": val_acc, "val_worst_class_recall": worst_recall}
            if real_bench_acc is not None:
                metrics["real_bench_acc"] = real_bench_acc
            mlflow.log_metrics(metrics, step=epoch)
            epoch += 1

        if best["state"] is not None:
            best_state = best["state"]
            best_selection_metric = best["metric"][0]
            best_val_accuracy = best["val_acc"]
            best_val_acc_real_photos = best["real"]
            selection_label = ("val_acc en banco real externo (con guardarraíl anti clase-cero)"
                                if (real_bench is not None and args.select_by_real_bench)
                                else "val_acc sintético (con guardarraíl anti clase-cero)")
        else:
            print("\nAVISO: ninguna epoch cumplió el guardarraíl (alguna clase siempre en recall 0) "
                  "— cayendo al criterio antiguo (max val_acc sintético) como red de seguridad.")
            best_state = fallback["state"]
            best_selection_metric = fallback["val_acc"]
            best_val_accuracy = fallback["val_acc"]
            best_val_acc_real_photos = None
            selection_label = "val_acc sintético (fallback, guardarraíl no satisfecho)"

        model.load_state_dict(best_state)
        print(f"\nCheckpoint elegido por {selection_label}: {best_selection_metric:.4f} "
              f"(val_acc sintético de esa misma epoch: {best_val_accuracy:.4f})")
        mlflow.log_metric("best_val_acc", best_val_accuracy)
        mlflow.log_metric("best_selection_metric", best_selection_metric)
        if best_val_acc_real_photos is not None:
            mlflow.log_metric("best_val_acc_real_photos", best_val_acc_real_photos)

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
        suffix = args.run_name or args.model_type
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
            "best_val_accuracy_real_photos": best_val_acc_real_photos,
            "checkpoint_selection": selection_label,
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
