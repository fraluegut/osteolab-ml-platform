"""Entrena un clasificador a partir de la tabla de features geométricas de
OpenCV (`shared/cv_extractor/build_dataset.py`), en vez de píxeles crudos como
hace `train.py`.

Por defecto clasifica `bone_group` (9 grupos morfológicos: cráneo, mandíbula/
maxilar, hueso largo, hueso plano, pelvis, sacro, vértebra, costilla, hueso
pequeño de mano/pie) en vez de `bone` (21 huesos canónicos finos): con solo 1-2
especímenes físicos reales por hueso individual, un clasificador de 21 clases
no puede generalizar más allá de "reconozco este objeto concreto" (ver aviso
de fuga más abajo); agrupando por forma general, cada grupo reúne entre 2 y 9
especímenes distintos, que es lo mínimo razonable para intentar generalizar.
Pasar `--target bone` para entrenar el clasificador fino en su lugar.

No toca el pipeline de `train.py` (dataset, modelo y outs distintos: éste
guarda en `models/geometric_*.joblib` y usa su propio experimento de MLflow),
para no romper el stage de DVC ya existente.

Trazabilidad dataset→modelo→evaluación (2026-07-25): con `--dataset-path` es
fácil entrenar sobre variantes del dataset (p.ej. limpio vs limpio+ruido) sin
que quede constancia de cuál se usó en cada run de MLflow más allá del
accuracy — dos runs con el mismo `n_features` son indistinguibles en la UI.
Aquí se loguea `dataset_path`/`n_train_rows`/`n_train_specimens` como params
y se escribe además `models/geometric_training_meta_<target>.json` con esos
mismos datos + el `run_id` — así `evaluate.py`/`evaluate_real_photos.py`
pueden leer ese fichero y loguear en SU PROPIA run de qué entrenamiento viene
el modelo que están evaluando, sin tener que adivinarlo por la fecha.
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import os
import joblib
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from training.group_splits import subject_ids

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PARAMS_PATH = BASE_DIR / "params.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_geometric")

# img_width/img_height excluidas (2026-07-25): constantes exactas (512x512) en todos los renders
# de entrenamiento, así que su desviación típica es ~0 — cualquier foto real de otra resolución
# produce un z-score enorme en esas dos columnas sin que la forma del hueso tenga nada que ver
# (ver HISTORIAL.md, bug de escala del híbrido). No aportan nada en entrenamiento (son constantes,
# ningún split puede usarlas para discriminar) y arriesgan mucho fuera de él.
NON_FEATURE_COLS = {"species", "bone", "bone_group", "specimen", "view", "source_path", "found",
                     "img_width", "img_height", "segmented_path"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="bone_group", choices=["bone_group", "bone"],
                   help="Columna a clasificar: bone_group (9 grupos, por defecto) o bone (21 huesos finos)")
    p.add_argument("--dataset-path", type=Path, default=None,
                   help="CSV alternativo a geometric_dataset.path de params.yaml (p.ej. para probar un dataset aumentado sin tocar el archivo de params)")
    p.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                   help="tag de MLflow para identificar esta run luego en la UI (repetible), p.ej. --tag dataset_variant=combined_with_noise")
    return p.parse_args()


def load_dataset(csv_path: Path, target_col: str):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    groups = subject_ids(df)
    print(f"Dataset cargado: X={X.shape}, y={y.shape} ({target_col}), {len(set(y))} clases")
    return X, y, groups, feature_cols, df


def main():
    t_start = time.monotonic()
    args = parse_args()
    logger.info("[1/7] Cargando parámetros y dataset (target=%s)", args.target)

    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    dataset_params = params["geometric_dataset"]
    training_params = params["geometric_training"]
    model_params = params["geometric_model"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(f"osteolab-bone-classification-geometric-{args.target}")

    csv_path = args.dataset_path if args.dataset_path is not None else BASE_DIR / dataset_params["path"]
    X, y, groups, feature_cols, df = load_dataset(csv_path, args.target)

    if len(X) == 0:
        raise ValueError(f"No se encontraron filas en {csv_path}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=training_params["random_state"],
    )

    # Evaluación honesta: cada fold deja FUERA especímenes físicos completos
    # (no vistas sueltas) — a diferencia del split de arriba, aquí un
    # espécimen nunca aparece a la vez en fit y en predict. Esto es lo que de
    # verdad responde "¿generaliza a un hueso que no ha visto nunca?", en vez
    # de "¿reconoce este objeto concreto desde otro ángulo?".
    #
    # GroupKFold sin shuffle reparte los folds según el orden en que aparecen
    # los especímenes en los datos (no al azar) — con solo 2-3 especímenes en
    # varias clases, una única partición así puede ser una muestra con mucha
    # suerte o mala suerte, no un resultado estable. Se repite con varias
    # semillas (shuffle=True + random_state distinto cada vez) y se promedia,
    # para no sacar conclusiones de una partición concreta.
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    n_repeats = 10
    cv_model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=training_params["random_state"],
    )

    repeat_accuracies = []
    per_class_recalls = {c: [] for c in le.classes_}
    for seed in range(n_repeats):
        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_preds = cross_val_predict(cv_model, X, y_encoded, groups=groups, cv=gkf)
        repeat_accuracies.append(accuracy_score(y_encoded, cv_preds))
        rep_report = classification_report(
            y_encoded, cv_preds, labels=range(len(le.classes_)), target_names=le.classes_,
            output_dict=True, zero_division=0,
        )
        for c in le.classes_:
            per_class_recalls[c].append(rep_report[c]["recall"])

    cv_accuracy = float(np.mean(repeat_accuracies))
    cv_accuracy_std = float(np.std(repeat_accuracies))
    print(f"\n=== Evaluación con espécimen completo fuera "
          f"(StratifiedGroupKFold×{n_repeats} barajados, {n_splits} folds cada uno) ===")
    print(f"Accuracy por repetición: {[round(a, 3) for a in repeat_accuracies]}")
    print("\nRecall medio ± desviación por grupo (a través de las 10 repeticiones):")
    for c in le.classes_:
        recalls = per_class_recalls[c]
        print(f"  {c:<20} {np.mean(recalls):.3f} ± {np.std(recalls):.3f}  (min={min(recalls):.2f}, max={max(recalls):.2f})")
    print(f"\nAccuracy (espécimen fuera): {cv_accuracy:.4f} ± {cv_accuracy_std:.4f}")

    n_train_specimens = int(len(set(groups)))
    try:
        dataset_path_display = str(csv_path.relative_to(BASE_DIR))
    except ValueError:
        dataset_path_display = str(csv_path)

    with mlflow.start_run():
        mlflow.log_params({
            "n_features": len(feature_cols),
            "test_size": training_params["test_size"],
            "random_state": training_params["random_state"],
            "model_name": model_params["name"],
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "dataset_path": dataset_path_display,
            "n_train_rows": len(df),
            "n_train_specimens": n_train_specimens,
            "cv_strategy": "RepeatedStratifiedGroupKFold",
            "cv_folds": n_splits,
            "cv_repeats": n_repeats,
            "group_key": "species::specimen",
        })
        for tag in args.tag:
            if "=" not in tag:
                raise SystemExit(f"--tag mal formado (esperaba KEY=VALUE): {tag!r}")
            key, value = tag.split("=", 1)
            mlflow.set_tag(key, value)

        # La CV estima generalizacion. Despues se reentrena el artefacto final
        # con todos los sujetos; no se guarda un modelo entrenado sobre solo el
        # 80% ni se publica una metrica optimista basada en vistas mezcladas.
        model.fit(X, y_encoded)

        mlflow.log_metric("accuracy_specimen_held_out_mean", cv_accuracy)
        mlflow.log_metric("accuracy_specimen_held_out_std", cv_accuracy_std)
        for class_name in le.classes_:
            mlflow.log_metric(f"cv_recall_{class_name}_mean", float(np.mean(per_class_recalls[class_name])))
            mlflow.log_metric(f"cv_recall_{class_name}_std", float(np.std(per_class_recalls[class_name])))

        print(f"Modelo final reentrenado con las {len(df)} filas / {n_train_specimens} sujetos.")
        print(f"Accuracy CV (sujeto fuera): {cv_accuracy:.4f} ± {cv_accuracy_std:.4f}")

        importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda t: -t[1])
        print("\nTop 10 features más importantes:")
        for name, imp in importances[:10]:
            print(f"  {name}: {imp:.4f}")

        model_path = MODEL_DIR / f"geometric_model_{args.target}.joblib"
        encoder_path = MODEL_DIR / f"geometric_encoder_{args.target}.joblib"
        columns_path = MODEL_DIR / f"geometric_feature_columns_{args.target}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(le, encoder_path)
        joblib.dump(feature_cols, columns_path)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(encoder_path), artifact_path="artifacts")

        run_id = mlflow.active_run().info.run_id
        print(f"\nModelo guardado en: {MODEL_DIR}")
        print(f"Run ID: {run_id}")

        # Deja constancia en disco (no solo en MLflow) de qué dataset/run produjo
        # el .joblib que hay ahora mismo en MODEL_DIR — evaluate.py y
        # evaluate_real_photos.py lo leen para loguear en su propia run de qué
        # entrenamiento viene el modelo que están evaluando.
        meta = {
            "target": args.target,
            "dataset_path": dataset_path_display,
            "n_train_rows": len(df),
            "n_train_specimens": n_train_specimens,
            "mlflow_run_id": run_id,
            "mlflow_experiment": mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            "accuracy_specimen_held_out_mean": cv_accuracy,
            "accuracy_specimen_held_out_std": cv_accuracy_std,
            "cv_strategy": "RepeatedStratifiedGroupKFold",
            "cv_folds": n_splits,
            "cv_repeats": n_repeats,
            "group_key": "species::specimen",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "tags": {t.split("=", 1)[0]: t.split("=", 1)[1] for t in args.tag},
        }
        meta_path = MODEL_DIR / f"geometric_training_meta_{args.target}.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Metadatos de entrenamiento guardados en: {meta_path}")

    # PCA de referencia (solo para bone_group: es lo que se enseña en la UI
    # como "dónde cae esta foto respecto a lo que ya conocemos"). Se
    # estandariza antes de proyectar porque las features vienen en escalas
    # muy distintas (área en px² frente a ratios 0-1 o momentos de Hu en
    # escala logarítmica) — sin esto, PCA solo vería la varianza de las
    # features de mayor magnitud (área, bbox en píxeles) e ignoraría el resto.
    if args.target == "bone_group":
        pca_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=training_params["random_state"])),
        ])
        # PCA no admite NaN (a diferencia de RandomForestClassifier, que sí
        # los tolera de forma nativa) — un puñado de filas se quedan sin
        # `ellipse_ratio` cuando el contorno tiene <5 puntos (ver
        # `_ellipse_ratio` en extract.py). Imputar con la media de columna,
        # igual que ya se hace en inferencia (`_fallback_means`).
        col_means = np.nanmean(X, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(X))
        X_pca = X.copy()
        X_pca[nan_rows, nan_cols] = np.take(col_means, nan_cols)
        coords = pca_pipeline.fit_transform(X_pca)
        explained = pca_pipeline.named_steps["pca"].explained_variance_ratio_
        print(f"\nPCA de referencia: {explained[0]*100:.1f}% + {explained[1]*100:.1f}% "
              f"= {explained.sum()*100:.1f}% de varianza explicada en 2D")

        pca_path = MODEL_DIR / "geometric_pca_pipeline.joblib"
        joblib.dump(pca_pipeline, pca_path)

        reference_df = df[["species", "bone", "bone_group", "specimen", "view"]].copy()
        reference_df["pca_x"] = coords[:, 0]
        reference_df["pca_y"] = coords[:, 1]
        reference_path = BASE_DIR / "data/processed/bone_geometric_pca_reference.csv"
        reference_df.to_csv(reference_path, index=False)
        print(f"PCA guardado en: {pca_path}")
        print(f"Coordenadas de referencia guardadas en: {reference_path}")


if __name__ == "__main__":
    main()
