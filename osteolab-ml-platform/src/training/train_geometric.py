"""Entrena un clasificador a partir de la tabla de features geométricas de
OpenCV (`src/cv_extractor/build_dataset.py`), en vez de píxeles crudos como
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
"""
import argparse
from pathlib import Path
import os
import joblib
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PARAMS_PATH = BASE_DIR / "params.yaml"

NON_FEATURE_COLS = {"species", "bone", "bone_group", "specimen", "view", "source_path", "found"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="bone_group", choices=["bone_group", "bone"],
                   help="Columna a clasificar: bone_group (9 grupos, por defecto) o bone (21 huesos finos)")
    return p.parse_args()


def load_dataset(csv_path: Path, target_col: str):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    groups = df["specimen"].to_numpy()  # para detectar fuga de espécimen entre train/test
    print(f"Dataset cargado: X={X.shape}, y={y.shape} ({target_col}), {len(set(y))} clases")
    return X, y, groups, feature_cols


def main():
    args = parse_args()

    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    dataset_params = params["geometric_dataset"]
    training_params = params["geometric_training"]
    model_params = params["geometric_model"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(f"osteolab-bone-classification-geometric-{args.target}")

    csv_path = BASE_DIR / dataset_params["path"]
    X, y, groups, feature_cols = load_dataset(csv_path, args.target)

    if len(X) == 0:
        raise ValueError(f"No se encontraron filas en {csv_path}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y_encoded,
        groups,
        test_size=training_params["test_size"],
        random_state=training_params["random_state"],
        stratify=y_encoded,
    )

    # Aviso de metodología: con un solo espécimen físico por clase (24 vistas,
    # todas del mismo hueso real), un split por fila reparte vistas del MISMO
    # espécimen entre train y test — el modelo puede "reconocer" ese objeto
    # concreto en vez de generalizar a huesos nuevos. No hay forma de evitarlo
    # con los datos actuales (dejar 0 vistas de esa clase en train o en test
    # no es mejor), pero la accuracy de estas clases hay que leerla con cautela
    # hasta que haya >1 espécimen por hueso.
    leaking_specimens = set(groups_train) & set(groups_test)
    if leaking_specimens:
        print(f"AVISO: {len(leaking_specimens)} espécimen(es) con vistas en train Y test "
              f"(clases con un solo espécimen disponible): {sorted(leaking_specimens)}")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

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
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    cv_model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=training_params["random_state"],
    )
    cv_preds = cross_val_predict(cv_model, X, y_encoded, groups=groups, cv=gkf)
    cv_accuracy = accuracy_score(y_encoded, cv_preds)
    cv_report = classification_report(
        y_encoded, cv_preds, labels=range(len(le.classes_)), target_names=le.classes_,
        output_dict=True, zero_division=0,
    )
    print(f"\n=== Evaluación con espécimen completo fuera (GroupKFold, {n_splits} folds) ===")
    print(classification_report(
        y_encoded, cv_preds, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0,
    ))
    print(f"Accuracy (espécimen fuera): {cv_accuracy:.4f}")

    with mlflow.start_run():
        mlflow.log_params({
            "n_features": len(feature_cols),
            "test_size": training_params["test_size"],
            "random_state": training_params["random_state"],
            "model_name": model_params["name"],
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
        })

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(
            y_test, preds, labels=range(len(le.classes_)), target_names=le.classes_,
            output_dict=True, zero_division=0,
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("accuracy_specimen_held_out", cv_accuracy)
        for class_name in le.classes_:
            mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
            mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
            mlflow.log_metric(f"f1_{class_name}", report[class_name]["f1-score"])

        print(classification_report(
            y_test, preds, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0,
        ))
        print(f"Accuracy (split por vista): {accuracy:.4f}")
        print(f"Accuracy (espécimen fuera, más honesto): {cv_accuracy:.4f}")

        importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda t: -t[1])
        print("\nTop 10 features más importantes:")
        for name, imp in importances[:10]:
            print(f"  {name}: {imp:.4f}")

        model_path = MODEL_DIR / f"geometric_model_{args.target}.joblib"
        encoder_path = MODEL_DIR / f"geometric_encoder_{args.target}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(le, encoder_path)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(encoder_path), artifact_path="artifacts")

        print(f"\nModelo guardado en: {MODEL_DIR}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
