"""Entrena un clasificador de hueso canónico (21 clases) a partir de la tabla
de features geométricas de OpenCV (`src/cv_extractor/build_dataset.py`), en
vez de píxeles crudos como hace `train.py`.

No toca el pipeline de `train.py` (dataset, modelo y outs distintos: éste
guarda en `models/geometric_*.joblib` y usa su propio experimento de MLflow),
para no romper el stage de DVC ya existente.
"""
from pathlib import Path
import os
import joblib
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PARAMS_PATH = BASE_DIR / "params.yaml"

NON_FEATURE_COLS = {"species", "bone", "specimen", "view", "source_path", "found"}


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].to_numpy()
    y = df["bone"].to_numpy()
    groups = df["specimen"].to_numpy()  # para detectar fuga de espécimen entre train/test
    print(f"Dataset cargado: X={X.shape}, y={y.shape}, {len(set(y))} clases")
    return X, y, groups, feature_cols


def main():
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    dataset_params = params["geometric_dataset"]
    training_params = params["geometric_training"]
    model_params = params["geometric_model"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("osteolab-bone-classification-geometric")

    csv_path = BASE_DIR / dataset_params["path"]
    X, y, groups, feature_cols = load_dataset(csv_path)

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
        for class_name in le.classes_:
            mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
            mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
            mlflow.log_metric(f"f1_{class_name}", report[class_name]["f1-score"])

        print(classification_report(
            y_test, preds, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0,
        ))
        print(f"Accuracy: {accuracy:.4f}")

        importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda t: -t[1])
        print("\nTop 10 features más importantes:")
        for name, imp in importances[:10]:
            print(f"  {name}: {imp:.4f}")

        joblib.dump(model, MODEL_DIR / "geometric_model.joblib")
        joblib.dump(le, MODEL_DIR / "geometric_encoder.joblib")

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(MODEL_DIR / "geometric_model.joblib"), artifact_path="artifacts")
        mlflow.log_artifact(str(MODEL_DIR / "geometric_encoder.joblib"), artifact_path="artifacts")

        print(f"\nModelo guardado en: {MODEL_DIR}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
