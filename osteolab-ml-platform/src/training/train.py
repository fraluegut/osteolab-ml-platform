from pathlib import Path
import os
import joblib
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/raw/bones"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PARAMS_PATH = BASE_DIR / "params.yaml"

IMG_SIZE = (32, 32)

def load_dataset():
    X = []
    y = []

    for class_dir in DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        print(f"Cargando clase: {label}")

        for img_path in list(class_dir.glob("*"))[:1000]:
            try:
                img = Image.open(img_path).convert("L")
                img = img.resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32).flatten() / 255.0
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset cargado: X={X.shape}, y={y.shape}")
    return X, y


def main():
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    training_params = params["training"]
    model_params = params["model"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("osteolab-bone-classification")

    X, y = load_dataset()

    if len(X) == 0:
        raise ValueError("No se encontraron imágenes en data/raw/bones")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=training_params["test_size"],
        random_state=training_params["random_state"],
        stratify=y_encoded,
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = LogisticRegression(
        max_iter=model_params["max_iter"],
        solver=model_params["solver"],
    )

    with mlflow.start_run():
        mlflow.log_params({
            "image_size": training_params["image_size"],
            "test_size": training_params["test_size"],
            "max_images_per_class": training_params["max_images_per_class"],
            "random_state": training_params["random_state"],
            "model_name": model_params["name"],
            "max_iter": model_params["max_iter"],
            "solver": model_params["solver"],
        })

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)

        mlflow.log_metric("accuracy", accuracy)
        for class_name in le.classes_:
            mlflow.log_metric(f"precision_{class_name}", report[class_name]["precision"])
            mlflow.log_metric(f"recall_{class_name}", report[class_name]["recall"])
            mlflow.log_metric(f"f1_{class_name}", report[class_name]["f1-score"])

        print(classification_report(y_test, preds, target_names=le.classes_))

        joblib.dump(model, MODEL_DIR / "model.joblib")
        joblib.dump(le, MODEL_DIR / "encoder.joblib")

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(MODEL_DIR / "model.joblib"), artifact_path="artifacts")
        mlflow.log_artifact(str(MODEL_DIR / "encoder.joblib"), artifact_path="artifacts")

        print(f"Modelo guardado en: {MODEL_DIR}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()   