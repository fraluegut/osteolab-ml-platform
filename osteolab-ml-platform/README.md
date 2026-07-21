# OsteoLab ML Platform

Plataforma de machine learning para clasificar imágenes de huesos (cráneo, fémur, húmero), con tracking de experimentos, versionado de datos y una API + UI de demo.

## Stack

| Componente | Tecnología | Puerto |
|---|---|---|
| UI de demo | Streamlit | `8501` |
| API de inferencia | FastAPI | `8000` |
| Tracking de experimentos | MLflow | `5000` |
| Backend de MLflow | PostgreSQL | `5432` (interno) |
| Entrenamiento | scikit-learn | — |
| Versionado de datos/pipeline | DVC | — |

## Requisitos

- Docker Engine + Docker Compose (plugin `docker compose`).
  - En WSL2 sin Docker Desktop: `sudo apt-get install docker.io docker-compose-v2 && sudo systemctl start docker`.
  - Con Docker Desktop: activa la integración WSL para esta distro.

## Cómo levantarlo

Desde la raíz del proyecto (`osteolab-ml-platform/`):

```bash
chmod +x run.sh
./run.sh
```

Esto construye y levanta los 4 servicios: `postgres`, `mlflow`, `app` (API) y `ui` (Streamlit).

Alternativa directa con Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Para ver logs, parar o limpiar:

```bash
docker compose -f docker/docker-compose.yml logs -f
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml down -v   # además borra datos de Postgres/MLflow
```

## Probar que funciona

- **UI (recomendado)**: abre `http://localhost:8501`, sube una imagen de hueso y mira la predicción y las probabilidades por clase.
- **API directa**:
  ```bash
  curl http://localhost:8000/health
  curl -X POST "http://localhost:8000/predict" -F "file=@/ruta/a/tu/imagen.jpg"
  ```
- **MLflow UI**: `http://localhost:5000` — experimentos, métricas y modelos registrados.

## Pipeline de filtrado en 3 pasos

`streamlit_app.py` encadena tres capas independientes antes de mostrar un resultado:

1. **CLIP zero-shot** (`/filter/clip`) — primera capa, sin entrenamiento propio.
2. **ResNet18 entrenado** (`/filter/is-bone`) — segunda confirmación de si es hueso.
3. **Clasificador multi-clase** (`/predict`) — tipo de hueso final (craneo/femur/humero).

Cada capa es un módulo aislado, sin compartir código ni estado con las demás.

### Paso 1 — Filtro CLIP zero-shot: ¿es un hueso? + sugerencia de tipo

Primera capa del pipeline. Usa CLIP (`openai/clip-vit-base-patch32`, vía
`transformers`) en modo zero-shot: compara la imagen contra descripciones de texto
en inglés, sin necesitar entrenamiento ni dataset propio. Decide si la imagen es o
no un hueso y, si lo es, sugiere el tipo (craneo/femur/humero) comparando contra las
mismas clases del clasificador multi-clase.

- **Código**: `src/clip_filter/` (`model.py`, `predict.py`).
- **Umbral de confianza**: si la probabilidad de la sugerencia supera
  `SUGGESTION_CERTAIN_THRESHOLD` (0.7 por defecto, en `src/clip_filter/model.py`), se
  marca como `certain: true` y se propone el tipo de hueso directamente; si no,
  se muestra como una simple sugerencia a confirmar con los siguientes pasos.
- **Cómo se llama**: la API expone `POST /filter/clip`, y `streamlit_app.py` lo
  invoca primero, antes que el filtro ResNet18.

### Paso 2 — Filtro ResNet18 entrenado: ¿es un hueso?

Filtro binario aislado que confirma si la imagen subida es o no un hueso, ya
entrenado con datos propios. Modelo distinto al de CLIP (ResNet18 en vez de
zero-shot), dataset distinto, carpeta de modelos distinta y endpoint de API
distinto. No comparte código ni estado con `src/training` / `src/inference` /
`src/clip_filter`.

- **Código**: `src/bone_filter/` (`model.py`, `train.py`, `predict.py`).
- **Dataset**: vuelca tus imágenes en `data/raw/bone_filter/bone/` y
  `data/raw/bone_filter/not_bone/` (instrucciones en el `README.md` de esa carpeta).
- **Entrenar** (usa tu GPU si hay CUDA disponible; si no, cae a CPU):
  ```bash
  python src/bone_filter/train.py
  # con más control:
  python src/bone_filter/train.py --epochs 15 --batch-size 64 --lr 1e-4
  ```
  Guarda los pesos en `models/bone_filter/bone_filter_resnet18.pt` y el mapeo de
  clases en `models/bone_filter/labels.json`. Si hay un servidor MLflow disponible,
  registra también la corrida en el experimento `osteolab-bone-filter` (sin mezclarse
  con `osteolab-bone-classification`).
- **Cómo se llama**: la API expone `POST /filter/is-bone`, y `streamlit_app.py` lo
  invoca después de CLIP; solo si ambos filtros dicen que sí es un hueso, se llama
  después a `/predict` para clasificar el tipo.

## Entrenar el modelo

Con el stack levantado:

```bash
docker compose -f docker/docker-compose.yml exec app python src/training/train.py
```

Esto:

- Lee imágenes desde `data/raw/bones` (una carpeta por clase).
- Entrena según `params.yaml` (tamaño de imagen, algoritmo, hiperparámetros, nº máx. de imágenes por clase).
- Guarda `models/model.joblib` y `models/encoder.joblib`.
- Registra parámetros, métricas (accuracy, precision/recall/F1 por clase) y el modelo en MLflow.

El pipeline también se puede reproducir con DVC (`dvc.yaml` define la etapa `train`, sus dependencias y parámetros):

```bash
docker compose -f docker/docker-compose.yml exec app dvc repro
```

Cómo elige el modelo la API/UI en cada predicción (`src/inference/predict.py`):

1. Si hay un modelo en Producción en el Model Registry de MLflow (`osteolab-bone-classifier/Production`), lo usa.
2. Si no, hace fallback al modelo local en `models/`.

## Estructura del proyecto

```text
app/
  main.py                # API FastAPI (/health, /version, /predict, /filter/clip, /filter/is-bone)
  streamlit_app.py       # UI de demo (sube imagen -> CLIP -> ResNet18 -> clasificador)
src/
  training/train.py      # Entrenamiento + logging a MLflow (clasificador multi-clase)
  inference/predict.py   # Carga de modelo (MLflow o local) + inferencia (clasificador multi-clase)
  clip_filter/             # Filtro CLIP zero-shot "¿es un hueso?" + sugerencia de tipo
    model.py               # Prompts, umbral de confianza y carga del modelo CLIP
    predict.py             # Inferencia, llamada desde app/main.py::/filter/clip
  bone_filter/            # Filtro binario "¿es un hueso?" (ResNet18 entrenado), aislado del resto
    model.py               # Definición del modelo
    train.py               # Script de entrenamiento (usa GPU si hay CUDA)
    predict.py             # Inferencia, llamada desde app/main.py::/filter/is-bone
airflow/dags/             # DAG de orquestación del pipeline de entrenamiento
docker/                   # Dockerfiles y docker-compose.yml
data/raw/bones/            # Dataset, una carpeta por clase (versionado con DVC)
data/raw/bone_filter/       # Dataset del filtro bone/not_bone (vuelca tus imágenes aquí)
models/                    # Modelo local (model.joblib, encoder.joblib)
models/bone_filter/         # Pesos del filtro (bone_filter_resnet18.pt, labels.json)
params.yaml                # Configuración de dataset/entrenamiento/modelo
dvc.yaml / dvc.lock         # Pipeline reproducible de DVC
```

## Dataset esperado

```text
data/raw/bones/
	craneo/
	femur/
	humero/
```

Versionado con DVC (`data/raw/bones.dvc`); usa `dvc pull` / `dvc repro` para sincronizar datos y reproducir el pipeline si tienes un remote de DVC configurado.
