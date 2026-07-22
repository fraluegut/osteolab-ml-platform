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

## Pipeline

`streamlit_app.py` encadena varias capas independientes antes de mostrar un resultado:

1. **CLIP zero-shot** (`/filter/clip`) — primera capa, sin entrenamiento propio: ¿es un hueso? + sugerencia de tipo.
2. **Extracción de features con OpenCV** (`/features/extract`) — localiza el hueso y mide su forma.
3. **ResNet18 entrenado** (`/filter/is-bone`) — segunda confirmación de si es hueso (resultado adicional).
4. **Clasificador multi-clase** (`/predict`) — tipo de hueso final craneo/femur/humero (resultado adicional).

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

### Paso 2 — Extracción de features geométricas con OpenCV

Segunda capa del pipeline, tras confirmar con CLIP que hay un hueso. Segmentación
clásica (Otsu + contornos, sin aprendizaje): localiza el contorno del hueso y mide
su forma. Pensado como la fuente de features numéricas para un modelo posterior
(y para comparar más adelante con lo que se genere sintéticamente en Blender).

- **Código**: `src/cv_extractor/extract.py`.
- **Geometría básica**: `bounding_box_px`, `rotated_box_px` (rectángulo alineado con
  el eje principal del hueso, no con la imagen), `area_px`, `area_ratio`, `aspect_ratio`,
  `extent`, `solidity`.
- **Forma**: `elongation` (PCA sobre la nube de puntos del contorno), `ellipse_ratio`
  (elipse ajustada), `circularity` (4π·área/perímetro²), `convexity` (perímetro de la
  envolvente convexa / perímetro del contorno), `hu_moments` (los 7 momentos de Hu,
  invariantes a traslación/escala/rotación, con la transformación logarítmica estándar).
- **Perfil de anchura**: `width_profile` (anchura en 20 cortes perpendiculares al eje
  principal, de un extremo al otro) y sus estadísticos `mean_width`, `std_width`,
  `max_width`, `min_width`.
- **Curvatura**: `curvature_mean`, `curvature_max` de la línea central que resulta de
  unir el punto medio de cada corte del perfil de anchura (curvatura de Menger, 1/px).
- **Visual**: `annotated_image_base64` (medidas pegadas a cada lado del hueso) y
  `mask_overlay_base64` (región detectada resaltada, para verificar la segmentación).
- Todas las medidas son relativas a la imagen (píxeles, ratios): no hay forma de
  derivar medidas reales (cm) sin una referencia de escala en la foto, así que ese
  cálculo no se intenta. El contexto opcional que aporte el usuario (medidas reales,
  peso, tipo...) se transporta aparte, sin mezclarse en el cálculo.
- **Cómo se llama**: la API expone `POST /features/extract` (imagen + campos de
  contexto opcionales por `multipart/form-data`), y `streamlit_app.py` lo invoca
  tras el formulario de contexto opcional.

### Paso 3 — Filtro ResNet18 entrenado: ¿es un hueso?

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
  invoca junto con `/predict` (Paso 4) tras la extracción de OpenCV, agrupados en la
  UI bajo "Resultados adicionales".

## Itinerario del flujo (archivo por archivo, función por función)

La sección anterior explica el pipeline a alto nivel; esta es la traza completa de
qué se ejecuta, en qué archivo y en qué función, desde que alguien sube una foto en
la UI hasta que ve el resultado final.

### 0. El usuario sube la imagen — `app/streamlit_app.py`

- Se sube desde la barra lateral con `st.file_uploader(...)` → `uploaded_file`.
- Se calcula `file_id = hashlib.md5(file_bytes).hexdigest()`: si es distinto al de
  la última imagen procesada, `_reset_state()` limpia `st.session_state` y vuelve
  a `stage = "idle"` (descarta resultados de la imagen anterior).
- `_file_payload()` empaqueta `(nombre, bytes, content_type)`: es lo que se reenvía
  por HTTP a la API en cada paso siguiente, sin volver a tocar el archivo subido.

### 1. Botón "Procesar" → filtro CLIP

`streamlit_app.py` hace `requests.post(f"{API_URL}/filter/clip", ...)`.

`app/main.py::filter_clip()` valida que sea una imagen y llama a
`src/clip_filter/predict.py::classify_image(image_file)`:

1. `_load()` — la primera vez, carga CLIP (`src/clip_filter/model.py::load_clip()`,
   `openai/clip-vit-base-patch32` vía `transformers`) y precalcula, con
   `_embed_descriptions()`, los embeddings de texto de las descripciones fijas
   definidas en `model.py` (`BONE_DESCRIPTIONS`, `NOT_BONE_DESCRIPTIONS`,
   `BONE_TYPE_DESCRIPTIONS`). Solo ocurre una vez por proceso: se cachea en
   variables de módulo.
2. `_image_embedding(image_file)` — embedding normalizado de la imagen subida.
3. `_softmax_over_groups(image_embed, _bone_text_embeds, _not_bone_text_embeds)` —
   similitud coseno de la imagen contra las descripciones de "hueso" y "no hueso",
   softmax conjunta y suma de probabilidad por grupo → `is_bone`.
4. Si `is_bone` es `False`, `classify_image` devuelve el resultado ahí mismo
   (`suggestion: None`) y el flujo se detiene: Streamlit muestra el aviso y no
   llama a nada más.
5. Si es `True`, repite el paso 3 pero contra `_type_text_embeds` (craneo/femur/
   humero) para la sugerencia de tipo, marcando `certain: True` si supera
   `SUGGESTION_CERTAIN_THRESHOLD` (0.7, en `model.py`).

Streamlit guarda la respuesta en `st.session_state.clip_result`, pasa a
`stage = "clip_checked"` y hace `st.rerun()`.

### 2. Formulario de contexto → extracción con OpenCV + resultados adicionales

Si CLIP dijo que sí es un hueso, Streamlit muestra el formulario opcional
(`st.form("context_form")`). Al enviarlo (`submitted`), encadena tres llamadas:

**a) `POST /features/extract`** — `app/main.py::features_extract()` separa los
campos de contexto (los vacíos se descartan, no se usan para nada) y llama a
`src/cv_extractor/extract.py::extract_features(image_file)`:

1. Decodifica la imagen con `cv2.imdecode`.
2. `_segment(gray)` — `GaussianBlur` + umbral de Otsu, se invierte si hace falta
   (se asume que el fondo ocupa más superficie que el hueso) y se limpia con
   `morphologyEx` (close + open) para quitar ruido.
3. `_largest_contour(binary)` — `cv2.findContours` y se queda con el de mayor
   área. Si no hay nada suficientemente grande, devuelve `{"found": False}` y
   termina aquí.
4. Con el contorno localizado: geometría base vía `cv2.contourArea`,
   `cv2.arcLength`, `cv2.boundingRect`, `cv2.minAreaRect` + `cv2.boxPoints`
   (rectángulo rotado, alineado con el hueso y no con la imagen), `cv2.convexHull`.
5. Descriptores de forma (todos en `shape_stats`): `area_ratio`, `aspect_ratio`,
   `extent` y `solidity` calculados inline; `_elongation(contour)` (PCA sobre la
   nube de puntos del contorno); `_ellipse_ratio(contour)` (`cv2.fitEllipse`);
   `circularity` y `convexity` calculados inline a partir de área y perímetro.
6. `_hu_moments(contour)` — `cv2.moments` + `cv2.HuMoments`, con la
   transformación logarítmica estándar para que sean comparables entre sí.
7. `_width_profile_and_curvature(contour, box_points)` — proyecta el contorno
   sobre el eje principal del rectángulo rotado, lo corta en 20 posiciones
   equiespaciadas, mide la anchura en cada corte y arma la línea central uniendo
   los puntos medios; `_menger_curvature()` calcula la curvatura en cada punto
   de esa línea a partir de 3 puntos consecutivos.
8. `_dimensioned_image(...)` — dibuja con Pillow (`ImageDraw`) el contorno, el
   bounding box con "Ancho"/"Alto" pegados a cada lado (`_text_with_outline()`
   para que se lea sobre cualquier fondo), el rectángulo rotado y una franja
   inferior con el resto de estadísticas; `_encode_png()` lo codifica a base64.
9. `_mask_overlay_image(...)` — la imagen original con la máscara binaria
   resaltada en color, para verificar la segmentación a simple vista.

`extract_features` devuelve un único diccionario con todo lo anterior;
`features_extract()` lo empaqueta junto con el contexto del usuario en
`{"cv_features": ..., "context": ...}`.

**b) `POST /filter/is-bone`** (resultado adicional, se pide en la misma tanda
para no tener que volver a llamarlo luego) — `app/main.py::filter_is_bone()` →
`src/bone_filter/predict.py::is_bone(image_file)`:

- `_load()` carga, una vez por proceso, los pesos entrenados
  (`models/bone_filter/bone_filter_resnet18.pt` + `labels.json`) sobre la
  arquitectura de `src/bone_filter/model.py::build_model()` (ResNet18 con la
  última capa cambiada a 2 clases).
- Aplica la transformación de `torchvision` (resize 224 + normalización
  ImageNet) y pasa la imagen por el modelo; `torch.softmax` da la probabilidad
  de "bone" vs "not_bone".

**c) `POST /predict`** (solo si el filtro anterior también dice que es hueso) —
`app/main.py::predict()` → `src/inference/predict.py::predict_image(image_file)`:

- El modelo (`models/model.joblib` + `models/encoder.joblib`) se carga una única
  vez al importar el módulo, vía `_get_model()` → `_load_local_model()` o, si hay
  `MLFLOW_TRACKING_URI`, `_load_mlflow_model()` (con fallback a local si falla).
- Reduce la imagen a escala de grises 32×32, la aplana a vector y llama a
  `model.predict()` / `predict_proba()` (regresión logística de scikit-learn)
  para obtener la clase (craneo/femur/humero) y sus probabilidades.

Streamlit guarda `cv_result` y `extra_result` en `session_state`, pasa a
`stage = "processed"` y hace `st.rerun()`.

### 3. Resultado en pantalla — `app/streamlit_app.py`

Bloque `if st.session_state.stage == "processed":`

- Pinta las dos imágenes de OpenCV (`annotated_image_base64`,
  `mask_overlay_base64`), las rejillas de métricas, el gráfico de
  `width_profile` (`st.line_chart`) y el JSON completo en un expander.
- El contexto que rellenó el usuario, si lo hizo.
- En un expander aparte, "Resultados adicionales": el filtro ResNet18 y, si
  coincide en que es hueso, la predicción del clasificador multi-clase con su
  gráfico de barras.
- Botón "Analizar otra imagen" → `_reset_state()` + `st.rerun()`, vuelta al paso 0.

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
  main.py                # API FastAPI (/health, /version, /predict, /filter/clip, /features/extract, /filter/is-bone)
  streamlit_app.py       # UI de demo (sube imagen -> CLIP -> OpenCV -> ResNet18 + clasificador)
src/
  training/train.py      # Entrenamiento + logging a MLflow (clasificador multi-clase)
  inference/predict.py   # Carga de modelo (MLflow o local) + inferencia (clasificador multi-clase)
  clip_filter/             # Filtro CLIP zero-shot "¿es un hueso?" + sugerencia de tipo
    model.py               # Prompts, umbral de confianza y carga del modelo CLIP
    predict.py             # Inferencia, llamada desde app/main.py::/filter/clip
  cv_extractor/            # Localización y features geométricas con OpenCV, sin aprendizaje
    extract.py              # Segmentación, medidas de forma, perfil de anchura y curvatura
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
