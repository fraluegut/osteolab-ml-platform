# OsteoLab ML Platform

Plataforma de machine learning para clasificar imágenes de huesos por **grupo morfológico**
(cráneo, mandíbula/maxilar, hueso largo, hueso plano, pelvis, sacro, vértebra, costilla, hueso
pequeño de mano/pie), con tracking de experimentos, versionado de datos y una API + UI de demo.

Las mallas 3D, el render multi-vista y el dataset base vienen de la subcarpeta
`base_datos_osea/` de este mismo repo (ver su `HISTORIAL.md` para el porqué de la taxonomía de
9 grupos y de por qué se abandonó clasificar por hueso exacto o por especie).

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

- **UI (recomendado)**: abre `http://localhost:8501`, sube una imagen de hueso y mira el grupo morfológico predicho, las probabilidades y dónde cae en el espacio PCA respecto a lo ya conocido.
- **API directa**:
  ```bash
  curl http://localhost:8000/health
  curl -X POST "http://localhost:8000/classify" -F "file=@/ruta/a/tu/imagen.jpg"
  curl http://localhost:8000/pca/reference   # las ~1300 vistas de referencia, en coordenadas PCA
  ```
- **MLflow UI**: `http://localhost:5000` — experimentos, métricas y modelos registrados.

## Pipeline

`streamlit_app.py` encadena dos capas independientes antes de mostrar un resultado:

1. **CLIP zero-shot** (`/filter/clip`) — primera capa, sin entrenamiento propio: ¿es un hueso? + sugerencia de tipo. Si dice que no, el flujo se detiene ahí.
2. **Medir + clasificar** (`/classify`) — localiza el hueso con OpenCV, mide su forma (Hu moments, ratios, perfil de anchura...) y con esas mismas medidas: (a) da una probabilidad por grupo morfológico con un Random Forest, y (b) proyecta la medida en un espacio PCA 2D ajustado sobre ~1300 vistas ya conocidas, para poder dibujarla junto a ellas.

Cada capa es un módulo aislado, sin compartir código ni estado con las demás. (El filtro binario
ResNet18 `/filter/is-bone`, entrenado aparte, sigue disponible como endpoint suelto pero ya no
forma parte del flujo por defecto de la UI — ver más abajo.)

### Paso 1 — Filtro CLIP zero-shot: ¿es un hueso? + sugerencia de tipo

Primera capa del pipeline. Usa CLIP (`openai/clip-vit-base-patch32`, vía
`transformers`) en modo zero-shot: compara la imagen contra descripciones de texto
en inglés, sin necesitar entrenamiento ni dataset propio. Decide si la imagen es o
no un hueso y, si lo es, sugiere el grupo morfológico comparando contra las mismas
9 clases del clasificador geométrico (cráneo, mandíbula/maxilar, hueso largo...).

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

### Paso 3 — Clasificación por grupo morfológico + posición en el espacio de features

Con las features geométricas del Paso 2 ya calculadas, un Random Forest entrenado
sobre esas mismas medidas (`src/training/train_geometric.py`) da una probabilidad
por cada uno de los 9 grupos morfológicos, y un PCA (2 componentes, ajustado sobre
las mismas features estandarizadas) proyecta la medida en 2D para poder dibujarla
junto a las ~1300 vistas ya conocidas.

- **Código**: `src/inference/predict_geometric.py`.
- **Por qué grupos y no huesos exactos ni especies**: con 1-9 especímenes físicos
  reales por hueso (ver `base_datos_osea/HISTORIAL.md`), un clasificador de 21 huesos
  finos o de especie no puede generalizar más allá de "reconozco este objeto
  concreto" — confirmado empíricamente con `GroupKFold` dejando especímenes
  completos fuera. Agrupando por forma general cada grupo reúne entre 2 y 9
  especímenes, lo mínimo razonable para intentar generalizar.
- **Modelo**: `models/geometric_model_bone_group.joblib` (Random Forest) +
  `models/geometric_encoder_bone_group.joblib` (`LabelEncoder`) +
  `models/geometric_feature_columns_bone_group.joblib` (orden exacto de columnas
  que espera el modelo — se guarda en el entrenamiento y se reutiliza aquí para que
  la imagen subida se mida columna a columna igual que los datos de entrenamiento).
- **PCA de referencia**: `models/geometric_pca_pipeline.joblib`
  (`StandardScaler` + `PCA(n_components=2)`, ajustado sobre las mismas features
  que el modelo — sin estandarizar antes, el PCA solo vería la varianza de las
  features de mayor magnitud en píxeles e ignoraría ratios/momentos de Hu) y
  `data/processed/bone_geometric_pca_reference.csv` (coordenadas ya calculadas
  de cada vista de entrenamiento, para pintar de fondo).
- **Cómo se llama**: la API expone `POST /classify` (mide + clasifica + proyecta
  en un único endpoint) y `GET /pca/reference` (las coordenadas de fondo, se piden
  una sola vez, cacheadas en el proceso con `functools.lru_cache`).

### Filtro ResNet18 entrenado: ¿es un hueso? (endpoint suelto, no en el flujo por defecto)

Filtro binario aislado, entrenado con datos propios (ResNet18, no zero-shot).
Sigue disponible en la API pero `streamlit_app.py` ya no lo llama por defecto — el
flujo actual usa solo CLIP como puerta de entrada. No comparte código ni estado
con el resto de módulos.

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
  con `osteolab-bone-classification-geometric-bone_group`).
- **Cómo se llama**: la API expone `POST /filter/is-bone`, independiente del resto.

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

### 2. Si CLIP dice que sí es un hueso → medir y clasificar

Streamlit llama directamente a `POST /classify` (sin formulario intermedio).
`app/main.py::classify()` → `src/inference/predict_geometric.py::classify_image(image_file)`,
que hace dos cosas con la misma imagen:

**a) Medir con OpenCV** — llama a
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

`extract_features` devuelve un único diccionario con todo lo anterior.

**b) Clasificar + proyectar** — con ese mismo diccionario, `classify_image`:

1. `flatten_features(raw_features)` (reutilizada literalmente de
   `src/cv_extractor/build_dataset.py`, el mismo script que construyó la tabla
   de entrenamiento) aplana los dicts anidados (`hu_moments`, `width_profile`,
   `bounding_box_px`...) a columnas sueltas, descartando las dos imágenes en
   base64 (no son features, son solo para mostrar).
2. Construye el vector de entrada en el **mismo orden de columnas** que se usó
   al entrenar (`geometric_feature_columns_bone_group.joblib`) — crítico: un
   `RandomForestClassifier` no sabe qué significa cada columna, solo su
   posición, así que si el orden no coincide exactamente las probabilidades
   salen sin sentido.
3. `model.predict_proba(row)` → probabilidad por cada uno de los 9 grupos.
4. `pca_pipeline.transform(row)` → el mismo vector, escalado y proyectado con
   el PCA ajustado en el entrenamiento → `{"x": ..., "y": ...}`.

`classify_image` devuelve `{"found", "prediction", "probabilities", "pca_point", "cv_features"}`
(o solo `{"found": False, "cv_features": ...}` si OpenCV no pudo aislar un contorno).

Streamlit guarda el resultado completo en `st.session_state.classify_result` y
hace `st.rerun()` — ya no hay `stage = "processed"` con llamadas encadenadas,
una sola respuesta trae todo lo necesario para pintar los pasos 2 y 3.

### 3. Resultado en pantalla — `app/streamlit_app.py`

- **Paso 2 en pantalla**: las dos imágenes de OpenCV (`annotated_image_base64`,
  `mask_overlay_base64`), las rejillas de métricas, el gráfico de
  `width_profile` (`st.line_chart`) y el JSON completo en un expander — igual
  que antes, solo que los datos ahora vienen de `classify_result["cv_features"]`
  en vez de una llamada aparte a `/features/extract`.
- **Paso 3 en pantalla** (solo si `features["found"]`): `st.metric` con el
  grupo más probable, `st.bar_chart` con las 9 probabilidades, y un scatter de
  Plotly (`px.scatter`) con los puntos de `GET /pca/reference` (cacheados con
  `@st.cache_data(ttl=3600)`, coloreados por `bone_group`) más un marcador de
  estrella para el punto de la imagen subida (`fig.add_scatter(...)`).
- Botón "Analizar otra imagen" → `_reset_state()` + `st.rerun()`, vuelta al paso 0.

## Entrenar el modelo

Con el stack levantado:

```bash
docker compose -f docker/docker-compose.yml exec app python -m src.training.train_geometric
```

Esto:

- Lee `data/processed/bone_geometric_features.csv` (generada por
  `python -m src.cv_extractor.build_dataset` a partir de los renders de
  `base_datos_osea` — no se regenera sola, hay que correr ese script después de
  añadir mallas/renders nuevos ahí).
- Entrena un Random Forest sobre `bone_group` (9 clases) según `params.yaml`
  (`geometric_training`, `geometric_model`).
- Evalúa dos veces: un split aleatorio por fila (optimista, deja fugarse vistas
  del mismo espécimen entre train/test) y un `GroupKFold` repetido 10 veces con
  especímenes completos fuera (la lectura honesta de si generaliza) — ambos se
  imprimen y se registran en MLflow.
- Guarda `models/geometric_model_bone_group.joblib`,
  `models/geometric_encoder_bone_group.joblib`,
  `models/geometric_feature_columns_bone_group.joblib`,
  `models/geometric_pca_pipeline.joblib` y
  `data/processed/bone_geometric_pca_reference.csv`.
- Acepta `--target bone` para entrenar en su lugar el clasificador de 21 huesos
  finos (no recomendado como modelo de producción — ver el porqué en
  `base_datos_osea/HISTORIAL.md` — pero útil para comparar).

El pipeline también se puede reproducir con DVC (`dvc.yaml` define la etapa
`train_geometric`, sus dependencias y parámetros):

```bash
docker compose -f docker/docker-compose.yml exec app dvc repro
```

`src/inference/predict_geometric.py` carga los 4 artefactos del modelo una
única vez al importar el módulo (variables de módulo, no por request) y falla
con un mensaje explícito si no existen todavía (hay que entrenar primero).

## Estructura del proyecto

```text
base_datos_osea/           # Repo hijo (git subtree): catálogo, descarga y render de las mallas 3D.
                            # Ver su README.md/HISTORIAL.md — no se toca desde aquí salvo para leer renders/.
app/
  main.py                    # API FastAPI (/health, /version, /classify, /pca/reference, /filter/clip, /features/extract, /filter/is-bone)
  streamlit_app.py           # UI de demo (sube imagen -> CLIP -> medir+clasificar+PCA)
src/
  training/train_geometric.py   # Entrenamiento (Random Forest + PCA de referencia) + logging a MLflow
  inference/predict_geometric.py  # Carga de modelo/PCA + inferencia sobre una imagen subida
  clip_filter/             # Filtro CLIP zero-shot "¿es un hueso?" + sugerencia de tipo (9 grupos)
    model.py               # Prompts, umbral de confianza y carga del modelo CLIP
    predict.py             # Inferencia, llamada desde app/main.py::/filter/clip
  cv_extractor/            # Localización y features geométricas con OpenCV, sin aprendizaje
    extract.py              # Segmentación, medidas de forma, perfil de anchura y curvatura
    build_dataset.py         # Recorre los renders de base_datos_osea -> tabla CSV de features + bone_group
  bone_filter/            # Filtro binario "¿es un hueso?" (ResNet18 entrenado), endpoint suelto
    model.py               # Definición del modelo
    train.py               # Script de entrenamiento (usa GPU si hay CUDA)
    predict.py             # Inferencia, llamada desde app/main.py::/filter/is-bone
airflow/dags/             # DAG de orquestación del pipeline de entrenamiento
docker/                   # Dockerfiles y docker-compose.yml
data/processed/bone_geometric_features.csv    # Tabla de entrenamiento (generada, versionada en git: es pequeña)
data/processed/bone_geometric_pca_reference.csv  # Coordenadas PCA de esa misma tabla (generada al entrenar)
data/raw/bone_filter/       # Dataset del filtro bone/not_bone (vuelca tus imágenes aquí)
models/                    # geometric_model_bone_group.joblib, geometric_encoder_bone_group.joblib,
                            # geometric_feature_columns_bone_group.joblib, geometric_pca_pipeline.joblib
                            # (todos gitignored, regenerables con train_geometric.py)
models/bone_filter/         # Pesos del filtro (bone_filter_resnet18.pt, labels.json)
params.yaml                # Configuración de dataset/entrenamiento/modelo geométrico
dvc.yaml                   # Pipeline reproducible de DVC (dvc.lock se regenera con `dvc repro`)
```

## Dataset

No hay un `data/raw/<clase>/` local: las imágenes viven como renders en
`base_datos_osea/renders/<especie>/<hueso>/<espécimen>/*.png` (subcarpeta de
este repo, 24 vistas por espécimen). Para (re)generar la tabla de
entrenamiento después de añadir mallas/renders nuevos ahí (rutas por defecto,
no hace falta pasarlas si se ejecuta desde la raíz del repo):

```bash
python -m src.cv_extractor.build_dataset
python -m src.training.train_geometric
```

`data/processed/bone_geometric_features.csv` sí se versiona en git (es una
tabla de números, pequeña — no las imágenes en sí).
