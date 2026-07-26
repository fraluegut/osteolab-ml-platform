# Historial del proyecto

Relato cronológico de decisiones, problemas encontrados y cómo se resolvieron. A diferencia del
README (que describe el *estado actual* y se reescribe), este documento **se añade, no se
reescribe** — es el guión completo de cómo se ha llegado hasta aquí, pensado para reconstruir el
razonamiento al final del proyecto sin tener que releer toda la conversación.

Dos repos involucrados: `base_datos_osea` (este, cataloga/descarga/renderiza) y
`osteolab-ml-platform` (entrena el clasificador a partir de lo que este repo produce).

---

## 2026-07-22 — Arranque del catálogo (sesión previa, resumido)

- Se construyó el pipeline MorphoSource + Smithsonian: `catalog.py` (clasifica y cataloga una
  especie), `gap_check.py` (arbitra un ganador por hueso entre especies), `download.py` (descarga
  lo seleccionado). 21 huesos objetivo definidos en `scripts/bones.py::TARGET_BONES`.
- **Problema encontrado**: el repo vivía en `/dev/base_datos_osea` — `/dev` es `devtmpfs`, no
  sobrevive a un reinicio de WSL. **Solución**: mover todo a `/root/dev/base_datos_osea` (disco
  real).
- **Problema encontrado**: el fémur de *H. sapiens* de MorphoSource (`000031310`) resultó ser un
  fragmento (cabeza + diáfisis, sin cóndilos distales), etiquetado simplemente `"femur"` sin ningún
  indicio de fragmento en el texto — el flag `is_partial` de MorphoSource no lo detectó. Se verificó
  midiendo el perfil de anchura a lo largo del eje principal (crece por un lado y se corta, en vez
  de ensanchar por los dos extremos). **Conclusión**: los metadatos de texto de MorphoSource no son
  fiables para detectar fragmentos sin avisar.
- **Solución adoptada**: pivotar a **Sketchfab** como fuente principal para huesos humanos —
  permite ver una miniatura de cada modelo *antes* de descargar, así que cada candidato se verifica
  visualmente (mirando la miniatura) antes de aceptarlo. Se dejaron 3 candidatos verificados
  (fémur, húmero, cráneo) pero sin descargar al cortar la sesión.

---

## 2026-07-23 — Continuación: completar Sketchfab, corregir el render, primer lote

### Descarga y verificación visual de Sketchfab (14 huesos)
- Se ejecutó `catalog`/`download` de los 3 candidatos pendientes, y se buscó/verificó uno a uno el
  resto de huesos de *H. sapiens* que venían de naledi/MorphoSource: mandíbula, escápula,
  clavícula, radio, cúbito, tibia, fíbula, rótula, sacro, vértebra, costilla — 11 más, 14 en total.
- **Maxilar sin reemplazo**: los 2 candidatos de Sketchfab revisados resultaron fragmentos
  unilaterales (falta un lado de la arcada) — rechazados, se mantuvo la malla de naledi.

### Bug en `render_bone.py`: mallas glTF invisibles o mal encuadradas
- **Problema**: al probar el primer `.glb` de Sketchfab, el render salía en blanco. El importador
  de glTF de Blender anida la malla bajo Empties con escala (~0.009) y traslación no triviales
  (Sketchfab exporta con su propio sistema de unidades) — el código leía `obj.data.vertices`
  asumiendo que ya estaban en espacio del mundo, así que el centrado/encuadre calculaba mal la
  posición y el radio.
- **Solución**: tras importar, aplanar el transform con `parent_clear(type="CLEAR_KEEP_TRANSFORM")`
  + `transform_apply()` antes de leer los vértices — corrige tanto glTF como cualquier otro formato.
  Verificado que no rompe el pipeline `.ply`/`.stl` existente (mismo resultado que antes).

### Auditoría geométrica de las 46 mallas descargadas
- Se renderizó 1 vista rápida de cada malla y se revisó visualmente. Se encontraron **7
  fragmentos/mallas mal etiquetadas** que ni MorphoSource ni el clasificador de texto habían
  avisado:
  - `homo_sapiens/cranium/000678985` y `homo_sapiens/femur/000031310` — ya documentados arriba.
  - `homo_naledi/scapula/000026236` — el nombre de fichero ya decía "fragment"; confirmado (blob
    irregular, sin glenoides/acromion reconocibles).
  - `homo_naledi/fibula/000100688` — carpeta decía "Distal"; confirmado fragmento distal only.
  - `homo_naledi/humerus/000026234` y `homo_naledi/tibia/000014985` — fragmentos (cabeza proximal
    suelta / diáfisis sin epífisis), confirmados con varios ángulos de cámara adicionales.
  - `homo_naledi/radius/000026239` (fichero literalmente llamado `uw-102-025-ulna-...`, ya
    sospechoso) — al renderizarlo era un fragmento pequeño irreconocible, ni radio ni cúbito.
- Las 5 mallas de naledi se dejaron en disco pero fuera del render (sus huesos ya estaban cubiertos
  por el reemplazo de Sketchfab en *H. sapiens*).

### Bug en el script de render por lotes: `cut -d/` con el índice de campo mal
- **Problema**: el primer lote (39 huesos) se guardó todo bajo `renders/meshes/<especie>/<hueso>/`
  en vez de `renders/<especie>/<hueso>/<espécimen>/` — el script usaba `cut -d/ -f2,3,4` sobre una
  ruta que empezaba por `data/meshes/...`, así que el campo 2 era literalmente la palabra "meshes",
  no la especie. Se detectó al revisar la estructura de carpetas resultante.
- **Solución**: corregir a `-f3,4,5`, borrar el árbol mal generado y volver a lanzar (rápido, ~30s
  por hueso). Resultado: **39/39 huesos, 936 imágenes**, sin fallos.

### Contaminación por carpetas de render obsoletas
- Al construir la tabla de features en `osteolab-ml-platform` por primera vez, salieron 972
  imágenes en vez de 936 — 3 carpetas de una sesión anterior (12 vistas, para los 2 fragmentos de
  MorphoSource ya excluidos + el húmero fragmentado de naledi) seguían en `renders/` y se colaron.
  **Solución**: borrarlas y regenerar.

---

## 2026-07-23 (continuación) — De "21 huesos finos" a "9 grupos morfológicos"

### Por qué se abandonó clasificar por hueso exacto
- Con el pipeline de features geométricas de OpenCV (`extract_features()`: Hu moments, ratios,
  perfil de anchura) montado y entrenado sobre las 21 clases finas, la accuracy en un split
  aleatorio por fila era razonable (76%), **pero** casi todos los huesos solo tenían 1 espécimen
  físico real (24 vistas del mismo objeto) — un split por fila reparte vistas del MISMO hueso entre
  train y test, así que la accuracy medía sobre todo "reconozco este objeto concreto desde otro
  ángulo", no generalización real.
- **Decisión del usuario**: no tiene sentido perseguir 21 clases finas por especie/edad — mejor
  agrupar en categorías morfológicas generales, que es como realmente se identifica un hueso
  encontrado en la práctica (cráneo, costilla, vértebra, hueso largo, hueso plano, pelvis, hueso
  pequeño de mano/pie...). Esto además soluciona de rebote el problema de fuga: un grupo "hueso
  largo" con 6-9 especímenes distintos (fémur+tibia+húmero+radio+cúbito+fíbula+clavícula) da mucha
  más señal de generalización que un hueso individual con 1 espécimen.
- **Taxonomía acordada** (`BONE_GROUPS` en `scripts/bones.py`, duplicada en `build_dataset.py` del
  otro repo): 9 grupos. Decisiones no obvias, todas confirmadas explícitamente por el usuario:
  mandíbula/maxilar aparte de cráneo (forma muy distinta), sacro aparte de vértebra (aunque sean
  vértebras fusionadas), clavícula con hueso largo (no con la cintura escapular), rótula con hueso
  pequeño (por forma, no por función anatómica).

### Evaluación honesta: `GroupKFold` con espécimen completo fuera
- Entrenando sobre los 9 grupos con un split aleatorio por fila: **83.8%** de accuracy. Pero el
  mismo problema de fuga seguía ahí a nivel de espécimen.
- Se añadió una evaluación con `GroupKFold` (agrupando por `specimen`, no por fila) que garantiza
  que un hueso físico nunca está a la vez en train y test. Resultado: la accuracy real cae a
  **65.5%**, y los 3 grupos con solo 2 especímenes (`costilla`, `sacro`, `vertebra`) se desploman a
  **0% de recall** en algunos folds — en esas particiones el modelo entrena sin haber visto nunca
  un ejemplo de esa clase (los 2 especímenes caen en el mismo fold de test, o el modelo solo ve 1
  espécimen en train, señal demasiado débil).
- **Por qué importa esto y qué NO lo arregla**: renderizar más vistas del mismo hueso no ayuda —
  son más copias de la misma forma, no variación real. Lo que enseña a generalizar es tener
  especímenes físicos *distintos*.

### 2º y 3er espécimen para los grupos más débiles
- `hueso_plano` (escápula) y `pelvis` solo tenían 1 espécimen — se buscó y verificó visualmente un
  2º en Sketchfab para cada uno (escápula de Eric Bauer, pelvis de Oregon State University).
- `costilla`, `sacro` y `vertebra` tenían 2 especímenes (el mínimo que sigue dando 0% de recall en
  algunos folds) — se buscó y verificó un 3º para cada uno, buscando además diversidad de forma
  real dentro de la clase, no solo "otro ejemplo cualquiera": una costilla 12ª (forma muy distinta
  a la costilla central ya usada), un sacro con cóccix de un autor independiente, una vértebra
  torácica (T5/T6) de un autor distinto al ya usado.
- Estado tras esto: **1056 imágenes renderizadas**, todos los grupos con ≥2 especímenes, la mayoría
  con 3+. Pendiente re-ejecutar la evaluación `GroupKFold` con estos datos nuevos para confirmar si
  sube el recall de los 3 grupos débiles.

### 3er espécimen para costilla/sacro/vértebra — resultado sorprendente
- Se buscó y verificó un 3er espécimen para los 3 grupos débiles (costilla: 12ª costilla para dar
  diversidad de forma real; sacro: autor independiente con cóccix; vértebra: torácica T5/T6 de otra
  fuente). Total tras esto: **1056 imágenes**, todos los grupos con ≥2 especímenes.
- **Se detectó un problema metodológico antes de fiarse del número**: `GroupKFold` de sklearn no
  baraja por defecto — el reparto en folds depende del orden en que aparecen los especímenes en los
  datos, no es aleatorio. Con solo 2-3 especímenes en varias clases, una sola partición podía ser
  una muestra con mucha suerte o mala suerte. **Solución**: repetir la validación 10 veces con
  `shuffle=True` y semilla distinta cada vez, y reportar la media ± desviación por clase en vez de
  un único número.
- **Resultado honesto (10 repeticiones)**: pasar de 2 a 3 especímenes **no arregló el problema** —
  sacro incluso empeoró (0.4% de recall medio). Los grupos con muchos especímenes (`hueso_largo`:
  9, `mandibula_maxilar`: 9) generalizan con desviación baja (≤0.04) — consistente y estable, no
  ruido. Los grupos débiles (`costilla`: 0.18±0.12, `pelvis`: 0.19±0.03, `hueso_plano`: 0.32±0.11,
  `vertebra`: 0.06±0.03, `sacro`: 0.004±0.006) siguen mal incluso con 3 especímenes.
- **Conclusión práctica, más dura de lo esperado**: el umbral real para generalizar parece estar
  hacia **6+ especímenes por grupo**, no 2-3. Añadir uno más no es suficiente para los grupos
  anatómicamente más variables/complejos (sacro y vértebra en particular, que tienen mucha
  variación de forma incluso dentro del mismo individuo según la posición vertebral/costal).

### Segunda ronda: empujar hacia 6+ especímenes — la hipótesis se confirma
- Se buscaron y verificaron 11 especímenes más para los 4 grupos débiles: 2 costillas más (1ª y
  2ª posición, formas muy distintas entre sí), 3 sacros más (incluido uno arqueológico
  documentado, St. Nicolas Kirk), 3 vértebras más (incluida el atlas C1, forma de anillo sin
  cuerpo vertebral — mucha diversidad de forma dentro de la clase), 3 escápulas más.
  - **Pelvis se quedó en 2**: no se encontraron más modelos de pelvis bilateral completa en
    Sketchfab que no llevaran fémures o vértebras lumbares fusionados en la misma malla (varios
    candidatos de MRI/CT los incluían — rechazados por contaminar la clase con forma de otros
    huesos). Los únicos "limpios" que había eran huesos coxales unilaterales (un solo lado), que
    no es la misma clase que la pelvis bilateral ya usada.
  - **Un modelo con las 2 escápulas (izquierda y derecha) en una sola malla se descartó** — no
    encaja con la convención de "una malla = un hueso" del pipeline de render.
- Total tras esto: **1320 imágenes**. Especímenes por grupo: `hueso_largo` 9, `mandibula_maxilar`
  9, `hueso_pequeno` 7, `cranio` 6, `sacro` 6, `vertebra` 6, `costilla` 5, `hueso_plano` 5,
  `pelvis` 2.
- **Resultado (GroupKFold×10, espécimen fuera): la hipótesis se confirma con creces.**
  `hueso_plano` (2→5 especímenes) pasó de 32% a **68% de recall medio** — ya está a la altura de
  los grupos "buenos". `sacro` (2→6) pasó de 0.4% a 23%. `vertebra` (2→6) de 6% a 31%. `costilla`
  (2→5) de 18% a 31%. La accuracy global bajó ligeramente (61%→58%), pero no por regresión: los
  especímenes nuevos meten diversidad de forma real y más difícil (el anillo del atlas, la
  costilla 1ª/12ª con curvatura atípica), así que es un dataset más honesto, no peor.
  `pelvis` (sigue en 2) quedó como el eslabón claramente más débil (recall bajó a 9%) — siguiente
  objetivo si aparece un 3er modelo de pelvis bilateral limpio.

### De "modelo entrenado en un fichero" a "modelo servido de verdad"
- Se descubrió que, aunque todo el trabajo de features geométricas + modelo por grupos estaba
  hecho, **la API real (`/predict`) seguía sirviendo el modelo antiguo**: regresión logística
  sobre píxeles 32×32 en gris, 3 clases (craneo/fémur/húmero), entrenado con un dataset de renders
  de una sesión mucho anterior sin relación con el catálogo curado. Si un usuario subía la foto de
  cualquier otro hueso, la API igualmente respondía "craneo", "fémur" o "húmero" porque no podía
  devolver ninguna otra etiqueta.
- **Se retiró el pipeline viejo por completo** (no solo se dejó de llamar): `src/inference/predict.py`,
  `src/training/train.py`, `data/raw/bones.dvc`, `data/sanity_check.py`, el stage viejo de
  `dvc.yaml`/`dvc.lock`, las secciones `dataset`/`training`/`model` de `params.yaml`, y los
  `.joblib` del modelo viejo.
- **Se conectó el modelo de grupos a la API de verdad**: `POST /classify` mide la imagen con
  OpenCV, la clasifica con el Random Forest de 9 grupos, y además proyecta la misma medida en un
  PCA 2D (ajustado sobre las ~1300 vistas de entrenamiento) para poder dibujarla junto a lo ya
  conocido — `GET /pca/reference` expone esas coordenadas de fondo. La sugerencia de tipo de CLIP
  (zero-shot, primera capa del pipeline) también se actualizó de los 3 huesos viejos a los 9 grupos,
  para que no diera pistas inconsistentes con el clasificador real.
- **Verificación**: probado de punta a punta con la API real corriendo (`uvicorn`) y un test con
  `streamlit.testing.v1.AppTest` que simula subir una imagen real y pulsar "Procesar" contra la API
  en marcha — sin excepciones, con la predicción y el punto PCA correctos. No se pudo comprobar
  visualmente en un navegador real (no hay herramienta de navegador en este entorno) — se dijo
  explícitamente en vez de asumir que se ve bien.

### Protección del trabajo con git
- Ninguno de los dos repos tenía forma de recuperar el trabajo si algo fallaba (recordando el
  incidente de `/dev` vs `/root/dev` de la sesión anterior). Se inicializó git en `base_datos_osea`
  (no lo tenía) y se comiteó el trabajo pendiente en `osteolab-ml-platform` (sí lo tenía, con
  remoto en GitHub, pero cambios sin comitear). `data/meshes/` y `renders/` quedan fuera de git
  (2.6GB+ de binarios regenerables desde los scripts); los `.joblib` de modelos entrenados también
  quedan fuera (regenerables desde el CSV de features, que sí se versiona). Sin `push` a ningún
  remoto todavía — solo local.

---

## 2026-07-23 (cont.) — Consolidación: fusión de los dos repos en uno

- El usuario pidió aplanar y fusionar: `base_datos_osea` debía pasar a vivir dentro de
  `osteolab-ml-platform`, todo en un único repo, y subido a GitHub.
- **Aplanado previo necesario**: `osteolab-ml-platform` tenía un anidamiento redundante
  (`osteolab-ml-platform/osteolab-ml-platform/`, la raíz real del repo git y una subcarpeta con el
  mismo nombre conteniendo el proyecto de verdad). Se movió todo un nivel arriba con `git mv`
  (detectado como renames 100%, historial intacto).
- **Fusión con `git subtree add --prefix=base_datos_osea`** (no un simple copy-paste): trae el
  historial completo de los 7 commits de `base_datos_osea` encadenado dentro del historial de
  `osteolab-ml-platform`, en vez de perderlo.
- **Lo que git subtree NO trae** (por diseño, solo mueve lo trackeado): `data/meshes/` (3.0GB),
  `renders/` (327MB) y `venv/` (188MB), todos gitignorados. Se copiaron a mano a la nueva ubicación
  y se verificó recuento de ficheros idéntico origen/destino antes de dar el movimiento por bueno.
- **Se encontró y arregló un problema arrastrado de la primera mudanza** (`/dev` → `/root/dev`,
  documentado en el punto 1 de este historial): el `venv/` copiado tenía shebangs apuntando a
  `/dev/base_datos_osea/venv/bin/python3` — una ruta que ya no existía ni siquiera antes de la
  fusión de hoy, nunca se había verificado a fondo (solo se probó `python -c "import ..."`, que no
  pasa por el shebang). Se recreó el venv desde cero en la ubicación nueva en vez de arrastrar el
  problema una vez más.
- **Rutas absolutas actualizadas**: `build_dataset.py` tenía `/root/dev/base_datos_osea/renders`
  quemado como ruta por defecto — se cambió a relativa (`Path(__file__).resolve().parents[2] /
  "base_datos_osea" / "renders"`), portable si alguien más clona el repo. Se regeneró la tabla de
  features y se reentrenó para que `source_path` reflejase la ruta nueva (mismos resultados que
  antes, mismos datos, solo cambia dónde se buscan).
- Contenedores Docker parados antes de mover directorios (tenían bind mounts activos a las rutas
  viejas) y reconstruidos después.

---

## 2026-07-23 (cont.) — Banco de evaluación fijo + MLflow persistente (EN CURSO, cortado a mitad)

**Motivación del usuario**: quiere un set de imágenes conocidas para evaluar el flujo de forma
comparable a lo largo del tiempo, con los resultados guardados en MLflow — en parte para tener
esa comparación, en parte como ejercicio de aprender la herramienta.

- **Diseño**: `build_dataset.py` ahora reparte las imágenes en dos CSV en vez de uno:
  `bone_geometric_features.csv` (entrenamiento) y `bone_geometric_benchmark.csv` (banco fijo,
  nunca se usa para entrenar). El banco reserva **1 espécimen completo por grupo** (8 de 9 —
  `pelvis` se deja fuera a propósito, solo tiene 2 especímenes y ya sabemos que es el más débil,
  restarle uno lo dejaría en 1 solo). Especímenes elegidos (`BENCHMARK_SPECIMENS` en
  `build_dataset.py`): el de *H. naledi* donde lo hay (mide "generaliza a la otra especie del
  dataset", más honesto que un ejemplo cualquiera de sapiens), y donde no hay naledi (cráneo,
  hueso_plano) el primate menos representado o el espécimen de peor calidad.
- **Nuevo script `src/training/evaluate.py`**: carga el modelo ya entrenado + el banco fijo,
  calcula accuracy/precision/recall por grupo, y **loguea una corrida de MLflow** en un experimento
  aparte (`osteolab-benchmark`, distinto de `osteolab-bone-classification-geometric-bone_group`
  donde ya se logueaba cada entrenamiento) — pensado para ejecutarse cada vez que cambie el modelo
  y comparar en la UI de MLflow (`http://localhost:5000`) si mejora o empeora sobre el mismo examen.
- **Se regeneró la tabla** con el reparto nuevo: 1128 filas de entrenamiento (antes 1320) + 192 del
  banco (8 especímenes × 24 vistas, ninguno perdido).

### Pendiente para retomar mañana (cortado aquí a petición del usuario)
1. **Reentrenar contra el MLflow real** (no el sqlite temporal que se usó y se borraba todo el
   rato durante la sesión de hoy): `MLFLOW_TRACKING_URI=http://localhost:5000
   python -m src.training.train_geometric` — los contenedores de Docker (`postgres`, `mlflow`, `app`,
   `ui`) deberían seguir arriba, comprobar con `docker ps`. **Importante**: el modelo que hay ahora
   mismo en `models/geometric_model_bone_group.joblib` (y el que sirve el contenedor `app`) se
   entrenó con la tabla VIEJA de 1320 filas (antes de separar el banco) — no está roto, pero está
   desactualizado respecto al CSV de entrenamiento actual (1128 filas). Hay que reentrenar antes de
   confiar en el número que dé `evaluate.py` la primera vez.
2. **Ejecutar `python -m src.training.evaluate`** contra ese modelo recién entrenado — primera
   entrada de referencia en el experimento `osteolab-benchmark` de MLflow.
3. **Mirar la UI de MLflow** (`http://localhost:5000`) juntos — parte del objetivo es que el
   usuario aprenda a leer/comparar corridas ahí, no solo que el número exista.
4. Tras esto, **redesplegar el contenedor `app`** (`docker compose up --build -d app` desde
   `osteolab-ml-platform/docker/`) para que sirva el modelo reentrenado — como ya pasó una vez hoy,
   un `restart` normal no basta, hace falta rebuild porque el código no está en un volumen.
5. Commitear `build_dataset.py`, `evaluate.py`, y los dos CSV regenerados (están sin comitear
   ahora mismo, `git status` limpio lo confirma).

### Pregunta abierta sin resolver (el usuario la lanzó, no se ha decidido)
El usuario preguntó si es buen momento para generar renders con más ruido (fondos/iluminación
variados) y entrenar una CNN con un backbone grande. Recomendación dada (pendiente de que el
usuario elija cómo seguir):
- **Ruido/aumento de datos: sí, más urgente de lo que parece.** La segmentación actual
  (`_segment()` en `extract.py`, umbral de Otsu) está calibrada para render limpio sobre fondo
  blanco — una foto real de usuario (fondo de mesa, sombras) probablemente rompe la segmentación
  antes de llegar siquiera al clasificador. Cuello de botella distinto y anterior al de "cuántos
  especímenes hay".
- **CNN: todavía no, como decisión de ingeniería** — toda la sesión de hoy demostró (con
  `GroupKFold` y espécimen fuera) que lo que limita la generalización es el nº de especímenes
  físicos, no la potencia del modelo; una CNN no arregla eso y con pocos especímenes por clase
  puede memorizar peor (textura/iluminación de un render concreto) que las features geométricas
  actuales, que ya ignoran color/textura por diseño.
- Sí se ofreció como **ejercicio de aprendizaje aparte** (no sustituyendo lo actual): fine-tuning
  de un backbone pequeño, comparado de forma justa contra el modelo actual sobre el mismo banco
  fijo de MLflow en vez de a ciegas.

---

## 2026-07-24 (cont.) — Banco de fotos reales, bug de NaN, segmentación combinada, y arranque de renders con ruido para CNN (EN CURSO, cortado a mitad a petición del usuario — se va a dormir)

**Motivación**: tras desplegar el modelo de la Ronda 3 (commit `4ce41f9`), el usuario probó la app
con una foto real suya (cráneo de resina) y la segmentación falló (solo detectó ~40% del cráneo
como hueso). Esto llevó a construir un banco de evaluación nuevo con fotos reales, y de ahí a la
conclusión de que hace falta un enfoque combinado CNN + geométrico entrenado con datos que se
parezcan más a fotos reales.

### 1. Banco de fotos reales (`data/real_world_benchmark/`, no versionado — ver `.gitignore`)
- 3 agentes en paralelo buscaron fotos reales (Wikimedia Commons, verificadas visualmente una a
  una) para los 9 grupos morfológicos. **Solo 1 de los 3 completó** (25 fotos: cranio, mandíbula,
  hueso_largo) — los otros 2 (hueso_pequeño/plano/pelvis, y sacro/vértebra/costilla) murieron por
  límite de sesión de la API antes de guardar nada. **Pendiente: relanzarlos** para completar los
  6 grupos que faltan.
- Nuevo script `src/training/evaluate_real_photos.py`: a diferencia de `evaluate.py` (que lee
  features ya extraídas de un CSV), este vuelve a correr `extract_features` desde la imagen cruda
  — prueba la segmentación de verdad, no solo el clasificador. Loguea en MLflow, experimento
  `osteolab-benchmark-real-photos`.
- **Resultado**: 28% accuracy (vs 75-87% en los bancos sintéticos). `hueso_largo` a 0% recall
  (0 de 9), `cranio` a 25% (confundido con `hueso_plano`, el mismo fallo que vio el usuario en
  vivo). Con texto/marca de agua superpuesto: 17% vs 38% sin texto. Fondo tipo "estudio con
  degradado" (el más común en las fotos de huesos largos, de colecciones osteológicas reales):
  0% accuracy en las 8 imágenes con ese fondo.

### 2. Bug real encontrado y arreglado: crash por NaN en producción
`ellipse_ratio` (y otras features con división guardada: `extent`, `solidity`, `circularity`,
`convexity`, `aspect_ratio`) pueden salir `None` si el contorno detectado es degenerado (pocos
puntos, lado corto ~0) — nunca pasa en los renders limpios de entrenamiento (verificado: 0 NaN en
`bone_geometric_features.csv`) pero sí en 2 de 25 fotos reales, y el `/classify` de la API real
habría devuelto un 500 sin control. **Arreglado** en `src/inference/predict_geometric.py`: se
precalcula la media de cada columna sobre la tabla de entrenamiento (`_fallback_means`) y se usa
para imputar cualquier `None`/NaN antes de pasar la fila al modelo/PCA.

### 3. Segmentación combinada (Otsu + adaptativo) — mejora real pero insuficiente
`_segment()` en `src/cv_extractor/extract.py` ahora combina umbral global (Otsu, bueno en renders
limpios) con umbral adaptativo local (capta variación de sombreado real) + cierre morfológico
agresivo (25×25) + relleno del contorno externo con huecos reales restaurados. Verificado sin
regresión: mismo accuracy en banco de renders (0.7361→0.7384 CV, 0.75 banco fijo, ambos estables).
**Pero no mejoró el banco de fotos reales (sigue en 0.28)** — dos causas nuevas descubiertas:
- El cierre agresivo puede fusionar dos huesos separados en la misma foto (ej. "dos fémures" de
  una colección) a través de un fondo con degradado luminoso, inventando una forma falsa (un
  "puente" entre ambos) — nunca pasa en renders (siempre 1 hueso por imagen).
- Incluso con buena segmentación, la silueta de una foto real (ángulo de cámara, perspectiva)
  sigue sin parecerse lo bastante a un render limpio como para que el clasificador la reconozca.
- Conclusión: el cuello de botella no es el umbral de segmentación, es que el modelo (geométrico
  o cualquier CNN futura) solo ha visto renders de estudio — confirma con datos la sospecha de
  ayer sobre necesitar renders con ruido.

### 4. Decisión: CNN + geométrico combinado, entrenado con renders "ruidosos"
El usuario decidió pasar a un modelo combinado (CNN con backbone moderno preentrenado + features
geométricas), comparando escenarios en MLflow, en vez de seguir puliendo solo la segmentación.
Se avisó del riesgo: una CNN entrenada con los mismos renders limpios de hoy probablemente
memorizaría igual de mal las peculiaridades del render — el orden acordado fue **primero generar
renders con ruido realista, luego entrenar la CNN sobre eso**. DVC se deja para después (está a
medio montar: `dvc.yaml`/`.dvc/config` existen pero el paquete no está instalado y el remoto
apunta a una ruta de otra máquina que no existe aquí).

Lista de ideas de ruido que dio el usuario, con las decididas para la v2 del render:
- ✅ Fondo con foto real compuesta (no procedural) — en curso, ver más abajo.
- ✅ Sombra de contacto real (shadow catcher de Cycles sobre un plano) — en curso.
- ✅ Encuadre no siempre centrado (jitter de cámara) — en curso.
- ✅ Dominantes de color / temperatura de luz variada — ya en v1, se amplía en v2.
- ⏸️ Degradación de calidad (reducir y volver a ampliar resolución, simular compresión JPEG) —
  decidido hacerlo como post-proceso en Python (barato), no en Blender. Pendiente de implementar.
- ⏸️ Objetos distractores en la escena (martillo, piedras, pelota) para forzar a segmentar solo
  el hueso — con buen criterio (primitivas procedurales en Blender, sin assets externos), pero
  **pospuesto**: más esfuerzo/riesgo que el resto, prioridad menor.
- ⏸️ Deformación del propio hueso (tipo Substance Painter: desgaste, óxido) — **pospuesto
  explícitamente**: riesgo de corromper la forma real que el modelo (sobre todo el geométrico)
  necesita aprender correctamente; si se hace, debe ser un desplazamiento muy sutil, no una
  distorsión de forma.
- 📝 Muestreo aleatorio de un subconjunto de vistas por espécimen (no las 24 completas) +
  barajar el orden al entrenar, para evitar que el modelo aprenda patrones de la secuencia de
  ángulos — es una decisión del script de entrenamiento de la CNN (dataloader), no del render;
  pendiente de aplicar cuando se escriba ese script.

**Progreso técnico de la v2** (`base_datos_osea/blender/render_bone_noisy.py`, commiteado tal cual
está, incompleto):
- v1 (fondo procedural sólido/degradado + luz solar direccional aleatoria + tono de hueso
  aleatorio) se probó sobre un fémur, se veía bien, pero el usuario pidió la ampliación de arriba
  antes de lanzar el lote completo — **se abortó el lote de v1 a mitad** (solo 2 especímenes
  parciales, `renders_noisy/` se borró por completo, no queda nada a medio hacer en disco).
- Se lanzó un agente para buscar ~20-25 fotos de fondo genéricas (mesa, tierra, mármol, tela,
  taller, hierba, hormigón, alfombra) en `data/backgrounds/` (no versionado) — **se paró a mitad**
  (0 fotos conseguidas todavía) para no dejar trabajo sin supervisión mientras el usuario dormía.

### Pendiente para retomar mañana (cortado aquí a petición del usuario)
1. Relanzar el agente de fondos genéricos (`data/backgrounds/`, ~20-25 fotos, ver categorías
   arriba) — no se guardó nada, hay que repetirlo desde cero.
2. Terminar la v2 de `render_bone_noisy.py`: falta el plano `shadow catcher`, el jitter de
   encuadre/cámara, ampliar los presets de temperatura de color, y el render con
   `film_transparent=True` + canal alpha para poder componer sobre las fotos de fondo reales.
3. Escribir el script de composición en Python (RGBA del render + fondo real elegido al azar →
   imagen final RGB) — incluir ahí la degradación de calidad (resize down/up) como paso barato.
4. Lanzar el lote completo sobre los 122 especímenes (10 vistas ruidosas cada uno, ~1h estimada
   según la prueba de v1: ~25s/espécimen con 6 vistas).
5. Relanzar los 2 agentes de fotos reales que fallaron ayer por límite de sesión (hueso_pequeño/
   plano/pelvis, sacro/vertebra/costilla) para completar el banco de 9 grupos (hoy solo hay 3).
6. Diseñar y escribir el script de entrenamiento de la CNN (backbone preentrenado vía `timm`,
   no instalado todavía; hay GPU CUDA disponible) + combinación con las features geométricas +
   logging de "escenarios" comparables en MLflow.
7. Redesplegar el contenedor `app` (`docker compose up --build -d app`) — sigue sirviendo el
   modelo de hace 2 horas, anterior al fix de NaN y a la segmentación combinada.
8. DVC sigue pendiente (decisión explícita de dejarlo para después).

---

## 2026-07-25 — Continuación: fondos reales + v2 de renders con ruido + composición

Retoma los puntos 1-3 del pendiente de ayer.

- **`render_bone_noisy.py` v2 terminado** (puntos 2 del pendiente): añadido un plano
  `is_shadow_catcher` bajo el hueso (sombra de contacto real, alpha parcial donde cae sombra) +
  `film_transparent=True` con salida `RGBA` (el render ya no lleva fondo "quemado", se compone
  después) + jitter del punto al que apunta la cámara (~15% del radio, así el encuadre no está
  siempre centrado) + paleta de temperaturas de color con 8 presets con nombre (incandescente,
  halógeno, luz día, nublado, sombra abierta, hora dorada, fluorescente, flash) en vez del jitter
  lineal de un solo tono blanco de la v1. Probado con un fémur (3 vistas, GPU CUDA): renderiza sin
  error, alpha con zona parcial (sombra) + zona opaca (hueso) confirmado con PIL/numpy, y
  verificado visualmente componiendo a mano sobre un color plano.
- **Script de composición nuevo**: `scripts/compose_noisy.py` (punto 3 del pendiente). Por cada
  render RGBA bajo `<renders-dir>/especie/hueso/especimen/*_viewNNN.png` (misma estructura que
  `renders/`, la que espera `build_dataset.py` del repo padre): elige un fondo real al azar,
  recorta una región cuadrada aleatoria de ese fondo (45%-100% del lado corto, para no repetir
  siempre el mismo encuadre de fondo) y compone el render encima con `Image.alpha_composite`;
  luego reduce y vuelve a ampliar la resolución (factor aleatorio 0.35-0.85, simula borrosidad de
  cámara mediocre) y guarda como `.jpg` con calidad JPEG aleatoria (35-85) — la propia compresión
  al guardar ya aporta los artefactos de bloque, sin necesidad de una doble pasada de recodificado.
  Escribe en `<out-dir>` con la misma estructura de carpetas y nombre de fichero que la entrada
  (solo cambia `.png`→`.jpg`), para que un futuro `--renders-dir` del entrenamiento pueda apuntar
  directamente ahí. Probado de punta a punta con los renders de prueba del fémur + 3 fondos
  sintéticos: genera los `.jpg`, tamaño y modo correctos, sombra y blur visibles al inspeccionar.
- **`.gitignore`**: añadidas `data/backgrounds/` y `renders_noisy/` (no versionadas, igual que
  `renders/` y `data/meshes/` — se habían quedado fuera del `.gitignore` original).
- **Agente de fondos genéricos relanzado y completado** (punto 1 del pendiente): 25 fotos CC0
  descargadas de Poly Haven (`api.polyhaven.com`/`dl.polyhaven.org`, escaneos de materiales
  reales, 2048px, sin atribución requerida — Unsplash/Pexels bloquearon la descarga directa por
  curl en esta sesión, requieren JS/login) a `data/backgrounds/`, verificadas magic bytes + PIL,
  sin marca de agua/texto/gente/animales. Repartidas en las 8 categorías: mesa 4, tierra 3,
  mármol 3, tela 3, taller 3, hierba 3, hormigón 3, alfombra 3 (78MB total). **Alfombra es la más
  floja**: Poly Haven solo tiene un activo de alfombra real (`dirty_carpet`); las otras 2 son
  texturas de tela suave (lana bouclé, borreguito) que leen como alfombra pero no lo son
  literalmente — revisar con una búsqueda de stock dedicada si se necesita más realismo ahí.
- **Lote completo lanzado** (nuevo script `scripts/render_noisy_batch.py`, un proceso Blender por
  espécimen — evita el bug de índices por posición del intento anterior usando
  `Path.relative_to(MESHES_DIR).parts[:3]`): 122/122 especímenes, 10 vistas c/u, **0 fallos**,
  1220 PNG RGBA en `renders_noisy/` (138MB).
- **Bug real encontrado y arreglado al revisar el resultado a ojo**: el color del hueso, la luz y
  el fondo real se sortean cada uno por su lado (nada los coordina) — inspección visual de una
  muestra encontró casos con el hueso casi invisible sobre el fondo (p.ej. tibia gris sobre tierra
  marrón, sacro gris oscuro sobre baldosa). Medido con un script de contraste (diferencia de
  luminancia media hueso-vs-fondo sobre 200 renders al azar): **25.3% por debajo de 25/255 de
  contraste, 15.8% por debajo de 15/255** — no es "dificultad realista" a propósito, es señal
  perdida en 1 de cada 4 imágenes. Arreglado en `compose_noisy.py` con `pick_background_crop`:
  prueba hasta 6 candidatos de fondo/recorte y usa el primero que supere el contraste mínimo (o el
  mejor visto si ninguno lo supera) — no descarta nunca el render, solo elige mejor entre
  opciones. Recompuesto el dataset completo con el fix (rápido, no requiere Blender, solo
  Python sobre los PNG ya renderizados): contraste bajo bajó de 25.3%→0.7%, contraste muy bajo de
  15.8%→0%. Verificado visualmente que los dos casos peores detectados antes ahora se ven bien.
- **Resultado final**: `renders_noisy_composited/` — 1220 `.jpg`, 30MB, misma estructura
  especie/hueso/especimen que `renders/`, listo para que `build_dataset.py` (o una variante) lo
  use como fuente adicional de entrenamiento.

### Evaluación: ¿ayuda realmente el dataset con ruido? (osteolab-ml-platform, no este repo)

El usuario pidió medir si todo lo de arriba mejora algo de verdad, no solo generar los datos.
Metodología: recrear el `.venv` del repo padre (el que había estaba roto — `pyvenv.cfg` apuntaba
a la ruta pre-aplanado `osteolab-ml-platform/osteolab-ml-platform/`, de antes del `flatten` — y sin
`pip`), levantar los contenedores (`postgres`/`mlflow`/`app`/`ui` estaban parados desde el reinicio
de la máquina), y regenerar `bone_geometric_features.csv` desde cero con el `extract.py` actual
(el CSV existente databa de las 22:47 de ayer, ANTES del commit de las 23:07 que combinó Otsu +
adaptativo — comparar contra el modelo viejo habría sido injusto en ambas direcciones).

Se añadió `--dataset-path` a `train_geometric.py` (antes solo leía `params.yaml`, sin forma de
apuntar a un CSV alternativo sin tocar el archivo de params) y `--ext`/regex genérico a
`build_dataset.py` (antes solo aceptaba `.png`, hacía falta para leer los `.jpg` de
`renders_noisy_composited`). Con eso: features del set con ruido extraídas igual que las limpias
(excluye los mismos `BENCHMARK_SPECIMENS` automáticamente, misma estructura de carpetas — 98.2%
de las 1220 imágenes se segmentaron bien, 1109 filas de entrenamiento) + concatenadas con las
2544 limpias = 3653 filas combinadas.

**Comparación ANTES (solo renders limpios) vs DESPUÉS (limpios + ruido), mismo código, dos
entrenamientos consecutivos logueados en los mismos experimentos de MLflow para poder comparar
en la UI**:

| Métrica | Antes | Después |
|---|---|---|
| GroupKFold espécimen-fuera (interno) | 0.7384 | 0.6450 |
| Banco fijo (renders limpios, especímenes nunca vistos) | 0.7500 | 0.7731 |
| **Banco de fotos reales (25 fotos, 3 grupos)** | **0.2800** | **0.4000** |

El accuracy interno (GroupKFold) BAJA con el dataset combinado — esperable, el set con ruido es
genuinamente más difícil (esa es la idea), así que el promedio interno cae aunque el modelo
generalice mejor a lo que de verdad importa. Las dos métricas externas (los exámenes que nunca
entran en entrenamiento) mejoran, y la de fotos reales lo hace con fuerza: **+12 puntos (0.28→0.40,
+43% relativo)**. Desglose relevante:
- `hueso_largo`: 0%→78% recall en fotos reales (0/9 → 7/9 correctas).
- Fondo tipo **"estudio (fondo degradado)"** (el caso que en producción rompía la segmentación,
  motivo original de todo este esfuerzo): **0%→87.5%** (0/8 → 7/8 correctas) — el problema
  concreto que se quería arreglar, arreglado y medido.
- `cranio` empeora: 50%→0% recall (4/8 antes, 0/8 después) — regresión real, aunque sobre una
  muestra muy pequeña (8 fotos) como para sacar conclusiones fuertes todavía.
- El banco de fotos reales solo cubre 3 de 9 grupos (`cranio`, `hueso_largo`,
  `mandibula_maxilar` — ver pendiente de ayer sobre completarlo a 9), así que esta lectura es
  parcial: no hay forma de saber todavía si sacro/costilla/pelvis (los grupos que peor iban)
  mejoran o no con fotos reales.
- Los `models/geometric_*.joblib` actuales en disco son los del entrenamiento CON ruido (el de
  después) — el contenedor `app` los monta en vivo (`volumes: ../models:/app/models`) pero el
  proceso ya en marcha tiene el modelo viejo cacheado en memoria desde el import, así que servir
  el nuevo requiere reiniciar el contenedor, no solo que cambien los ficheros.

### Trazabilidad MLflow + redeploy (mismo día, a petición del usuario)

- **`train_geometric.py`** ahora loguea `dataset_path`/`n_train_rows`/`n_train_specimens` como
  params y admite `--tag KEY=VALUE` (repetible) para etiquetar la run (p.ej.
  `dataset_variant=combined_with_noise`); además escribe
  `models/geometric_training_meta_<target>.json` (dataset, filas, `run_id`, experimento,
  accuracy espécimen-fuera, timestamp) — es la fuente de verdad en disco de qué entrenamiento
  produjo el `.joblib` que hay ahora mismo en `models/`.
- **`evaluate.py`** y **`evaluate_real_photos.py`** leen ese JSON y loguean `training_run_id`/
  `training_dataset_path`/`training_n_rows` en su propia run — así cualquier run de evaluación en
  la UI de MLflow enlaza directamente a qué entrenamiento (y qué dataset) le dio el modelo que
  evaluó, sin tener que adivinarlo por fecha.
- Las 6 runs de la comparación de esta sesión (2 entrenamientos + 2×2 evaluaciones) se etiquetaron
  a posteriori con `dataset_variant=clean_only` / `combined_with_noise` vía `MlflowClient.set_tag`,
  y se repitió el entrenamiento+evaluación de la variante con ruido ya con el logging nuevo
  (resultado idéntico: RandomForest con seed fija es determinista) para que el modelo que queda
  desplegado tenga su `training_meta` completo.
- **Redeploy**: `docker compose up --build -d app ui` (postgres/mlflow ya estaban al levantarlos
  antes en la sesión) — el contenedor `app` ahora sirve el modelo entrenado con ruido (el que dio
  0.40 en fotos reales). UI en `localhost:8501`, API en `localhost:8001/docs`, MLflow en
  `localhost:5000`.
- **Agentes relanzados** para completar el banco de fotos reales a los 9 grupos (pendiente de
  ayer): uno para hueso_pequeño/hueso_plano/pelvis, otro para sacro/vertebra/costilla — en curso
  al cortar esta entrada.

### Corte de luz y recuperación (mismo día, más tarde)

Se fue la luz mientras los 2 agentes de fotos reales corrían y con los contenedores Docker
levantados; la máquina se reinició y los 4 contenedores (`postgres`/`mlflow`/`app`/`ui`) quedaron
parados. Al recuperar la sesión: los agentes **sí habían terminado** antes del corte (el manifest
del banco quedó completo, 75 fotos en los 9 grupos, último fichero escrito a las 13:45) — no se
perdió trabajo, solo quedó pendiente el paso de evaluación. Se relevantaron los contenedores
(`docker compose up -d postgres mlflow app ui`) y se completó el pendiente #1: correr
`evaluate_real_photos.py` contra el banco ya completo, con el mismo modelo en producción
(`combined_with_noise`, el de las 12:02).

**Resultado con el banco completo (75 fotos, 9 grupos) vs. la lectura parcial de antes (3/9
grupos, accuracy 0.40)**: **accuracy 0.2667** — baja porque los 6 grupos nuevos añadidos son
justo los que peor rendían ya en el banco fijo interno:

| Grupo | Recall | Precision |
|---|---|---|
| `hueso_largo` | 0.78 | 0.29 |
| `mandibula_maxilar` | 0.38 | 0.75 |
| `hueso_plano` | 0.50 | 0.29 |
| `hueso_pequeno` | 0.44 | 0.17 |
| `vertebra` | 0.25 | 1.00 |
| `cranio` | 0.00 | 0.00 |
| `costilla` | 0.00 | 0.00 |
| `pelvis` | 0.00 | 0.00 |
| `sacro` | 0.00 | 0.00 |

`costilla`/`pelvis`/`sacro` en 0% recall confirma que el cuello de botella es el mismo de siempre
(pocos especímenes en entrenamiento, `pelvis` con solo 4) y que la lectura parcial de 0.40 era
optimista por casualidad de qué 3 grupos se midieron primero. El dato bueno se mantiene: el
desglose por tipo de fondo vuelve a confirmar que **"estudio (fondo degradado)"** (el caso que
rompía en producción) sale con 0.80 de accuracy (8/10) — el problema original que motivó todo
este esfuerzo sigue arreglado, el problema nuevo que revela el banco completo es distinto (falta
de especímenes en los grupos débiles, no ruido de fondo). Corrida logueada en MLflow
(`osteolab-benchmark-real-photos`, run `28004d9e82a6421e89ad6a15d77b53f6`).

### CNN vs. geométrico (pendiente #3, resuelto): ¿ayuda de verdad una CNN?

Implementado en `osteolab-ml-platform` (no este repo): `src/training/cnn_dataset.py` (manifest =
las mismas tablas de `build_dataset.py`, no hace falta recorrer `renders/` de nuevo),
`src/training/train_cnn.py` (backbone `resnet18` de `timm`, 2 fases — cabeza congelada 8 épocas,
luego fine-tune del backbone a LR bajo 15 épocas más —, class weights por el desbalance de
`pelvis`), `src/inference/predict_cnn.py` + `evaluate_cnn.py` / `evaluate_cnn_real_photos.py`
(mismos bancos que el geométrico, para comparar de verdad). Dos variantes:
- **`cnn_only`**: aprende solo de píxeles, no necesita segmentación en absoluto.
- **`hybrid`**: embedding del backbone concatenado con el mismo vector de features geométricas
  del RandomForest (sí necesita segmentación) + cabeza MLP pequeña.

Split especímen-fuera único (no 10×CV repetido como el geométrico — inviable en tiempo para una
red; se acepta más varianza en el número a cambio de que entrenar sea viable). GPU RTX 4080,
cada entrenamiento completo (23 épocas) tardó **~3-4 minutos**.

**Resultado — misma tabla, tres exámenes, el geométrico es el `combined_with_noise` ya
documentado arriba (0.6450 / 0.7731 / 0.2667):**

| Modelo | CV interno (espécimen fuera) | Banco limpio (renders, especímenes nunca vistos) | **Fotos reales (75, 9 grupos)** |
|---|---|---|---|
| Geométrico (RandomForest) | 0.6450 | 0.7731 | 0.2667 |
| **CNN sola** (`resnet18`) | 0.6830 | 0.7546 | **0.4933** |
| Híbrido (CNN + geométrico) | 0.7009 | **0.8102** | **0.1333** |

Lectura, y es la parte importante: **el híbrido gana claramente en los dos bancos sintéticos
(interno y banco limpio) y luego se DESPLOMA en fotos reales — peor incluso que el geométrico
solo**, casi siempre prediciendo `sacro` (recall 1.00 mecánico con precisión 0.14, el resto de
grupos en 0.00). Hipótesis más probable: el modelo aprende en entrenamiento a fiarse mucho de las
features geométricas (allí son una señal limpia y fiable, vienen de renders bien segmentados) y
en una foto real esas mismas features salen distorsionadas (segmentación real, con sombras/fondo,
nunca es tan limpia como un render) — el modelo las sigue creyendo y la fusión ingenua las
arrastra a una predicción degenerada, en vez de dejar que la señal de textura de la CNN compense.
Es la misma lección que ya dejó esta sesión con el banco de fotos reales completo: **el banco
limpio sintético puede ser activamente engañoso** — aquí no solo "optimista", sino que ordena los
3 modelos justo al revés que el examen que de verdad importa.

**La CNN sola es la que de verdad ayuda**: 0.2667→0.4933 en fotos reales (+85% relativo), y a
diferencia del geométrico (`costilla`/`pelvis`/`sacro`/`cranio` en 0% recall) **ningún grupo cae a
0%** — el peor es `hueso_pequeno` con 0.22. Contrapartida real: en el desglose por fondo, el caso
`estudio (fondo degradado)` (el bug de producción original) baja de 0.80 (geométrico) a 0.40 (CNN
sola) — la CNN generaliza mejor en conjunto pero no domina en todos los subcasos, y esa es
justo el subcaso que motivó el esfuerzo inicial.

**Conclusión para el pendiente**: sí, una CNN ayuda — pero la versión ingenua de "combinar todo"
(el híbrido) es la peor opción de las tres en el examen que importa, pese a ganar en los dos que
no importan tanto. Si se quiere seguir por aquí, el candidato razonable a explorar es CNN sola,
no el híbrido tal cual está — y antes de considerar reemplazar el modelo en producción, estaría
bien: (a) confirmar que 0.49 en 75 fotos no es ruido de muestra pequeña, y (b) si se insiste en
combinar ambas señales, hacerlo de forma que el modelo pueda desconfiar de las features
geométricas cuando huelan a fuera-de-distribución, no concatenarlas a ciegas.

Artefactos: `models/cnn_model_{cnn_only,hybrid}_bone_group.pt` + encoder/scaler/columnas,
`models/cnn_training_meta_{cnn_only,hybrid}_bone_group.json`. MLflow: experimento nuevo
`osteolab-bone-classification-cnn-bone_group` para las runs de entrenamiento; las de evaluación
reusan `osteolab-benchmark` / `osteolab-benchmark-real-photos` (mismo sitio que el geométrico, con
`model_type` como param/tag) para poder comparar las tres filas una al lado de otra en la UI.
No se ha desplegado ningún modelo CNN a producción — el `app`/`ui` en Docker siguen sirviendo el
geométrico, esto ha sido solo el experimento de medición que pedía el pendiente.

### Auditoría manual del banco de fotos reales: un tercio estaba contaminado

El usuario revisó a ojo las fotos del banco e identificó el problema con precisión: varias
imágenes tienen **más de un hueso** (p.ej. `costilla` con una foto de la caja torácica entera, o
varias costillas sueltas fotografiadas juntas; `mandibula_maxilar` con un cráneo completo en vez
de solo la mandíbula) — el clasificador (y su segmentación) miden ahí un objeto que no es "un
hueso aislado", ensuciando cualquier lectura de accuracy/recall. Revisadas las 75 fotos una a una:

- **24 de 75 (32%) tenían más de un elemento** o eran un montaje de varias imágenes: cajas
  torácicas/costillas sueltas en grupo (`costilla`: 6 de 8), manos/pies articulados o láminas con
  varios huesos de la mano bajo un mismo texto (`hueso_pequeno`: 6 de 9), un cráneo con mandíbula
  (`mandibula_maxilar`: 1 de 8), 4 cráneos en fila (`cranio`: 1 de 8), pares de fémur/tibia/radio
  fotografiados juntos (`hueso_largo`: 5 de 9), 2 escápulas + un montaje de 5 fases de
  descomposición (`hueso_plano`: 2 de 8), un pelvis+vértebras+hueso extra del mismo museo de Hong
  Kong que ya había dado problemas en costilla, duplicado en 2 ficheros (`pelvis`: 2 de 9), 2
  diagramas con 2 sacros cada uno (`sacro`: 2 de 8), y un montaje de 3 vértebras (`vertebra`: 1 de
  8).
- **6 de esas 24 se pudieron recuperar recortando**, no solo descartar: 3 pares de huesos
  (`hueso_largo_femur_01`, `hueso_largo_tibia_02`, `hueso_plano_03`) tenían los dos objetos
  separados y sin solape de caja delimitadora — detectados por contorno (Otsu + cierre morfológico,
  el mismo enfoque que `extract.py`) y recortados automáticamente a 1 hueso por imagen, verificado
  visualmente. El resto (costillas en abanico, pares de huesos que sí se solapan en su caja
  delimitadora, fotos con texto ocupando todo el encuadre) no se pudo separar de forma fiable sin
  arriesgarse a crear recortes igual de contaminados, así que se descartaron sin más — quedaron
  movidos a `/tmp/.../removed_benchmark_images/` (fuera del repo, no se borraron por si hacen
  falta) en vez de borrados.
- **Resultado**: banco de 75 → 55 fotos. `costilla` queda en solo 2 imágenes y `hueso_pequeno` en
  3 — demasiado pocas para una lectura de recall fiable por sí solas, pero ya no mezclan señal de
  "un hueso" con señal de "varios huesos a la vez".

**Los 3 modelos mejoran solo por limpiar el examen, sin tocar el modelo**:

| Modelo | Antes (75, contaminado) | Después (55, limpio) |
|---|---|---|
| Geométrico | 0.2667 | **0.3273** |
| CNN sola | 0.4933 | **0.5273** |
| Híbrido (bug de escala aún sin arreglar) | 0.1333 | 0.2000 |

El híbrido sigue roto (sube un poco pero sigue pegado al azar, 0.11) — el bug de escala de las
features geométricas no tiene nada que ver con la limpieza de imágenes, sigue pendiente si se
quiere arreglar. Geométrico y CNN sola sí ganan de verdad: no cambiaron nada del modelo, solo
dejó de penalizárseles por acertar "razonablemente" sobre una foto que en realidad tenía 2-5
huesos y una sola etiqueta.

**Recall por grupo, banco limpio (n pequeño en varios grupos, leer con cautela)**:

| Grupo | n | Geométrico | CNN sola |
|---|---|---|---|
| costilla | 2 | 0.00 | 0.50 |
| cranio | 7 | 0.00 | 0.57 |
| hueso_largo | 8 | **0.88** | 0.38 |
| hueso_pequeno | 3 | 0.67 | 0.67 |
| hueso_plano | 8 | 0.62 | 0.62 |
| mandibula_maxilar | 7 | 0.43 | **0.71** |
| pelvis | 7 | 0.00 | 0.29 |
| sacro | 6 | 0.00 | 0.33 |
| vertebra | 7 | 0.14 | **0.71** |

Nota honesta: no es "la CNN gana en todo" — en `hueso_largo` el geométrico (0.88) supera bastante
a la CNN (0.38), al revés que en `cranio`/`mandibula_maxilar`/`vertebra`. Con `n` de 2-8 fotos por
grupo cualquiera de estos números puede moverse mucho con una sola foto más o menos; sirven para
ver la tendencia (CNN mejor en conjunto, 0.53 vs 0.33 global), no para afirmar diferencias grupo
a grupo con confianza.

### Arreglado el bug de escala del híbrido (el usuario insistió en que no podía quedar así)

Dos cambios en `src/cv_extractor/extract.py` / `src/training/{train_geometric,cnn_dataset}.py`,
sin re-extraer ni tocar los CSV de entrenamiento (los renders ya son 512×512, así que ambos
cambios son no-op ahí — solo afectan a la medición de fotos reales):

1. **`_resize_to_canonical()`**: antes de segmentar/medir nada, la imagen se reescala (conservando
   proporción, sin deformar) para que su lado mayor mida 512px — la misma resolución de los
   renders. Confirmado con el mismo script de diagnóstico de antes: el z-score máximo de las
   features geométricas sobre las 55 fotos reales bajó de **4096 a 347**.
2. **Excluir `img_width`/`img_height` del feature set**: con o sin el resize, estas dos columnas
   son EXACTAMENTE 512 en el 100% de las filas de entrenamiento (desviación típica ≈0), así que
   cualquier desviación por pequeña que sea revienta el z-score — no es una feature "ruidosa", es
   una feature rota por diseño para cualquier imagen que no sea también 512×512 exacto. Se
   añadieron a `NON_FEATURE_COLS` y se reentrenaron el geométrico y el híbrido (rápido: el
   geométrico en segundos, el híbrido ~4 min en la RTX 4080).

**Resultado en fotos reales (banco limpio de 55, mismo que arriba)**:

| Modelo | Banco limpio sintético | Antes del fix | Después del fix |
|---|---|---|---|
| Geométrico | 0.7731 (sin cambios) | 0.3273 | 0.2727 |
| CNN sola | 0.7546 (sin cambios, no usa geometría) | 0.5273 | 0.5273 (no afectada) |
| **Híbrido** | 0.8056 (antes 0.8102, ruido) | 0.2000 | **0.3455** |

El híbrido pasa de "prácticamente azar" (0.20, cerca del 0.11 de 9 clases al azar) a un
clasificador que de verdad funciona (0.35) — sigue por debajo de la CNN sola (0.53), no se ha
vuelto el mejor modelo, pero ya no está roto. El geométrico baja ligeramente (0.33→0.27): no es
una regresión causada por el fix, es varianza normal de reentrenar un RandomForest con
`random_state` fijo pero distinto número de columnas candidatas por split (las dos columnas
quitadas eran inertes en entrenamiento — constantes, ningún árbol podía partir por ellas — así
que en teoría no deberían cambiar nada, pero cambian qué índices caen en cada muestreo aleatorio
de features por nodo). Con 55 fotos de banco, una diferencia de 3 aciertos ya mueve el % varios
puntos — no se puede leer como "el geométrico empeoró de verdad".

### Cascada geométrico→CNN: aprovechar el ratio de `hueso_largo` sin arrastrar el resto

Pregunta del usuario: en vez de fusionar geométrico+CNN a ciegas (lo que falló hoy, ver arriba),
¿por qué no usar la geometría solo donde de verdad es fiable (p.ej. `hueso_largo`, muy elongado
por definición anatómica) y dejar la CNN para el resto? Está bien fundamentado en bibliografía:
la fusión por concatenación es la forma más débil de combinar modalidades (tanto en fusión
multimodal general como en imagen médica combinando radiómica+CNN); lo que mejor funciona es
fusión tardía o, como aquí, una **cascada/"reject option"** (patrón clásico desde Jacobs et al.
1991 "mixture of experts" y Chow 1970 sobre clasificación con opción de rechazo) — usar una señal
barata y fiable cuando decide con confianza, delegar en el modelo caro el resto. Y en morfometría
ósea los ratios de forma llevan décadas siendo un rasgo diagnóstico serio (osteología clásica), no
un sustituto pobre de la CNN.

**Antes de implementar nada, se puso a prueba la idea con los datos** (no solo con la teoría):
- `elongation`/`aspect_ratio` para `hueso_largo`: un stump de 1 sola variable (umbral=0.900 en
  `elongation`, ajustado SOLO con datos de entrenamiento) da 88% de accuracy binaria en
  entrenamiento y **93% sobre las 55 fotos reales, con solo 1 falso positivo en 47 fotos de otras
  categorías**. Generaliza de maravilla.
- Se probó el mismo enfoque en otras categorías con features que parecían igual de prometedoras
  en el banco limpio (`n_holes` para `sacro`/`vertebra`, `circularity`/`solidity` para
  `cranio`/`hueso_plano`, todas 88-92% de accuracy binaria EN ENTRENAMIENTO) — **todas se rompen
  en fotos reales** (16-80%, hasta 46 falsos positivos de 49 para `sacro`). Motivo: contar
  agujeros o medir circularidad exacta depende de un contorno fino y limpio; la segmentación
  sobre una foto real (sombras, textura, fondo) ensucia esos detalles finos mucho más que la
  envolvente general del objeto (lo que mide `elongation`). **`hueso_largo` es un caso especial
  por lo robusta que es esa métrica concreta, no la prueba de que "cualquier ratio geométrico
  sirve de atajo"** — cualquier regla nueva de este tipo debe validarse igual, contra fotos
  reales y mirando falsos positivos, no solo accuracy en el banco sintético.
- También se probó usar la confianza (`predict_proba`) del propio RandomForest geométrico como
  compuerta genérica en vez de una regla hecha a mano — no funcionó: la confianza del bosque
  sobre fotos reales apenas se diferencia entre aciertos y fallos (0.34 vs 0.27 de media), y en
  `sacro` está "confiadamente equivocado" (0.40 de confianza media cuando falla) — exigir
  confianza>0.5 solo cubre el 7% de las fotos. La compuerta que funciona es la regla explícita
  sobre una feature concreta y validada, no la confianza agregada del modelo multiclase.

**Implementado**: `src/inference/predict_cascade.py` (si `elongation`>0.900 → `hueso_largo`
directo, si no → `cnn_only`) + `src/training/evaluate_cascade_real_photos.py`.

| Modelo | Banco limpio (216) | Fotos reales (55) |
|---|---|---|
| Geométrico | 0.7731 | 0.27-0.33 |
| CNN sola | 0.7546 | 0.5273 |
| Híbrido (con el fix de escala) | 0.8056 | 0.3455 |
| **Cascada (elongación → `hueso_largo`, resto → CNN)** | **0.7963** | **0.6000** |

La cascada es ahora el mejor modelo medido en fotos reales (+7 puntos sobre la CNN sola, +27
sobre el geométrico) y también mejora en el banco limpio — `hueso_largo` sube de 0.38 a 0.88 de
recall sin tocar nada del resto de categorías. No se ha desplegado a producción.

### FastSAM como segmentador: mejor a ojo, peor (o igual) medido

Propuesta del usuario: en vez de seguir buscando reglas geométricas sueltas, invertir en mejorar
la SEGMENTACIÓN de base (Otsu+adaptativo, frágil con sombras/textura reales) — con un contorno
limpio, un humano distinguiría la mayoría de huesos solo por la silueta. Instalado `ultralytics`
(trae FastSAM y también YOLO-seg) y añadido `_segment_fastsam()` a `extract.py` como backend
alternativo (prompt de un punto en el centro de la imagen, sin necesidad de conocer la clase) —
`extract_features(..., backend="classic"|"fastsam")`, propagado por `predict_geometric.py`,
`predict_cnn.py`, `predict_cascade.py` y los 3 `evaluate_*_real_photos.py` (`--segmenter`).

Inspección visual de 3 fotos difíciles: las máscaras de FastSAM se ven claramente mejor que Otsu
(contornos limpios, sin el ruido de sombras que rompía `n_holes`/`circularity`). **Medido en el
banco de 55, el resultado es el contrario de lo esperado**:

| Modelo | Segmentación clásica | FastSAM |
|---|---|---|
| Geométrico | 0.2727 | 0.2364 (peor) |
| Híbrido | 0.3455 | 0.3636 (algo mejor) |
| **Cascada** | **0.6000** | 0.5455 (peor) |

Motivo (hipótesis, no confirmada a fondo): los modelos están calibrados sobre las medidas EXACTAS
que produce Otsu sobre los renders de entrenamiento (ahí Otsu es casi perfecto — fondo plano,
sin sombras) — cambiar de segmentador desplaza sutilmente bordes/extremos finos y por tanto la
distribución de features, aunque el contorno sea "mejor" en un sentido absoluto. Una máscara más
fiel no ayuda si el modelo se entrenó para esperar el sesgo particular de Otsu. Para que FastSAM
pagara de verdad haría falta re-extraer features (también de los renders, para que train y test
usen el MISMO segmentador) y reentrenar — no se ha hecho, es una inversión bastante mayor que
cambiar el backend suelto. **Se mantiene la cascada con segmentación clásica (0.60) como mejor
configuración medida** — FastSAM no se adopta por ahora.

### Pipeline de fotorrealismo con Stable Diffusion + revisor manual

Petición del usuario: en vez de una GAN entrenada desde cero (inviable, ~75 fotos reales es muy
poco dato, colapsaría), usar un modelo de difusión ya entrenado (gratis, local, sin pagar) en modo
img2img para "fotorrealizar" los renders limpios que ya tenemos por miles — mantiene la forma 3D
correcta, sustituye textura/fondo/iluminación por algo realista. Con revisión manual antes de que
cuente para nada (un modelo de imagen genérico no sabe de anatomía ósea, va a generar bastante
basura junto a resultados buenos).

- `base_datos_osea/scripts/generate_ai_variants.py`: elige N especímenes al azar por
  `bone_group` (solo de `renders/` limpios, no de los ya compuestos con ruido) y genera un
  candidato por cada uno.
- **Primer intento con `stabilityai/sd-turbo`** (1-4 pasos, sin guidance, pensado para velocidad):
  descartado tras inspección visual — con tan pocos pasos y sin CFG el resultado era casi
  idéntico al render de entrada (mismo fondo blanco liso, sin textura nueva), el prompt apenas
  tenía margen para cambiar nada estructural.
- **SD1.5 clásico** (`stable-diffusion-v1-5/stable-diffusion-v1-5`, 30 pasos, guidance=7.5,
  strength=0.5) sí transforma de verdad: fondo con textura real (mármol, madera), tinción/pátina
  en el hueso, manteniendo la silueta reconocible. Más lento por imagen (segundos en vez de
  milisegundos) pero irrelevante en lotes de decenas/cientos.
- Lote piloto: 27 candidatos (3 especímenes × 9 grupos) en `data/ai_generated_review/` (no
  versionado) + `manifest.csv`. Inspección rápida de 2: el cráneo salió muy convincente (suturas
  craneales, hasta un soporte de vitrina inventado); un fémur salió con una textura anillada poco
  natural — exactamente el tipo de caso que debe rechazarse, confirma que la revisión manual hace
  falta, no se puede aceptar todo automáticamente.
- `base_datos_osea/scripts/review_ai_generated.py`: app Streamlit (puerto 8502, distinto del 8501
  de producción) que muestra original vs. generado lado a lado con botones Aceptar/Rechazar;
  decisión se guarda al momento en `manifest.csv` (se puede cerrar y retomar). Los aceptados se
  copian a `data/ai_generated_accepted/<bone_group>/`.
- **Pendiente de la usuaria**: revisar los 27 candidatos del lote piloto vía Streamlit antes de
  generar más o de plantear usarlos en entrenamiento.

### Descriptores de Fourier del contorno (propuesta del usuario): probado, descartado — pero saca un bug real

Propuesta: en vez de seguir con reglas geométricas sueltas, usar descriptores elípticos de
Fourier (EFD) del contorno completo — técnica clásica en morfometría geométrica para distinguir
elementos óseos por silueta. Probado con `pyefd` (normalización Kuhl-Giardina correcta, no una
versión casera): **42-44% de accuracy en renders de entrenamiento, 17-19% en fotos reales** — muy
por debajo de las features ya existentes (64-71% interno). Combinarlo con las features actuales
tampoco ayudó en la prueba (54.7%, aunque sobre una muestra pequeña, no 100% comparable).

Conclusión: no es un fallo de implementación — la reconstrucción visual (`plot_efd`/
`reconstruct_contour`, ver captura generada) confirma que con 10-20 armónicos el contorno
reconstruido traza la silueta real con fidelidad cuando la segmentación es buena (cráneo, costilla,
fémur de foto real, todos con reconstrucciones muy fieles). Simplemente no aporta información que
no tuviéramos ya: `elongation`, `aspect_ratio`, momentos de Hu, etc. ya capturan "cómo es la
silueta" de otras formas. **No se sigue por esta vía.**

**Lo que sí valió la pena de la exploración**: la visualización reveló que 2 de las 8 muestras de
entrenamiento pintadas tenían la segmentación completamente rota (contorno = textura del fondo,
no el hueso) — en renders con ruido compuesto. Medido con `area_ratio` (ya calculado) como filtro
barato: **7.3% de las 3653 filas de `bone_geometric_features_combined.csv` (267 filas) tenían
`area_ratio` ≤ 0.02**, es decir, un contorno diminuto casi con toda seguridad espurio — 5.1% en
renders limpios (130/2544), 12.4% en los de ruido (137/1109), 18.0% en el banco de ruido excluido
(16/89). El banco limpio de evaluación (`bone_geometric_benchmark.csv`) salió limpio, 0%.

**Arreglado**: filtradas las 4 tablas (`bone_geometric_features.csv`,
`bone_geometric_features_noisy.csv`, `bone_geometric_benchmark.csv`,
`bone_geometric_benchmark_noisy_excluded.csv`) por `area_ratio > 0.02`, `combined.csv`
regenerado (3653→3386 filas), geométrico e híbrido reentrenados sobre los datos limpios:

| Modelo | Fotos reales, antes del filtro | Fotos reales, después |
|---|---|---|
| Geométrico | 0.27-0.33 (ruidoso entre reentrenos) | 0.29 (dentro del mismo ruido) |
| **Híbrido** | 0.3455 | **0.3818** |
| Cascada | 0.6000 | 0.6000 (no usa estas tablas — `cnn_only` + regla en vivo, sin cambios) |

El híbrido mejora de verdad (+3.6 puntos) con datos más limpios, sin cambiar nada del modelo. El
geométrico sigue en la misma banda de ruido que ya conocíamos de reentrenar un RandomForest. La
cascada, que sigue siendo el mejor resultado medido (0.60), no se ve afectada porque no depende de
estas tablas de entrenamiento.

### Pendiente
1. ~~Confirmar resultado de los 2 agentes de fotos reales y volver a correr `evaluate_real_photos.py`
   con el banco completo~~ — hecho, ver arriba. Nuevo pendiente que deja: `costilla`/`pelvis`/
   `sacro` en 0% recall en fotos reales — mismo problema de fondo que ya se sabía (pocos
   especímenes), pero ahora medido también fuera del banco fijo sintético.
2. Si se quiere reforzar "alfombra" con fotos reales de alfombra (no solo texturas de tela
   similares), buscar en Unsplash/Pexels manualmente (bloquearon la descarga automática por curl).
3. ~~Diseñar y escribir el entrenamiento de la CNN~~ — hecho. ~~Investigar y arreglar por qué el
   híbrido colapsa~~ — hecho (resize a 512px + excluir `img_width`/`img_height`, ver arriba):
   0.13→0.35 en fotos reales. Arreglo PARCIAL: solo se excluyeron las 2 columnas que eran
   constantes exactas en entrenamiento; las otras ~35 columnas en píxeles absolutos (bbox, área,
   perfil de anchura...) siguen ahí, mitigadas por el resize (z-score máximo bajó de 4096 a ~14)
   pero no eliminadas del todo — si se quiere apurar más, convertirlas a ratios (p.ej. bbox_width
   / img_width en vez de bbox_width a secas) sería el siguiente paso, sin falta de reentrenar en
   caliente ya que cambiaría otra vez el nº de columnas. Sigue pendiente decidir si merece la pena
   llevar `cnn_only` a producción (pierde la explicabilidad del PCA 2D que hoy se enseña en la UI,
   que es geométrico-only) y **redesplegar** — los `.joblib`/`.pt` en `models/` ya son las
   versiones nuevas pero el contenedor `app` sigue sirviendo las viejas hasta que se reinicie.
4. DVC sigue pendiente (decisión explícita de dejarlo para después).
5. Banco de fotos reales de `costilla` (2 fotos) y `hueso_pequeno` (3 fotos) quedó muy corto tras
   quitar las contaminadas con varios huesos — buscar fotos nuevas de costillas y huesos de
   mano/pie sueltos (no manos/pies articulados completos) para esos dos grupos.
6. Pregunta abierta del usuario (2026-07-25, sin decidir ni implementar): ¿merece la pena deformar
   las mallas 3D en el pipeline de render (estirar/comprimir de forma no uniforme, simular golpes/
   melladuras con un modificador Displace o Boolean, variar el color base del material) para
   cerrar más el hueco sim-to-real, además del ruido de fondo/luz/compresión que ya existe en
   `render_bone_noisy.py`? Viable técnicamente en Blender/bpy; no evaluado si aporta tanto como
   meter fotos reales directamente en el entrenamiento (ver conclusión de la sesión sobre qué
   harían los mejores estudios de este tipo).
7. La cascada (`predict_cascade.py`) es el mejor modelo medido hoy (0.60 en fotos reales) pero no
   se ha desplegado ni se ha decidido si sustituye al geométrico en producción. Si se retoma:
   probar si `costilla` (también elongado, ver tabla de `aspect_ratio` por grupo) se beneficia de
   una segunda regla (p.ej. curvatura, para separar "recto" de "curvado" dentro del grupo
   elongado) — no probado todavía, y con solo 2 fotos reales de costilla en el banco cualquier
   lectura ahí sería poco fiable hasta ampliarlo (ver punto 5).
8. Propuesta del usuario tras el resultado de FastSAM: re-extraer features geométricas (renders
   limpios + `renders_noisy_composited/`) usando `backend="fastsam"` en vez de Otsu, y reentrenar
   geométrico + híbrido con eso — así entrenamiento y evaluación usarían el MISMO segmentador
   (hoy el desajuste entre "Otsu en render limpio" vs. "FastSAM en foto real" es la hipótesis de
   por qué FastSAM midió peor pese a verse mejor a ojo). No implementado: requiere pasar FastSAM
   sobre ~4000 renders (factible, minutos en GPU) + reentrenar ambos modelos + re-evaluar los 3
   bancos. Es la inversión "mayor" que se mencionaba en la sección de FastSAM de arriba.
9. ~~Revisar los 27 candidatos de `data/ai_generated_review/`~~ — hecho, ver entrada 2026-07-26.
   Rechazados todos, pipeline de fotorrealismo con IA abandonado.

---

## 2026-07-24 — Ampliación grande de especímenes vía Sketchfab (Ronda 3) + banco sin naledi

**Continuación de la sesión anterior**: se retomó el pendiente de ayer (reentrenar contra MLflow
real + `evaluate.py`) y, al ver el resultado (0.43 accuracy en el banco fijo, con `sacro`/`costilla`
en 0-8% de recall), el usuario decidió ir a la raíz del problema en vez de solo medirlo: "quitar
los de naledi y buscar más huesos de sapiens/otras especies... ampliemos mucho con Sketchfab".

- **Diagnóstico previo (con el usuario)**: el 43% no era "modelo malo" sino un examen
  injusto — para `sacro`/`vertebra`/`costilla`/`hueso_pequeno` el espécimen del banco era
  *H. naledi* y había **0 especímenes naledi en entrenamiento** de esos huesos (confirmado
  contando especímenes por `bone_group`/`species` en el CSV). El banco medía "especie nunca
  vista", no "espécimen nunca visto".
- **Búsqueda de candidatos**: 4 agentes en paralelo (uno por subconjunto de grupos) buscaron en la
  Sketchfab Data API v3, verificaron cada candidato por miniatura (mismo método manual que rondas
  anteriores — ver docstring de `scripts/sketchfab.py`) y propusieron ~90 especímenes nuevos.
  Descubrieron 2 colecciones institucionales grandes sin explotar: **Andreas Kontny / Medical
  University Varna** (cubre casi todo `hueso_largo` + sacro/costilla/patella/pelvis) y **Oregon
  State University eCampus** (columna vertebral completa C1-L5 + sacros + metacarpianos/
  metatarsianos). Un agente se quedó colgado 600s y hubo que relanzarlo con timeouts más cortos.
  Se excluyeron 2 candidatos dudosos (un cráneo de bonobo que resultó ser un molde/`moulage`, y un
  maxilar que parecía un modelo didáctico retopologizado) — el resto (90 candidatos) se añadió a
  `CURATED_CANDIDATES` en `scripts/sketchfab.py`.
- **Descarga y render**: `sketchfab.py download` (16 de 90 fallaron por rate-limit 429 la primera
  vez, reintento automático los completó todos) + render en Blender (`/opt/blender/blender`, no
  está en el PATH) de las 90 mallas nuevas, 24 vistas cada una (~1 min/espécimen). Un primer
  intento del script de batch tenía un bug de índices (`cut -d/ -f2/f3` en vez de `f3/f4`, por el
  path extra de `data/meshes/`) que escribió una carpeta `renders/meshes/` corrupta — se detectó
  al momento por el nombre de carpeta raro en el log, se abortó y se borró antes de que avanzara
  más, sin pérdida de datos real.
- **Resultado en specimen-count**: hueso_largo 7→28 especímenes, hueso_pequeno 6→15, vertebra
  5→12, mandibula_maxilar 8→11, sacro 6→10, costilla 5→10, cranio 6→9, hueso_plano 5→8, pelvis 2→4.
- **Reentrenamiento (mismo banco, para medir solo el efecto de más datos)**: accuracy
  espécimen-fuera (GroupKFold) subió de **0.6285 → 0.7715**. Pero el banco fijo (todavía con
  especímenes naledi reservados) solo subió de 0.43 → 0.45 — confirmó que el naledi seguía siendo
  el cuello de botella real para sacro/costilla (0% recall persistente), no el volumen de datos.
- **Cambio de `BENCHMARK_SPECIMENS`** (en `osteolab-ml-platform/src/cv_extractor/build_dataset.py`,
  no en este repo): se sustituyó el espécimen naledi reservado en mandíbula/clavícula/sacro/
  vértebra/costilla/patella por un espécimen sapiens (ahora sobran, hay 10+ en varios grupos), y el
  naledi liberado pasa a entrenamiento — primera vez que esos huesos ven un naledi en train. Se
  eligió `metatarsal_1` (no `patella`) como reserva de `hueso_pequeno` porque patella solo tiene 2
  especímenes sapiens y perder uno para el banco lo habría dejado en 1. `pelvis` entra al banco por
  primera vez (antes excluido a propósito con solo 2 especímenes, ahora hay 4).
- **Resultado final tras el cambio de banco**: accuracy del banco fijo **0.4531 → 0.7454**. Sacro
  0%→88% recall, costilla 4-8%→54%, vertebra 17-29%→92%. Pelvis queda como el grupo más débil
  (21% recall, solo 3 especímenes en train tras reservar 1) — se sabía ya que era el más débil,
  sigue siéndolo, pero ahora al menos está medido en el banco en vez de excluido.
- **Redeploy**: `docker compose up --build -d app` (postgres/mlflow/app/ui habían quedado parados
  desde el reinicio de la máquina, se levantaron todos primero con `docker compose up -d`).

### Nota suelta: inconsistencia de nombre de especie
El fémur de gorila nuevo (`sketchfab_3fbe6a09153b4a0cb38d15005726d8ff`) quedó en
`data/meshes/gorilla_gorilla/` (slug de "Gorilla gorilla", 2 palabras) mientras el resto de mallas
de gorila del dataset usan `gorilla_gorilla_gorilla/` (trinomial, 3 palabras). No afecta al
entrenamiento (el clasificador usa `bone_group`, no la columna de especie) pero si se quiere
limpiar: renombrar la carpeta y el `species` en el CSV a `gorilla_gorilla_gorilla` para
consistencia.

### Pendiente / próximos pasos
- Comitear todo lo de esta sesión (`sketchfab.py`, `catalog/sketchfab_media.csv`, HISTORIAL.md,
  README.md de este repo; `build_dataset.py`, `evaluate.py`, los CSV regenerados en
  `osteolab-ml-platform`).
- `pelvis` sigue siendo el grupo más débil (21% recall) — sin más mallas de pelvis bilateral
  completa disponibles en Sketchfab tras 2 búsquedas, el siguiente paso ahí sería buscar en otra
  fuente (MorphoSource/Smithsonian) o aceptar el límite actual.
- Sigue pendiente la pregunta abierta de la sesión anterior (ruido/aumento de datos + CNN como
  ejercicio aparte) — no se ha tocado hoy.

---

## 2026-07-26 — Rechazado el pipeline de fotorrealismo con IA

Revisados a mano los 27 candidatos del lote piloto (`data/ai_generated_review/`, vía
`review_ai_generated.py` en Streamlit puerto 8502). **Veredicto del usuario: todos horribles.**
No fue un problema de un caso suelto (como el fémur con textura anillada ya detectado en la
sesión anterior) — el lote entero, incluido el cráneo que en su momento pareció convincente
(suturas craneales, soporte de vitrina inventado), no se sostiene a revisión completa.

**Decisión: se abandona por completo la vía de fotorrealizar renders con Stable Diffusion
img2img.** No se sigue con más lotes, ni con SD1.5 ni con ningún otro modelo de difusión para
este propósito.

**Limpieza**: eliminados `base_datos_osea/scripts/generate_ai_variants.py` y
`base_datos_osea/scripts/review_ai_generated.py` (código muerto), y los directorios locales
`data/ai_generated_review/` y `data/ai_generated_accepted/` (nunca estuvieron versionados). Se
revierten las entradas correspondientes de `.gitignore`.

Cierra el pendiente 9 de la sesión anterior. El pendiente 8 (reextraer con FastSAM + reentrenar)
sigue abierto y sin relación con esto.

---

## 2026-07-26 — Grounded-SAM: nuevo backend de segmentación, adoptado en todo el pipeline

### Inspector visual de features (`scripts/inspect_features.py`)
Para entender qué mide `extract_features()` exactamente y sobre qué superficie, se construyó un
Streamlit (puerto 8503) que muestra, para una imagen elegida a mano (foto real, render limpio o
con ruido): la superficie segmentada (máscara superpuesta), el contorno+bbox anotado, la tabla
completa de features, y el perfil de anchura. Se amplió luego para correr los tres backends de
segmentación a la vez sobre la misma imagen y compararlos lado a lado.

### Casos reales que motivaron abandonar el punto/umbral fijo
Revisando el banco de fotos reales con el inspector aparecieron varios fallos de segmentación no
detectados antes por ir directos a métricas agregadas:
- `cranio_05.jpg`: fondo gris casi del mismo tono que el hueso — con `classic` (Otsu+adaptativo)
  solo sobrevivió la cavidad nasal (alto contraste local), el resto del cráneo no se segmentó.
- `cranio_06.jpg` (momia envuelta): la textura de la envoltura tiene tanto microcontraste como el
  hueso — 224 contornos tras la apertura morfológica, el mayor solo cubre 18% del área real.
- `hueso_plano_08.jpg` (escápula, `fastsam`): la máscara nativa de FastSAM trae un corte recto de
  fábrica justo en el cuello fino de la escápula — limitación del decodificador de máscara a baja
  resolución en geometría cóncava/fina, no un bug del pipeline.
- `vertebra_05.jpg` (`fastsam`): el prompt de punto fijo en el centro de la imagen cayó dentro del
  foramen vertebral — FastSAM segmentó el agujero, no el hueso. Cualquier hueso con hueco central
  y vista centrada en él (vértebra, sacro, pelvis) es vulnerable a este fallo.
- `cranio_08.jpg`: primer plano macro de una cuenca ocular con raíces/tierra (foto de excavación in
  situ) — no es un fallo de segmentación, es una imagen no representativa del caso de uso real
  (nadie envía una foto así para identificar un hueso). Confirma que parte del ruido medido en el
  banco de fotos reales viene de la curación del banco, no solo del modelo — pendiente aparte, sin
  tocar todavía.

### Grounded-SAM (Grounding DINO + SAM por texto)
Propuesta del usuario: en vez de un punto fijo (asume hueso centrado, falla con huecos centrados),
usar detección por texto — Grounding DINO encuadra "bone" semánticamente, SAM segmenta dentro de
esa caja. Implementado en `_segment_grounded_sam` (`src/cv_extractor/extract.py`,
`IDEA-Research/grounding-dino-tiny` + `facebook/sam-vit-base`, vía `transformers`, ~0.3-0.5s/imagen
en GPU tras la carga inicial). Arregla el caso de `vertebra_05.jpg` de raíz: la caja encuadra la
vértebra entera, no un punto que pueda caer dentro del agujero.

**Dos bugs encontrados y arreglados durante la implementación:**
1. **Selección de máscara por `iou_score` en vez de por área**: SAM devuelve 3 máscaras candidatas
   (distinta granularidad) con un score que se autoasigna. En `mandibula_01.jpg` (fondo blanco puro,
   mandíbula ocupando ~90% del encuadre, caja de Grounding DINO casi tan grande como la imagen
   entera) la máscara correcta (silueta completa, 60% del área) puntuó 0.93, y dos sub-partes
   erróneas puntuaron 0.97 y 1.01 — SAM confía más en la lectura equivocada cuando la caja cubre
   casi todo el encuadre (ambigüedad conocida de SAM: no sabe si "el objeto" es la caja entera o
   una parte suya). Arreglado eligiendo la máscara de **mayor área** en vez de mayor score —
   coincide con la mejor máscara en todos los casos ya validados a ojo (vértebra, cráneo).
2. **Artefacto de borde que infla `area_ratio`**: en `hueso_plano_08.jpg`, la máscara de SAM traía
   una tira falsa de ~1px pegada al borde inferior de la imagen (artefacto de upsampling desde la
   rejilla de baja resolución) — apenas cambia el área en píxeles (52%) pero "engancha" el contorno
   externo al borde, y `cv2.contourArea` sobre ese contorno midió 98% en vez de 52%. Una apertura
   morfológica (kernel 3×3) tras obtener la máscara lo elimina sin tocar la silueta real.

### Decisión: `grounded_sam` backend por defecto en TODO el pipeline
Con los dos bugs arreglados, el usuario decidió adoptarlo como backend por defecto siempre, no solo
como opción. Cambiado el default en `extract_features()`, `predict_geometric.classify_image()`,
`predict_cnn.classify_image()`, `predict_cascade.classify_image()` y los `--segmenter` de los 3
scripts `evaluate_*_real_photos.py`. **Importante**: esto exige reentrenar geométrico e híbrido con
features re-extraídas también con `grounded_sam` — servir en producción un backend distinto del que
vio el modelo en entrenamiento es exactamente el desajuste que ya se sospechaba como causa de que
FastSAM midiera peor pese a verse mejor a ojo (pendiente 8, sesión anterior).

### Pipeline completo re-ejecutado con `grounded_sam`
1. `build_dataset.py` sobre `renders/` y `renders_noisy_composited/` — mejor tasa de contorno
   encontrado que `classic` (limpio: 2544/2760 vs 2414/2760; ruido: 1077/1220 vs 972/1220).
2. Filtro de calidad `area_ratio>0.02` (mismo criterio que el 2026-07-25): mucho menos ruido que con
   `classic` — 4.9%/3.2% de filas descartadas (limpio/ruido) frente al 5.1%/12.4% de antes. Combinado
   final: 3462 filas (antes 3386).
3. **Bug latente encontrado en `train_geometric.py`**: la generación de la tabla de referencia PCA
   (`pca_pipeline.fit_transform(X)`) usa el `X` crudo sin imputar — 8 filas se quedan sin
   `ellipse_ratio` (contorno con <5 puntos) y `PCA` no admite NaN a diferencia de
   `RandomForestClassifier` (que sí los tolera de forma nativa desde hace unas versiones de
   sklearn). Nunca se había disparado porque `classic` no producía esas filas; con `grounded_sam` sí.
   Arreglado imputando con la media de columna antes del PCA.
4. **Bug latente más serio en `train_cnn.py` (modelo híbrido)**: el mismo NaN en `ellipse_ratio`
   llega sin imputar al `StandardScaler` → al tensor de la CNN, y **colapsa el entrenamiento entero
   a `loss=nan` desde el primer batch que lo contiene** (visto en la primera corrida: accuracy de
   validación estancada en 0.19, recall 0 en 7 de 9 grupos). Arreglado con la misma imputación por
   media de columna antes de construir los datasets de train/val. Este bug es el mismo patrón que el
   colapso del híbrido documentado el 2026-07-25 (columnas en píxeles absolutos) — otra variante de
   "un valor raro en una fila cuela sin avisar y rompe todo el entrenamiento", vale la pena recordar
   este patrón para la próxima vez que el híbrido dé números sospechosamente malos.
5. MLflow/Postgres no estaban levantados (`docker compose up -d postgres mlflow` desde
   `docker/`) — sin esto, `train_geometric.py`/`train_cnn.py` se quedan colgados en silencio
   intentando conectar (sin ningún print antes, porque `mlflow.set_experiment()` va antes de cargar
   el dataset).
6. Reentrenado geométrico (`train_geometric.py`) e híbrido (`train_cnn.py --model-type hybrid`).
   `cnn_only` NO se reentrena — no depende del backend de segmentación (no llama a
   `extract_features` en ningún punto de su pipeline).

### Resultados: salto grande en fotos reales

| Modelo | Antes (classic, con filtro area_ratio) | Ahora (grounded_sam) |
|---|---|---|
| Geométrico | 0.27-0.33 | **0.5185** |
| Híbrido | 0.3818 | **0.6296** |
| Cascada (elongación+cnn_only) | 0.6000 (mejor histórico) | 0.5741 |
| Geométrico, banco sintético fijo | — | 0.7731 |

Geométrico e híbrido mejoran muchísimo — la segmentación deja de ser el cuello de botella que
enmascaraba la capacidad real de los modelos. **Pero la cascada, que era el mejor resultado medido
hasta ahora, ahora queda por DEBAJO del híbrido solo** (0.5741 vs 0.6296): la puerta de elongación
se diseñó para rescatar `hueso_largo` cuando el resto del pipeline iba mal (cnn_only solo: 0.5273
con classic) — con una segmentación tan buena, el híbrido ya no necesita ese parche, y forzar la
regla geométrica en vez de dejar decidir al modelo puede estar restando en vez de sumando. El umbral
`ELONGATION_THRESHOLD=0.900` tampoco se ha re-validado sobre la distribución de `elongation` que da
`grounded_sam` (se fijó mirando datos de `classic`).

### Limpieza del banco de fotos reales
`hueso_largo_femur_02.jpg` mostraba el MISMO fémur en dos vistas (anterior/posterior) apiladas en
una sola imagen — cualquier segmentación por objeto ve ahí dos blobs de hueso, no uno. Eliminada del
banco (`data/real_world_benchmark/`, no versionado) y su fila de `manifest.csv`.

### Pendiente
10. Con el híbrido ahora por delante de la cascada, replantear qué modelo servir en producción
    (`app/main.py` sirve hoy el geométrico simple, ni siquiera el híbrido/cascada — ver pendiente 3
    de la sesión anterior sobre esto) — y si se despliega, el contenedor `app` necesita reiniciarse
    para recoger los `.joblib`/`.pt` nuevos.
11. Re-validar `ELONGATION_THRESHOLD` (cascada) sobre la distribución de `elongation` de
    `grounded_sam`, no la de `classic` — puede que el umbral óptimo haya cambiado, o que la puerta ya
    no aporte nada por las razones de arriba.
12. `cranio_08.jpg` y similares (fotos de excavación in situ, primeros planos macro no
    representativos) siguen sin revisar/curar en el banco de fotos reales — quedó apuntado durante
    esta sesión pero no se llegó a hacer la pasada completa.

---

## 2026-07-26 (cont.) — La CNN pasa a entrenar y predecir sobre el recorte segmentado, no la foto completa

### Hallazgo: la CNN nunca había visto una imagen segmentada
Pregunta directa del usuario: "¿estamos segmentando todo?" — la respuesta, revisando el código, fue
que NO. `extract_features()` (con el backend que fuera) solo alimentaba el vector de features
geométricas (modelo geométrico, y el vector extra del híbrido); la rama de imagen de la CNN
(`cnn_dataset.py`/`predict_cnn.py`) siempre abría el fichero original tal cual y solo le aplicaba
resize+normalización — el fondo, real o sintético, llegaba intacto al backbone en el 100% de los
casos, tanto en entrenamiento como en inferencia. Esto conecta con la conversación previa de esa
misma sesión sobre domain gap (HDRIs, fondos reales, suciedad...): si el fondo nunca llega a la CNN,
gran parte de ese esfuerzo de fotorrealismo del render deja de ser necesario.

### Decisión y diseño
El usuario pidió recortar al bounding box del contorno Y pintar de gris lo de fuera de la máscara
(combinación de las dos opciones que se le plantearon), aplicado a AMBAS variantes (`cnn_only` e
`hybrid`, no solo la híbrida) tanto en entrenamiento como en evaluación/inferencia. Si la
segmentación falla en una fila de entrenamiento, esa fila se descarta (no se usa la imagen sin
segmentar como fallback — decisión explícita del usuario, evita reintroducir el problema que se
quería evitar).

Implementado en `src/cv_extractor/extract.py`:
- `_locate_bone()`: segmentación factorizada fuera de `extract_features` (antes duplicada), para
  poder reutilizarla desde una función nueva sin correr el backend dos veces sobre la misma imagen.
- `_crop_masked()` / `segment_and_crop()`: recorte al bounding box con fondo gris (128,128,128)
  fuera de la máscara.
- `extract_features(..., include_crop=True)`: añade una clave `"crop_image"` (PIL Image, no
  serializable — se hace `pop()` antes de devolver la respuesta en `predict_cnn.py` para que nunca
  llegue a un JSON) al resultado ya existente, reutilizando la MISMA pasada de segmentación en vez
  de correrla dos veces (una para features, otra para el recorte) cuando hace falta lo uno y lo otro
  (el modelo híbrido).

Nuevo `src/cv_extractor/build_segmented_crops.py`: precalcula el recorte para las 4 tablas de
features ya construidas (reutiliza que cada fila ya tiene `found=True` con `grounded_sam` por
construcción — no vuelve a decidir qué imágenes "se encuentran"), guarda los PNG en
`data/segmented_crops/` (mismo árbol relativo que `base_datos_osea/`, gitignored) y añade una
columna `segmented_path` a cada tabla. Resultado: 0 filas descartadas, 0 fallos sobre 3758 imágenes
(la segmentación es determinista, sin aleatoriedad, así que recomputar sobre las mismas imágenes que
ya se sabían "encontradas" no iba a fallar).

**Bug propio encontrado y arreglado**: la primera versión del script guardaba en `segmented_path` la
ruta SIN el prefijo `data/segmented_crops/` (apuntando por error a la ubicación original con
extensión `.png`, que nunca existe ahí — los renders con ruido son `.jpg`). Reventó el primer
intento de entrenamiento con `FileNotFoundError` en el primer batch. Los ficheros en sí se habían
guardado bien (el bug era solo en qué ruta se anotaba en el CSV); arreglado recalculando la columna
sin tener que re-segmentar las 3758 imágenes otra vez.

`src/training/cnn_dataset.py`: `BoneImageDataset` carga `segmented_path` en vez de `source_path`.
`src/inference/predict_cnn.py`: `classify_image()` llama a `extract_features(..., include_crop=True)`
SIEMPRE ahora (antes solo para `hybrid`) — `cnn_only` también pasa por segmentación por primera vez.

### Reentreno y resultados

| Modelo | Fotos reales (grounded_sam, imagen completa) | Fotos reales (grounded_sam, recorte segmentado) | Banco sintético fijo |
|---|---|---|---|
| cnn_only | — (no evaluado con grounded_sam+imagen completa, no se había reentrenado hasta ahora) | 0.5370 | 0.7685 |
| Híbrido | 0.6296 | 0.6296 | 0.8241 |
| Cascada | 0.5741 | 0.5741 | — |

A primera vista parece que el recorte segmentado "no cambió nada" en fotos reales (mismo accuracy
total en híbrido y cascada) — **es una coincidencia del total, no ausencia de efecto**: el
`classification_report` completo por grupo es distinto entre las dos corridas (ej. híbrido: costilla
0.50→0.00, cranio 0.43→0.57, sacro 0.83→0.50, mandíbula 0.57→0.71) — sube en unos grupos, baja en
otros, y la cuenta total (34/54) cuadra igual por casualidad. Con 54 fotos repartidas en 9 grupos
(2-8 por grupo), un empate exacto en el total agregado no es señal fiable de nada; hace falta un
banco de fotos reales más grande para leer esto con confianza — mismo motivo por el que se está
ampliando el banco de fotos reales para entrenamiento (ver pendientes de la sesión).

En el banco sintético SÍ hay mejora clara y consistente (cnn_only 0.7685, híbrido 0.8241,
recordando que antes de esta sesión ni siquiera se media esto con grounded_sam) — pero ese banco
tiene fondos ya bastante triviales de por sí, así que es la lectura menos informativa de las dos.

### Pendiente
13. El resultado en fotos reales del recorte segmentado es inconcluyente con el tamaño de banco
    actual (empate por coincidencia, per-grupo con ruido real) — repetir esta comparación cuando el
    banco de fotos reales de entrenamiento/evaluación sea más grande.
14. Propuesta del usuario (sin implementar, de acuerdo en el orden): probar DINOv3
    (`transformers` ya trae `DINOv3ViTModel`/`DINOv3ConvNextModel`, confirmado disponible en este
    entorno) como backbone alternativo a `resnet18` para `cnn_only`/`hybrid` — motivación: features
    autosupervisadas suelen generalizar mejor que un ResNet supervisado en ImageNet cuando hay poco
    dato real y mucho salto de dominio, que es justo la situación de este proyecto. Requiere cambiar
    `train_cnn.py` de `timm` a `transformers` para el backbone (no es solo un flag). Se decidió
    esperar a tener un número limpio del recorte segmentado antes de cambiar también el backbone, para
    no mezclar dos variables a la vez.

---

## 2026-07-26 (cont.) — Migración a BlenderProc: implementada y validada, regeneración a escala EN CURSO

### Qué se pidió y por qué
Propuesta del usuario: unir el fix de `has_real_texture()` (commit `5dbca0d`) con una migración a
BlenderProc, dándole margen para configurarlo "que aporte mucho valor" en vez de una migración 1:1.
La razón de mayor peso para migrar: hoy `extract_features()` tiene que ADIVINAR el contorno del
hueso sobre el render ya compuesto (Otsu/FastSAM/grounded_sam) — falla en huesos pequeños/lejanos de
bajo contraste (~12% de renders limpios descartados por esto, ver sección de "renders en blanco" más
arriba). BlenderProc puede exportar la máscara exacta desde la propia geometría 3D, sin adivinar nada.

### Implementado (`base_datos_osea/blender/render_bone_bproc.py`)
1. **Máscara ground-truth exacta**: con `film_transparent=True`, el canal alfa del render YA es la
   silueta real del objeto — se guarda como `<vista>_mask.png` (alpha>250, umbral que excluye la
   sombra de contacto de alpha parcial). Nueva función `extract_features_from_mask()` en
   `extract.py` (+ `_compute_all_features()` factorizado de `extract_features` para no duplicar
   lógica) mide sobre esa máscara en vez de correr un backend de segmentación — usada por el nuevo
   `src/cv_extractor/build_dataset_bproc.py`.
2. **HDRIs reales**: el downloader `blenderproc download haven` colgó/fue demasiado lento en este
   entorno (probado con timeout de 90s, sin resultado, y de 60s directamente colgado) — en vez de
   insistir, se usó la API pública de Poly Haven (`api.polyhaven.com`, misma fuente, CC0) directamente
   con `requests` para bajar 14 HDRIs 1k a mano (`base_datos_osea/hdris/`, gitignored): 6 de estudio
   fotográfico + 6 exteriores nublados + 2 de taller/workshop — variedad real de iluminación/reflejos
   en vez del `World` de color sólido/degradado de antes.
3. **Suciedad procedural en vez de descarga de texturas PBR externas**: `blenderproc download
   cc_textures` (AmbientCG) no tiene forma de acotar el descargo (todo el catálogo, sin filtro) —
   dado el tiempo disponible en esta sesión, se implementó `apply_procedural_dirt()`: ruido de
   Blender (`ShaderNodeTexNoise` + `ColorRamp`) modulando color y roughness en parches, no un tono
   plano único — sigue siendo un sustituto de las texturas PBR reales de la propuesta original, pero
   sin depender de una descarga externa no acotada. Se aplica SOLO a mallas sin
   `has_real_texture()`, exactamente igual que en `render_bone_noisy.py`.
4. **Óptica variable**: focal length 24-70mm por vista (antes FOV fijo a 40°), con la distancia de
   cámara recalculada por focal para mantener el hueso ocupando una fracción similar del encuadre.
5. **Oclusiones/objetos de laboratorio (punto 5 de la lista original)**: NO implementado — sin una
   fuente de assets 3D CC0 de "objetos de laboratorio" a mano y sin tiempo para prepararlos en esta
   sesión, tal y como se avisó de antemano que podía pasar. Queda pendiente.

### Problemas encontrados y resueltos durante la implementación
- `blenderproc run` exige `import blenderproc as bproc` como PRIMERA línea del script (antes de
  cualquier otro import, incluso el docstring del módulo) — falla con un error explícito si no.
- Los argumentos del script van después de `--` en la CLI, pero `blenderproc run` internamente ya
  los separa de los suyos (`sys.argv = sys.argv[:1] + unknown_args`) — NO queda un `--` literal que
  buscar en `sys.argv` como sí pasa invocando `blender --python ... --` a pelo.
- `bpy.ops.object.join()` para fusionar las sub-mallas del glTF en un único objeto falla con
  "context is incorrect" incluso con `bpy.context.temp_override(...)` — el contexto headless de
  blenderproc no expone lo que el operador espera. Solución: NO fusionar — operar sobre la LISTA de
  sub-objetos (bbox/centroide agregados en espacio de mundo, material aplicado a cada uno), mismo
  resultado visual sin el operador problemático.
- Bug propio: mover `.location` de TODOS los objetos (incluidos los Empty padres de la jerarquía del
  glTF) por `-centroide` desplaza a los hijos DOS VECES (una vez por la propia asignación, otra vez
  por la herencia del transform del padre ya movido) — el hueso aparecía diminuto en una esquina.
  Arreglado sin tocar ninguna `.location`: la cámara apunta directamente al centroide en espacio de
  mundo, en vez de recentrar la escena.
- `bproc.camera.set_intrinsics_from_blender_params()` no acepta `sensor_width` como parámetro (sí
  `lens`/`lens_unit`) — el sensor se fija aparte con `cam.data.sensor_width` tras la llamada.
- Nodos Empty de la jerarquía del glTF (agrupación/transform, sin malla) rompían `has_real_texture()`
  y el cómputo de bbox con `AttributeError` (`.data` es `None`) — filtrados con `type == "MESH" and
  data is not None` antes de procesar.

Validado visualmente (igual que se hizo con `render_bone_noisy.py`): un cráneo de macaca con textura
real (conserva la fotogrametría, incluida una etiqueta de catálogo manuscrita) y un fémur sin
textura de origen (suciedad procedural en parches, no un tono plano) — cada uno sigue el camino
correcto.

### Estado de la regeneración a escala: EN CURSO, no completada en esta sesión
Un primer intento de regenerar los 122 especímenes completos (mismo criterio que
`renders_noisy_composited/`, 10 vistas cada uno) resultó demasiado lento para esta sesión: ~2.5-3
min/espécimen medido en la práctica (arranque de `blenderproc run` + carga de malla + HDRI por
vista), muy por encima de la estimación inicial (~6s/vista según un test de timing con 10 vistas de
un único espécimen, que no capturaba bien el overhead de arrancar Blender fresco por invocación) —
a ese ritmo, el lote completo tardaría 5-6 horas, inviable dentro de esta sesión.

**Validación a escala reducida**: se generó un lote de 9 especímenes (uno por `bone_group`, 6 vistas
cada uno) para probar el pipeline completo extremo a extremo — `build_dataset_bproc.py` extrajo 39
filas válidas de 54 imágenes (15 descartadas sin máscara/contorno válido) usando la máscara
ground-truth. **Decisión**: no se reentrenó ni reevaluó con esta muestra de 39 filas — comparar
contra los ~1043 filas del `renders_noisy_composited/` actual no sería una lectura justa (tamaño de
muestra demasiado distinto), así que reportar un número aquí habría sido engañoso.

**Lote completo relanzado en background** (`base_datos_osea/scripts/render_noisy_batch_bproc.sh`,
proceso `nohup` desacoplado de esta sesión, sigue corriendo tras cerrarla): 122 especímenes × 6
vistas (reducido de 10 a 6 por tiempo, especímenes físicos distintos siguen siendo lo que importa,
no vistas por especímen — mismo criterio que ya se estableció el 2026-07-23). Log en
`/tmp/render_noisy_bproc.log`.

### Pendiente (para retomar cuando el lote termine)
15. Comprobar que `base_datos_osea/scripts/render_noisy_batch_bproc.sh` terminó (`grep BATCH_DONE
    /tmp/render_noisy_bproc.log`), luego:
    `python -m src.cv_extractor.build_dataset_bproc --renders-dir base_datos_osea/renders_noisy_bproc
    --out data/processed/bone_geometric_features_noisy_bproc.csv`, recombinar con
    `bone_geometric_features.csv` (limpio, sin tocar) en `bone_geometric_features_combined.csv`,
    aplicar el filtro `area_ratio>0.02` de siempre, regenerar recortes segmentados
    (`build_segmented_crops.py`), reentrenar geométrico + cnn_only + híbrido, reevaluar en los 3
    bancos y comparar contra los números ya documentados hoy (geométrico 0.52, híbrido 0.63, cascada
    0.57 en fotos reales) — todo con el mismo rigor de siempre, sin dar nada por sentado.
16. Decidir si el nuevo `renders_noisy_bproc/` SUSTITUYE a `renders_noisy_composited/` (y por tanto
    a `render_bone_noisy.py`+`render_noisy_batch.py`+`compose_noisy.py`) o convive con él — no
    decidido todavía, pendiente de ver si los números justifican el reemplazo completo.
17. Oclusiones/objetos de laboratorio (punto 5 de la propuesta original) sigue sin implementar —
    necesitaría una fuente de assets 3D CC0 que no se buscó en esta sesión.
18. `blenderproc download haven`/`download cc_textures` colgaron o fueron poco fiables en este
    entorno — si se retoma esa vía en vez de la API de Poly Haven a mano, investigar por qué antes
    de asumir que es solo lentitud de red (la API REST de Poly Haven respondió en <1s en la misma
    sesión, así que no parece ser conectividad general).
