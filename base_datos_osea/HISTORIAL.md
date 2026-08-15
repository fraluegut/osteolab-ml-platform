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
15. ~~Comprobar que el lote terminó y re-extraer/reentrenar/reevaluar~~ — hecho, ver siguiente
    sección. El lote acabó bastante antes de lo estimado (~15-20 min reales para 101/122
    especímenes, no las 5-6h proyectadas — la estimación de 2.5-3 min/espécimen venía de la muestra
    de prueba pequeña, con overhead de arranque no representativo del lote ya caliente).
16. ~~Decidir si `renders_noisy_bproc/` sustituye o convive con `renders_noisy_composited/`~~ —
    decidido: CONVIVEN, ver siguiente sección (sustituir del todo midió peor en fotos reales).
17. Oclusiones/objetos de laboratorio (punto 5 de la propuesta original) sigue sin implementar —
    necesitaría una fuente de assets 3D CC0 que no se buscó en esta sesión.
18. `blenderproc download haven`/`download cc_textures` colgaron o fueron poco fiables en este
    entorno — si se retoma esa vía en vez de la API de Poly Haven a mano, investigar por qué antes
    de asumir que es solo lentitud de red (la API REST de Poly Haven respondió en <1s en la misma
    sesión, así que no parece ser conectividad general).

### Regeneración a escala terminada: 101/122 especímenes, un bug de fuga encontrado y arreglado
El lote (`render_noisy_batch_bproc.sh`) terminó con 101/122 especímenes — los 21 restantes son
todos `homo_naledi` sin malla utilizable bajo `data/meshes/` (`SIN MALLA` en el log), consistente
con la limitación conocida desde el inicio del proyecto (esas mallas se catalogaron pero se
excluyeron del render porque Sketchfab ya cubría esos huesos en *H. sapiens*) — no es un fallo
nuevo del pipeline BlenderProc.

**Bug de fuga de especímenes encontrado antes de reentrenar**: `build_dataset_bproc.py` no
reservaba los 9 especímenes del banco de evaluación fijo (`BENCHMARK_SPECIMENS`, la misma
constante que sí respeta `build_dataset.py`) — los 9 estaban en la tabla de entrenamiento nueva.
Arreglado añadiendo la misma lógica de reserva + `--out-benchmark`
(`bone_geometric_benchmark_noisy_bproc_excluded.csv`); confirmado 0 solapamiento tras el arreglo.
De paso, otro bug menor: `train_geometric.py` y `evaluate.py` tienen cada uno su propia copia de
`NON_FEATURE_COLS` (no comparten la de `cnn_dataset.py`) y ninguna incluía la columna
`segmented_path` añadida en la sesión anterior — reventaba con `ValueError: could not convert
string to float` al intentar usarla como feature numérica. Arreglado en los dos sitios.

### Resultado: BlenderProc solo (sustituyendo) empeora en fotos reales pese a mejorar en sintético
Primera medición, sustituyendo `renders_noisy_composited/` (clásico) por `renders_noisy_bproc/`
(BlenderProc) del todo en `bone_geometric_features_combined.csv`:

| Modelo | Antes (clásico+grounded_sam) | Solo BlenderProc (sustituyendo) |
|---|---|---|
| Geométrico, espécimen fuera (sintético) | 0.6588 | **0.7172** |
| Híbrido, banco fijo (sintético) | 0.8241 | **0.8843** |
| Geométrico, fotos reales | 0.5185 | 0.5185 (igual en bruto; balanced accuracy 0.5423→0.5046) |
| cnn_only, fotos reales | 0.5370 | **0.4259** |
| Híbrido, fotos reales | 0.6296 | **0.5370** |
| Cascada, fotos reales | 0.5741 | **0.4815** |

Mejora clara y consistente en el banco sintético, pero empeora (o se queda igual) en los 4 modelos
sobre fotos reales — el patrón clásico de sobreajustar a la distribución sintética sin que se
traslade a la distribución real, justo lo que esta sesión entera intenta evitar. Hipótesis principal
(barata de comprobar, no llegó a probarse por separado): el volumen de datos "con ruido" se
desplomó al sustituir — 1043 filas del pipeline clásico a solo 381 con BlenderProc (menos
especímenes por el límite de tiempo, más el filtro `area_ratio`). Aunque las máscaras nuevas son
más exactas, hay bastante menos variedad para que la CNN generalice. Reportado al usuario sin
maquillar como si fuera una mejora — decisión explícita de no comittear esto como estado final.

### Decisión del usuario: combinar ambos ruidos en vez de sustituir
En vez de que BlenderProc reemplace al pipeline clásico, se suman las filas de los dos
(`bone_geometric_features_noisy.csv` + `bone_geometric_features_noisy_bproc.csv`) en el
`combined.csv`: 2419 limpias + 1043 ruido clásico + 381 ruido BlenderProc = 3843 filas. Reentrenados
los 3 modelos sobre esto:

| Modelo | Antes (clásico) | Solo BlenderProc | **Clásico + BlenderProc combinado** |
|---|---|---|---|
| Geométrico, fotos reales | 0.5185 | 0.5185 | 0.4815 (baja un poco) |
| cnn_only, fotos reales | 0.5370 | 0.4259 | **0.5741** (máximo de la sesión) |
| Híbrido, fotos reales | 0.6296 | 0.5370 | **0.6852** (máximo de la sesión) |
| Cascada, fotos reales | 0.5741 | 0.4815 | **0.6296** (máximo de la sesión) |

Combinar en vez de sustituir es un acierto claro para cnn_only, híbrido y cascada — recupera el
volumen que faltaba y suma las máscaras exactas donde las hay. El geométrico es el único que baja
ligeramente (0.5185→0.4815); no investigado por qué (podría ser simplemente ruido de un banco de
54 fotos, o que el modelo geométrico sea más sensible a la distribución de features exactas del
ruido nuevo) — pendiente si se quiere profundizar, pero no bloquea nada: el híbrido, que es el
mejor modelo medido con diferencia, mejora con este cambio.

**Estado final de esta sub-sesión**: `renders_noisy_bproc/` CONVIVE con `renders_noisy_composited/`
(pendiente 16 resuelto así) — no se sustituye nada del pipeline clásico, se suma.

### Giro de prioridad del usuario: fotos reales en entrenamiento pasa a ser lo fundamental
Con BlenderProc solo (sustituyendo) empeorando en fotos reales pese a mejorar en sintético, el
usuario concluyó — con razón — que es la prueba más directa de que seguir puliendo solo el lado
sintético (más domain randomization, backbone nuevo, etc.) tiene un techo. Reformulado: meter fotos
reales en el entrenamiento deja de ser "el siguiente paso cuando haya tiempo" y pasa a ser la
prioridad, por delante de cualquier otro ajuste del lado sintético.

Estado en el momento del giro: 12 fotos reales aceptadas en `data/real_photos_accepted/` (4 cranio,
1 hueso_largo, 3 pelvis, 4 vertebra — 5 de los 9 grupos en cero), 95 candidatas pendientes de
revisar en `data/real_photos_review/` (11 costilla, 14 cranio, 23 hueso_largo, 3 hueso_pequeno, 9
hueso_plano, 13 mandibula_maxilar, 4 pelvis, 6 sacro, 12 vertebra) sin curar. **El entrenamiento
sigue siendo 100% sintético a día de hoy** — las fotos reales solo se usan para evaluación, nunca
han entrado en `bone_geometric_features_combined.csv`.

### Pendiente
19. ~~Construir el pipeline de ingesta de fotos reales aceptadas hacia entrenamiento~~ — hecho, ver
    siguiente sección. Sigue pendiente curar las 95 candidatas en `data/real_photos_review/`.
20. Investigar por qué el geométrico baja ligeramente (0.5185→0.4815) al combinar ambos ruidos —
    no bloqueante (el híbrido es el mejor modelo con diferencia), pero no explicado todavía.

### Primera vez con fotos reales en entrenamiento: salto grande en los 4 modelos

Nuevo `src/cv_extractor/build_dataset_real_photos.py`: mismo formato de fila que
`build_dataset.py`/`build_dataset_bproc.py` (para poder concatenar directo en `combined.csv`), pero
a partir de `data/real_photos_accepted/manifest.csv` en vez de renders — especie="real",
hueso=`bone_group` (no hay hueso canónico fino en fotos ya clasificadas por grupo), especimen=nombre
de fichero. Las 12 fotos aceptadas hasta ahora (4 cranio, 1 hueso_largo, 3 pelvis, 4 vertebra)
pasaron todas la segmentación (`found=True`, ninguna descartada por `area_ratio`).

`combined.csv` pasa a ser: 2419 limpio + 1043 ruido clásico + 381 ruido BlenderProc + **12 fotos
reales** = 3855 filas — la primera vez en todo el proyecto que el entrenamiento deja de ser 100%
sintético. Reentrenados los 3 modelos, reevaluados en fotos reales:

| Modelo | Sin fotos reales (solo sintético) | Con 12 fotos reales |
|---|---|---|
| Geométrico | 0.5185 | **0.5926** |
| cnn_only | 0.5741 | **0.6667** |
| Híbrido | 0.6852 | **0.7593** |
| Cascada | 0.6296 | **0.7222** |

Sube en los 4 modelos, de forma consistente — no es ruido de un banco pequeño (el patrón está en
los 4 a la vez, no es una casualidad de uno solo, a diferencia de los empates puntuales de sesiones
anteriores). **12 fotos reales (0.3% del dataset combinado) mueven más el resultado que todo lo
demás hecho en esta sesión junto** (Grounded-SAM, recorte segmentado, BlenderProc). Confirma de
forma contundente la intuición del usuario: el domain gap sim-to-real no se cierra puliendo más el
lado sintético, se cierra con datos reales de verdad, aunque sean pocos.

Híbrido en 0.7593 es, con diferencia, el mejor resultado medido en todo el proyecto hasta ahora
(el anterior mejor era la cascada a 0.60, hace dos sesiones).

### Pendiente
21. Curar las 95 fotos candidatas en `data/real_photos_review/` (11 costilla, 14 cranio, 23
    hueso_largo, 3 hueso_pequeno, 9 hueso_plano, 13 mandibula_maxilar, 4 pelvis, 6 sacro, 12
    vertebra) — con la señal de que 12 fotos ya mueven tanto, aceptar aunque sea una fracción de
    las 95 debería seguir sumando. Máxima prioridad de la siguiente sesión.
22. Con `data/real_photos_accepted/` ganando peso, plantearse separar una parte como banco de
    evaluación adicional (no solo entrenamiento) para no depender únicamente de las 54 fotos
    congeladas de `data/real_world_benchmark/` — no decidido todavía, el volumen actual (12) es
    demasiado pequeño para plantearlo en serio.

---

## 2026-08-03 — Investigación: por qué el híbrido empeoró tras sumar las 37 fotos + fuga train/val/test

Petición del usuario: entender por qué el híbrido empeoró en el commit `c29bb88` (0.772→0.702 en
el banco real de 57 fotos) y auditar si en algún punto se mezclan train/validación/test.

### Causa del empeoramiento del híbrido
Confirmado que ya estaba explicado en el propio mensaje del commit `c29bb88`: al sumar las 37 fotos
reales curadas, geométrico y cnn_only mejoran, pero el híbrido empeora — `mandibula_maxilar` y
`sacro` pierden recall **pese a no haber recibido ninguna foto nueva** (mandibula_maxilar tiene 0
fotos reales en todo `real_photos_accepted/`; sacro tiene 8). Confirmado además que la cabeza de
clasificación es compartida entre las 9 clases y solo se afinan 15 epochs — la señal apunta a que
el fine-tuning corto descalibra la cabeza globalmente cuando el resto de clases reciben datos más
heterogéneos, no a un problema de datos específico de esas 2 clases. Decisión del usuario: no
revertir (los modelos no están versionados en git).

### Auditoría de fuga train/val/test — sin fuga real
- `train_cnn.py` reparte por espécimen (`GroupShuffleSplit`) con `RuntimeError` explícito si hay
  overlap train/val. `train_geometric.py` usa `GroupKFold` con el mismo criterio (aviso, no
  excepción, para el caso borde de una clase con un único espécimen). Ambos guards activos y
  correctos.
- Cruce `combined.csv` (entrenamiento) × `bone_geometric_benchmark.csv` (banco sintético limpio) ×
  `real_world_benchmark/` (63 fotos reales, el "examen honesto") × `real_photos_accepted` (las 49
  fotos que SÍ entraron a entrenamiento): 1 solo espécimen compartido entre `combined` y el banco
  sintético (`nmnhvz_7049315`), pero contribuye `bone_group` distinto en cada lado (mandíbula vs
  cráneo del mismo animal) — silueta distinta, no es fuga real. Los nombres de archivo de
  `real_world_benchmark/` y `real_photos_accepted/` coinciden por convención de numeración
  (`cranio_01.jpg`, etc.) pero son fotos distintas — verificado por hash y resolución, falsa alarma.
- **Conclusión: no hay fuga.** La caída del híbrido es un problema de selección/calibración de
  checkpoint, no de contaminación de datos.

### Decisión y cambio: seleccionar el checkpoint por val_acc en fotos reales, no por val_acc sintético
`train_cnn.py` guardaba el mejor checkpoint por `val_acc` sobre el split sintético completo — nunca
miraba cómo se comportaba en foto real durante el propio entrenamiento. Como `real_photos_accepted`
ya cae parcialmente en val por el split de espécimen (12 de 49 fotos reales, en 5 de los 9 grupos,
con random_state=42), se usa ese subconjunto como criterio de selección cuando existe (fallback a
val_acc sintético si el split lo deja vacío). Implementado en `train_cnn.py`:
`real_val_df`/`val_acc_real_photos` por epoch, nuevo campo `checkpoint_selection` en el meta.json,
métricas nuevas en MLflow (`val_acc_real_photos`, `best_selection_metric`,
`best_val_acc_real_photos`). Aviso explícito en el propio commit/código: es un proxy pequeño y
ruidoso (12 filas), no sustituye la evaluación posterior contra el banco de 63 fotos de
`evaluate_cnn_real_photos.py`.

### Primer intento fallido: seleccionar solo por val_acc_real_photos regala una clase entera
Primer reentreno del híbrido con el criterio nuevo: `val_acc_real_photos` sube de 0.833 a 0.917 en
la última epoch, pero `mandibula_maxilar` cae a **recall 0.000** en el val sintético completo (las
demás 8 clases se mantienen razonables, cranio la más baja con 0.538) y el banco sintético limpio
baja de 0.889 a 0.861. Causa: `mandibula_maxilar` no tiene NINGUNA foto real en el proxy de 12 filas
(0 de 49), así que maximizar ese proxy a ciegas no la protege — puede sacrificarla entera sin que la
métrica de selección se entere. Confirma el aviso que ya llevaba el propio cambio ("proxy pequeño y
ruidoso").

**Guardarraíl añadido**: no aceptar como "mejor" ninguna epoch que deje alguna clase en recall 0 en
el val sintético (que sí cubre casi todas las clases), calculado gratis a partir de los mismos
preds/labels que ya produce `run_epoch` para el val. Con fallback al criterio antiguo (max val_acc
sintético) si ninguna epoch lo cumple.

**Segundo bug, encontrado al probar el guardarraíl**: con `random_state=42` el split de espécimen
deja a `mandibula_maxilar` con **0 filas en val** (11 especímenes en total, ninguno cayó en el 20%
de val) — `recall_score(labels=range(num_classes))` trata esa clase ausente como recall=0 con
`zero_division=0`, así que el guardarraíl se disparaba en TODAS las epochs (clase fantasma) y caía
siempre al fallback, sin filtrar nada de verdad. Corregido: `recall_score` limitado a
`labels=np.unique(val_labels)` (solo clases con al menos 1 fila real en val). Nota aparte, más de
fondo: como `mandibula_maxilar` no aparece nunca en val con este split, el propio `val_acc` global
tampoco tiene NUNCA señal de esa clase durante el entrenamiento — ni el criterio viejo ni el nuevo
la vigilan de verdad; solo el banco sintético externo (`bone_geometric_benchmark.csv`) y el banco
real la evalúan. Especímenes por grupo en `combined.csv`: hueso_plano 9, **mandibula_maxilar 11**,
cranio 15, hueso_pequeno 15, pelvis 17, costilla 18, sacro 18, vertebra 25, hueso_largo 34 — con tan
pocos especímenes y un split no estratificado por clase, quedarse sin representación en val es
plausible para las clases pequeñas (no solo mandibula_maxilar; pelvis con 2 filas y vertebra con 4
en este split concreto también son casi tan frágiles).

### Propuesta 2 implementada: `lr_head` más bajo en fase 2
Nuevo flag `--lr-head-finetune` (por defecto `lr_head/10`, es decir `1e-4` con los defaults
actuales) — en fase 1 la cabeza parte de cero y necesita `lr_head` completo para moverse rápido,
pero en fase 2 ya está entrenada y competir a la misma LR que el backbone recién descongelado
parecía ser parte de por qué se descalibraba con datos heterogéneos nuevos. `checkpoint_selection`
en el meta.json ahora refleja cuál de los 3 caminos se usó (real_photos / fallback sintético).

### Contra el examen real (banco de 63 fotos): el proxy de 12 fotos empeora, no mejora
Con el guardarraíl corregido + desempate por val_acc arreglado, se corrió por fin
`evaluate_cnn_real_photos.py --model-type hybrid` (el examen de verdad, no el val sintético) contra
el checkpoint elegido por `val_acc_real_photos`: **0.6491 accuracy** (balanced 0.5747) — peor que el
0.702 ya regresado del commit `c29bb88` y muy por debajo del 0.772 original. `mandibula_maxilar`
mejora bastante (recall 0.62, confirma que el 0.000 que salía en el val sintético era el artefacto
de la clase sin filas en val, no una regresión real) pero `costilla` y `hueso_pequeno` caen a
recall 0.00 — regresiones nuevas que no estaban antes.

Conclusión: el checkpoint elegido por el proxy de 12 fotos resultó ser de fase 1 (época 3, backbone
TODAVÍA CONGELADO) — con solo 12 filas, `val_acc_real_photos` empata constantemente (10/12, 11/12) y
el desempate por val_acc sintético más alto no protege de elegir una época demasiado temprana, donde
el backbone solo tiene features genéricas de ImageNet sin adaptar al dominio de huesos. Tres
iteraciones seguidas de este enfoque (versión rota, con guardarraíl, con desempate) dieron las 3 un
resultado peor que el baseline en el banco real — **revertido**: la selección de checkpoint vuelve a
ser por val_acc sintético (620 filas, el criterio que sí dio 0.772 originalmente) + el guardarraíl
anti clase-cero se mantiene como red de seguridad adicional (no hace daño, y sigue calculando/
logueando `val_acc_real_photos` por epoch solo para tenerlo a la vista). Lección: un proxy de 12
imágenes no tiene suficiente señal para pilotar la selección de checkpoint por sí solo — haría falta
uno bastante más grande (o el propio banco de 63, con el coste de correr Grounding DINO+SAM cada
epoch) para que esta idea funcione.

### Propuesta 2 (lr_head_finetune) probada aislada: mismo 0.6491 que el intento fallido
Reentreno con el criterio de selección YA revertido a val_acc sintético (el bueno) + guardarraíl +
`lr_head_finetune` aislado: esta vez el checkpoint elegido es de fase 2, época 9 (backbone
totalmente afinado, val_acc sintético 0.7484 — el pico real de toda la corrida, banco limpio
0.8426, mucho más sano que las corridas anteriores). Pero contra el banco real de 63 fotos:
**0.6491 accuracy** — el mismo número exacto que el intento fallido anterior (que había elegido un
checkpoint completamente distinto, de fase 1 con el backbone aún congelado), con una matriz de
confusión casi idéntica (`costilla` sigue en recall 0, mismos patrones de confusión).

**Hallazgo real, más importante que cualquiera de los cambios probados hoy**: `train_cnn.py` NUNCA
fijaba semilla aleatoria — `--random-state` solo controla el split train/val
(`GroupShuffleSplit`), pero la inicialización de la cabeza (`nn.Linear` con pesos aleatorios) y el
orden/augmentation de los batches (`DataLoader(shuffle=True)`, sin `torch.manual_seed`) no tenían
ninguna semilla. Que dos checkpoints tan distintos den el mismo accuracy con la misma matriz de
confusión es la prueba de que la varianza entre corridas domina por encima de cualquier cambio que
se estuviera probando (selección de checkpoint, `lr_head_finetune`) — los números 0.649/0.649/0.702
/0.772 vistos hoy pueden ser en gran parte ruido de inicialización, no señal real de qué cambio
ayuda o no. **Corregido**: `torch.manual_seed(args.random_state)` +
`np.random.seed(args.random_state)` al principio de `main()`.

Conclusión honesta de toda la sesión: no se puede afirmar con una sola corrida por configuración si
`lr_head_finetune` ayuda o no — hace falta repetir cada config con varias semillas y comparar
medias, ahora que el entrenamiento sí es reproducible corrida a corrida. No hecho todavía (coste de
tiempo/cómputo, pendiente decidir con el usuario si se invierte en ello).

### Pendiente
23. ~~Confirmar que el guardarraíl + selección por fotos reales mejora el banco real~~ — probado y
    revertido: empeora (0.649 vs 0.702 regresado vs 0.772 original). Guardarraíl anti clase-cero
    conservado como red de seguridad; selección vuelve a ser por val_acc sintético.
24. ~~Bajar `lr_head` en fase 2~~ — implementado (`--lr-head-finetune`, default `lr_head/10`), un
    solo run dio 0.6491 en el banco real (igual que el intento fallido con checkpoint distinto) —
    no concluyente por el problema de semilla del punto 27.
25. Plantearse un split estratificado por clase (no solo por espécimen) para `train_cnn.py` —
    con clases de 9-11 especímenes, un `GroupShuffleSplit` puro puede dejarlas sin ninguna fila en
    val por pura mala suerte del `random_state` (pasó con `mandibula_maxilar` en este `random_state`
    42), y entonces ni el criterio de selección ni las métricas de val ven nada de esa clase durante
    todo el entrenamiento. No implementado.
26. Si se quiere retomar la idea del proxy de fotos reales para seleccionar checkpoint, hacerlo con
    bastantes más de 12 imágenes (o correr el banco de 63 fotos real como criterio, no solo como
    examen final) y restringir la búsqueda a épocas de fase 2 (backbone ya descongelado) para no
    poder elegir por accidente una época de fase 1 todavía sin adaptar. No implementado.
27. ~~Fijar semilla aleatoria en `train_cnn.py`~~ — hecho: nuevo flag `--model-seed` (separado de
    `--random-state`, que sigue controlando solo el split) para poder repetir el mismo split con
    distinta inicialización.

### Experimento final: 3 semillas de modelo, split fijo — `lr_head_finetune` bajo PIERDE, revertido

Con la semilla ya separable, se corrieron 6 entrenamientos (`baseline`: `lr_head_finetune=lr_head`,
comportamiento de siempre · `fix`: `lr_head_finetune=lr_head/10`, la propuesta 2) × 3
`--model-seed` (42/43/44, split `random_state=42` fijo en los 6), cada uno evaluado contra el banco
real de 63 fotos:

| model_seed | baseline (real_acc) | fix (real_acc) |
|---|---|---|
| 42 | **0.7018** | 0.6842 |
| 43 | 0.6667 | 0.6491 |
| 44 | 0.6842 | 0.6491 |
| **media** | **0.684** | 0.661 |

`baseline` gana en las 3 semillas — señal consistente, no ruido de una sola corrida. La hipótesis de
la propuesta 2 (bajar la LR de la cabeza en fase 2 para que no se descalibre) resultó equivocada:
**empeora** el accuracy real en promedio ~2.3 puntos. Revertido: `--lr-head-finetune` por defecto
vuelve a ser `lr_head` (el flag sigue existiendo, documentado con este resultado, por si se quiere
reabrir la comparación con otro dataset/backbone). El guardarraíl anti clase-cero se mantiene (no
participó en la diferencia, es red de seguridad neutral).

Dato adicional útil: `baseline` con `model_seed=42` da 0.7018 — prácticamente igual al 0.702 ya
documentado en el commit `c29bb88` tras sumar las 37 fotos reales. Confirma que el estado actual del
dataset (con las 37 fotos) efectivamente asienta el híbrido en ~0.68-0.70 en el banco real, no en el
0.772 de antes de esas fotos — ese descenso es real y viene de los datos/el fine-tuning corto, no de
ningún bug de esta sesión. Modelo final en disco: reentrenado con `baseline` + `model_seed=42`
(la mejor combinación medida) para dejar el mejor checkpoint conocido guardado.

### Pendiente
28. El hueco 0.772 (antes de las 37 fotos) → ~0.68-0.70 (media actual) sigue sin explicación
    completa más allá de "cabeza compartida se descalibra con fine-tuning corto y datos más
    heterogéneos" (hipótesis original del commit `c29bb88`, no refutada pero tampoco un fix
    encontrado todavía). Ideas no probadas: más epochs de fine-tuning con early stopping por banco
    real completo (caro, requiere correr Grounding DINO+SAM cada pocas epochs), o class-balanced
    sampling (ver propuesta 3 de la sesión, no implementada).
29. Split estratificado por clase (pendiente #25) y ampliar el proxy de fotos reales en val más allá
    de 12 filas (pendiente #26) siguen abiertos, sin tocar en esta sesión.

### Segunda vuelta a "seleccionar por fotos reales": banco EXTERNO precomputado, no proxy en val

Pendiente #26 implementado: en vez del proxy de 12 filas dentro de val, `train_cnn.py` ahora
segmenta UNA VEZ (Grounding DINO+SAM, antes de entrenar — la segmentación no depende de los pesos)
el banco externo completo de `data/real_world_benchmark/` (57-63 fotos, el mismo que usa
`evaluate_cnn_real_photos.py`) y cachea los tensores; cada epoch solo hace un forward pass barato
sobre ellos (`eval_real_bench`, sin resegmentar). Se usa como criterio de selección principal
(`real_bench_acc`), con desempate por val_acc sintético en empates y el guardarraíl anti clase-cero
de siempre como red de seguridad adicional.

Resultado (`model_seed=42`): checkpoint elegido en fase 2 época 3, **real_bench_acc=0.7018** en el
banco real — confirmado corriendo `evaluate_cnn_real_photos.py` después (mismo 0.7018, balanced
0.7044). Por primera vez en toda la sesión NINGUNA clase queda en recall 0 en el examen real
(mandibula_maxilar 0.50, sacro 1.00, macro avg 0.69) — más sano que cualquier intento anterior con
el proxy de 12 filas o con `lr_head_finetune`.

**Aviso metodológico importante, sin resolver todavía**: al seleccionar el checkpoint maximizando
`real_bench_acc` sobre EXACTAMENTE el mismo banco de 63 fotos que luego usa
`evaluate_cnn_real_photos.py` para "examinar", el número deja de ser una medida honesta de
generalización — es circular (el banco de examen se usó también para elegir el modelo). El 0.7018
puede reflejar en parte sobreajuste a esas 63 fotos concretas, no mejora real frente a fotos nuevas
que el modelo no ha visto ni de refilón durante la selección.

### Pendiente #30 resuelto: partido el banco, medido honesto — la vía se descarta

`manifest.csv` de `data/real_world_benchmark/` ganó una columna `split` (`selection`/`held_out`,
random_state=42, 1 foto por `bone_group` reservada como `held_out` — con `costilla` en 3 y
`hueso_pequeno` en 4 fotos totales, un 10%/90% puro habría dejado alguna clase en 0 en el held_out,
igual que pasó con `mandibula_maxilar` en el val sintético). 54 filas `selection`, 9 `held_out`.
`train_cnn.py::precompute_real_bench` ahora solo usa `selection`; `evaluate_cnn_real_photos.py`
gana `--split {all,selection,held_out}` para poder examinar solo con lo nunca visto.

Reentrenado con `model_seed=42` seleccionando por `real_bench_acc` sobre `selection` (48 imágenes
tras descartar 6 sin segmentar): checkpoint elegido con real_bench_acc(selection)=0.75. Medido
honesto:

| split | accuracy | balanced_accuracy | n |
|---|---|---|---|
| `selection` (optimista, es lo que decidió el checkpoint) | 0.75 | 0.71 | 48 |
| `held_out` (honesto, nunca visto) | **0.4444** | 0.4444 | 9 |

**Conclusión**: el 0.70-0.75 que parecía la mejor cifra de toda la sesión era en gran parte
sobreajuste a las 63 fotos del banco (usadas para elegir el checkpoint) — medido de forma honesta
contra las 9 fotos held_out, da 0.4444, peor que el 0.684-0.702 del criterio de val_acc sintético
documentado con 3 semillas. Con solo 9 fotos held_out el margen de error es enorme (cada foto vale
~11 puntos), así que tampoco se puede afirmar con certeza que esta vía sea peor en el límite — pero
no hay ninguna evidencia de que ayude, y sí una señal honesta preocupante. **Descartada como
default**: nuevo flag `--select-by-real-bench` (default `False`) para poder reabrir esta vía a
propósito si el banco real crece lo bastante como para tener un held_out con más de 1 foto/clase;
`real_bench_acc` se sigue calculando y logueando siempre (banco precomputado una vez, gratis) para
tenerlo a la vista sin que decida. Selección de checkpoint vuelve a ser val_acc sintético +
guardarraíl anti clase-cero — el único criterio validado con evidencia sólida esta sesión. Modelo
final en disco: reentrenado con esta configuración + `model_seed=42`.

### Pendiente
31. Si se quiere reabrir `--select-by-real-bench`, primero ampliar `data/real_world_benchmark/`
    (más fotos por grupo, especialmente costilla=3 y hueso_pequeno=4) para que el held_out pueda
    tener más de 1 foto/clase y el número honesto deje de tener tanto margen de error.

### Lote de 21 fotos de mandíbula pegadas por el usuario: 0 aceptadas

El usuario pegó 21 fotos buscadas en `Fotos entrenamiento/mandibula/` para cerrar el hueco de
`mandibula_maxilar` (0 fotos reales de entrenamiento). Revisadas una a una: **0 aceptadas**.
Desglose — especie incorrecta ~17 (mandíbulas de perro, lobo, gato/mustélido y varios
ungulados tipo ciervo/cabra/vaca, algunas con nombre de fichero ya delatándolo:
`dog-jaw-or-wolf-jaw...webp`, `animal-jawbone-teeth...webp`), ilustraciones/diagramas con
etiquetas quemadas en la imagen ~3 (una CGI dental, dos diagramas anatómicos con flechas/texto),
figura de artículo científico con anotaciones ~1 (`vf-223-article-3-figure-3b-eng.png`, mismo
problema de copyright que la figura de learnmuscles.com rechazada el 2026-08-02).

Guardadas las 21 crudas en `base_datos_osea/Fotos entrenamiento/mandibula/` (gitignored, igual que
el resto de esa carpeta — provenance de lo revisado, se acepte o no). `manifest.csv` y
`data/real_photos_accepted/` sin tocar — nada que meter a entrenamiento de este lote.

**Causa probable**: buscar "mandibula" (sin más contexto) devuelve sobre todo resultados
veterinarios/zoológicos — a diferencia de "cráneo humano" o términos con "human"/"osteológico"
explícito, que en las tandas anteriores dieron mucho mejor ratio de aceptación (fuente
predominante: Wikimedia Commons, ver `source_url` en el resto del manifest).

### Pendiente
32. ~~`mandibula_maxilar` sigue en 0 fotos reales de entrenamiento~~ — corregido el criterio de
    curación (ver más abajo), 5 fotos aceptadas de ese mismo lote.

### Corrección de criterio: especie NO descalifica, solo el grupo anatómico importa

El usuario frenó la revisión anterior: rechazar por "especie incorrecta" (perro/lobo/ciervo) era un
error — el objetivo es pasar el filtro por grupo anatómico (¿se reconoce la forma de mandíbula/
cráneo/etc.?), la especie es un problema aparte que se resuelve después, no bloqueante aquí. No es
un cambio de código (nunca hubo filtro de especie en `extract_features`/`build_dataset_real_photos.py`,
solo era juicio manual de curación) — corregido también en memoria persistente para no repetirlo.

Re-revisadas las 21 fotos de mandíbula con el criterio correcto: **5 aceptadas** (de 0 antes).
Motivos de rechazo de las 16 restantes, todos independientes de especie: ilustración/diagrama/CGI
(4), marca de agua o stock sin licencia clara (3), figura de paper con anotaciones superpuestas —
flechas/números/letras en post-producción (5), especímenes solapados en una imagen (1), ambigüedad/
imposible identificar un hueso único con certeza (1), figura de artículo científico con copyright
explícito (1), borderline con anotación mínima descartado por consistencia (1).

Aceptadas (`data/real_photos_accepted/mandibula_maxilar/mandibula_01..05.jpg`) — 3 de las 5 son de
especie NO primate (perro/lobo, y dos probables carnívoros pequeños), aceptadas a propósito bajo el
criterio nuevo. `bone_geometric_features_real_photos.csv` 49→54 filas, `combined.csv` 3892→3897.

Reentrenado el híbrido (criterio validado: val_acc sintético + guardarraíl, model_seed=42):
- Banco sintético limpio: 0.8704 (sube).
- `mandibula_maxilar` sigue en 0.000 en la tabla de val sintético — mismo artefacto de siempre (el
  split de espécimen mandó las 5 fotos nuevas a train, 0 a val; no significa que el modelo falle,
  solo que no hay medición ahí).
- **Examen honesto (`--split held_out`, 9 fotos nunca vistas)**: accuracy 0.6667 (antes 0.4444 con
  el mismo held_out) — `mandibula_maxilar` acierta 1/1 en el held_out. Mejora, pero n=9 sigue
  siendo demasiado pequeño para afirmar nada con confianza (~11 puntos de margen por foto).

### Pendiente
33. Los lotes huérfanos de fotos ya descargadas pero nunca fusionadas (`manifest_hueso_largo.csv`
    23 filas, `manifest_hueso_plano.csv` 3, `manifest_sacro.csv` 4, `manifest_vertebra.csv` 1 —
    ver pendiente de la sección anterior) necesitan revisión con el criterio corregido (algunas
    notas ya dicen "escápula de ciervo/perro", que ahora no descalifica) antes de fusionar. Uno de
    esos CSV tiene además una fila mal formada (columna con coma sin comillas) a arreglar al pasar.

### Segunda corrección de criterio: tampoco rechazo por ilustración/marca de agua/anotación — decide el usuario

Tras aceptar 5 de 21 (tras corregir el criterio de especie), el usuario frenó otra vez: seguía
rechazando por mi cuenta (ilustración/CGI, marca de agua, anotaciones de figura de paper). Corregido
de raíz: **no rechazo nada** — el usuario decide qué entra, asume el permiso/criterio de cada
imagen. Mi única tarea activa es dividir imágenes con más de un hueso físico en recortes separados
(hay precedente ya en el manifest: `costilla_104/105`, recortes de una foto con 3 costillas).
Confirmado además que ya había precedente de aceptar cosas "no ortodoxas": `cranio_101.jpg` es una
RÉPLICA (cast Bone Clones) de *Australopithecus afarensis* — ni hueso real ni Homo sapiens, y está
aceptada desde antes. Guardado en memoria persistente para no repetir el error una tercera vez.

Re-revisado el lote completo de 21 con este criterio: las 8 ya aceptadas se mantienen, más 15
adicionales (10 imágenes de un solo hueso sin más — incluye el CGI dental, los 2 diagramas
anatómicos con texto, la de marca de agua visible, las 2 con puntos numerados de medición, la
completa vista inferior con arco dental etiquetado, la figura de artículo con flechas — y 2 imágenes
con dos mandíbulas apiladas divididas en 4 recortes, `images (5).jpg`→`mandibula_20/21`,
`images (15).jpg`→`mandibula_22/23`). `images (10).jpg` resultó ser 2 VISTAS del mismo espécimen
(oclusal + lateral), no 2 huesos — no se divide. Total: **23 fotos de mandibula_maxilar** aceptadas
de las 21 originales (2 se convirtieron en 4 recortes). `bone_geometric_features_real_photos.csv`
49→72 filas (todas segmentaron bien, 0 sin segmentar), `combined.csv` 3892→3915.

Reentrenado el híbrido (`model_seed=42`, criterio de selección val_acc sintético + guardarraíl):
- `mandibula_maxilar` recall **0.807** en val sintético — señal real esta vez (37 especímenes en
  val, antes 33; con solo 5 fotos previas ninguna había caído en val por azar del split).
- `sacro` cae a recall 0.000 — pero es 1 solo espécimen en val (fallo puntual, no patrón amplio;
  mismo problema estructural del split no estratificado ya documentado en el pendiente #25, no una
  regresión nueva). El guardarraíl detectó correctamente que ninguna epoch tenía todas las clases
  con recall>0 y cayó al fallback (max val_acc), como está diseñado.
- Banco sintético limpio: **0.8981** (sube).
- **Examen honesto (`--split held_out`, 9 fotos nunca vistas): 0.7778** — progresión de la sesión:
  0.4444 (antes de tocar mandíbula) → 0.6667 (5 fotos) → 0.7778 (23 fotos). `mandibula_maxilar`
  sigue acertando 1/1. Ya cerca del 0.772 original de antes de la regresión de las 37 fotos —
  aunque n=9 sigue siendo pequeño, la tendencia con cada ronda de fotos reales añadidas es
  consistentemente positiva, coherente con el patrón de toda la sesión (más fotos reales, no más
  algoritmo, es lo que mueve el número).

### Pendiente
34. ~~Confirmar la mejora con más fotos reales en otros grupos débiles~~ — hecho a lo grande, ver
    más abajo.
35. Aplicar el criterio corregido (sin rechazo unilateral) a los lotes huérfanos del pendiente #33
    cuando se retomen.

### Barrido completo: 138 fotos nuevas en 5 grupos a la vez, delegado en 5 forks en paralelo

El usuario dejó las mismas carpetas crudas de siempre (`Fotos entrenamiento/costilla|craneo|pelvis|
sacro|vertebra`, más "nuevos 1" con PDFs de papers de referencia, sin fotos) y pidió aplicar el
criterio ya corregido (sin rechazo unilateral, solo dividir imágenes con más de un hueso) a TODO,
no solo a mandíbula. Confirmado por hash MD5 que las 5 carpetas son bit a bit idénticas a las que ya
estaban en `base_datos_osea/Fotos entrenamiento/` de la sesión anterior — no había fotos nuevas en
sí, pero sí fotos que en su momento se rechazaron con el criterio viejo (especie/ilustración/marca
de agua/anotación) y que ahora tocaba reconsiderar.

Delegado en 5 forks en paralelo (uno por grupo, mismo protocolo que mandíbula: comparar contra lo ya
aceptado para no duplicar, copiar/recortar el resto, anotar en `manifest.csv`). Resultado:

| Grupo | Antes | Después | Nuevas | Notas |
|---|---|---|---|---|
| cranio | 5 | 72 | +67 | 4 collages divididos en recortes individuales (uno de ~22 cráneos pequeños en baja resolución, calidad de recorte floja — señalado) |
| vertebra | 13 | 47 | +34 | 31 recortes de 6 imágenes multi-espécimen (grid learnmuscles.com, figuras de paper) + 3 sin recortar |
| sacro | 8 (+4 huérfanas) | 26 | +18 | Encontrados `sacro_001-004` en disco sin fila en manifest (huérfanos de antes, no tocados, quedan invisibles para el pipeline) |
| pelvis | 14 | 23 | +9 | 13 duplicados exactos descartados (mismas fotos Bone Clones ya aceptadas) |
| costilla | 8 | 18 | +10 | 6 imágenes con 7-24 costillas apiladas MUY densas dejadas sin recortar (no separables limpio de forma automática) |
| mandibula_maxilar | — | 23 | (ya hecho antes en la sesión) | |

Manifest.csv: 210 filas totales, sin duplicados de `(bone_group, filename)`, todos los ficheros
verificados en disco. `bone_geometric_features_real_photos.csv` regenerado: 210/210 segmentaron bien
(0 descartadas). `combined.csv`: 3915→4053 filas.

Reentrenado el híbrido (`model_seed=42`, criterio de selección val_acc sintético + guardarraíl):
- **Las 9 clases con recall > 0 por primera vez en toda la sesión** (antes `sacro`/`mandibula_maxilar`
  se turnaban para caer a 0 por n=1 en val) — guardarraíl satisfecho sin caer a fallback.
- `cranio` sube de 0.553 a 0.679, `mandibula_maxilar` 0.878, `hueso_pequeno` 0.890.
- Banco sintético limpio: 0.875.
- **Examen honesto (`--split held_out`, 9 fotos nunca vistas): 0.5556 — BAJA desde el 0.7778
  anterior**, pese a que todos los números sintéticos mejoran. Matriz de confusión: `costilla` y
  `hueso_pequeno` (antes acertaban) ahora fallan, `hueso_pequeno`→`cranio` y `pelvis`→`cranio` (este
  último ya fallaba antes, se mantiene). Sospecha: los 72 cráneos reales metidos de golpe (frente a
  1-26 del resto de grupos) desequilibran el modelo hacia sobre-predecir `cranio` cuando duda. Con
  n=9 cada fallo pesa ~11 puntos, así que no es concluyente estadísticamente, pero 2 fallos nuevos
  cayendo ambos hacia la misma clase sobre-representada no parece puro azar. **No confundir "sube en
  sintético" con "mejora de verdad"** — la lección de siempre en esta sesión, ahora con un ejemplo
  más grande y más claro que los anteriores.

### Pendiente
36. `sacro_001.jpg` a `sacro_004.jpg` existen en disco (`data/real_photos_accepted/sacro/`) pero NO
    tienen fila en `manifest.csv` — invisibles para `build_dataset_real_photos.py`. Añadir sus filas
    o confirmar por qué se dejaron fuera.
37. Imágenes densas con muchos especímenes dejadas sin recortar por los forks (6 de costilla, 4 de
    vertebra, 1 collage de cráneo de baja calidad) — recorte manual pendiente si se quiere ese
    volumen extra.
38. `hueso_plano` y `hueso_pequeno` siguen en 0 fotos reales de entrenamiento — ningún lote de esta
    sesión tocó esos grupos (el usuario no dejó fotos de esos dos). Siguen siendo la prioridad más
    clara para la próxima ronda.
39. **Investigar el desequilibrio de cráneo (72 fotos) sobre-prediciendo frente al resto** — probar
    class-balanced sampling (`WeightedRandomSampler`, ya propuesto y no implementado antes en la
    sesión) o simplemente capar cuántas fotos reales por grupo entran a cada epoch, para que un
    grupo con muchas más fotos no arrastre la predicción hacia sí cuando el modelo duda. Con n=9 en
    el held_out no se puede confirmar la causa con certeza — ampliar el banco held_out ayudaría
    también a diagnosticar esto (pendiente #30/#31).

### Cuarta ronda: los 3 grupos en cero (hueso_largo/hueso_plano/hueso_pequeno) rellenados

El usuario dejó fotos nuevas en `Fotos entrenamiento/hueso largo/` (femur+humerus), `Hueso plano/`
(Escapula+esternon) y `Huesos pequeños/` (Astragalo) — justo los 3 grupos que llevaban toda la
sesión en 0 fotos reales de entrenamiento. De paso apareció "nuevos2" (más PDFs de papers, sin
fotos, igual que "nuevos 1"). Los 6 grupos de la ronda anterior (costilla/cráneo/mandíbula/pelvis/
sacro/vértebra) no tenían cambios (hash idéntico a lo ya procesado).

Delegado en 3 forks paralelos, mismo protocolo de siempre (sin rechazo unilateral, solo dividir
imágenes con más de un hueso). Sorpresa aparte: `hueso_largo` y `hueso_plano` tenían lotes huérfanos
de una sesión previa (`manifest_hueso_largo.csv` 23 filas, `manifest_hueso_plano.csv` 3 filas —
pendiente #33/#36 de antes) con fotos YA copiadas a disco pero nunca fusionadas al `manifest.csv`
principal — los forks las fusionaron de paso antes de procesar lo nuevo.

| Grupo | Antes | Después | Detalle |
|---|---|---|---|
| hueso_largo | 1 | **90** | +23 fusión huérfanos, +66 nuevas (fémur+húmero, 20 recortes de fotos con varios huesos) |
| hueso_plano | 3 | **20** | +3 fusión huérfanos (escápula ciervo/perro/mamut), +17 nuevas (escápula+esternón) |
| hueso_pequeno | 0 | **38** | +38 nuevas (astrágalo), 30 de ellas recortes de 8 fotos con varios especímenes |

`manifest.csv`: 357 filas totales. `bone_geometric_features_real_photos.csv` regenerado: 357/357
segmentaron bien. `combined.csv`: 4053→4200 filas. **Por primera vez en toda la sesión, los 9 grupos
tienen fotos reales de entrenamiento** (ninguno en 0).

Reentrenado el híbrido (`model_seed=42`):
- val_acc sintético baja algo (0.6697 vs 0.7590 de la ronda anterior) — `mandibula_maxilar` cae
  fuerte a 0.333 recall, `pelvis` a 0.500. Mismo patrón de siempre: meter mucho volumen nuevo de
  golpe en unos grupos (hueso_largo pasa a ser el grupo con MÁS fotos reales, 90) desplaza el
  balance y afecta a otros grupos que no cambiaron.
- Banco sintético limpio: 0.8472 (baja un poco desde 0.8981/0.875 de rondas anteriores).
- **Examen honesto (`--split held_out`): 0.6667** — progresión completa de la sesión:
  0.4444 → 0.6667 → 0.7778 → 0.5556 → **0.6667**. Recupera algo desde el mínimo de la ronda
  anterior; `hueso_pequeno` (que fallaba en la ronda de cráneo) ahora acierta. Sigue sin superar el
  pico de 0.7778 (con solo mandíbula añadida, antes de los grandes lotes de cráneo/hueso_largo).

**Lectura de las 5 rondas de esta sesión**: añadir fotos reales SÍ ayuda de fondo (0.4444 inicial
→ 0.6667 final, +22 puntos netos), pero el ORDEN y el BALANCE entre grupos importa — meter un lote
enorme y desproporcionado de un solo grupo (72 cráneos, luego 90 huesos largos) empeora
temporalmente antes de que el conjunto se asiente. Con n=9 en el held_out cada ronda tiene mucho
ruido individual; la tendencia de fondo (varias rondas, no una sola) es la que hay que mirar.

### Pendiente
40. Class-balanced sampling (pendiente #39) gana urgencia: ahora hay grupos con 90 fotos reales
    (hueso_largo) y grupos con 18 (costilla) — el desequilibrio entre grupos de fotos reales es tan
    grande como el que había entre grupos sintéticos al principio del proyecto.
41. ~~Ampliar el banco held_out~~ — hecho a lo grande, ver más abajo.

### Quinta ronda: `WeightedRandomSampler` + examen final 8x más grande (de `real_world_benchmark`, n=9, a `real_photos_accepted`, n=72)

Dos piezas implementadas a la vez:

**1. `WeightedRandomSampler` (real vs sintético dentro de cada grupo).** Se confirmó que
`class_weight="balanced"` en la pérdida YA compensaba el desequilibrio ENTRE grupos (hueso_largo con
952 filas vs pelvis con 132) — eso no era nuevo. Lo que faltaba: DENTRO de cada grupo, las fotos
reales son minoría frente a las sintéticas (costilla: 18 reales de 360 filas, 5%; cada ronda de
fotos reales de esta sesión se diluye entre muchas más filas sintéticas del mismo grupo). Nuevo flag
`--real-photo-boost` (default 4.0): `WeightedRandomSampler` sobremuestrea las filas reales x4 dentro
de su grupo, sin tocar el balance entre grupos (eso sigue en la pérdida).

**2. Examen final mucho más grande, sacado del propio pool de entrenamiento.** El held_out de
`real_world_benchmark` (1 foto/grupo, n=9) tenía demasiado margen de error para juzgar nada con
confianza. Con `data/real_photos_accepted` ya en 357 fotos, tiene sentido reservar una porción fija
como examen — nuevo `manifest.csv` gana columna `split` (`train`/`test`, ~20% estratificado por
grupo, `random_state=42`, mínimo 3 por grupo): **72 fotos de examen** (4-18 por grupo) en vez de 9.
`build_dataset_real_photos.py` gana `--split {train,test,all}` (default `train`, lo único que entra
a `combined.csv`); `evaluate_cnn_real_photos.py` gana `--source {benchmark,accepted}` para poder
examinar contra cualquiera de los dos bancos.

Reentrenado el híbrido (`model_seed=42`, sampler activo, `combined.csv` con solo las 285 filas
`train` de fotos reales — las 72 de `test` nunca entran):
- `real_bench_acc` (banco viejo, 48 fotos `selection`) sube fuerte a 0.8542 en la última epoch —
  el sampler sí mueve la aguja en ese banco.
- Val sintético: `sacro` cae a 0.083 recall (raro, aislado) pero el resto razonable
  (cranio 0.762, hueso_largo 0.865, hueso_pequeno 0.842, hueso_plano 0.861, vertebra 0.816). Banco
  sintético limpio: 0.7917 (baja un poco).
- **Examen honesto NUEVO (`--source accepted --split test`, 72 fotos): accuracy 0.8472, balanced
  0.7631** — el número más fiable de toda la sesión con diferencia (8x más muestra que cualquier
  medida anterior). `hueso_largo` (18/18), `pelvis` (5/5) y `sacro` (5/5) perfectos; `cranio` 13/14;
  `vertebra` 8/9. Débiles de verdad: `costilla` (1/4 recall) y `hueso_plano` (1/4 recall) — aquí el
  n ya es suficiente para tomárselo en serio, no es ruido de 1 sola foto.

**Nota importante sobre esta cifra**: el examen de `accepted/test` viene de la MISMA fuente/estilo
que las fotos de entrenamiento (mismas tandas de búsqueda de imágenes, fondos parecidos) — es un
examen honesto en el sentido de que el modelo nunca vio esas fotos concretas, pero no es tan
"externo" como `real_world_benchmark` (fuente totalmente distinta, Wikimedia Commons). Mantener
ambos bancos: `accepted/test` (n=72) para seguimiento fino por grupo epoch a epoch, y
`real_world_benchmark/held_out` (n=9, cuando crezca) como control de generalización a fuentes que ni
siquiera aportaron nada al pool de entrenamiento.

### Pendiente
42. ~~`costilla` con recall 0.25~~ — resuelto, ver más abajo (sube a 0.89).
43. Considerar ampliar también `real_world_benchmark` (sigue en n=9 el held_out) para tener un
    segundo control totalmente externo con más muestra.

### Sexta ronda: segmentación multi-instancia + 5 métricas de forma nuevas — costilla 0.25→0.89

Propuesta del usuario: para las fotos de costilla con VARIOS huesos sueltos en la misma imagen (ya
descartadas enteras en la ronda anterior por no poder recortarse limpio con un rectángulo), segmentar
cada hueso por separado y pintar el resto de gris — y de paso, añadir métricas de forma pensadas para
distinguir costilla (muy tortuosa) de hueso_largo (casi recto): tortuosidad, giro acumulado,
consistencia del giro, ajuste a arco, ajuste a línea.

**1. Segmentación multi-instancia** (`segment_instances()` en `extract.py`, nuevo
`segment_instances_cli.py`): en vez de quedarse con la caja de mayor score de Grounding DINO (como
`_segment_grounded_sam`), detecta TODAS las cajas por encima de umbral, segmenta cada una con SAM por
separado, descarta duplicados casi idénticos (IoU de máscara), y devuelve un recorte por hueso con
todo lo demás (fondo + los otros huesos) pintado de gris — mismo criterio que ya existía para
`costilla_104/105` (recortes manuales de una foto con 3 costillas), ahora automatizado.

Probado sobre las 7 fotos de costilla más densas (7-24 costillas cada una, descartadas enteras antes):
42 recortes candidatos generados, revisados uno a uno con un contact sheet — **28 limpios** (un hueso
por recorte), 14 descartados (varias costillas fusionadas en una caja, típico cuando están muy juntas
o solapadas — las fotos de "parrilla"/ribcage completo casi no separan bien, esperable). `costilla`
pasa de 18 a 46 fotos reales (37 train / 9 test tras re-estratificar el split solo para este grupo).

**2. Cinco métricas de forma nuevas**, calculadas sobre la misma `centerline` que ya usaba
`curvature_mean/max` (sin segmentar de nuevo): `tortuosity` (longitud del eje / distancia entre
extremos), `total_turning_angle` (suma de giros, radianes), `turning_consistency` (0-1, si el giro es
siempre hacia el mismo lado como un arco, o se cancela como en una recta con ruido), `circle_fit_residual`
(RMS al mejor círculo ajustado, /radio) y `line_fit_residual` (RMS a la mejor recta ajustada). La
`centerline` cruda resultó ruidosa punto a punto (viene de punto medio de anchura por corte, sensible
al borde de la máscara) — se suaviza con media móvil antes de calcular estas 5, sin tocar
`curvature_mean/max` (que se dejan como estaban).

**Efectos secundarios encontrados al re-extraer TODO el dataset sintético** (necesario para que las 5
columnas nuevas tengan valores reales en sintético, no solo en fotos reales):
- `base_datos_osea/renders/` había crecido desde la última vez que se generaron estas tablas (más
  especímenes acumulados con el tiempo, nunca se había vuelto a extraer) — 2419→2544 filas limpias,
  1043→1077 ruido clásico, 381→414 BlenderProc. Más volumen sintético de regalo, no relacionado con
  las métricas nuevas.
- **Bug encontrado**: `build_segmented_crops.py` (añade la columna `segmented_path` que necesita
  `cnn_dataset.py` para cargar el recorte de imagen) no incluía `bone_geometric_features_noisy.csv`
  en su lista `TABLES` — el CSV commiteado SÍ tenía esa columna (se había añadido en algún momento
  anterior sin dejar el cambio en el script). Al regenerar `bone_geometric_features*.csv` desde cero,
  esa tabla se quedó sin la columna y **rompió el entrenamiento** (crash en el primer batch). Arreglado
  añadiendo la tabla a `TABLES` y re-corriendo — 0 filas perdidas al recortar de nuevo.

Reentrenado el híbrido (`model_seed=42`, todo lo demás igual que la ronda anterior):
- Val sintético: `costilla` sube a 0.816 recall (antes bailaba entre 0.5-0.7 según la ronda). Banco
  limpio: 0.838.
- **Examen honesto (`accepted/test`, ahora 77 fotos: 72→77 por las nuevas costillas): accuracy 0.8442,
  balanced 0.7959** (similar al 0.8472/0.7631 de la ronda anterior en global, pero mejor balanced).
  **`costilla`: recall 0.25 → 0.89 (8/9)** — el punto débil más claro de la sesión, resuelto.
  `hueso_plano` sigue flojo (0.25, n=4) — ahora el punto débil claro que queda, sin tocar en esta
  ronda (no se le sumaron fotos ni se le aplicó segmentación multi-instancia).

### Pendiente
44. `hueso_plano` (recall 0.25, n=4) es ahora el punto débil objetivo más claro — mismo patrón que
    costilla antes de esta ronda: candidato a segmentación multi-instancia si hay fotos con varias
    escápulas/esternones juntos, o simplemente más fotos reales (sigue con pocas, 20 en total).
45. Aplicar `segment_instances()` a las fotos densas ya identificadas de otros grupos (vertebra tenía
    4 imágenes muy densas sin recortar, pendiente #37) ahora que la herramienta existe y funciona.
46. Considerar métricas de forma equivalentes para otros grupos con geometría distintiva no capturada
    aún (ej. número de bifurcaciones del esqueleto, propuesta del usuario no implementada por
    complejidad/prioridad más baja — huesos con forma en Y o ramificada, ninguno de los 9 grupos
    actuales la tiene marcadamente, bajo valor esperado por ahora).

### Pipeline paralelo "blackbg": fondo negro en vez de gris, para comparar — empate técnico

Propuesta del usuario: un pipeline COMPLETO en paralelo (sin tocar el de siempre) donde el fondo de
todos los recortes segmentados sea negro puro en vez del gris neutro habitual (`CROP_BG_GRAY=128`),
aplicado a TODO — sintético (train/val), fotos reales (entrenamiento/examen), y el examen final.

**Implementación, atajo clave**: `_crop_masked` usa máscara DURA (sin difuminado en el borde), así
que cada píxel de fondo en un recorte ya generado es EXACTAMENTE `(128,128,128)` — no hace falta
volver a segmentar nada con Grounding DINO+SAM, basta con sustituir ese color por `(0,0,0)` en los
PNG ya existentes (resultado matemáticamente idéntico a haber corrido la segmentación desde cero con
`bg_gray=0`). Nuevo `shared/cv_extractor/build_blackbg_variant.py`: recolorea los recortes de las 6
tablas (limpio/ruido clásico/ruido bproc/reales train/reales test/banco sintético) a
`data/segmented_crops_blackbg/`, escribe CSVs paralelos con sufijo `_blackbg` — 0 filas perdidas,
todas las 4343+216 filas convertidas en segundos (vs ~25-30 min si hubiera hecho falta resegmentar).

Para el examen final (fotos que se segmentan EN VIVO, sin recorte cacheado, ej.
`real_world_benchmark`), `extract.py` gana un parámetro `bg_gray` enhebrado por `extract_features`/
`extract_features_from_mask`/`_compute_all_features` (default sigue en 128, no cambia nada del
pipeline normal). `train_cnn.py` gana `--run-name` (nombre de fichero de salida distinto de
`--model-type`, para no pisar `cnn_model_hybrid_bone_group.pt`) y `--bg-gray`. `predict_cnn.py`
(`_load`/`classify_image`) y `evaluate_cnn_real_photos.py` ganan los mismos dos flags a juego.
`.gitignore` gana `data/segmented_crops_blackbg/` (el patrón exacto anterior no cubría variantes).

**Resultado — entrenado igual que el mejor híbrido actual** (`model_seed=42`, mismo sampler, mismas
features nuevas, guardado como `hybrid_blackbg` sin pisar el modelo normal):
- Val sintético: banco limpio 0.824 (vs 0.838 gris) — comparable, algo por debajo.
- **Examen honesto (77 fotos): accuracy 0.8442 (IDÉNTICA a la gris), balanced accuracy 0.8102 (vs
  0.7959 gris, +1.4 puntos)**. Por grupo: `costilla` perfecta 9/9 (vs 8/9 gris), `mandibula_maxilar`
  perfecta 5/5 (vs 4/5), `sacro` igual 5/5 — pero `cranio` baja a 0.71 (vs 0.79) y `vertebra` a 0.78
  (vs 0.89). **`hueso_plano` exactamente igual de mal (0.25) en los dos** — confirma que ese fallo no
  tiene que ver con el fondo.

**Conclusión honesta**: empate técnico. Quitar el fondo del todo no arruina el modelo ni lo dispara
— cambia QUÉ clases van mejor/peor (costilla y mandíbula mejoran, cráneo y vértebra empeoran un
poco), sin ganador claro en la cifra global. Con n=77 (y menos por grupo) la diferencia de 1-2 fotos
por clase ya explica buena parte del vaivén — no hay señal fuerte de que uno sea sistemáticamente
mejor que el otro con los datos actuales.

### Pendiente
47. Si se quiere seguir explorando el pipeline blackbg, entrenar con 2-3 semillas más (como se hizo
    para descartar `lr_head_finetune`) antes de sacar conclusiones firmes — con un solo run cada uno,
    la diferencia de 1.4 puntos en balanced accuracy está dentro del ruido esperado de inicialización
    ya documentado en esta sesión (pendiente #21 y el hallazgo de `--model-seed`).

## 2026-08-13 — Fix de agrupación por sujeto, gated_hybrid, comparación ROC, app de revisión + DVC

Sesión de cierre de varios frentes abiertos, sin entrenar nada nuevo salvo lo ya hecho para el
pipeline blackbg (ver arriba).

**1. Agrupación train/val por `species::specimen`, no solo `specimen`** (`training/group_splits.py`,
nuevo, con tests en `tests/test_group_splits.py`): `specimen` por sí solo puede colisionar entre
especies o colecciones distintas — la clave combinada evita juntar dos sujetos distintos por
casualidad de numeración. Centraliza `subject_ids`/`stratified_group_folds`/`subject_holdout`,
usado ahora desde `train_cnn.py` (vía `cnn_dataset.specimen_split`, que delega en
`subject_holdout`) y `train_geometric.py` (que además pasa de un split train/test optimista a
**solo CV honesta** — `StratifiedGroupKFold` repetida — y reentrena el modelo final publicado con
el 100% de los datos, ya no se guarda ni se reporta accuracy de un modelo entrenado solo con el 80%).

**Auditoría de leakage repetida con la clave nueva** (motivada por dudas del usuario sobre si había
fuga): `combined.csv` (4343 filas / 418 sujetos) partido con `subject_holdout` da 0 sujetos
compartidos entre train y val. Cruce contra el banco sintético limpio: 1 sujeto compartido
(`macaca_radiata::nmnhvz_7049315`) pero con `bone_group` distinto en cada lado (mandíbula en
`combined`, cráneo en el banco) — mismo animal, silueta distinta, no es fuga real (ya se había visto
esto el 2026-08-03 con la clave vieja; se confirma con la nueva). Cruce contra el examen honesto
(77 fotos reales de test): 0 sujetos compartidos. **Conclusión: sigue sin haber fuga real.**

**2. `gated_hybrid`** (`train_cnn.py`, `shared/inference/predict_cnn.py`): arquitectura nueva,
mezcla de expertos — rama CNN, rama geométrica y una puerta aprendida (`sigmoid` sobre la
concatenación de ambos embeddings) que decide, muestra a muestra, cuánto pesa cada rama. Pérdidas
auxiliares (0.25× cada rama por separado, sumadas a la pérdida de la salida fusionada) para que
ninguna rama quede atrofiada aunque la puerta se incline pronto hacia la otra. `predict_cnn.py`
expone el peso de la puerta como `modality_weights` en la respuesta, para poder ver qué modelo confía
más en apariencia vs en forma en cada foto. **Implementado pero sin entrenar ni evaluar todavía** —
pendiente decidir si vale la pena frente al híbrido normal.

**3. Comparación AUC/ROC entre las 4 variantes** (`training/plot_roc_cnn.py`,
`training/plot_roc_curves.py`, salida en `reports/roc_curves/`): sobre el val sintético, macro AUC
geometric 0.977, cnn_only 0.966, hybrid 0.990, **hybrid_blackbg 0.993** (el mejor, por poco) — coherente
con el empate técnico ya documentado en el examen honesto de fotos reales (sección de arriba).

**4. Infra Docker**: `Dockerfile` instala `torch`/`torchvision` cpu explícitamente antes de
`requirements.txt` y suma `libgl1`/`libglib2.0-0` (runtime de opencv, faltaban). `README.md`
documenta que en Docker la API queda expuesta en el puerto `8017` (antes decía `8000`, que sigue
siendo el puerto correcto solo para correr sin Docker).

**5. App de revisión "Tinder" del backlog de fotos candidatas** (`scripts/review_app.py`, Streamlit
en el puerto 8502): se descubrió que `data/real_photos_review/manifest.csv` tiene **803 fotos
candidatas, todas en `status=pending`** — nunca se había revisado el lote entero, la promoción a
`real_photos_accepted/` siempre fue 100% manual (mirar archivo por archivo, copiar a mano, editar
el CSV a mano). Por grupo: pelvis 185, cranio 126, costilla 123, mandibula_maxilar 112,
**hueso_plano 84** (el grupo más flojo del modelo hoy — pendiente #44), vertebra 69, sacro 56,
hueso_pequeno 30, hueso_largo 18. La app muestra una foto a la vez con botones + atajos de teclado
(`→` aceptar, `←` descartar, `↓` duda con nota obligatoria); aceptar mueve el archivo y añade la
fila al manifest de `real_photos_accepted/` con el split elegido a mano (recordando en pantalla
cuántas train/test tiene ya el grupo); descartar/duda solo cambian `status`, sin mover archivo. Cola
reanudable sin estado propio (se recalcula del CSV en cada carga).

**6. DVC**: `data/real_photos_review/`, `data/real_photos_accepted/`, `data/segmented_crops/` y
`data/real_world_benchmark/` pasan de estar en `.gitignore` sin ningún respaldo a estar versionadas
con DVC (remote local ya configurado en `/root/dvc-storage` desde antes, solo se usaba para 1 stage
de pipeline) — primer `dvc push`, 6040 ficheros subidos. `data/segmented_crops_blackbg/` y
`data/bone_ingest/` se quedan fuera por ahora (experimental/staging transitorio).

### `gated_hybrid` entrenado y evaluado — pierde contra el híbrido normal

Entrenado igual que el mejor híbrido (`model_seed=42`, mismo dataset/sampler), requirió levantar
`postgres`+`mlflow` de `docker compose` que estaban parados (el primer intento crasheó en
`mlflow.set_experiment` por `localhost:5000` sin escuchar, sin llegar a entrenar nada).

Val sintético 0.7217, banco fijo limpio 0.8333 — cifras razonables en sintético. Pero en el
**examen honesto (77 fotos reales de test)**: accuracy **0.8052**, balanced accuracy **0.7757** —
peor que el híbrido normal (0.8442 / 0.7959) y que `hybrid_blackbg` (0.8442 / 0.8102). `hueso_plano`
sigue en recall 0.25 (1/4), sin mejora — la puerta aprendida no arregla el punto débil de siempre.
Con una sola semilla no se puede descartar que sea ruido de inicialización, pero no hay ninguna
señal a favor de la complejidad extra (más parámetros, pérdidas auxiliares, forward más caro) frente
al híbrido simple. **Decisión: no se adopta.** No vale la pena gastar más semillas en esto salvo que
cambie algo de fondo (más datos reales, por ejemplo) que pueda cambiar la conclusión.

### Pendiente
49. Backlog de 803 fotos candidatas en `data/real_photos_review/` sin triar — usar
    `scripts/review_app.py` para ir aceptando/descartando, empezando por `hueso_plano` (84
    candidatas) que es el grupo con peor recall hoy (pendiente #44).

## 2026-08-13 (sesión larga) — Rediseño completo del render sintético: escena de laboratorio

Motivación: comparando sintético vs real con la app nueva (`scripts/compare_synth_real_app.py`,
sesión anterior) quedó claro que el 63% del sintético (`renders/`) es silueta plana sin sombreado —
"de risa" según el usuario. Petición: rehacer la escena desde cero imitando una foto de laboratorio
real (mesa, hueso soltado con física, cámara de persona de pie, luz de estudio, suciedad con
sentido físico) — no otro parche sobre la escena vieja.

**`base_datos_osea/blender/render_bone_lab.py`** (nuevo, no toca `render_bone.py` ni
`render_bone_bproc.py` — pipeline paralelo, mismo patrón que el experimento blackbg). Usa
`blender --background --python`, NO `blenderproc run` (la física necesita un único objeto de
colisión por hueso vía `bpy.ops.object.join()`, poco fiable bajo blenderproc sin contexto de UI).

**Iteración larga con el usuario, bugs reales encontrados y corregidos**:
- **Malla no recentrada** tras importar: el pivote heredado de Sketchfab podía quedar a decenas de
  unidades del origen — mesa y punto de caída no coincidían con dónde estaba la geometría de
  verdad. `center_on_bbox()` lo arregla.
- **Escala inconsistente entre mallas** (bug de fondo grave): radio local visto entre 2 y 106
  unidades según la malla, sin relación con el tamaño real del hueso — a escala grande el solver de
  rigid body fallaba EN SILENCIO (objeto "flotando" a medio caer, sin error). `normalize_scale()`
  fuerza un radio fijo (10 unidades) al importar, antes de todo lo demás — con esto, luces/mesa/
  física se calibran una vez y sirven para cualquier malla.
- **`MESH` (colisión exacta) inestable** para huesos delgados/curvos: una costilla rodaba como
  mecedora sin parar nunca (la fricción frena el deslizamiento, no la rodadura) y se salía de una
  mesa de 32x su radio — error numérico, no vuelco físico real. `CONVEX_HULL` (más simple, estándar
  para cuerpos dinámicos) asienta de forma estable.
- **Huesos alargados "de pie" a medio volcar**: amortiguación angular insuficiente + pocos frames
  dejaba al fémur balanceado sobre un extremo en vez de completar el vuelco. Subida la amortiguación
  angular y los frames de caída.
- **`has_natural_resting_orientation()`**: el chequeo de "¿está de pie?" solo cubría huesos tipo
  barra (un eje largo). Una escápula (forma de plato, un eje CORTO) fallaba el mismo chequeo sin
  que existiera para su caso — ampliado a cubrir también forma de plato.
- **Escápula que nunca asienta tumbada por más reintentos que se den**: no es mala suerte, es SU
  mínimo de energía estable con `CONVEX_HULL` para esa malla en concreto. Se probó a forzar la
  rotación a mano (matriz de cambio de base) pero `obj.matrix_world` no reflejaba el cambio al
  leerlo justo después sin volver a pasar por `settle_drop` (con o sin rigid body de por medio,
  ninguna combinación de `view_layer.update()`/`frame_set()` lo sincronizaba de forma fiable) —
  descartado por frágil. Solución robusta: si tras los reintentos normales ninguno pasa el chequeo
  estricto, se acepta el último intento tal cual (con tal de que siga sobre la mesa) en vez de
  forzar o descartar la caída.
- **Sin sombra de contacto** (se veía "flotando"): margen de colisión del solver deja un hueco de
  aire pequeño pero visible — "encaje" final que corrige solo la Z (traslación pura) tras asentar.
- **Luces**: primer intento con 1 sola luz salió ínfimo de energía (~40W a 60 unidades de distancia,
  caída inversa al cuadrado de Cycles) — subida a millones de vatios en unidades de escena
  normalizada. 2-3 luces IGUALES se cancelan entre sí y aplanan (probado); la solución fue una key
  dominante + secundaria a intensidad claramente distinta (nunca cerca de 1:1).
- **Material "de plástico"**: rugosidad demasiado baja (mínimo 0.35) daba brillo especular duro.
  Subida a 0.55-0.9. Además, si la malla trae SU PROPIO material descargado (fotogrametría real,
  se conserva para no perder textura), ese material puede traer un acabado PBR genérico/plástico de
  fábrica — `force_bone_reflectivity()` fuerza mate+sin metal en TODOS los materiales del objeto
  pase lo que pase, conservando color/textura.
- **Color de mancha verdoso** (efecto hongo) daba tinte antinatural a la pieza entera — cambiado a
  marrón tierra.
- **Craquelado (Voronoi) demasiado basto** y **polvo (acumulado en caras hacia arriba) cubriendo
  media pieza en huesos redondeados** (un cráneo sin superficie plana de verdad hace que "algo
  hacia arriba" sea casi toda la bóveda) — ambos apretados con `POWER` alto sobre la señal y techo
  de opacidad bajo, para que sean detalle sutil, no una capa que tape el hueso.
- **Encuadre con demasiado aire**: `--fill` subido de 0.72 (heredado de los otros scripts) a 0.97 —
  la gente hace la foto de cerca.
- **Mesa demasiado limpia**: acabado tipo tela/terciopelo oscuro (ruido de pliegues suave + Sheen),
  no un plano perfecto ni una textura "quemada".
- **Malla de cráneo mala descartada**: `sketchfab_32a69b1fc0ba45a2ac2d91889657407f` daba una forma
  deformada/hinchada pase lo que pase con el material — se confirmó comparando contra otras 2 mallas
  de cráneo disponibles, borrada de `data/meshes/` (no versionada en git, sin pérdida).
- **`LIGHT_PRESETS`**: 5 combinaciones de luz (neutro/tungsteno_calido/calido_suave/frio_exterior/
  ambiente_mixto) para que el usuario comparase — `calido_suave` fue la elegida (cálida, foco más
  grande/suave que el resto, sombra menos dura que un estudio de examen).

**Herramientas nuevas de apoyo**: `scripts/lab_render_viewer.py` (visor Streamlit local, puerto
8505, selector de espécimen + checkbox de máscara) y `base_datos_osea/scripts/render_lab_batch.py`
(runner por lote, mismo patrón que `render_noisy_batch.py` — un proceso Blender por espécimen).

### Pendiente
50. ~~Lote completo lanzado...~~ RESUELTO, ver abajo.

### Resultado del render de laboratorio: mejora clara en el examen honesto

Lote completo (121 especímenes de `data/meshes/`, `render_lab_batch.py`, `--light-preset
calido_suave`, 2 caídas x 3 vistas = 6 vistas/espécimen): **121/121 sin fallos**, 726 imágenes,
672 filas útiles (54 reservadas como banco de evaluación por `BENCHMARK_SPECIMENS`, igual criterio
que las otras ramas). `build_dataset_lab.py` (nuevo, mismo patrón que `build_dataset_bproc.py`:
usa la máscara ground-truth que ya trae cada vista, sin resegmentar) + recortes vía
`build_segmented_crops.py` (tabla `"mask"`, 0 fallos) + `build_combined_lab_variant.py` (misma
receta que `combined.csv` pero sustituyendo `renders/` por `renders_lab/`: 2471 filas final).

Entrenado el híbrido igual que el mejor actual (`model_seed=42`, `--run-name hybrid_lab`).
Val sintético bajó (0.5984 vs ~0.75+ de antes) — pero es una comparación contra el banco VIEJO de
renders planos (`bone_geometric_benchmark.csv`), dominio distinto al que entrenó ahora, no es la
métrica que importa. **Examen honesto (77 fotos reales) — comparación**:

| Modelo | Accuracy | Balanced accuracy |
|---|---|---|
| hybrid (el de referencia hasta hoy) | 0.8442 | 0.7959 |
| hybrid_blackbg | 0.8442 | 0.8102 |
| gated_hybrid | 0.8052 | 0.7757 |
| **hybrid_lab** | **0.8701** | **0.8323** |

**Mejor resultado de toda la sesión.** Por grupo: costilla, pelvis y sacro perfectos (1.0),
hueso_largo 0.94, vertebra 0.89, cranio 0.86, mandibula_maxilar sube a 0.80 (venía flojeando).
**`hueso_plano` sigue en recall 0.25 (1/4)** — sin cambio, confirma (tercera vez que se ve el mismo
patrón: aquí, con blackbg, y con gated_hybrid) que es un problema de volumen de foto real para ese
grupo, no de arquitectura ni de realismo del render. Con una sola semilla no se puede descartar
ruido de inicialización del todo, pero la mejora (+2.6 accuracy, +3.6 balanced) es mayor que el
ruido típico visto en este proyecto (±1-2 puntos entre semillas).

### Pendiente
51. Confirmar la mejora con 2-3 semillas más antes de adoptar `hybrid_lab` como modelo por defecto
    (mismo criterio que se aplicó para descartar `lr_head_finetune` — un solo run no basta para
    separar señal de ruido de inicialización).
52. Curar el backlog de fotos reales de `hueso_plano` (pendiente #44/#49) sigue siendo la palanca
    más clara para ese grupo en concreto — el render de laboratorio no lo mueve.
53. Si `hybrid_lab` se confirma, plantear renderizar más vistas por espécimen (ahora solo 6) o sumar
    más especímenes/mallas por grupo — el volumen sintético actual (672 filas) es mucho menor que
    el de `renders/` (2544) y aun así ganó; más volumen con la escena nueva podría dar más margen.

### `hybrid_lab` confirmado con 3 semillas y promovido a modelo por defecto

Pendiente #51 resuelto. Mismo split (`--random-state 42`), solo cambia `--model-seed` (1 y 7,
sumados al 42 original) para separar señal de ruido de inicialización (mismo patrón que la sesión
2026-08-03). Examen honesto (77 fotos, `--split test`) por semilla:

| Semilla | Accuracy | Balanced accuracy |
|---|---|---|
| 42 (original) | 0.8701 | 0.8323 |
| 1 | 0.8571 | 0.8314 |
| 7 | 0.8831 | 0.8242 |

Las 3 semillas quedan por encima de `hybrid` (0.8442/0.7959) por un margen mayor que el ruido de
init visto dentro del propio `hybrid_lab` (~0.026 de spread). **Mejora real, no ruido.**
`hueso_plano` sigue débil en las 3 semillas (recall 0.50/0.67/0.40) — confirma otra vez que es
volumen de foto real, no arquitectura ni semilla.

**`hybrid_lab` (semilla 42, el run original, sin cherry-picking de la mejor semilla) promovido a
modelo por defecto** — pasa a ser `cnn_model_hybrid_bone_group.pt` y compañía. El `hybrid` anterior
se conserva como backup en `models/cnn_*_hybrid_pre_lab_bone_group.*` (encoder/scaler/
feature_columns/model/meta) por si hace falta revertir. `mlflow` (servidor en `localhost:5000`)
estaba caído en esta sesión — reentrenar apuntando a `sqlite:///mlruns_local/mlflow.db` vía
`MLFLOW_TRACKING_URI` funciona sin servidor levantado (el backend de fichero plano `./mlruns` está
bloqueado por MLflow, "maintenance mode", desde esta versión).

### Pendiente
54. Curar backlog de fotos reales de `hueso_plano` sigue siendo la palanca más clara (ver #52) —
    ahora con `hybrid_lab` como base.
55. Evaluar #53 (más vistas/especímenes por grupo en el render de laboratorio) como siguiente paso
    de mejora sintética.

