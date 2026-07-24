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
