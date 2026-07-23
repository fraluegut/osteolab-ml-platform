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
