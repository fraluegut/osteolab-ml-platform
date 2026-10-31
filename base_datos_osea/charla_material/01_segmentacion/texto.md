# Eje 1 — Segmentación: ¿la CNN veía hueso, o veía todo?

## Qué es este eje

Antes de clasificar, hay que saber qué parte de la imagen es hueso. El primer experimento con CNN
entrenaba sobre la foto o el render completo, fondo incluido. La mejora real vino de dos cambios
encadenados: primero acertar el método de segmentación (de un umbral fijo a detección por texto),
y después, por separado, hacer que la CNN viera de verdad el recorte segmentado en vez de la imagen
entera — dos bugs distintos, corregidos en el mismo día.

## Línea temporal

- **2026-07-26 — Grounded-SAM adoptado como backend de segmentación por defecto.**
  Se abandona el punto/umbral fijo (Otsu+adaptativo, o un punto central con FastSAM) por detección
  por texto: Grounding DINO encuadra "bone" semánticamente, SAM segmenta dentro de esa caja.
  Motivado por fallos reales encontrados con el inspector de features (`scripts/inspect_features.py`):
  - `cranio_05.jpg`: fondo gris casi del mismo tono que el hueso — con Otsu clásico solo sobrevivía
    la cavidad nasal, el resto del cráneo no se segmentaba.
  - `cranio_06.jpg` (momia envuelta): la textura de la envoltura generaba tanto microcontraste como
    el hueso — 224 contornos tras la apertura morfológica, el mayor solo cubría 18% del área real.
  - `vertebra_05.jpg`: el prompt de punto fijo centrado caía dentro del foramen vertebral — FastSAM
    segmentaba el agujero, no el hueso. Cualquier hueso con hueco central y vista centrada en él
    (vértebra, sacro, pelvis) era vulnerable a este fallo. Grounded-SAM lo arregla de raíz: la caja
    encuadra la vértebra entera.
  - `hueso_plano_08.jpg` (escápula, FastSAM): la máscara nativa traía un corte recto de fábrica justo
    en el cuello fino de la escápula — limitación del decodificador en geometría cóncava/fina.

- **2026-07-26 (mismo día, después) — La CNN pasa a entrenar y predecir sobre el recorte
  segmentado, no la foto completa.**
  Hallazgo: aunque ya existía `extract_features()` con segmentación, la rama de imagen de la CNN
  nunca la usaba — abría el fichero original tal cual (resize + normalización), fondo real o
  sintético incluido, tanto en entrenamiento como en inferencia. Es decir: **la primera CNN no
  usaba el recorte bueno — eso vino después, el mismo día pero como cambio separado.** Se implementó
  recorte al bounding box del contorno + fondo gris (128,128,128) fuera de la máscara, aplicado a
  `cnn_only` e `hybrid`, en entrenamiento y en inferencia.

## Métricas: antes → después

**Cambio de backend de segmentación (classic → grounded_sam), fotos reales:**

| Modelo | Antes (classic + filtro area_ratio) | Después (grounded_sam) |
|---|---|---|
| Geométrico | 0.27–0.33 | **0.5185** |
| Híbrido | 0.3818 | **0.6296** |
| Cascada (elongación+cnn_only) | 0.6000 (mejor histórico hasta entonces) | 0.5741 (baja — la cascada dejó de aportar) |
| Geométrico, banco sintético fijo | — | 0.7731 |

Tasa de contorno encontrado (`build_dataset.py`): limpio 2544/2760 (grounded_sam) vs 2414/2760
(classic); con ruido 1077/1220 vs 972/1220. Filas descartadas por `area_ratio>0.02`: 4.9%/3.2%
(limpio/ruido) frente a 5.1%/12.4% con classic.

**Cambio a recorte segmentado en la CNN (mismo backend grounded_sam), fotos reales:**

| Modelo | Imagen completa | Recorte segmentado | Banco sintético fijo |
|---|---|---|---|
| cnn_only | — (no evaluado antes) | 0.5370 | 0.7685 |
| Híbrido | 0.6296 | 0.6296 (total igual, por coincidencia) | 0.8241 |
| Cascada | 0.5741 | 0.5741 | — |

El total del híbrido empata, pero el desglose por grupo cambia de verdad (costilla 0.50→0.00,
cráneo 0.43→0.57, sacro 0.83→0.50, mandíbula 0.57→0.71) — con 54 fotos en 9 grupos, el empate en
el agregado es casualidad estadística, no ausencia de efecto. En el banco sintético, donde hay más
datos, la mejora es clara y consistente.

## Los 3 backends, cabeza a cabeza, con el modelo de producción actual (nuevo, corrido en vivo)

Las tablas de arriba comparan backends con checkpoints antiguos y un banco pequeño (54 fotos). Para
la charla hace falta la comparación definitiva: **el modelo híbrido que está en producción ahora
mismo** (`hybrid_lab`, el confirmado con 3 semillas, ver `project_hybrid_model_regression`) contra
las **83 fotos reales del split `test`** de `data/real_photos_accepted` (nunca vistas en
entrenamiento — examen honesto, no el banco `benchmark` de 9 fotos). Corrido con
`training.evaluate_cnn_real_photos --model-type hybrid --run-name hybrid_lab --source accepted
--split test`, un backend distinto cada vez, sin tocar nada más.

| Backend | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Precision (weighted) | Recall (weighted) | F1 (weighted) | Balanced acc. | Top-3 | Sin segmentar |
|---|---|---|---|---|---|---|---|---|---|---|
| **classic** (Otsu+adaptativo) | 0.4699 | 0.44 | 0.47 | 0.39 | 0.55 | 0.47 | 0.44 | 0.4741 | 0.7349 | 0/83 |
| **fastsam** (punto central) | 0.5904 | 0.61 | 0.53 | 0.53 | 0.74 | 0.59 | 0.62 | 0.5941 | 0.9351 | 6/83 (7.2%) |
| **grounded_sam** (DINO+SAM) | **0.8675** | **0.85** | **0.83** | **0.81** | **0.89** | **0.87** | **0.86** | **0.8315** | **0.9759** | 0/83 |

`macro` = media simple entre las 9 clases (cada grupo pesa igual, aunque tenga 4 o 20 fotos).
`weighted` = media ponderada por nº de fotos de cada grupo en el banco (lo que domina el
accuracy global). Recall macro ≈ balanced accuracy (por diseño, son casi la misma cosa). Con
grounded_sam, precision y recall suben casi a la par (0.85/0.83 macro) — no es un backend que
"acierte por casualidad" prediciendo siempre la clase mayoritaria, que es lo que sí pasa un poco
con classic (precision weighted 0.55 muy por encima de precision macro 0.44: acierta mejor en las
clases grandes, mal reparto entre clases).

De 0.47 a 0.87 de accuracy — **+40 puntos**, no un matiz. FastSAM queda a mitad de camino (0.59):
mejor que Otsu puro pero muy por detrás de grounded_sam, consistente con el fallo del foramen visto
en `comparacion_backends/vertebra_05_foramen/`.

Desglose por grupo (recall), classic → fastsam → grounded_sam:

| Grupo | classic | fastsam | grounded_sam |
|---|---|---|---|
| costilla | 1.00 | 0.56 | **1.00** |
| cráneo | 0.60 | 0.55 | 0.85 |
| hueso_largo | 0.22 | 0.61 | **0.94** |
| hueso_pequeño | 0.00 | 0.62 | 0.75 |
| hueso_plano | 0.00 | 0.25 | 0.25 (el más débil con diferencia, en los 3 backends) |
| mandíbula_maxilar | 1.00 | 0.40 | 0.80 |
| pelvis | 0.60 | 1.00 | **1.00** |
| sacro | 0.40 | 0.80 | **1.00** |
| vértebra | 0.44 | 0.56 | 0.89 |

`hueso_plano` sigue siendo el grupo débil pase lo que pase con la segmentación (0.00 → 0.25 → 0.25)
— confirma que ahí el cuello de botella es volumen de especímenes físicos, no el backend de
segmentación (coincide con lo ya documentado en HISTORIAL para ese grupo).

Logs completos (matriz de confusión + classification_report de sklearn) en
`comparacion_modelo_real/eval_classic.txt`, `eval_fastsam.txt`, `eval_grounded_sam.txt`.

## Comparación paso a paso de los 4 métodos (nuevo, corrido en vivo con los modelos reales)

`comparacion_backends/<ejemplo>/` — una carpeta por imagen de ejemplo, con 6 ficheros SUELTOS
(no un collage): `1_original.jpg`, `2_classic_otsu.jpg`, `3_fastsam.jpg`, `4_grounding_dino.jpg`,
`5_grounded_sam.jpg`, `6_recorte_final.jpg`. Cada imagen lleva su propia franja translúcida abajo
con el nombre del método en blanco — pensadas para usarse una a una en la diapositiva (construir
la secuencia con builds/animación) en vez de un panel fijo. Generadas corriendo las funciones
reales de `shared/cv_extractor/extract.py` (`_segment`, `_segment_fastsam`,
`_segment_grounded_sam`) sobre fotos del propio dataset — mismo código que producción, no una
simulación. Cada carpeta trae un `NOTA.txt` con el motivo de por qué se eligió esa foto.

5 ejemplos:

- **`hueso_largo_004/`** — fémur sobre trípode, fondo de laboratorio con reflejos (caso "normal").
  Otsu pinta medio fondo de rojo (el microcontraste de los reflejos activa el umbral adaptativo);
  FastSAM y grounded_sam aíslan el hueso limpio porque está centrado.
- **`vertebra_05_foramen/`** — **la imagen exacta citada en HISTORIAL.md (2026-07-26)** que motivó
  adoptar Grounding DINO: FastSAM, con su punto fijo en el centro, cae dentro del foramen
  vertebral y segmenta el agujero (relleno amarillo hueco, no el hueso). Grounded-SAM encuadra
  primero toda la vértebra por semántica ("bone", score 0.81) y segmenta la silueta completa. El
  ejemplo más fuerte: muestra el bug concreto que forzó el cambio de backend.
- **`hueso_plano_08_sam_artefacto/`** — escápula, caso citado en HISTORIAL: SAM corta recto el
  cuello fino de la escápula (limitación del decodificador en geometría cóncava/fina). Ver si el
  artefacto sigue visible en `5_grounded_sam.jpg` con el modelo actual.
- **`mandibula_01_caja_casi_toda_imagen/`** — caso citado en HISTORIAL: la caja de DINO cubre casi
  toda la imagen y SAM, por `iou_score`, podía elegir mal la máscara — motivó el fix de "elegir por
  área, no por iou_score" (ver código de `_segment_grounded_sam`). En el run actual el recorte sale
  limpio: buen ejemplo de "bug ya corregido, verificar que sigue corregido".
- **`cranio_143_objeto_pequeno_fondo_madera/`** — cráneo pequeño sobre fondo de madera con otro
  cráneo detrás (posible distractor para Grounding DINO: dos objetos "bone" candidatos en la
  misma foto).

## Precisión visual sobre el dataset de fotos reales (nuevo, inspección manual)

`found=True` en el CSV solo dice "encontró algún contorno" — no dice si es el contorno correcto.
Sobre `data/processed/bone_geometric_features_real_photos.csv` el backend grounded_sam encuentra
contorno en el **100% de las 349 fotos reales aceptadas** (los 9 grupos al 1.00). Pero esa métrica
no ve fallos de "recortó el objeto equivocado" — para eso hace falta mirar.

Se generó un muestreo visual: 8 pares (original | recorte segmentado fondo negro) por cada uno de
los 9 grupos (`ejemplos_precision/grid_<grupo>.jpg`, 72 ejemplos en total) y se inspeccionaron a
ojo. Patrón encontrado: la segmentación falla sobre todo cuando la foto **no es un objeto limpio
sobre fondo simple** — fotos de catálogo/producto con reglas, logos de stock, paneles
multi-imagen o maniquíes con elementos de medida confunden a Grounding DINO, que prioriza el
elemento más "cosa reconocible" de la imagen en vez del hueso.

8 fallos y 3 aciertos representativos quedaron recortados en `casos_concretos/`:

- `fallo_vertebra_102.jpg` — la máscara cubre la vértebra entera en negro sólido, sin hueco del
  foramen: pierde la forma interna que el clasificador usa.
- `fallo_vertebra_133.jpg` — foto con regla + 3 fragmentos óseos: recorta solo un fragmento
  pequeño, pierde el hueso principal de la fila.
- `fallo_vertebra_125.jpg` — panel multi-objeto con escala en mm: recorta el objeto equivocado del
  panel.
- `fallo_pelvis_117.jpg` — maniquí gris con anillo de medida superpuesto: la caja de Grounding DINO
  prioriza el anillo, no la pelvis.
- `fallo_pelvis_111.jpg` — foto de catálogo con etiqueta de escala ("2½ in"): recorta la etiqueta,
  no el hueso.
- `fallo_pelvis_113.jpg` — imagen de stock con marca de agua "alamy": recorta el logo, no el dibujo
  anatómico.
- `fallo_cranio_161.jpg` — fondo de tela oscura con texto de marca de agua superpuesto: recorte
  ruidoso, incluye fondo.
- `fallo_cranio_150.jpg` — solo recorta una esquina del cráneo, pierde la mayor parte del hueso.
- `bien_vertebra_116.jpg`, `bien_pelvis_104.jpg`, `bien_cranio_108.jpg` — objeto único sobre fondo
  limpio o neutro: recorte preciso en los tres casos.

**Lectura para la charla**: la tasa de "encontró algo" (100%) es engañosa — la tasa real de
"encontró lo correcto" depende de la limpieza de la foto de origen, no está medida con un número
único todavía (haría falta etiquetar manualmente los 349 pares original/recorte con acierto
binario para dar un porcentaje). Buen punto para dejar como pregunta abierta o próximo paso:
"medir precisión de localización, no solo tasa de contorno encontrado".

## Imágenes incluidas

1. `01_caso_cranio_05_fondo_confunde_otsu.jpg` — fondo gris que casi funde con el hueso, Otsu
   clásico solo rescataba la cavidad nasal.
2. `02_caso_cranio_06_momia_envuelta.jpg` — textura de la envoltura genera falso microcontraste,
   224 contornos espurios con el método clásico.
3. `03_caso_vertebra_05_fastsam_segmenta_agujero.jpg` — el punto fijo centrado cae en el foramen
   vertebral, FastSAM segmenta el agujero en vez del hueso.
4. `04_caso_hueso_plano_08_artefacto_sam.jpg` — escápula con corte recto de fábrica en el cuello
   fino, limitación del decodificador de SAM en geometría cóncava.
5. `05_resultado_recorte_segmentado_cranio.png` — ejemplo de recorte final (fondo gris, bounding
   box ajustado) tal como lo ve la CNN tras el fix del mismo día.
6. `06_resultado_recorte_segmentado_vertebra.png` — mismo tipo de resultado para vértebra.

Para el collage de 5 paneles (foto original / Otsu puro / segmentación clásica / DINO / SAM) que
sirvió de referencia: no existe guardado en disco — se generó y mostró en el chat de una sesión,
no se persistió como fichero. Si se quiere recrear para la charla, usar
`scripts/inspect_features.py` (Streamlit, corre los tres backends a la vez sobre la misma imagen)
contra una de las fotos de `data/real_world_benchmark/`.
