# Eje 2 — Sim2real: de render de juguete a render que engaña

## El proceso, explicado como pipeline de render profesional (nuevo, empezar la charla por aquí)

Lo que se ha hecho encaja con el flujo estándar de cualquier estudio de VFX/arte 3D al preparar un
asset para render fotorrealista: no se trabajan luz y material a la vez, se añaden en capas, una
decisión a la vez, para poder juzgar cada una por separado. `progresion_profesional/` muestra las
5 capas, **mismo hueso, misma cámara, cambia una sola cosa entre imagen e imagen**:

1. **`1_clay_sin_luz.jpg` — Clay render.** Solo geometría: material gris mate uniforme, luz de
   relleno genérica sin ambiente real. Es lo primero que se mira en cualquier pipeline 3D — si la
   forma no se lee bien aquí (siluetas raras, proporciones rotas), nada de lo que viene después lo
   arregla. Equivalente a nuestro `render_bone.py` original (fondo blanco, sin ruido).
2. **`2_clay_mas_hdri.jpg` — + Iluminación real (HDRI).** Mismo material gris de clay, pero ahora
   la luz viene de un HDRI real (foto panorámica 360° de un entorno real, usada como fuente de luz
   y reflejos — no es un foco inventado). Se nota sobre todo en cómo cae la sombra de contacto y en
   los brillos sutiles de la superficie curva; el objeto en sí sigue sin tener "identidad" de
   material.
3. **`3_material_completo_mas_hdri.jpg` — + Material (shading).** Aquí es donde entra de verdad el
   shader: color base sacado de una paleta real (KMeans sobre 426 fotos de hueso ya segmentadas),
   variación de rugosidad, bump/normal por porosidad, y la textura real escaneada encima
   (`bone_material.py`, commit `71dcce3`). El material ya "vende" que es hueso, no plástico pintado.
4. **`4_suciedad_curvatura_mas_pose_fisica.jpg` — + Suciedad por curvatura + composición física.**
   La capa final: un mapa de curvatura de la propia malla (nodo *Pointiness* de Blender — el
   equivalente exacto al *Curvature map* que usa Substance Painter para dirigir el *dirt*) hace que
   la suciedad se acumule en huecos y bordes desgastados, no al azar; y el hueso ya no "flota" en el
   render, se suelta con física real sobre una mesa y se fotografía desde el ángulo en que lo haría
   una persona. Esta era la imagen que mejor engañaba al ojo antes de la capa 5 (abajo) — junta
   luz + material + suciedad dirigida + composición creíble, pero el specular seguía plano.
5. **`5_specular_ior_mas_clearcoat.jpg` — + Specular/IOR por zona + barniz/pulido (Coat), capa
   nueva (2026-10-23).** Hasta aquí el `Specular IOR Level` del BSDF se quedaba en su valor por
   defecto (0.5), igual en un hueco poroso que en una superficie compacta — no es físico: lo poroso
   dispersa la luz (mate, specular bajo), lo compacto refleja algo más. Se reutiliza la misma señal
   de cavidad que ya dirige la suciedad (`cavity_ramp`) para variar el specular por zona, y un 30%
   de las piezas sortea además un barniz/pulido sutil (`Coat Weight` + `Coat Roughness`) — simula
   tanto una pieza de museo barnizada como hueso pulido por manipulación repetida, no todas las
   piezas lo tienen. Comparar de cerca con la fase 3 (mismo seed, mismo encuadre): el brillo a lo
   largo de la diáfisis ya no es uniforme.

**Importante — esta capa 6 NO llega todavía a la escena de laboratorio.** `render_bone_lab.py` (la
fase 4, la de la mesa) tiene su propio material separado (`apply_two_layer_material`), no usa
`bone_material.py` — así que el specular/clearcoat nuevo solo está en producción vía
`render_bone_bproc.py`. Si se quiere la combinación "mesa + specular por zona", hay que portar la
capa 6 a `apply_two_layer_material` o unificar ambos materiales — pendiente, no hecho en esta
sesión.

6. **`6_fix_bump_textura_real.jpg` — fix real de la textura escaneada (2026-10-23).** El usuario
   notó que "la textura de hueso no va bien" — diagnóstico confirmado con un render de prueba
   (textura cruda conectada SOLA a Base Color, sin mezclar con nada más): salía casi plana/uniforme,
   sin apenas poro visible, pase lo que pase con la escala de repetición (probado ×1/×5/×15, sin
   cambio perceptible). Causa: la textura real solo alimentaba el *color*; el bump de la capa 3
   venía de un Voronoi procedural sin relación con el relieve fotografiado real. Una foto diffuse de
   superficie porosa tiene poco contraste de color por sí sola — se lee como "porosa" sobre todo por
   la sombra que cada poro proyecta, no por su tinte. Fix en `bone_material.py` (commit pendiente):
   se deriva un Bump propio de la textura (su canal de color como altura) encadenado con el bump
   procedural existente, más un boost de contraste local antes de teñir. Comparar de cerca con la
   fase 3: ahora sí se ve grano/poro real en vez de una franja lisa con un solo brillo especular.

**Por qué en este orden y no otro**: en producción real de VFX se separa así para poder diagnosticar
problemas — si algo se ve mal con el material puesto, se vuelve al clay+HDRI para comprobar si el
problema es de luz o de shader. Aplicado a nuestro caso: el salto de accuracy real medido (ver eje
de segmentación) vino sobre todo de la fase 2→algo-parecido-a-3 (máscara y recorte, no del detalle
de material) — la capa 4 es la que más vende visualmente en una charla, pero su aporte a la métrica
todavía no está medido (ver sección de métricas más abajo).

## Qué es este eje

El dataset de entrenamiento es sobre todo sintético (renders 3D), pero el modelo se evalúa contra
fotos reales. Cerrar esa distancia (domain gap) pasó por tres etapas: renders limpios sin ruido,
renders con ruido y composición sobre fondos reales + escena de laboratorio, y finalmente un
material de superficie mucho más realista (paleta real + capas de conservación) — esta última
etapa es visual, hecha para la charla, y no está integrada en el pipeline de entrenamiento.

## Línea temporal

- **(sesión previa) — Render limpio, fondo blanco uniforme.** Primera generación de renders.
  Buen contorno pero nada de variación de iluminación/fondo/suciedad — el "render de juguete".

- **2026-07-25 — Renders con ruido + composición sobre fondos reales (v2).** Fondos variados,
  iluminación distinta, composición sobre fotografías de fondo reales en vez de blanco liso.

- **2026-07-26 — Migración a BlenderProc.** Máscara ground-truth exacta desde la geometría 3D
  (canal alfa, `film_transparent=True`) en vez de adivinar el contorno con Otsu/SAM sobre el render
  ya compuesto — antes se perdía ~12% de renders limpios por bajo contraste en huesos pequeños o
  lejanos.

- **2026-08-13 — Rediseño completo del render sintético: escena de laboratorio.** Composición más
  elaborada (mesa, utillaje, iluminación de estudio) en vez de solo fondo+HDRI. Esta es la etapa que
  da el salto de "objeto flotando sobre una foto" a "objeto en una escena creíble".

- **2026-10-22 — Material procedural multicapa (`bone_material.py`), preparado para la charla.**
  Motivación explícita: el render seguía viéndose "plástico pintado" porque `apply_procedural_dirt`
  (2026-07-26) solo modulaba color base + roughness, sin bump/normal ni subsurface. **Importante:**
  ya se sabía desde el 2026-08-14 que más realismo de render no mueve el recall de `hueso_plano` —
  el cuello de botella ahí es volumen de foto real, no fidelidad del shader. Este trabajo es un
  experimento aparte (`base_datos_osea/blender/`), sin tocar el pipeline de entrenamiento ni
  `render_bone_bproc.py` en ese momento.
  - Paleta real extraída de 426 fotos reales ya segmentadas (KMeans, 8 colores de 21.7M píxeles de
    hueso real), no inventada a mano.
  - Material multicapa: color macro de la paleta real + manchas/vetas micro + porosidad (bump real
    vía Voronoi → Normal del BSDF, la capa que antes faltaba del todo) + roughness de dos octavas +
    subsurface sutil.
  - Integrado en producción el mismo día en una sesión posterior: `render_bone_bproc.py` ya llama a
    `apply_layered_bone_material` en vez de `apply_procedural_dirt`.
  - Tras feedback del usuario ("poca suciedad, poco realismo — a veces tienen manchas de moho, de
    minerales, de contacto"): tres capas nuevas (commit `f2024e8`) — mineralización (manchas grandes
    no ligadas a cavidad), moho (presente en ~40% de las piezas, no todas), mancha de contacto
    (asimétrica, no centrada). Todo escalado por un único factor `weathering` por pieza, para que el
    mismo código produzca tanto piezas casi limpias como muy desgastadas.
  - **Textura real de superficie** (commit `71dcce3`): propuesta del usuario de partir de una
    textura de hueso real (porosa, tileable, aportada por el usuario, CC0) en vez de solo ruido
    procedural, con las capas anteriores (AO, manchas, bump) encima. Rompe a propósito la garantía
    de "100% procedural sin descargas" que tenía el módulo — decisión consciente, documentada en el
    docstring.

## Métricas: qué se movió y qué no

El ruido de render y la escena de laboratorio sí movieron métricas de evaluación en fotos reales
(ver tabla del eje de segmentación: banco sintético fijo pasa de no medirse con grounded_sam a
0.7685–0.8241 según modelo, tras el salto de segmentación + recorte). El material procedural de
octubre **no se ha medido todavía contra el banco real** — es un experimento visual hecho para
la charla, pendiente #77 del historial es justo "relanzar el dataset sintético completo con el
material actualizado si se decide que reemplaza a los renders ya generados". Y ya se sabe de
antemano (2026-08-14) que el cuello de botella de `hueso_plano` es volumen de foto real, no
fidelidad de shader — así que no se espera que esta etapa mueva esa métrica por sí sola.

## Progresión visual (nuevo): el MISMO fémur a través de las 5 etapas

`progresion/` — en vez de ejemplos sueltos de huesos distintos, esta carpeta sigue **el mismo
espécimen** (`homo_sapiens/femur/sketchfab_0647ff9a00924f40a06e08361cea176c`) desde el render más
cutre hasta el material de octubre, para que la comparación sea limpia (mismo hueso, misma pose
aproximada) y no haya que fiarse de la memoria para ver el salto. Imágenes reales en disco de cada
etapa — no se generó nada nuevo, ya estaban en `renders/`, `renders_noisy_composited/`,
`renders_noisy_bproc/`, `renders_lab/` y `blender/textures/`.

1. `1_estudio_limpio_render_bone.jpg` — **etapa 1**, `render_bone.py` (sesión previa): fondo blanco
   liso, luz suave omnidireccional. El "render de juguete" — buen contorno, cero variación.
2. `2_ruido_compuesto_render_bone_noisy.jpg` — **etapa 2** (2026-07-24), `render_bone_noisy.py`:
   compuesto sobre fondo fotográfico real, con sombra de contacto. Primer intento de romper la
   limpieza de estudio.
3. `3_blenderproc_hdri_render_bone_bproc.jpg` — **etapa 3** (2026-07-26), `render_bone_bproc.py`:
   migración a BlenderProc, HDRI real para reflejos/iluminación ambiente, suciedad procedural
   todavía solo color+roughness. Fondo a cuadros = canal alfa real (máscara ground-truth exacta,
   la mejora clave de esta etapa).
4. `4_escena_laboratorio_render_bone_lab.jpg` — **etapa 4** (2026-08-13), `render_bone_lab.py`:
   física real (el hueso se SUELTA sobre la mesa, pose de reposo creíble) + suciedad dirigida por
   curvatura de la malla (nodo Pointiness, igual que un mapa de curvatura de Substance Painter —
   la suciedad se concentra en huecos/zonas cóncavas, no al azar).
5. `5_textura_real_superficie_octubre.jpg` — **etapa 5** (2026-10-22), `bone_material.py`: ya no es
   un render completo sino la textura base en sí — superficie porosa escaneada real (CC0) con las
   capas de mineralización/moho/mancha de contacto encima, sustituye al ruido procedural puro.
6. `6_referencia_foto_real.jpg` — foto real de fémur (banco de fotos aceptadas), para comparar el
   punto de llegada contra el objetivo real.

**Para la charla**: contar que el salto más grande en métricas vino de la etapa 3 (máscara exacta +
segmentación con recorte, ver eje de segmentación) — las etapas 4 y 5 son mejoras de verosimilitud
visual, importantes para la narrativa pero sin medición propia todavía contra el banco real (ver
sección de métricas arriba).

## Progresión alternativa (nuevo): fémur de gorila, pose flotante → vista de mesa

`progresion_gorila/` — mismo espécimen (`gorilla_gorilla/femur/sketchfab_3fbe6a09153b4a0cb38d15005726d8ff`)
a través de **5 fases reales** (no 4 — la malla original seguía en disco en
`data/meshes/gorilla_gorilla/femur/.../gorilla_femur_vcu_3d_4321.glb`, así que la fase del material
de octubre se pudo renderizar de verdad, en vivo, con `blenderproc run blender/test_bone_material.py`
contra el código actual de `bone_material.py` — no es un recorte reciclado de otra sesión).
Ordenado como pidió el usuario: primero las fases con **pose flotante ilógica** (mismo encuadre
aproximado — el hueso aparece suspendido en el aire sin relación física con lo que le rodea, propio
de un render de entrenamiento, no de una foto real), y al final la **vista de mesa**, la única con
una pose de reposo físicamente lógica.

1. `1_pose_flotante_estudio_limpio.jpg` — fase 1, `render_bone.py`: fondo blanco liso.
2. `2_pose_flotante_ruido_compuesto.jpg` — fase 2 (2026-07-24): mismo encuadre flotante, ya con
   ruido y fondo real compuesto.
3. `3_pose_flotante_blenderproc_hdri.jpg` — fase 3 (2026-07-26): mismo encuadre, BlenderProc +
   HDRI real (fondo a cuadros = canal alfa, el hueso sigue sin apoyarse en nada).
4. `4_pose_flotante_material_octubre.jpg` — fase 4 (2026-10-22), **render nuevo generado para esta
   charla**: mismo encuadre flotante, pero con `apply_layered_bone_material` actual — paleta real +
   manchas/vetas + porosidad por bump + textura real escaneada encima. Comparar directamente con la
   fase 3: mismo tipo de pose, solo cambia el material.
5. `5_vista_mesa_escena_laboratorio.jpg` — fase 5 (2026-08-13), `render_bone_lab.py`: el hueso se
   SUELTA de verdad sobre una mesa (rigid body) y se fotografía desde arriba como lo haría una
   persona de pie — la única fase en la que la pose tiene sentido físico, no solo visual (usa el
   material anterior a octubre, generado antes de esa mejora — si se quisiera "todo junto", habría
   que re-renderizar `render_bone_lab.py` con el material de octubre, no hecho todavía).

Útil para la charla como remate: "las fases 1 a 4 mejoran textura y luz, pero el objeto sigue
flotando en el vacío — el salto real a 'foto de verdad' es cuando además la física decide la pose".

## Imágenes incluidas

1. `01_render_limpio_sin_ruido.png` — etapa 1, fondo blanco, sin ruido ni composición.
2. `02_render_ruido_compuesto.jpg` — etapa 2/3, ruido + composición sobre fondo real (post
   BlenderProc).
3. `03_render_escena_laboratorio.png` — etapa 4, escena de laboratorio completa (2026-08-13).
4. `04_textura_real_superficie_hueso.png` — la textura base real (CC0, aportada por el usuario)
   usada desde el commit `71dcce3` como capa de superficie.
5. `05_foto_real_hueso_plano.jpg` / `06_foto_real_vertebra.jpg` — ejemplos del lado "real" del
   gap, del banco `real_world_benchmark/`, para comparar contra los renders.

No se encontró en disco el `collage_variabilidad.png` citado en el historial (4 seeds del mismo
húmero, de casi limpio a muy manchado) ni el antes/después del material en las 4 mallas de prueba
(húmero, fémur humano, fémur de gorila, cráneo de bos taurus) — se mostraron en el chat de la
sesión pero no se guardaron como fichero. Si se quieren para la charla, regenerar con
`base_datos_osea/blender/test_bone_material.py` / `test_bone_material_before.py`.
