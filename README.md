# base_datos_osea

Catálogo y descarga ordenada de mallas 3D de huesos desde varias fuentes abiertas, pensado como
fuente de datos para `osteolab-ml-platform` (renders en Blender → entrenamiento del clasificador
de huesos).

## Estado y próximos pasos (última sesión: 2026-07-23)

**Léeme primero si retomas esto.** Resumen de por dónde íbamos:

1. **El repo se movió de sitio.** Antes vivía en `/dev/base_datos_osea` — un error mío: `/dev` es
   `devtmpfs` (sistema de ficheros de dispositivos del kernel), **no persiste tras reiniciar WSL**.
   Ahora vive en `/root/dev/base_datos_osea` (disco real, junto a `osteolab-ml-platform` y el resto
   de proyectos). Si algo referencia la ruta vieja, está obsoleto.
   - **Verificado 2026-07-23**: el `venv/` sobrevivió el `mv` desde `/dev` sin problemas
     (`venv/bin/python -c "import morphosource"` funciona).
   - Blender vive en `/opt/blender` (disco real, no se movió, no debería tener problemas).

2. **Se encontró un problema serio de calidad de datos.** El fémur de *Homo sapiens* descargado de
   MorphoSource (`000031310`) resultó ser un **fragmento** (cabeza + diáfisis, sin cóndilos
   distales) — se verificó midiendo el perfil de anchura a lo largo del eje principal: crece de
   ~5mm en un extremo a ~60mm en el otro y ahí se corta, en vez de ensancharse en los dos extremos
   como un fémur completo. MorphoSource lo etiquetó simplemente `"femur"`, sin ninguna palabra como
   `"proximal"`/`"fragment"` que el clasificador de texto de `bones.py` pudiera detectar — **ojo,
   el flag `is_partial` de MorphoSource/`bones.py` NO es fiable para pillar fragmentos sin avisar**.
   El cráneo de sapiens (`000678985`) sí estaba correctamente marcado parcial desde el principio
   (solo hueso frontal) — ese no es sorpresa, está documentado más abajo.

3. **Por eso se decidió pivotar a Sketchfab** como fuente principal para huesos humanos: permite
   ver una miniatura de cada modelo (`GET /v3/models/{uid}` trae `thumbnails.images`) **antes** de
   descargar nada, así que cada candidato se revisa visualmente (Claude mirando la miniatura con
   `Read`) antes de aceptarlo — verificación real de forma, no solo de metadatos de texto.
   `scripts/sketchfab.py` (subcomandos `catalog`/`download`) ya funciona; búsqueda/miniaturas no
   necesitan token, la descarga sí (`SKETCHFAB_API_TOKEN` en `.env`, ya puesto).
   - **14 de 21 huesos objetivo ya verificados visualmente y descargados** vía Sketchfab, todos
     *Homo sapiens* reales, en `data/meshes/homo_sapiens/<hueso>/sketchfab_<uid>/` (formato
     `.glb`, no `.ply` como MorphoSource/Smithsonian — **pendiente confirmar que
     `blender/render_bone.py` importa `.glb` sin cambios**): cráneo, fémur, húmero, mandíbula,
     escápula, clavícula, radio, cúbito, tibia, fíbula, rótula, sacro, vértebra (lumbar), costilla
     (central típica). La mayoría son escaneos reales de especímenes de UNCG Imaging Lab (mismo
     laboratorio que ya hizo el cráneo) o de Eric Bauer, con proveniencia documentada.
   - **Maxilar: sin reemplazo.** Se revisaron 2 candidatos en Sketchfab
     (`47125f083e464d87b23ec748b2679983`, `e59a4be88e8545268dda24c72eed2a0c`) y ambos resultaron
     fragmentos unilaterales (falta un lado de la arcada) — rechazados. Sigue cubierto por
     *H. naledi*.
   - **Pendiente**: quedan por buscar/verificar en Sketchfab los huesos pequeños de mano/pie que
     hoy vienen de *H. sapiens* (MorphoSource) o *H. naledi*: pelvis, metacarpiano I, metatarsiano
     I, falange proximal I, trapecio, sesamoideo. Mismo proceso: buscar → mirar miniatura → añadir
     a `CURATED_CANDIDATES` en `scripts/sketchfab.py` solo si se ve completo.

4. **Auditoría geométrica: hecha para todo lo que se iba a renderizar.** Se revisó visualmente
   (Claude renderizando 1 vista de cada malla y mirándola) las 46 mallas descargadas antes del
   render por lotes. Se encontraron y excluyeron 7 fragmentos/mallas mal etiquetadas que
   `bones.py`/MorphoSource no habían marcado como parciales:
   - `homo_sapiens/cranium/000678985` y `homo_sapiens/femur/000031310` (MorphoSource, ya
     documentados arriba) — reemplazados por sus equivalentes de Sketchfab.
   - `homo_naledi/scapula/000026236` — el propio nombre de fichero decía "fragment"; confirmado
     visualmente (blob irregular sin glenoides/acromion reconocibles).
   - `homo_naledi/fibula/000100688` — carpeta decía "Distal"; confirmado fragmento distal only.
   - `homo_naledi/humerus/000026234` y `homo_naledi/tibia/000014985` — fragmentos (cabeza proximal
     suelta y diáfisis sin epífisis respectivamente), confirmados con varios ángulos de cámara.
   - `homo_naledi/radius/000026239` (fichero `uw-102-025-ulna-...`, el nombre ya no cuadraba con
     la carpeta) — al renderizarlo resultó ser un fragmento pequeño irreconocible, ni radio ni
     cúbito real. Metadatos de MorphoSource no fiables una vez más.
   Estas 5 mallas de naledi quedan en disco (`data/meshes/homo_naledi/...`) pero fuera del render;
   sus huesos ya están cubiertos por el reemplazo de Sketchfab en *H. sapiens*. Las 3 carpetas
   obsoletas de `renders/` (12 vistas, sesión previa) ya se borraron — contaminaban la tabla de
   features con fragmentos etiquetados como buenos (ver punto 6).

5. **Render de Blender: hecho para las 39 mallas verificadas.** `blender/render_bone.py` ahora
   soporta `.glb`/`.gltf` además de `.stl`/`.ply`/`.obj` (el importador de glTF de Blender anida la
   malla bajo Empties con escala/traslación no triviales — sin aplanar ese transform antes de leer
   `obj.data.vertices`, el hueso salía invisible o mal encuadrado; se corrigió aplicando
   `parent_clear` + `transform_apply` tras la importación, para cualquier formato). Con eso:
   **39/39 huesos renderizados, 24 vistas c/u (936 imágenes) en `renders/<especie>/<hueso>/<id>/`**,
   sin fallos. Cubre las 11 especies/huesos de Smithsonian, 8 de naledi (tras excluir los 5
   fragmentos) y 20 de sapiens (14 Sketchfab + 6 MorphoSource ya buenos de antes: metacarpiano I,
   metatarsiano I, pelvis, falange proximal I, sesamoideo, trapecio).
   - **Pendiente**: los huesos pequeños de mano/pie aún sin buscar en Sketchfab (ver punto 3) —
     una vez se añadan, re-renderizarlos igual que el resto.
   - **Pendiente**: decidir qué hacer con maxilar — sigue siendo naledi (única opción tras
     rechazar los 2 candidatos de Sketchfab), sin verificación geométrica adicional del perfil de
     anchura (parece razonablemente completo a ojo, con arcada dental visible).

6. **El entrenamiento real vive en `osteolab-ml-platform`, no aquí** — este repo solo cataloga,
   descarga y renderiza. `src/cv_extractor/build_dataset.py` (en ese otro repo) recorre `renders/`
   de aquí, pasa cada imagen por `extract_features()` (Hu moments, ratios, perfil de anchura) y
   escribe `data/processed/bone_geometric_features.csv`. `src/training/train_geometric.py` entrena
   sobre esa tabla.
   - **Se abandonó clasificar por los 21 huesos finos**: con 1-2 especímenes físicos reales por
     hueso, un clasificador de 21 clases no puede generalizar más allá de "reconozco este objeto
     concreto desde otro ángulo" (confirmado empíricamente, ver debajo). Se pasó a clasificar por
     **9 grupos morfológicos** (`BONE_GROUPS` en `scripts/bones.py`, duplicado en
     `build_dataset.py` del otro repo — mantener ambos en sync): `cranio`, `mandibula_maxilar`,
     `hueso_largo` (fémur/tibia/fíbula/húmero/radio/cúbito/clavícula), `hueso_plano` (escápula),
     `pelvis`, `sacro`, `vertebra`, `costilla`, `hueso_pequeno` (metacarpiano I/metatarsiano
     I/falange proximal I/trapecio/sesamoideo/rótula). Decisiones no obvias: mandíbula/maxilar
     aparte de cráneo (forma muy distinta), sacro aparte de vértebra (aunque sean vértebras
     fusionadas), clavícula con hueso largo (no con la cintura escapular), rótula con hueso pequeño
     (por forma, no por función).
   - **2 grupos solo tenían 1 espécimen físico** (`hueso_plano`/escápula y `pelvis`) — se buscó y
     verificó visualmente un 2º espécimen en Sketchfab para cada uno: escápula de Eric Bauer
     (`0745bbb368b4401db89e73babe440ee8`, 9.8M caras) y pelvis de Oregon State University
     (`35586f343d9c4c6eb813f9006f036595`, ambos ilion + sacro). Ya descargados y renderizados
     (`renders/` ahora tiene **984 imágenes**, no 936).
   - **Hallazgo clave (por qué importa el nº de especímenes)**: un split aleatorio por fila da
     83.8% de accuracy en 9 clases, pero reparte vistas del MISMO espécimen entre train y test —
     mide sobre todo memorización de objetos concretos. Con `GroupKFold` dejando **especímenes
     completos** fuera (nunca el mismo hueso físico en train y test a la vez), la accuracy real cae
     a **65.5%**, y los grupos con solo 2 especímenes (`costilla`, `sacro`, `vertebra`) se
     desploman a **0% de recall** en algunos folds — el modelo nunca vio ningún ejemplo de esa
     clase al entrenar ese fold. Los grupos con más especímenes (`hueso_largo`: 9, `mandibula_maxilar`:
     9, `hueso_pequeno`: 7, `cranio`: 6) sí generalizan razonablemente incluso en esta evaluación
     honesta (F1 0.67–0.87).
   - **Pendiente / próximo paso claro**: buscar un 3º+ espécimen en Sketchfab para `costilla`,
     `sacro` y `vertebra` (0% recall honesto) y luego `pelvis`/`hueso_plano` (2 especímenes, algo
     mejor pero aún pocos) — mismo proceso de siempre (buscar → mirar miniatura → verificar
     completo). Sin esto, la accuracy de esos 3 grupos no significa nada todavía.

No se scrapea ninguna web directamente (varias tienen protección anti-bot). Todo pasa por APIs
oficiales:

- **[MorphoSource](https://www.morphosource.org)** — vía su
  [API REST oficial](https://morphosource.stoplight.io/docs/morphosource-api) y el paquete oficial
  [`morphosource`](https://github.com/Imageomics/pyMorphoSource). Fuente principal, huesos
  individuales de homínidos (*Homo sapiens*, *H. naledi*, *H. neanderthalensis*).
- **[Smithsonian Open Access](https://www.si.edu/openaccess)** — vía `api.si.edu`. Solo aporta
  cráneo/mandíbula (es lo único que tiene digitalizado en 3D de primates), pero en CC0 y con
  descarga directa sin aprobación. Se usa para dar diversidad de especies en esas dos clases, no
  para tapar huecos de otros huesos.
- **NIH 3D Print Exchange** — investigado y descartado: el sitio se rediseñó (ahora
  `3d.nih.gov`, Next.js) y la API pública documentada (`3dprint.nih.gov/developer`) está muerta
  (404 confirmado en 2026-07-22). No es scriptable sin scraping de un sitio con JS del lado
  cliente, así que se dejó fuera.

## Cómo funciona

Tres scripts encadenados:

1. **`scripts/catalog.py`** — consulta la API (búsqueda, no requiere API key) para UNA especie,
   clasifica cada resultado en un hueso canónico (`scripts/bones.py`) y escribe/actualiza
   `catalog/<especie>_media.csv`: una fila por media de MorphoSource, con hueso, lado, tipo de
   media, especie, licencia, titular de derechos, tipo de acuerdo de uso, tamaño de archivo, URL,
   etc. Marca `selected_for_download=True` en el mejor candidato de cada hueso *dentro de esa
   especie*.
2. **`scripts/gap_check.py`** — mira TODOS los `catalog/*_media.csv` a la vez y arbitra entre
   especies: para cada hueso deja seleccionado un único ejemplar (el más completo disponible en
   cualquiera de las especies catalogadas), desmarcando el resto. Así nunca se descargan dos
   especies distintas para tapar el mismo hueso, y si un hueso ya está `done`, esa descarga se
   respeta (no se reemplaza por una "mejor" que aparezca después).
3. **`scripts/download.py`** — lee un catálogo y descarga los ficheros marcados como seleccionados
   (solo los de visibilidad `open`), los descomprime en `data/meshes/<especie>/<hueso>/<media_id>/`
   y anota `download_status`/`local_path`/`downloaded_at` de vuelta en el CSV. Persiste el CSV
   después de cada ítem, así que una interrupción a mitad de descarga no pierde lo ya bajado.

Todo es seguro de re-ejecutar: ni `catalog.py` ni `gap_check.py` pisan el estado de descargas ya
hechas.

## Por qué solo "Open Download"

MorphoSource marca cada media como `open` o `restricted_download`. Lo segundo requiere que un
curador humano apruebe tu solicitud desde la web — no es automatizable. Por eso este catálogo
**solo indexa y descarga media `open`**.

## Estado actual (2026-07-22)

21 de 21 huesos objetivo cubiertos, combinando 2 especies del género *Homo*:

| Hueso | Especie usada | Nota |
|---|---|---|
| cráneo | *Homo sapiens* | parcial (solo hueso frontal) |
| fémur | *Homo sapiens* | completo |
| metacarpiano I | *Homo sapiens* | completo |
| metatarsiano I | *Homo sapiens* | completo |
| pelvis | *Homo sapiens* | completo |
| falange proximal I | *Homo sapiens* | completo |
| trapecio | *Homo sapiens* | completo |
| sesamoideo | *Homo sapiens* | completo |
| húmero, radio, cúbito, clavícula, escápula, tibia, fíbula, maxilar, mandíbula, rótula, sacro, costilla, vértebra | *Homo naledi* | completos (fósiles de la Cámara Dinaledi, Sudáfrica) |

*Homo sapiens* solo tenía en abierto los 8 primeros huesos (dientes, tejido blando y variantes de
pose de un mismo estudio de pelvis excluidos — ver `catalog/homo_sapiens_media.csv`, columna
`category`). El resto no existe en abierto para *H. sapiens*, así que se completó con
*Homo naledi* (233 media abiertos, con casi todo el esqueleto post-craneal digitizado) y, en menor
medida, *Homo neanderthalensis* (18 media, perdió el arbitraje de `gap_check.py` frente a naledi
en todos los huesos donde competían). *Homo sp.* (identificación indeterminada) aporta una
mandíbula completa alternativa no usada (naledi ya cubre mandíbula).

**Mezclar especies para entrenar el clasificador tiene una implicación real**: un húmero de
*H. naledi* no tiene la misma morfología que uno de *H. sapiens* (naledi es más pequeño y con
proporciones distintas). Válido como dataset inicial "hay o no hay hueso X" / forma general, pero
si el objetivo es clasificar específicamente huesos *humanos modernos*, esos 13 huesos de naledi
son marcadores de bajo esfuerzo, no sustitutos — la vía correcta a medio plazo sigue siendo pedir
acceso `restricted_download` a especímenes de *H. sapiens* en morphosource.org.

### Smithsonian: diversidad de cráneo/mandíbula entre primates

`scripts/smithsonian.py` cataloga y descarga cráneo + mandíbula de 6 especies de primate curadas
para dar variedad taxonómica (gran simio, simio menor, mono del Viejo Mundo, mono del Nuevo Mundo,
prosimio): *Pan troglodytes verus*, *Gorilla gorilla gorilla*, *Symphalangus syndactylus*,
*Macaca radiata*, *Alouatta palliata*, *Lemur catta*. Estado actual: **11/12 descargados** (falta
el cráneo de *Lemur catta*, esa especie solo tiene mandíbula digitalizada en 3D en Smithsonian).
Todo en formato `.ply`, licencia **CC0** (dominio público, sin restricciones — mejor que
MorphoSource). Ver `catalog/smithsonian_primates_media.csv`.

A diferencia de MorphoSource, aquí no hay "un ganador por hueso": cada especie se descarga sin
competir con las demás (no pasa por `gap_check.py`), porque el objetivo es diversidad de especies
en cráneo/mandíbula, no completar huesos que faltan.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Necesitas dos API keys (ninguna requiere aprobación manual):

- **MorphoSource** (para descargar, no para catalogar): cuenta en morphosource.org → Perfil →
  Advanced → "View API Key".
- **Smithsonian** (para catalogar; la descarga de ficheros no necesita key): formulario de un
  campo en https://api.data.gov/signup/, la key llega al instante.

Ponlas en `.env` en la raíz del repo (no versionado, ver `.gitignore`):

```
MORPHOSOURCE_API_KEY=tu_key
SI_API_KEY=tu_key
```

## Uso

**MorphoSource** (huesos individuales de homínidos):

```bash
# 1. Catalogar una especie
python scripts/catalog.py --taxon sapiens --species-name "Homo sapiens"
python scripts/catalog.py --taxon naledi --species-name "Homo naledi"
python scripts/catalog.py --taxon neanderthalensis --species-name "Homo neanderthalensis"
# especies sin epíteto GBIF propio (aparecen como "Homo sp." en physical_object_taxonomy_name):
python scripts/catalog.py --taxon Homo --species-name "Homo sp." --exact-taxonomy-name "Homo sp."

# 2. Arbitrar entre todas las especies catalogadas (evita duplicar huesos entre especies)
python scripts/gap_check.py

# 3. Descargar lo seleccionado en cada catálogo
python scripts/download.py --catalog catalog/homo_sapiens_media.csv
python scripts/download.py --catalog catalog/homo_naledi_media.csv
```

**Smithsonian** (diversidad de cráneo/mandíbula entre primates):

```bash
python scripts/smithsonian.py catalog
python scripts/smithsonian.py download
```

## Licencias

Cada fila del catálogo trae su propia `license` — no hay una licencia global del repo.

- **MorphoSource**: también trae `ip_holder`, `permits_commercial_use` y `permits_3d_use` por
  ítem. Algunos (p.ej. el fémur de *H. sapiens*, `000031310`) no tienen licencia Creative Commons
  asignada (`license` vacío) y solo declaran `copyright_statement: In Copyright` — en esos casos
  manda `permits_commercial_use`/`permits_3d_use`, no asumas dominio público. La mayoría del
  material en abierto está en **CC BY-NC 4.0** (sin uso comercial, impresión 3D limitada).
- **Smithsonian**: todo lo descargado es **CC0** (dominio público, sin restricción alguna).

Válido para entrenar modelos y renderizar en Blender para este proyecto, pero revisa el CSV antes
de redistribuir nada o darle uso comercial (por el material de MorphoSource).

## Estructura

```text
scripts/
  bones.py        # taxonomía de huesos canónicos + clasificador de texto libre (compartido)
  catalog.py       # MorphoSource: construye/actualiza el CSV de UNA especie (sin descargar)
  gap_check.py      # MorphoSource: arbitra selección ENTRE catálogos de varias especies
  download.py        # MorphoSource: descarga lo seleccionado en un CSV dado (requiere API key)
  smithsonian.py       # Smithsonian: catálogo + descarga de cráneo/mandíbula por especie
catalog/
  homo_sapiens_media.csv           # el "registro tabla": una fila por media de MorphoSource
  homo_naledi_media.csv
  homo_neanderthalensis_media.csv
  homo_sp_media.csv
  smithsonian_primates_media.csv
data/meshes/                # mallas descargadas (no versionado en git, son binarios grandes)
  <especie>/<hueso>/<media_id o record_id>/
```
