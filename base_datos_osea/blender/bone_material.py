"""Material procedural de hueso, multicapa, para los renders sintéticos
(render_bone_bproc.py). Reemplaza a apply_procedural_dirt (color+roughness
planos) por un acabado físico con más capas, usando la paleta real extraída
por extract_bone_palette.py en vez de colores fijos a mano.

Determinista (mismo seed -> mismo material). 100% procedural, sin assets
externos: la base de color es ruido de Blender (Noise -> ColorRamp) teñido
con 2-3 colores de la paleta real extraída de fotos (extract_bone_palette.py),
no una textura fotografiada. Se probó una textura real de superficie
(textures/bone_base_diffuse.png, CC0) proyectada en triplanar sobre esta
base -- descartada 2026-10-24, el resultado se veía peor que el ruido
procedural (demasiado plana/artificial incluso con el bump derivado de su
canal de color, ver HISTORIAL.md). El código de esa capa ya no está aquí,
solo el archivo de textura se queda en disco sin usarse.

Capas:
  1. Color base — ruido procedural (ColorRamp con paradas en 0/0.5/1),
     teñido con 2-3 colores de la paleta real extraída de fotos
     (extract_bone_palette.py) para variar tono entre piezas/especies sin
     tener que fotografiar cada una.
  2. Manchas/vetas micro — suciedad localizada oscura, concentrada en
     cavidades (AO), mezclada SOBRE el color base.
  2b. Mineralización/moho/mancha de contacto — variabilidad de estado de
      conservación (weathering), no todas las piezas igual de "limpias".
  3. Porosidad (bump) — relieve físico procedural: Voronoi de alta
     frecuencia -> Bump -> Normal del BSDF, más marcado en cavidades y
     crestas.
  4. Roughness combinando macro+micro (dos octavas, no una sola).
  5. Subsurface — hueso deja pasar algo de luz en bordes finos; opaco total
     se ve falso. Peso bajo, tintado con el color macro.
  6. Specular/IOR por zona + barniz/pulido (clearcoat) — el specular del
     Principled BSDF se queda en 0.5 (dieléctrico por defecto) en las capas
     1-5, igual en poro que en hueso compacto, lo cual no es físico: una
     superficie porosa dispersa la luz (specular bajo, se ve mate) y una
     zona compacta/pulida la refleja más (specular más alto). Reutiliza la
     misma señal de cavidad (`cavity_ramp`) que ya dirige dónde va la
     suciedad, esta vez para bajar el specular en huecos. Un subconjunto de
     piezas (30%) sortea además un barniz/pulido sutil (Coat), simulando
     tanto una pieza de museo preparada (barnizada) como hueso natural
     pulido por manipulación repetida — no todas las piezas lo tienen.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import bpy
import numpy as np

PALETTE_PATH = Path(__file__).resolve().parent / "bone_palette.json"

_palette_cache: dict | None = None


def load_palette() -> dict:
    global _palette_cache
    if _palette_cache is None:
        if not PALETTE_PATH.exists():
            raise FileNotFoundError(
                f"Falta {PALETTE_PATH} — ejecuta antes "
                "`python3 base_datos_osea/blender/extract_bone_palette.py`"
            )
        _palette_cache = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
    return _palette_cache


def _weighted_pick(rng: random.Random, clusters: list[dict], k: int) -> list[dict]:
    """k colores de la paleta CON diversidad de luminosidad.

    Un pick puramente proporcional al peso real puede sacar k tonos muy
    parecidos (los 3 clusters más pesados del banco de fotos reales son
    justo los 3 más claros -- sumados, más de la mitad del peso total), y
    entonces el material sale casi plano sin que ningún parámetro esté mal:
    simplemente le tocó un sorteo con poca variedad. Para que la capa macro
    SIEMPRE tenga contraste de verdad, se reparte la paleta en k franjas de
    luminosidad (clara/media/oscura/...) y se sortea un color por franja,
    con probabilidad proporcional al peso DENTRO de esa franja -- conserva
    el "qué tono de claro es más probable" del dato real sin perder la
    variedad clara-a-oscura.

    Se descartan los clusters casi negros (suma de canales < 0.3) antes de
    repartir en franjas -- la paleta extraída de fotos trae algún cluster
    de sombra/fondo contaminado (visto: uno a 0.05/0.04/0.03, casi negro),
    y si cae en la franja oscura se usa como TONO BASE del hueso, que no
    debe pasar de marrón oscuro/amarillento -- negro en la capa 1 solo se
    debe ver si lo aporta el moho/mancha de la capa 2b, no el color macro."""
    MIN_LUMA = 0.3
    filtered = [c for c in clusters if sum(c["rgb"]) >= MIN_LUMA]
    pool = sorted(filtered or clusters, key=lambda c: sum(c["rgb"]))  # oscuro -> claro
    n = min(k, len(pool))
    bands = np.array_split(np.arange(len(pool)), n) if n > 1 else [np.arange(len(pool))]

    picked = []
    for band_idx in bands:
        band = [pool[i] for i in band_idx]
        weights = [c["weight"] for c in band]
        total = sum(weights)
        r = rng.uniform(0, total)
        acc = 0.0
        for c, w in zip(band, weights):
            acc += w
            if r <= acc:
                picked.append(c)
                break
        else:
            picked.append(band[-1])
    return picked


def build_layered_bone_material(
    rng: random.Random,
    bone_size: float = 1.0,
    stop_after: str | None = None,
    palette: dict | None = None,
) -> "bpy.types.Material":
    """Construye el material multicapa UNA VEZ. `rng` debe ser un
    random.Random ya sembrado por el caller, para que todo el render sea
    reproducible con el mismo --seed. `bone_size` es una medida real del
    tamaño de la malla (p.ej. el `radius` que ya calcula render_bone_bproc.py)
    -- hace falta para que los parámetros Scale de los nodos de ruido, que
    están pensados para coordenadas normalizadas 0-1, no queden casi sin
    variación visible al aplicarse sobre World en metros reales.

    Antes se llamaba una vez POR CADA sub-malla del glTF (un hueso importado
    no es un único objeto, son ~30 piezas separadas) -- cada una se quedaba
    con su propio material independiente, con sus propias coordenadas de
    ruido "Generated" normalizadas al bounding box de esa pieza suelta. El
    resultado: cortes de color visibles exactamente en las uniones entre
    piezas, como un patchwork mal cosido, en vez de una variación continua
    de tono a lo largo de todo el hueso. Fix: UN material compartido, con
    las coordenadas de ruido explícitamente en espacio World (continuo para
    todas las piezas a la vez, a diferencia del espacio local/Generated que
    cada sub-malla trae de fábrica).

    `stop_after` es solo para material de diagnóstico/charla (progresiones
    "una capa a la vez"): corta la construcción justo después de conectar
    esa capa al BSDF y devuelve el material tal cual, sin las capas
    posteriores. Valores: "color" (capa 1), "stains" (+ capa 2), "weathering"
    (+ capa 2b), "bump" (+ capa 3). None (por defecto) = material completo,
    comportamiento sin cambios.

    `palette` permite pasar una paleta ya cargada (mismo formato que
    load_palette()) en vez de la de producción -- solo para comparar/
    depurar con otra gama de color (p.ej. una paleta sacada de una sola
    foto de referencia). None (por defecto) = load_palette(), sin cambios."""
    if palette is None:
        palette = load_palette()

    mat = bpy.data.materials.new("bone_layered")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    # Coordenadas compartidas: posición en espacio World (nodo Geometry), no
    # el "Generated"/"Object" de ShaderNodeTexCoord -- esos normalizan al
    # bounding box o al espacio local de CADA objeto por separado, que es
    # justo lo que causaba los cortes entre piezas. ShaderNodeTexCoord no
    # expone una salida "World" directa en esta versión de Blender;
    # Geometry.Position sí da la posición real ya transformada, continua
    # entre todas las piezas sin importar el origen/transform local de cada
    # una.
    #
    # Esa posición viene en metros reales (un fémur puede medir ~0.4m, un
    # hueso pequeño ~0.02m) -- un Mapping node la reescala por 1/bone_size
    # antes de los nodos de ruido, para que los valores de Scale (pensados
    # para 0-1) den la misma densidad de variación visual sin importar el
    # tamaño real de la pieza.
    geometry = nodes.new("ShaderNodeNewGeometry")
    mapping = nodes.new("ShaderNodeMapping")
    inv_size = 1.0 / max(bone_size, 1e-6)
    mapping.inputs["Scale"].default_value = (inv_size, inv_size, inv_size)
    links.new(geometry.outputs["Position"], mapping.inputs["Vector"])
    world_coord = mapping.outputs["Vector"]

    # ── Capa 1: color macro, de la paleta real (2-3 tonos mezclados) ──
    macro_colors = _weighted_pick(rng, palette["clusters"], k=3)
    macro_noise = nodes.new("ShaderNodeTexNoise")
    links.new(world_coord, macro_noise.inputs["Vector"])
    macro_noise.inputs["Scale"].default_value = rng.uniform(1.5, 4.0)  # baja frecuencia = manchas grandes
    macro_noise.inputs["Detail"].default_value = rng.uniform(1.0, 3.0)
    macro_noise.inputs["Roughness"].default_value = rng.uniform(0.3, 0.6)

    # El Fac de un Noise Texture de pocas octavas rara vez visita los
    # extremos 0/1 -- se queda concentrado cerca de 0.5. Poner ahí los
    # colores de la paleta (como se hacía antes) significa que casi toda la
    # superficie cae en el color del medio y los extremos casi no se ven --
    # el material sale plano aunque los colores elegidos sean bien
    # distintos. Un Map Range estira el rango que el ruido SÍ visita de
    # verdad a 0-1 completo, para que el ColorRamp (con paradas en 0/0.5/1)
    # aproveche todo el contraste disponible.
    macro_remap = nodes.new("ShaderNodeMapRange")
    macro_remap.inputs["From Min"].default_value = 0.35
    macro_remap.inputs["From Max"].default_value = 0.65
    macro_remap.clamp = True
    links.new(macro_noise.outputs["Fac"], macro_remap.inputs["Value"])

    macro_ramp = nodes.new("ShaderNodeValToRGB")
    macro_ramp.label = "channel_macro_clean"
    macro_ramp.color_ramp.interpolation = "EASE"  # transición suave, no manchas con borde duro
    macro_ramp.color_ramp.elements[0].position = 0.0
    macro_ramp.color_ramp.elements[0].color = (*macro_colors[0]["rgb"], 1.0)
    macro_ramp.color_ramp.elements[1].position = 1.0
    macro_ramp.color_ramp.elements[1].color = (*macro_colors[1]["rgb"], 1.0)
    if len(macro_colors) > 2:
        mid = macro_ramp.color_ramp.elements.new(0.5)
        mid.color = (*macro_colors[2]["rgb"], 1.0)
    links.new(macro_remap.outputs["Result"], macro_ramp.inputs["Fac"])

    if stop_after == "color":
        links.new(macro_ramp.outputs["Color"], bsdf.inputs["Base Color"])
        return mat

    # ── Señal de cavidad (como un mapa de AO/curvatura de Substance
    # Painter) -- la suciedad y la porosidad de un hueso real NO están
    # repartidas al azar: se acumulan en huecos, grietas y surcos.
    #
    # Primer intento con `Pointiness` del nodo Geometry (curvatura POR
    # VÉRTICE): en mallas escaneadas/retopologizadas de Sketchfab con poca
    # densidad de vértices sale casi plano en toda la pieza -- no hay
    # suficiente resolución local para que Blender detecte huecos/crestas
    # reales, confirmado con un render de diagnóstico (el patrón de Voronoi
    # salía parejo en toda la superficie, sin ninguna zona más marcada que
    # otra). Fix: Ambient Occlusion (`ShaderNodeAmbientOcclusion`, ray-
    # traced) en vez de curvatura por vértice -- no depende de la densidad
    # de la malla, encuentra huecos/grietas REALES por oclusión entre
    # superficies (igual que el "dirt pass" de un bake de AO en Substance
    # Painter). AO bajo = superficie ocluida por otra cercana = hueco real.
    ao_node = nodes.new("ShaderNodeAmbientOcclusion")
    ao_node.inputs["Distance"].default_value = max(bone_size * 0.08, 0.002)
    ao_node.samples = 8

    cavity_ramp = nodes.new("ShaderNodeValToRGB")
    cavity_ramp.label = "channel_cavity_dirtpass"
    cavity_ramp.color_ramp.elements[0].position = 0.35  # más contraste que el AO crudo
    cavity_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)  # ocluido/hueco = máxima suciedad
    cavity_ramp.color_ramp.elements[1].position = 0.75
    cavity_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)  # expuesto = limpio
    links.new(ao_node.outputs["AO"], cavity_ramp.inputs["Fac"])

    # Las crestas de inserción muscular sí dependen de curvatura (convexo),
    # no de oclusión -- Pointiness se deja aquí como contribución secundaria
    # y débil (mejor que nada, aunque sea sutil en estas mallas).
    ridge_ramp = nodes.new("ShaderNodeValToRGB")
    ridge_ramp.color_ramp.elements[0].position = 0.55
    ridge_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ridge_ramp.color_ramp.elements[1].position = 0.75
    ridge_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # convexo marcado = cresta
    links.new(geometry.outputs["Pointiness"], ridge_ramp.inputs["Fac"])

    # ── Capa 2: manchas/vetas, concentradas en las cavidades (no al azar) ──
    micro_noise = nodes.new("ShaderNodeTexNoise")
    links.new(world_coord, micro_noise.inputs["Vector"])
    micro_noise.inputs["Scale"].default_value = rng.uniform(10.0, 28.0)
    micro_noise.inputs["Detail"].default_value = rng.uniform(3.0, 6.0)
    micro_noise.inputs["Roughness"].default_value = rng.uniform(0.5, 0.8)

    # La mancha aparece donde HAY cavidad Y el ruido orgánico también marca
    # esa zona -- sin el ruido, la suciedad seguiría el relieve de forma
    # demasiado perfecta/geométrica; sin la cavidad, sería ruido puro sin
    # relación con la forma (el problema original).
    stain_mix = nodes.new("ShaderNodeMixRGB")
    stain_mix.blend_type = "MULTIPLY"
    stain_mix.inputs["Fac"].default_value = 1.0
    links.new(micro_noise.outputs["Fac"], stain_mix.inputs["Color1"])
    links.new(cavity_ramp.outputs["Color"], stain_mix.inputs["Color2"])

    # BUG encontrado 2026-10-24 (se verificó aislando este canal con un
    # render de diagnóstico): un ColorRamp no extrapola -- un Fac por debajo
    # de la posición del primer stop se clampea al color de ESE stop, no a
    # "sin cambio". stain_mix vale ~0 (negro) en case toda la superficie
    # expuesta (cavity_ramp~0 ahi, ver más arriba), así que con el oscuro en
    # el stop bajo y el blanco en el alto, el resultado era justo al revés
    # de lo que dice el comentario: oscuro en case TODA la pieza, y solo
    # "se libra" (blanco/transparente) justo en las cavidades reales -- la
    # suciedad salía en todas partes MENOS donde debía concentrarse.
    stain_ramp = nodes.new("ShaderNodeValToRGB")
    stain_ramp.label = "channel_stain_dark_stop"
    stain_ramp.color_ramp.elements[0].position = rng.uniform(0.15, 0.3)
    stain_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)  # transparente (no tapa el macro)
    stain_ramp.color_ramp.elements[1].position = 0.6
    stain_ramp.color_ramp.elements[1].color = (0.16, 0.11, 0.07, 1.0)  # veta marrón oscuro (no negro), solo donde hay cavidad real -- pedido 2026-10-26
    links.new(stain_mix.outputs["Color"], stain_ramp.inputs["Fac"])

    mix_color = nodes.new("ShaderNodeMixRGB")
    mix_color.blend_type = "MULTIPLY"
    mix_color.inputs["Fac"].default_value = rng.uniform(0.25, 0.5)  # algo más visible que antes, ahora que tiene lógica
    links.new(macro_ramp.outputs["Color"], mix_color.inputs["Color1"])
    links.new(stain_ramp.outputs["Color"], mix_color.inputs["Color2"])

    if stop_after == "stains":
        links.new(mix_color.outputs["Color"], bsdf.inputs["Base Color"])
        return mat

    # ── Capa 2b: estado de conservación -- un hueso real casi nunca está
    # "limpio de museo" homogéneo. Varía mucho pieza a pieza: mineralización
    # (óxidos de hierro/manganeso infiltrados desde el sedimento, parches
    # grandes de borde difuso, tono óxido/negruzco) y moho/biopelícula
    # (colonias puntuales, verdosas-negras, de borde más duro que un ruido
    # orgánico difuso). Antes solo existía la capa 2 (veta ligada a
    # cavidad/AO) con opacidad baja (0.25-0.5) -- consistentemente sutil, sin
    # variedad entre piezas. Aquí cada material sortea su propio "nivel de
    # desgaste" (weathering) para que, con el mismo código, salgan tanto
    # huesos casi limpios como piezas muy manchadas -- la variedad es el
    # punto, no un valor fijo "bien calibrado".
    weathering = rng.uniform(0.15, 1.0)

    # Mineralización: manchas grandes de borde difuso, NO ligadas a
    # cavidad/AO (a diferencia de la capa 2) -- en hueso real, la
    # infiltración mineral desde el sedimento tizna superficies planas igual
    # que huecos, a veces más. Voronoi de baja frecuencia + Smooth F1 da
    # blotches orgánicos de tamaño grande en vez del patrón puntual de un
    # Noise.
    mineral_voronoi = nodes.new("ShaderNodeTexVoronoi")
    links.new(world_coord, mineral_voronoi.inputs["Vector"])
    mineral_voronoi.inputs["Scale"].default_value = rng.uniform(0.8, 2.5)  # blotches grandes
    mineral_voronoi.feature = "SMOOTH_F1"

    mineral_ramp = nodes.new("ShaderNodeValToRGB")
    mineral_ramp.color_ramp.interpolation = "EASE"
    mineral_ramp.color_ramp.elements[0].position = rng.uniform(0.2, 0.4)
    # óxido de hierro/manganeso: de marrón-rojizo a casi negro según la pieza
    mineral_tone = rng.uniform(0.0, 1.0)
    mineral_dark = (
        0.10 + 0.08 * mineral_tone,
        0.05 + 0.02 * mineral_tone,
        0.03,
        1.0,
    )
    mineral_ramp.color_ramp.elements[0].color = mineral_dark
    mineral_ramp.color_ramp.elements[1].position = rng.uniform(0.55, 0.75)
    mineral_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # transparente
    links.new(mineral_voronoi.outputs["Distance"], mineral_ramp.inputs["Fac"])

    mineral_mix = nodes.new("ShaderNodeMixRGB")
    mineral_mix.blend_type = "MULTIPLY"
    mineral_mix.inputs["Fac"].default_value = weathering * rng.uniform(0.35, 0.8)
    links.new(mix_color.outputs["Color"], mineral_mix.inputs["Color1"])
    links.new(mineral_ramp.outputs["Color"], mineral_mix.inputs["Color2"])

    # Moho/biopelícula: colonias puntuales, no un degradado difuso --
    # Voronoi de alta frecuencia recortado con un umbral duro (no EASE) para
    # que salgan manchas redondeadas discretas, verdosas-negras, en vez de
    # una tinta uniforme. Solo aparece si esta pieza "le tocó" moho en el
    # sorteo (has_mold) -- no todas las piezas deben tenerlo, igual que en
    # la realidad no todo hueso expuesto desarrolla biopelícula visible.
    has_mold = rng.random() < 0.4
    if has_mold:
        mold_voronoi = nodes.new("ShaderNodeTexVoronoi")
        links.new(world_coord, mold_voronoi.inputs["Vector"])
        mold_voronoi.inputs["Scale"].default_value = rng.uniform(3.0, 7.0)
        mold_voronoi.feature = "F1"

        mold_ramp = nodes.new("ShaderNodeValToRGB")
        mold_ramp.color_ramp.elements[0].position = rng.uniform(0.04, 0.12)  # colonias pequeñas y dispersas
        mold_ramp.color_ramp.elements[0].color = (0.14, 0.09, 0.05, 1.0)  # marrón oscuro, no negro -- pedido 2026-10-26
        mold_ramp.color_ramp.elements[1].position = rng.uniform(0.15, 0.22)
        mold_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # transparente
        links.new(mold_voronoi.outputs["Distance"], mold_ramp.inputs["Fac"])

        # El moho se concentra en zonas húmedas/ocluidas (cavidades), igual
        # que la suciedad de la capa 2 -- se modula con la misma cavity_ramp.
        mold_mask = nodes.new("ShaderNodeMixRGB")
        mold_mask.blend_type = "ADD"
        mold_mask.inputs["Fac"].default_value = 0.5
        mold_mask.inputs["Color1"].default_value = (0.5, 0.5, 0.5, 1.0)
        links.new(cavity_ramp.outputs["Color"], mold_mask.inputs["Color2"])

        mold_lerp = nodes.new("ShaderNodeMixRGB")  # blanquea el ramp de moho donde la máscara dice "seco"
        mold_lerp.blend_type = "LIGHTEN"
        links.new(mold_ramp.outputs["Color"], mold_lerp.inputs["Color1"])
        links.new(mold_mask.outputs["Color"], mold_lerp.inputs["Color2"])

        mold_mix = nodes.new("ShaderNodeMixRGB")
        mold_mix.blend_type = "MULTIPLY"
        mold_mix.inputs["Fac"].default_value = weathering * rng.uniform(0.4, 0.9)
        links.new(mineral_mix.outputs["Color"], mold_mix.inputs["Color1"])
        links.new(mold_lerp.outputs["Color"], mold_mix.inputs["Color2"])
        final_color_output = mold_mix.outputs["Color"]
    else:
        final_color_output = mineral_mix.outputs["Color"]

    # Contacto/absorción puntual (p.ej. apoyo contra tierra/metal/otro
    # hueso): un parche oscuro ASIMÉTRICO, no centrado -- se logra
    # desplazando el centro del ruido con un offset aleatorio por eje en vez
    # de usar world_coord sin más (que siempre está centrado en el origen de
    # la malla).
    contact_offset = nodes.new("ShaderNodeVectorMath")
    contact_offset.operation = "ADD"
    contact_offset.inputs[1].default_value = (
        rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)
    )
    links.new(world_coord, contact_offset.inputs[0])

    contact_noise = nodes.new("ShaderNodeTexNoise")
    links.new(contact_offset.outputs["Vector"], contact_noise.inputs["Vector"])
    contact_noise.inputs["Scale"].default_value = rng.uniform(0.6, 1.4)  # una sola mancha grande, no un patrón
    contact_noise.inputs["Detail"].default_value = rng.uniform(1.0, 2.0)

    contact_ramp = nodes.new("ShaderNodeValToRGB")
    contact_ramp.color_ramp.elements[0].position = rng.uniform(0.55, 0.7)  # umbral alto -- mancha chica y localizada
    contact_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    contact_ramp.color_ramp.elements[1].position = rng.uniform(0.72, 0.85)
    contact_ramp.color_ramp.elements[1].color = (0.18, 0.12, 0.08, 1.0)  # mancha de contacto marrón oscuro (no negro) -- pedido 2026-10-26
    links.new(contact_noise.outputs["Fac"], contact_ramp.inputs["Fac"])

    contact_mix = nodes.new("ShaderNodeMixRGB")
    contact_mix.label = "channel_final_dirty"
    contact_mix.blend_type = "MULTIPLY"
    contact_mix.inputs["Fac"].default_value = weathering * rng.uniform(0.3, 0.7)
    links.new(final_color_output, contact_mix.inputs["Color1"])
    links.new(contact_ramp.outputs["Color"], contact_mix.inputs["Color2"])

    links.new(contact_mix.outputs["Color"], bsdf.inputs["Base Color"])

    if stop_after == "weathering":
        return mat

    # ── Capa 3: porosidad física (bump real), más marcada en las cavidades
    # (poros reales se notan más en huecos) y en las crestas de inserción
    # muscular (rugosidad por tensión mecánica repetida) ──
    pore_voronoi = nodes.new("ShaderNodeTexVoronoi")
    links.new(world_coord, pore_voronoi.inputs["Vector"])
    pore_voronoi.inputs["Scale"].default_value = rng.uniform(40.0, 90.0)
    pore_voronoi.feature = "SMOOTH_F1"

    pore_mask = nodes.new("ShaderNodeMixRGB")  # cavidad + cresta = "zonas con relieve", valle liso queda fuera
    pore_mask.blend_type = "ADD"
    pore_mask.inputs["Fac"].default_value = 0.6
    links.new(cavity_ramp.outputs["Color"], pore_mask.inputs["Color1"])
    links.new(ridge_ramp.outputs["Color"], pore_mask.inputs["Color2"])

    pore_mix = nodes.new("ShaderNodeMixRGB")
    pore_mix.label = "channel_pore_bump"
    pore_mix.blend_type = "MULTIPLY"
    pore_mix.inputs["Fac"].default_value = 1.0
    links.new(pore_voronoi.outputs["Distance"], pore_mix.inputs["Color1"])
    links.new(pore_mask.outputs["Color"], pore_mix.inputs["Color2"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = rng.uniform(0.12, 0.3)  # algo más fuerte, ahora dirigido
    bump.inputs["Distance"].default_value = rng.uniform(0.3, 0.8)
    links.new(pore_mix.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    if stop_after == "bump":
        return mat

    # ── Capa 4: roughness combinando macro + micro (dos octavas) ──
    rough_mix = nodes.new("ShaderNodeMixRGB")
    rough_mix.blend_type = "ADD"
    rough_mix.inputs["Fac"].default_value = 0.5
    rough_ramp_macro = nodes.new("ShaderNodeValToRGB")
    rough_ramp_macro.color_ramp.elements[0].position = 0.4
    rough_ramp_macro.color_ramp.elements[0].color = (0.3, 0.3, 0.3, 1.0)
    rough_ramp_macro.color_ramp.elements[1].position = 1.0
    rough_ramp_macro.color_ramp.elements[1].color = (0.75, 0.75, 0.75, 1.0)
    links.new(macro_noise.outputs["Fac"], rough_ramp_macro.inputs["Fac"])

    rough_ramp_micro = nodes.new("ShaderNodeValToRGB")
    rough_ramp_micro.color_ramp.elements[0].position = 0.4
    rough_ramp_micro.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    rough_ramp_micro.color_ramp.elements[1].position = 1.0
    rough_ramp_micro.color_ramp.elements[1].color = (0.2, 0.2, 0.2, 1.0)
    links.new(micro_noise.outputs["Fac"], rough_ramp_micro.inputs["Fac"])

    links.new(rough_ramp_macro.outputs["Color"], rough_mix.inputs["Color1"])
    links.new(rough_ramp_micro.outputs["Color"], rough_mix.inputs["Color2"])
    links.new(rough_mix.outputs["Color"], bsdf.inputs["Roughness"])

    # ── Capa 5: subsurface sutil, tintado con el macro ──
    if "Subsurface Weight" in bsdf.inputs:  # nombre del socket varía por versión de Blender
        bsdf.inputs["Subsurface Weight"].default_value = rng.uniform(0.02, 0.08)
    elif "Subsurface" in bsdf.inputs:
        bsdf.inputs["Subsurface"].default_value = rng.uniform(0.02, 0.08)
    if "Subsurface Radius" in bsdf.inputs:
        avg = [sum(c["rgb"][i] for c in macro_colors) / len(macro_colors) for i in range(3)]
        bsdf.inputs["Subsurface Radius"].default_value = (avg[0] * 0.5, avg[1] * 0.4, avg[2] * 0.3)

    # ── Capa 6: specular/IOR por zona ──
    # cavity_ramp ya vale 1.0 en huecos (máxima suciedad) y 0.0 en superficie
    # expuesta (ver capa 2) -- reusar esa misma señal para el specular: hueco
    # = dispersa la luz = specular bajo (mate); expuesto/compacto = algo más
    # de brillo especular, como la cortical lisa de un hueso real.
    spec_ramp = nodes.new("ShaderNodeValToRGB")
    spec_ramp.label = "channel_specular_zone"
    spec_ramp.color_ramp.elements[0].position = 0.0  # expuesto/compacto
    spec_ramp.color_ramp.elements[0].color = (
        rng.uniform(0.45, 0.6),) * 3 + (1.0,)
    spec_ramp.color_ramp.elements[1].position = 1.0  # hueco/poroso
    spec_ramp.color_ramp.elements[1].color = (rng.uniform(0.1, 0.22),) * 3 + (1.0,)
    links.new(cavity_ramp.outputs["Color"], spec_ramp.inputs["Fac"])

    specular_socket = "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    if specular_socket in bsdf.inputs:
        links.new(spec_ramp.outputs["Color"], bsdf.inputs[specular_socket])

    # ── Capa 6b: barniz/pulido (Coat) en un subconjunto de piezas ──
    # No todo hueso tiene esta capa -- pieza de museo barnizada o hueso
    # pulido por manipulación repetida, frente a una pieza en bruto sin
    # tratar. Mismo patrón que has_mold: sorteo binario por pieza, no un
    # valor fijo "bien calibrado" para todas.
    has_polish = rng.random() < 0.3
    coat_weight_socket = "Coat Weight" if "Coat Weight" in bsdf.inputs else "Clearcoat"
    coat_rough_socket = "Coat Roughness" if "Coat Roughness" in bsdf.inputs else "Clearcoat Roughness"
    if has_polish and coat_weight_socket in bsdf.inputs:
        bsdf.inputs[coat_weight_socket].default_value = rng.uniform(0.15, 0.35)
        if coat_rough_socket in bsdf.inputs:
            bsdf.inputs[coat_rough_socket].default_value = rng.uniform(0.1, 0.25)

    return mat


def apply_layered_bone_material(
    objs: list,
    rng: random.Random,
    bone_size: float = 1.0,
    stop_after: str | None = None,
    palette: dict | None = None,
) -> None:
    """Construye el material UNA VEZ y lo asigna al MISMO datablock en todas
    las piezas de `objs` (lista de objetos bproc o bpy de una misma malla
    compuesta) -- ver la nota en build_layered_bone_material sobre por qué
    no puede ser un material nuevo por pieza. `bone_size`, `stop_after` y
    `palette` ver misma nota en build_layered_bone_material."""
    mat = build_layered_bone_material(rng, bone_size=bone_size, stop_after=stop_after, palette=palette)
    for obj in objs:
        mesh = obj.data if hasattr(obj, "data") else obj.blender_obj.data
        mesh.materials.clear()
        mesh.materials.append(mat)
