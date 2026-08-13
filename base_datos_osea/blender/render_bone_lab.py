"""Escena de laboratorio: mesa + caída física + cámara de persona de pie + luces
de laboratorio + material del hueso en 2 capas (base + suciedad por curvatura).

Sustituye la órbita completa alrededor del hueso (`render_bone.py`,
`render_bone_bproc.py`) por una escena que imita cómo se fotografiaría de
verdad un hueso en una mesa de laboratorio:

  1. El hueso se SUELTA sobre una mesa (rigid body, física real) y se deja
     asentar — pose de reposo creíble, no una orientación arbitraria fijada
     a mano. Se puede soltar varias veces (`--n-drops`) para variar pose y
     sombra sin repetir siempre la misma postura.
  2. La cámara está restringida a cómo fotografiaría una persona de pie
     mirando hacia abajo: elevación entre `--elevation-min-deg` y
     `--elevation-max-deg` (por defecto 55°-85° sobre la horizontal), nunca
     desde el lateral ni desde abajo. Focal (24-70mm) y sensor (23-36mm)
     variables por disparo, simulando cámaras distintas.
  3. Iluminación combinada: HDRI real de `base_datos_osea/hdris/` (ambiente y
     reflejos, igual que `render_bone_bproc.py`) MÁS 2-3 focos de área en
     ángulo oblicuo sobre la mesa (para que las imperfecciones de la
     superficie generen sombra propia, no se aplanen) MÁS un foco grande y
     suave de relleno (para que no haya sombras completamente negras).
  4. Material del hueso en 2 capas si no trae textura real
     (`has_real_texture`, mismo criterio que la rama bproc): capa base con
     imperfecciones (ruido modulando color/roughness) + capa de suciedad
     dirigida por CURVATURA real de la malla (nodo Geometry → Pointiness,
     análogo al mapa de curvatura de Substance Painter) — la suciedad se
     concentra en zonas cóncavas/huecos, con sentido físico, no repartida
     al azar.
  5. Tras cada caída se verifica que el hueso sigue sobre la mesa y que cada
     disparo de cámara lo deja encuadrado por completo (proyección de las
     esquinas del bbox al espacio de cámara) — reintenta si no.

Usa `blender --background --python` (NO `blenderproc run`): la física de
rigid body necesita un único objeto de colisión por hueso, y
`bpy.ops.object.join()` no es fiable bajo `blenderproc run` sin contexto de
UI real (ver comentario en `render_bone_bproc.py`). Las piezas buenas de la
rama bproc (HDRI, muestras de Cycles, intrínsecos de cámara) se replican a
mano con `bpy` puro — este script no depende del paquete `blenderproc`.

Uso:
    blender --background --python base_datos_osea/blender/render_bone_lab.py -- \\
        --mesh data/meshes/homo_sapiens/femur/.../malla.glb \\
        --out-dir base_datos_osea/renders_lab --prefix homo_sapiens_femur_xxx \\
        --n-drops 3 --views-per-drop 4 --seed 0
"""

import argparse
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

HDRI_DIR = Path(__file__).resolve().parents[1] / "hdris"

BONE_COLOR_PALETTE = [
    (0.85, 0.82, 0.75), (0.62, 0.54, 0.42), (0.70, 0.60, 0.45),
    (0.45, 0.38, 0.30), (0.55, 0.52, 0.48), (0.75, 0.65, 0.50),
    (0.35, 0.28, 0.22), (0.80, 0.75, 0.68),
]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--n-drops", type=int, default=3)
    p.add_argument("--views-per-drop", type=int, default=4)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--fill", type=float, default=0.97)  # gente hace la foto muy cerca -- 0.88 seguía dejando aire de sobra
    p.add_argument("--elevation-min-deg", type=float, default=55.0)
    p.add_argument("--elevation-max-deg", type=float, default=85.0)
    p.add_argument("--drop-frames", type=int, default=90)
    p.add_argument("--max-drop-retries", type=int, default=4)
    p.add_argument("--max-frame-retries", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--light-preset", choices=list(LIGHT_PRESETS.keys()), default="neutro")
    return p.parse_args(argv)


# ---------------------------------------------------------------- malla ----

def import_mesh(path):
    """Copia de render_bone.py: importa, aplana transform (padres/escala del
    glTF) y funde en un único objeto — necesario para tener UN objeto de
    colisión de física por hueso."""
    ext = path.suffix.lower()
    before = set(bpy.data.objects)
    if ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"formato no soportado: {ext}")
    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]

    bpy.ops.object.select_all(action="DESELECT")
    for o in imported:
        o.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if len(imported) > 1:
        bpy.ops.object.join()
        imported = [bpy.context.view_layer.objects.active]
    return imported[0]


def has_real_texture(obj) -> bool:
    """Mismo criterio que render_bone_bproc.py, adaptado a objeto bpy puro
    (aquí ya viene fundido en uno solo por import_mesh, sin wrapper)."""
    mesh = obj.data
    if mesh is None:
        return False
    if mesh.color_attributes and len(mesh.color_attributes) > 0:
        return True
    for mat in mesh.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                return True
    return False


def force_bone_reflectivity(obj):
    """Un hueso real es mate y poroso, nunca brillante -- pero si la malla
    trae SU PROPIO material (has_real_texture=True, se conserva para no
    perder una textura fotográfica real capturada por fotogrametría), ese
    material puede traer un acabado PBR genérico/plástico tal cual vino del
    descargable (visto: un cráneo de Sketchfab salía con brillo especular
    duro, "de plástico"). Se fuerza aquí SIEMPRE, en TODOS los materiales
    del objeto sea cual sea su origen -- conserva color/textura, solo
    corrige cómo refleja la luz. En los materiales procedurales propios
    (Roughness ya conectado a un nodo) esto no tiene efecto: default_value
    se ignora en un input con link, es inofensivo pasar por aquí también."""
    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            continue
        if not bsdf.inputs["Roughness"].is_linked:
            bsdf.inputs["Roughness"].default_value = 0.65
        specular_input = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
        if specular_input is not None and not specular_input.is_linked:
            specular_input.default_value = 0.25
        if not bsdf.inputs["Metallic"].is_linked:
            bsdf.inputs["Metallic"].default_value = 0.0


def center_on_bbox(obj):
    """import_mesh() no recentra (a diferencia de render_bone.py, que usa el
    centro de masa) — la geometría puede quedar a decenas de unidades del
    origen local del objeto (visto en Sketchfab: pivote heredado de la
    escena original, no del hueso). Sin esto, obj.location deja de
    corresponder a dónde está realmente la malla, y la mesa/física quedan
    descolocadas respecto al hueso. Centro de bbox (no de masa): más simple
    y no exige malla cerrada/manifold, suficiente para posicionar la caída."""
    local_corners = [Vector(c) for c in obj.bound_box]
    local_center = sum(local_corners, Vector((0, 0, 0))) / 8
    obj.data.transform(Matrix.Translation(-local_center))
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()


def local_bbox_radius(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector((0, 0, 0))) / 8
    radius = max((c - center).length for c in corners)
    return center, radius


TARGET_RADIUS = 10.0


def normalize_scale(obj):
    """Las mallas de Sketchfab/MorphoSource NO comparten escala real entre
    sí (visto: costilla radius≈2, fémur≈12, escápula≈106, mismo tipo de
    fichero) -- a escalas grandes el solver de rigid body falla de forma
    silenciosa (el objeto se queda "flotando" a medio caer en vez de dar
    error, visto con la escápula). Forzar un radio fijo aquí hace que la
    física, las luces y la mesa se comporten igual de bien para cualquier
    malla, sin recalibrar nada por espécimen."""
    _, radius = local_bbox_radius(obj)
    if radius < 1e-9:
        return
    factor = TARGET_RADIUS / radius
    obj.scale = (factor, factor, factor)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


# ------------------------------------------------------- material 2 capas --

def apply_two_layer_material(obj, rng):
    """Capa base (imperfecciones, igual que render_bone_bproc.py) + capa de
    suciedad dirigida por curvatura real de la malla (Pointiness invertido:
    baja curvatura/zonas cóncavas = más suciedad, con sentido físico) mezclada
    con ruido para que no sea un realce uniforme sino manchas orgánicas."""
    mesh = obj.data
    mesh.materials.clear()
    mat = bpy.data.materials.new("bone_lab")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    base_color = rng.choice(BONE_COLOR_PALETTE)
    jitter = lambda c: max(0.0, min(1.0, c + rng.uniform(-0.04, 0.04)))
    base_rgba = (*[jitter(c) for c in base_color], 1.0)

    # --- capa base: imperfecciones (idéntico a apply_procedural_dirt) ---
    noise_base = nodes.new("ShaderNodeTexNoise")
    noise_base.inputs["Scale"].default_value = rng.uniform(8.0, 25.0)
    noise_base.inputs["Detail"].default_value = rng.uniform(2.0, 6.0)
    noise_base.inputs["Roughness"].default_value = rng.uniform(0.4, 0.8)

    ramp_base = nodes.new("ShaderNodeValToRGB")
    ramp_base.color_ramp.elements[0].position = rng.uniform(0.35, 0.55)
    ramp_base.color_ramp.elements[0].color = (0.05, 0.04, 0.03, 1.0)
    ramp_base.color_ramp.elements[1].position = 1.0
    ramp_base.color_ramp.elements[1].color = base_rgba
    links.new(noise_base.outputs["Fac"], ramp_base.inputs["Fac"])

    mix_base = nodes.new("ShaderNodeMixRGB")
    mix_base.blend_type = "MIX"
    mix_base.inputs["Fac"].default_value = rng.uniform(0.15, 0.45)
    mix_base.inputs["Color1"].default_value = base_rgba
    links.new(ramp_base.outputs["Color"], mix_base.inputs["Color2"])

    rough_ramp = nodes.new("ShaderNodeValToRGB")
    rough_ramp.color_ramp.elements[0].position = 0.4
    rough_ramp.color_ramp.elements[0].color = (0.55, 0.55, 0.55, 1.0)  # 0.35 salía con brillo de plástico -- hueso es mate
    rough_ramp.color_ramp.elements[1].position = 1.0
    rough_ramp.color_ramp.elements[1].color = (0.9, 0.9, 0.9, 1.0)
    links.new(noise_base.outputs["Fac"], rough_ramp.inputs["Fac"])

    # --- capa de suciedad: curvatura (Pointiness) + ruido, en huecos ---
    geometry = nodes.new("ShaderNodeNewGeometry")
    invert_point = nodes.new("ShaderNodeMath")
    invert_point.operation = "SUBTRACT"
    invert_point.inputs[0].default_value = 1.0
    links.new(geometry.outputs["Pointiness"], invert_point.inputs[1])

    noise_dirt = nodes.new("ShaderNodeTexNoise")
    noise_dirt.inputs["Scale"].default_value = rng.uniform(15.0, 40.0)
    noise_dirt.inputs["Detail"].default_value = rng.uniform(3.0, 8.0)

    combine_mask = nodes.new("ShaderNodeMath")
    combine_mask.operation = "MULTIPLY"
    links.new(invert_point.outputs["Value"], combine_mask.inputs[0])
    links.new(noise_dirt.outputs["Fac"], combine_mask.inputs[1])

    # Multiplicar dos señales 0-1 (curvatura×ruido) apaga demasiado el
    # resultado (0.5*0.5=0.25) -- POWER<1 aclara los tonos medios antes del
    # degradado, si no casi ninguna zona llegaba a cruzar el umbral y la
    # suciedad salía casi invisible.
    boost_mask = nodes.new("ShaderNodeMath")
    boost_mask.operation = "POWER"
    boost_mask.inputs[1].default_value = 0.32  # más agresivo que antes (0.45) -> más superficie "sucia"
    links.new(combine_mask.outputs["Value"], boost_mask.inputs[0])

    mask_ramp = nodes.new("ShaderNodeValToRGB")
    mask_ramp.color_ramp.elements[0].position = rng.uniform(0.08, 0.15)
    mask_ramp.color_ramp.elements[1].position = rng.uniform(0.4, 0.55)
    links.new(boost_mask.outputs["Value"], mask_ramp.inputs["Fac"])

    stain_color = (0.11, 0.08, 0.055, 1.0)  # tierra/marrón oscuro -- el verde-oliva anterior daba un tinte poco natural a la pieza entera
    mix_dirt = nodes.new("ShaderNodeMixRGB")
    mix_dirt.blend_type = "MIX"
    mix_dirt.inputs["Color2"].default_value = stain_color
    links.new(mix_base.outputs["Color"], mix_dirt.inputs["Color1"])
    links.new(mask_ramp.outputs["Color"], mix_dirt.inputs["Fac"])

    rough_dirt = nodes.new("ShaderNodeMixRGB")
    rough_dirt.blend_type = "MIX"
    rough_dirt.inputs["Color2"].default_value = (0.95, 0.95, 0.95, 1.0)
    links.new(rough_ramp.outputs["Color"], rough_dirt.inputs["Color1"])
    links.new(mask_ramp.outputs["Color"], rough_dirt.inputs["Fac"])
    links.new(rough_dirt.outputs["Color"], bsdf.inputs["Roughness"])

    # --- microfracturas: líneas finas en los bordes de celda de un Voronoi
    # ("Distance to Edge" da ~0 justo en el borde, alto lejos de él -- son
    # las grietas) oscurecidas sobre el color ya con suciedad, no sobre la
    # base limpia.
    crack_voronoi = nodes.new("ShaderNodeTexVoronoi")
    crack_voronoi.voronoi_dimensions = "3D"
    crack_voronoi.feature = "DISTANCE_TO_EDGE"
    crack_voronoi.inputs["Scale"].default_value = rng.uniform(45.0, 80.0)  # células más pequeñas -> grietas más finas y numerosas, no parches bastos

    crack_ramp = nodes.new("ShaderNodeValToRGB")
    crack_ramp.color_ramp.interpolation = "EASE"
    crack_ramp.color_ramp.elements[0].position = 0.0
    crack_ramp.color_ramp.elements[0].color = (0.45, 0.45, 0.45, 1.0)  # línea de grieta sutil, no negra
    crack_ramp.color_ramp.elements[1].position = rng.uniform(0.012, 0.025)  # muy fina
    crack_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(crack_voronoi.outputs["Distance"], crack_ramp.inputs["Fac"])

    crack_mix = nodes.new("ShaderNodeMixRGB")
    crack_mix.blend_type = "MULTIPLY"
    crack_mix.inputs["Fac"].default_value = 0.4  # sutil, no una capa de craquelado obvia
    links.new(mix_dirt.outputs["Color"], crack_mix.inputs["Color1"])
    links.new(crack_ramp.outputs["Color"], crack_mix.inputs["Color2"])

    # --- polvo: se posa por gravedad en caras que miran hacia ARRIBA, al
    # revés que la suciedad (que se concentra en huecos/grietas) -- normal
    # de Geometry en espacio de mundo, así que funciona bien aunque el
    # material se construya antes de que la física decida la orientación
    # final del hueso al caer.
    dust_dot = nodes.new("ShaderNodeVectorMath")
    dust_dot.operation = "DOT_PRODUCT"
    dust_dot.inputs[1].default_value = (0.0, 0.0, 1.0)
    links.new(geometry.outputs["Normal"], dust_dot.inputs[0])

    # Superficies muy redondeadas (cráneo) tienen medio casco entero
    # "mirando hacia arriba" de algún modo -- sin concentrar la señal, el
    # polvo cubre media pieza como si la hubieran rebozado en harina (visto
    # al probarlo). POWER alto deja solo las zonas CASI perfectamente
    # horizontales/hacia arriba.
    dust_sharpen = nodes.new("ShaderNodeMath")
    dust_sharpen.operation = "POWER"
    dust_sharpen.inputs[1].default_value = 18.0  # 6 seguía cubriendo medio cráneo -- solo casi-plano de verdad
    links.new(dust_dot.outputs["Value"], dust_sharpen.inputs[0])

    dust_noise = nodes.new("ShaderNodeTexNoise")
    dust_noise.inputs["Scale"].default_value = rng.uniform(25.0, 40.0)  # motas más finas, no manchones grandes
    dust_noise.inputs["Detail"].default_value = rng.uniform(4.0, 7.0)

    dust_combine = nodes.new("ShaderNodeMath")
    dust_combine.operation = "MULTIPLY"
    links.new(dust_sharpen.outputs["Value"], dust_combine.inputs[0])
    links.new(dust_noise.outputs["Fac"], dust_combine.inputs[1])

    dust_ramp = nodes.new("ShaderNodeValToRGB")
    dust_ramp.color_ramp.elements[0].position = rng.uniform(0.03, 0.06)
    dust_ramp.color_ramp.elements[1].position = rng.uniform(0.15, 0.25)
    links.new(dust_combine.outputs["Value"], dust_ramp.inputs["Fac"])

    # Techo de opacidad bajo aunque la máscara sature a blanco -- una capa
    # fina de polvo deja transparentar el hueso de debajo, no lo tapa (visto
    # con un cráneo: 0.45 seguía pareciendo pintado encima, no polvo real).
    dust_opacity = nodes.new("ShaderNodeMath")
    dust_opacity.operation = "MULTIPLY"
    dust_opacity.inputs[1].default_value = 0.22
    links.new(dust_ramp.outputs["Color"], dust_opacity.inputs[0])

    dust_color = (0.78, 0.75, 0.66, 1.0)  # ceniciento/calizo, mate
    dust_mix = nodes.new("ShaderNodeMixRGB")
    dust_mix.blend_type = "MIX"
    dust_mix.inputs["Color2"].default_value = dust_color
    links.new(crack_mix.outputs["Color"], dust_mix.inputs["Color1"])
    links.new(dust_opacity.outputs["Value"], dust_mix.inputs["Fac"])
    links.new(dust_mix.outputs["Color"], bsdf.inputs["Base Color"])

    dust_rough = nodes.new("ShaderNodeMixRGB")
    dust_rough.blend_type = "MIX"
    dust_rough.inputs["Color2"].default_value = (0.98, 0.98, 0.98, 1.0)  # polvo = muy mate
    links.new(rough_dirt.outputs["Color"], dust_rough.inputs["Color1"])
    links.new(dust_opacity.outputs["Value"], dust_rough.inputs["Fac"])
    links.new(dust_rough.outputs["Color"], bsdf.inputs["Roughness"])

    # --- porosidad: bump de alta frecuencia (hueso trabecular/esponjoso
    # real tiene poros pequeños por toda la superficie, no solo grietas) ---
    pore_noise = nodes.new("ShaderNodeTexNoise")
    pore_noise.inputs["Scale"].default_value = rng.uniform(60.0, 110.0)
    pore_noise.inputs["Detail"].default_value = rng.uniform(6.0, 10.0)
    pore_noise.inputs["Roughness"].default_value = 0.75

    pore_bump = nodes.new("ShaderNodeBump")
    pore_bump.inputs["Strength"].default_value = 0.35
    pore_bump.inputs["Distance"].default_value = 0.4
    links.new(pore_noise.outputs["Fac"], pore_bump.inputs["Height"])
    links.new(pore_bump.outputs["Normal"], bsdf.inputs["Normal"])

    mesh.materials.append(mat)


# --------------------------------------------------------- mesa + física ---

def make_table(radius):
    # Grande a propósito: huesos curvos (costilla) ruedan como mecedora con
    # poca fricción/amortiguación (necesaria para que otros no se queden
    # congelados a medio volcar) -- más mesa da margen para que la rueda se
    # frene sola antes del borde, en vez de tener que compensarlo solo con
    # más fricción (que sí frenaría el vuelco de huesos alargados).
    table_half = radius * 16.0
    thickness = radius * 0.4
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, -thickness / 2))
    table = bpy.context.active_object
    table.name = "lab_table"
    table.scale = (table_half, table_half, thickness / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    table.location = (0, 0, -thickness / 2)

    mat = bpy.data.materials.new("table_lab")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    # Casi negro a propósito -- fotos reales de espécimen suelen ser fondo
    # de estudio oscuro/negro (ver comparador), no una mesa gris uniforme.
    # Acabado tipo tela/terciopelo oscuro (fondo de estudio típico), no
    # plástico ni piedra: pliegues suaves y grandes (escala baja, sin
    # detalle fino -- con detalle alto se veía "quemado"/ruidoso, no tela),
    # muy mate (roughness alto) y sheen (reflejo rasante característico de
    # tejido con pelo, ej. terciopelo).
    noise_tex = nodes.new("ShaderNodeTexNoise")
    noise_tex.inputs["Scale"].default_value = 2.2
    noise_tex.inputs["Detail"].default_value = 2.0
    noise_tex.inputs["Roughness"].default_value = 0.5

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (0.012, 0.012, 0.013, 1.0)
    color_ramp.color_ramp.elements[1].color = (0.035, 0.033, 0.032, 1.0)
    links.new(noise_tex.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.06  # pliegues suaves de tela, no relieve duro
    links.new(noise_tex.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Roughness"].default_value = 0.85  # muy mate -- tela, no plástico
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Sheen Weight" in bsdf.inputs:  # nombre de input varía entre 4.0/4.x
        bsdf.inputs["Sheen Weight"].default_value = 0.6
    elif "Sheen" in bsdf.inputs:
        bsdf.inputs["Sheen"].default_value = 0.6
    table.data.materials.append(mat)

    bpy.context.view_layer.objects.active = table
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    table.rigid_body.collision_shape = "BOX"
    table.rigid_body.friction = 0.6
    return table, table_half


def add_bone_rigid_body(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    # MESH (forma cóncava exacta) es INESTABLE en la práctica para huesos
    # delgados y curvos (probado con una costilla: rueda como mecedora sin
    # parar nunca, se sale de una mesa de 32x el radio — error de contacto
    # numérico, no vuelco físico real). CONVEX_HULL, aunque simplifica la
    # concavidad, asienta de forma estable y creíble — es además la opción
    # estándar para cuerpos rígidos dinámicos por esta misma razón.
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.rigid_body.friction = 0.6
    obj.rigid_body.restitution = 0.15
    # Amortiguación angular alta a propósito: con menos, un hueso curvo
    # (costilla) rueda indefinidamente (la fricción frena el deslizamiento,
    # no la rodadura). Con drop_frames/drop_height generosos, un hueso
    # alargado (fémur) todavía tiene tiempo de completar el vuelco entero
    # en vez de quedarse "de pie" a medias.
    obj.rigid_body.linear_damping = 0.1
    obj.rigid_body.angular_damping = 0.45


def settle_drop(scene, obj, rng, table_half, drop_height, n_frames, forced_rotation=None):
    bpy.ops.ptcache.free_bake_all()
    scene.frame_set(scene.frame_start)
    obj.location = (
        rng.uniform(-table_half * 0.4, table_half * 0.4),
        rng.uniform(-table_half * 0.4, table_half * 0.4),
        drop_height,
    )
    obj.rotation_euler = forced_rotation if forced_rotation is not None else (
        rng.uniform(0, 2 * math.pi),
        rng.uniform(0, 2 * math.pi),
        rng.uniform(0, 2 * math.pi),
    )
    for f in range(scene.frame_start + 1, n_frames + 1):
        scene.frame_set(f)


def has_natural_resting_orientation(obj) -> bool:
    """bound_box es siempre en espacio LOCAL (no le afecta la rotación
    actual) -- mide la forma intrínseca de la malla, independiente de cómo
    haya caído. Dos formas tienen un modo de fallo "de pie"/en canto que
    hay que descartar: alargada tipo barra (fémur, húmero, costilla -- un
    eje claramente más LARGO que los otros dos) y plana tipo plato
    (escápula, costado del pelvis -- un eje claramente más CORTO, el
    grosor). Solo un hueso compacto de verdad (vértebra, carpiano...) no
    tiene una orientación "natural" clara y no se le aplica el chequeo --
    aplicárselo a un objeto sin eje dominante da falsos positivos (bbox del
    mundo con rotación no distingue tumbado de inclinado)."""
    corners = [Vector(c) for c in obj.bound_box]
    xs, ys, zs = [c.x for c in corners], [c.y for c in corners], [c.z for c in corners]
    dims = sorted([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)], reverse=True)
    elongated = dims[0] > dims[1] * 1.6  # eje largo dominante (barra)
    flat = dims[1] > dims[2] * 1.6  # eje corto dominante (plato)
    return elongated or flat


def bone_settled_on_table(obj, table_half, radius, elongated):
    center, corners_radius = local_bbox_radius(obj)
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    min_z = min(zs)
    off_table_xy = abs(center.x) > table_half * 0.9 or abs(center.y) > table_half * 0.9
    fell_through = min_z < -radius * 2.0
    standing_up = False
    if elongated:
        horizontal_extent = max(max(xs) - min(xs), max(ys) - min(ys))
        vertical_extent = max(zs) - min(zs)
        standing_up = vertical_extent > horizontal_extent * 0.6
    return not (off_table_xy or fell_through or standing_up), center, corners_radius


# ------------------------------------------------------------- iluminación -

# Galería de combinaciones de luz para comparar -- pedido explícito del
# usuario ("la luz es demasiado neutra, no parece real"). Cada preset toca:
# color de la key/relleno (temperatura), fuerza/tinte del HDRI ambiente, y
# el tamaño/potencia relativa de la key (más grande+débil = sombra más
# suave, tipo luz de habitación en vez de foco de estudio).
LIGHT_PRESETS = {
    "neutro": dict(
        key_color=(1.0, 1.0, 1.0), fill_color=(1.0, 1.0, 1.0),
        hdri_strength=0.08, hdri_tint=(1.0, 1.0, 1.0),
        key_energy_mult=1.0, key_size_mult=1.0,
    ),
    "tungsteno_calido": dict(
        key_color=(1.0, 0.80, 0.58), fill_color=(1.0, 0.85, 0.68),
        hdri_strength=0.10, hdri_tint=(1.0, 0.88, 0.72),
        key_energy_mult=1.0, key_size_mult=1.0,
    ),
    "calido_suave": dict(
        key_color=(1.0, 0.87, 0.72), fill_color=(1.0, 0.90, 0.78),
        hdri_strength=0.18, hdri_tint=(1.0, 0.92, 0.82),
        # radiancia de un foco de área cae con el TAMAÑO al cuadrado -- para
        # que 1.8x de tamaño no salga muy oscuro hace falta compensar con
        # más potencia, no menos (visto: 0.75/2.2 salió casi negro).
        key_energy_mult=3.0, key_size_mult=1.8,
    ),
    "frio_exterior": dict(
        key_color=(0.82, 0.90, 1.0), fill_color=(0.85, 0.92, 1.0),
        hdri_strength=0.14, hdri_tint=(0.85, 0.91, 1.0),
        key_energy_mult=1.0, key_size_mult=1.0,
    ),
    "ambiente_mixto": dict(
        key_color=(1.0, 0.93, 0.85), fill_color=(1.0, 0.93, 0.85),
        hdri_strength=0.35, hdri_tint=(1.0, 0.95, 0.88),
        key_energy_mult=0.55, key_size_mult=1.3,  # ambiente domina, key más discreta
    ),
}


def setup_world_hdri(hdri_path, preset):
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(str(hdri_path), check_existing=True)

    tint = nodes.new("ShaderNodeMixRGB")
    tint.blend_type = "MULTIPLY"
    tint.inputs["Fac"].default_value = 1.0
    tint.inputs["Color2"].default_value = (*preset["hdri_tint"], 1.0)
    links.new(env_tex.outputs["Color"], tint.inputs["Color1"])

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = preset["hdri_strength"]
    output = nodes.new("ShaderNodeOutputWorld")
    links.new(tint.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def add_lab_lights(center, radius, rng, preset):
    """Focos de techo en ángulo oblicuo (relieve/imperfecciones) + un
    relleno grande y suave (para que no haya sombras totalmente negras).
    Devuelve los objetos creados para poder borrarlos entre caídas (el
    centro del hueso se mueve en cada caída).

    3 focos a intensidades BIEN distintas (no 2-3 iguales -- eso sí se
    cancela entre sí y aplana, lo probamos): una key dominante que marca
    la sombra principal, una secundaria a ~35% para que no se lea como
    "una sola luz de estudio de examen", y un relleno débil de techo."""
    created = []
    key_azim = math.radians(rng.uniform(0, 360))
    ratios = [1.0, rng.uniform(0.3, 0.45)]  # key, secundaria -- nunca cerca de 1:1
    for i, ratio in enumerate(ratios):
        azim = key_azim if i == 0 else key_azim + math.radians(rng.uniform(90, 200))
        elev = math.radians(rng.uniform(30, 50))
        direction = Vector((math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)))
        loc = center + direction * radius * 5.0
        light_data = bpy.data.lights.new(f"key_{i}", type="AREA")
        light_data.color = preset["key_color"]
        light_data.energy = (rng.uniform(1800000, 2600000) * ratio * preset["key_energy_mult"]
                              * (radius ** 2) / (190.0 ** 2))
        # más pequeño -> sombra dura de verdad + algo de zona "quemada" en los brillos, como la foto real
        light_data.size = radius * 0.22 * preset["key_size_mult"]
        light_obj = bpy.data.objects.new(f"key_{i}", light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc
        light_obj.rotation_euler = (center - loc).normalized().to_track_quat("-Z", "Y").to_euler()
        created.append(light_obj)

    fill_loc = center + Vector((0, 0, radius * 6.0))
    fill_data = bpy.data.lights.new("fill", type="AREA")
    fill_data.color = preset["fill_color"]
    fill_data.energy = 40000 * (radius ** 2) / (190.0 ** 2)  # débil a propósito: solo levanta el negro puro, no compite con la key
    fill_data.size = radius * 6.0
    fill_obj = bpy.data.objects.new("fill", fill_data)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.location = fill_loc
    fill_obj.rotation_euler = (center - fill_loc).normalized().to_track_quat("-Z", "Y").to_euler()
    created.append(fill_obj)
    return created


def clear_lights(light_objs):
    for obj in light_objs:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.lights.remove(data)


# ------------------------------------------------------------------ cámara -

def make_camera(clip_end):
    cam_data = bpy.data.cameras.new("lab_cam")
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.01
    cam_data.clip_end = clip_end
    cam_obj = bpy.data.objects.new("lab_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def position_camera(cam, center, radius, elevation_deg, azimuth_deg, focal_mm, sensor_width, fill):
    cam.data.lens = focal_mm
    cam.data.sensor_width = sensor_width
    fov_half = math.atan(sensor_width / (2 * focal_mm))
    distance = radius / (fill * math.tan(fov_half))

    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    direction = Vector((math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)))
    cam.location = center + direction * distance
    look_dir = (center - cam.location).normalized()
    cam.rotation_euler = look_dir.to_track_quat("-Z", "Y").to_euler()
    return distance


def bone_in_frame(cam, scene, corners) -> bool:
    for c in corners:
        co = world_to_camera_view(scene, cam, c)
        if not (0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z > 0):
            return False
    return True


# --------------------------------------------------------------- render ----

def render_view(scene, table, out_path: Path):
    """Sin PIL a propósito: el Python embebido de Blender (`blender
    --background`) no lo trae, solo numpy — a diferencia de
    render_bone_bproc.py, que corre bajo `blenderproc run` (Python del venv
    del repo, con Pillow).

    Dos pasadas: (1) la foto "de verdad", con la mesa visible y opaca — es
    el objetivo, a diferencia de la rama bproc que aísla el hueso sobre
    fondo transparente. (2) una pasada barata solo para la máscara GT
    (mesa oculta, fondo transparente, pocas muestras — no importa el ruido,
    solo la silueta), para no perder el contrato de salida `_mask.png` que
    ya consumen/consumirán extract_features/build_dataset."""
    scene.render.film_transparent = False
    scene.render.filepath = str(out_path.with_suffix(""))
    bpy.ops.render.render(write_still=True)

    # bpy.data.images["Render Result"].pixels no es fiable en modo
    # --background (buffer no siempre realizado sin ventana/área real) —
    # se escribe a disco y se relee, igual que la imagen principal.
    original_samples = scene.cycles.samples
    table.hide_render = True
    scene.render.film_transparent = True
    scene.cycles.samples = 8
    alpha_tmp_path = out_path.parent / f"{out_path.stem}_alpha_tmp.png"
    scene.render.filepath = str(alpha_tmp_path.with_suffix(""))
    bpy.ops.render.render(write_still=True)

    alpha_img = bpy.data.images.load(str(alpha_tmp_path), check_existing=False)
    w, h = alpha_img.size
    pixels = np.array(alpha_img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    solid = (pixels[:, :, 3] > (250.0 / 255.0)).astype(np.float32)
    mask_rgba = np.stack([solid, solid, solid, np.ones_like(solid)], axis=-1)

    mask_path = out_path.parent / f"{out_path.stem}_mask.png"
    mask_img = bpy.data.images.new("mask_tmp", width=w, height=h, alpha=True)
    mask_img.pixels = mask_rgba.flatten().tolist()
    mask_img.filepath_raw = str(mask_path)
    mask_img.file_format = "PNG"
    mask_img.save()
    bpy.data.images.remove(alpha_img)
    bpy.data.images.remove(mask_img)
    alpha_tmp_path.unlink()

    table.hide_render = False
    scene.render.film_transparent = False
    scene.cycles.samples = original_samples


def main():
    args = parse_args()
    mesh_path = Path(args.mesh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False  # la mesa debe verse — render_view() gestiona la pasada de máscara aparte
    scene.view_settings.view_transform = "Standard"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type in ("CUDA", "OPTIX")
    scene.cycles.device = "GPU"

    obj = import_mesh(mesh_path)
    keep_original = has_real_texture(obj)
    center_on_bbox(obj)
    normalize_scale(obj)

    base_seed = args.seed or abs(hash(args.prefix)) % (2**31)
    rng_material = random.Random(base_seed)
    if not keep_original:
        apply_two_layer_material(obj, rng_material)
    force_bone_reflectivity(obj)

    _, radius = local_bbox_radius(obj)
    elongated = has_natural_resting_orientation(obj)
    table, table_half = make_table(radius)
    add_bone_rigid_body(obj)
    cam = make_camera(clip_end=radius * 50)

    hdri_files = sorted(HDRI_DIR.glob("*.hdr"))
    drop_height = radius * 3.0  # más recorrido de caída -> más energía/vuelco al impactar
    view_idx = 0
    current_lights = []

    for drop in range(args.n_drops):
        rng_drop = random.Random(base_seed + drop * 1000 + 1)
        settled = False
        center, settled_radius = None, radius
        for attempt in range(args.max_drop_retries):
            settle_drop(scene, obj, rng_drop, table_half, drop_height, args.drop_frames)
            ok, center, settled_radius = bone_settled_on_table(obj, table_half, radius, elongated)
            if ok:
                settled = True
                break
            print(f"AVISO: caída {drop} intento {attempt} fuera de mesa, reintentando")
        if not settled and center is not None:
            # Último recurso: alguna malla (visto con una escápula real) no
            # encuentra sola la pose tumbada por más reintentos que se den
            # -- da igual la rotación de partida, la física la vuelve a
            # dejar de canto siempre (es SU mínimo de energía estable con
            # CONVEX_HULL, no mala suerte de reintentos). Se probó a forzar
            # la rotación a mano (eje corto hacia arriba) pero
            # obj.matrix_world no reflejaba el cambio de forma fiable al
            # leerlo justo después sin volver a pasar por settle_drop -- en
            # vez de perseguir eso, se acepta el último intento tal cual
            # (ya viene de un settle_drop real, sin el problema de lectura)
            # con tal de que siga sobre la mesa, aunque no quede perfecto.
            off_table_xy = abs(center.x) > table_half * 0.9 or abs(center.y) > table_half * 0.9
            if not off_table_xy:
                settled = True
                print(f"AVISO: caída {drop} aceptada sin pose tumbada perfecta tras {args.max_drop_retries} intentos")
        if not settled:
            print(f"AVISO: caída {drop} descartada tras {args.max_drop_retries} intentos")
            continue

        # "Encaje" al final: el margen de colisión del solver (sobre todo
        # con CONVEX_HULL) deja un hueco de aire pequeño pero visible entre
        # el hueso y la mesa -- sin sombra de contacto, se ve "flotando" en
        # vez de posado. Solo se corrige la Z (traslación pura), no la
        # rotación/pose que ya dio la física.
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        gap = min(c.z for c in corners)
        obj.location.z -= gap
        bpy.context.view_layer.update()

        clear_lights(current_lights)
        light_preset = LIGHT_PRESETS[args.light_preset]
        if hdri_files:
            setup_world_hdri(rng_drop.choice(hdri_files), light_preset)
        current_lights = add_lab_lights(center, settled_radius, rng_drop, light_preset)
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]

        for shot in range(args.views_per_drop):
            rng_shot = random.Random(base_seed + drop * 1000 + 100 + shot)
            focal_mm = rng_shot.uniform(24, 70)
            sensor_width = rng_shot.uniform(23, 36)
            elevation_deg = rng_shot.uniform(args.elevation_min_deg, args.elevation_max_deg)
            azimuth_deg = rng_shot.uniform(0, 360)

            fill = args.fill
            for qa_attempt in range(args.max_frame_retries):
                position_camera(cam, center, settled_radius, elevation_deg, azimuth_deg, focal_mm, sensor_width, fill)
                if bone_in_frame(cam, scene, corners):
                    break
                fill *= 0.85  # aleja la cámara (encuadre menos apretado) y reintenta
            else:
                print(f"AVISO: caída {drop} vista {shot} no encaja del todo en cuadro, se acepta igual")

            out_path = out_dir / f"{args.prefix}_drop{drop:02d}_view{shot:03d}.png"
            render_view(scene, table, out_path)
            view_idx += 1

    print(f"RENDER_DONE views={view_idx} out={out_dir}")


if __name__ == "__main__":
    main()
