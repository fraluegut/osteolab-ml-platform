"""Variante "ruidosa" de `render_bone.py`: mismo rig de cámaras (esfera de
Fibonacci sobre el centro de la bbox), pero con fondo, iluminación y color de
material ALEATORIOS en cada vista, en vez de fondo blanco liso + luz suave
omnidireccional fija.

Motivación (2026-07-24): el banco de fotos reales mostró que el modelo
(geométrico y, en el futuro, cualquier CNN) solo ha visto renders de estudio
limpios — nunca un fondo con textura/degradado, ni una luz direccional que
deje la superficie del propio hueso con sombreado desigual (justo lo que
rompió la segmentación con la foto real de un cráneo). Este script genera
ADEMÁS de los renders limpios (no los sustituye) un set con esa variedad,
para entrenar con algo más parecido a una foto de usuario real:
  - Fondo: `film_transparent=True` + canal alpha — el render en sí NO lleva
    fondo "quemado", se compone después en Python sobre una foto de fondo
    real (ver `compose_noisy.py`). El `World` de abajo (color sólido o
    degradado) se mantiene solo por su efecto en la iluminación/reflejos,
    nunca es visible en el PNG final (alpha=0 fuera del hueso y su sombra).
  - Sombra de contacto real: un plano `is_shadow_catcher` bajo el hueso —
    con `film_transparent=True`, Cycles deja ese plano invisible salvo por
    la sombra que cae sobre él (alpha parcial), lista para componer sobre
    cualquier fondo real sin que se note un "corte" limpio del recorte.
  - Encuadre no siempre centrado: el punto al que apunta la cámara se
    desplaza con un jitter aleatorio (hasta ~15% del radio del hueso) en
    vez de apuntar siempre exactamente al centro de la bbox.
  - Luz: una fuente direccional (sol) con ángulo/intensidad aleatorios y
    color tomado de una paleta de temperaturas de color plausibles
    (incandescente, halógeno, luz día, nublado, sombra abierta, hora
    dorada, fluorescente, flash) + una luz de relleno tenue para que el
    lado oscuro no quede en negro absoluto.
  - Material: tono de "hueso" aleatorio dentro de un rango plausible (blanco,
    beige, marrón, grisáceo), no siempre el mismo tono fijo — SOLO para
    mallas sin textura/color de vértice de origen (ver `has_real_texture`,
    2026-07-26: sobre una muestra de 15 mallas, ~la mitad SÍ traían
    fotogrametría real que antes se tiraba sin comprobar con
    `materials.clear()`; ahora se conserva tal cual cuando existe).

Cada vista tiene su propia semilla derivada de (prefix, índice de vista), así
que la aleatoriedad es reproducible pero distinta vista a vista.

Uso:
    blender --background --python blender/render_bone_noisy.py -- \\
        --mesh data/meshes/homo_sapiens/femur/sketchfab_xxx/malla.glb \\
        --out-dir renders_noisy --prefix homo_sapiens_femur_sketchfab_xxx --n-views 10
"""

import argparse
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_bone import (  # noqa: E402
    import_mesh, center_on_mass, framing_target, fibonacci_sphere, make_camera, point_at,
    write_render_metadata,
)

# Paleta de fondos: tonos neutros/de mesa/estudio variados, no solo blanco.
BACKGROUND_PALETTE = [
    (0.92, 0.92, 0.90), (0.75, 0.72, 0.65), (0.55, 0.50, 0.45),
    (0.30, 0.30, 0.32), (0.15, 0.15, 0.16), (0.65, 0.60, 0.50),
    (0.40, 0.45, 0.42), (0.35, 0.40, 0.50), (0.80, 0.78, 0.70),
    (0.20, 0.22, 0.25),
]

# Paleta de tonos de hueso: blanco/beige/marrón/grisáceo, con algo de variación
# de saturación para simular hueso fresco, seco, fósil o mineralizado.
BONE_COLOR_PALETTE = [
    (0.85, 0.82, 0.75), (0.62, 0.54, 0.42), (0.70, 0.60, 0.45),
    (0.45, 0.38, 0.30), (0.55, 0.52, 0.48), (0.75, 0.65, 0.50),
    (0.35, 0.28, 0.22), (0.80, 0.75, 0.68),
]

# Paleta de temperaturas de color de la luz clave, de cálida a fría —
# aproximaciones RGB de fuentes reales (bombilla, luz día, sombra, flash),
# no solo un jitter de "más o menos amarillo" sobre un único blanco.
COLOR_TEMP_PRESETS = [
    (1.00, 0.85, 0.70),  # incandescente cálida
    (1.00, 0.93, 0.86),  # halógeno
    (1.00, 1.00, 1.00),  # luz día neutra
    (0.92, 0.96, 1.00),  # nublado frío
    (0.80, 0.90, 1.00),  # sombra abierta azulada
    (1.00, 0.78, 0.58),  # hora dorada / atardecer
    (0.88, 1.00, 0.90),  # fluorescente verdosa
    (0.95, 0.97, 1.00),  # flash de cámara neutro-frío
]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--fov-deg", type=float, default=40.0)
    p.add_argument("--fill", type=float, default=0.72)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def has_real_texture(obj) -> bool:
    """True si la malla importada trae textura de imagen o color por
    vértice reales (fotogrametría de origen) — comprobado sobre una muestra
    de 15 mallas del catálogo (2026-07-26): ~la mitad NO traen nada (mesh
    "en blanco", solo `NORMAL`+`POSITION`) y para esas el color procedural
    de `make_material` sigue haciendo falta, pero la otra mitad SÍ trae
    textura de imagen o `COLOR_0` — hasta ahora se tiraba siempre con
    `materials.clear()` sin comprobar. Si es True, el llamador debe dejar
    el material tal cual (no limpiarlo ni sustituirlo)."""
    if obj.data.color_attributes and len(obj.data.color_attributes) > 0:
        return True
    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                return True
    return False


def make_material(rng):
    mat = bpy.data.materials.new("bone_matte_noisy")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    color = rng.choice(BONE_COLOR_PALETTE)
    # pequeño jitter para no repetir literalmente el mismo tono en dos vistas
    jitter = lambda c: max(0.0, min(1.0, c + rng.uniform(-0.04, 0.04)))
    final_color = [jitter(c) for c in color]
    roughness = rng.uniform(0.35, 0.75)
    bsdf.inputs["Base Color"].default_value = (*final_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    meta = {
        "base_color_palette": list(color), "base_color_final": final_color, "roughness": round(roughness, 4),
    }
    return mat, meta


def setup_world(rng):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg = nodes["Background"]

    meta = {}
    if rng.random() < 0.5:
        # fondo sólido
        color = rng.choice(BACKGROUND_PALETTE)
        bg.inputs[0].default_value = (*color, 1.0)
        meta["background_type"] = "solido"
        meta["background_color"] = list(color)
    else:
        # degradado de dos tonos (justo el tipo de fondo que rompió la
        # segmentación en producción — se incluye a propósito)
        c1 = rng.choice(BACKGROUND_PALETTE)
        c2 = rng.choice(BACKGROUND_PALETTE)
        grad = nodes.new("ShaderNodeTexGradient")
        grad_type = rng.choice(["LINEAR", "SPHERICAL"])
        grad.gradient_type = grad_type
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*c1, 1.0)
        ramp.color_ramp.elements[1].color = (*c2, 1.0)
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, math.tau))
        coord = nodes.new("ShaderNodeTexCoord")
        links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], grad.inputs["Vector"])
        links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bg.inputs["Color"])
        meta["background_type"] = "degradado"
        meta["background_gradient_type"] = grad_type
        meta["background_colors"] = [list(c1), list(c2)]

    bg_strength = rng.uniform(0.5, 1.3)
    bg.inputs[1].default_value = bg_strength
    meta["background_strength"] = round(bg_strength, 4)
    return meta


def add_lights(center, radius, rng):
    # Luz clave direccional (sol): ángulo y potencia aleatorios por vista, para
    # generar sombreado desigual real sobre la superficie del hueso — a
    # propósito, es justo lo que faltaba en el rig limpio.
    sun_data = bpy.data.lights.new("key_sun", type="SUN")
    sun_energy = rng.uniform(1.5, 4.5)
    sun_angle_deg = rng.uniform(2, 15)  # sombras más o menos duras
    sun_data.energy = sun_energy
    sun_data.angle = math.radians(sun_angle_deg)
    base_color = rng.choice(COLOR_TEMP_PRESETS)
    jitter = lambda c: max(0.0, min(1.0, c + rng.uniform(-0.03, 0.03)))
    sun_data.color = tuple(jitter(c) for c in base_color)
    sun_obj = bpy.data.objects.new("key_sun", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (
        rng.uniform(0.2, 1.3), rng.uniform(0, math.tau), rng.uniform(0, math.tau)
    )

    # Relleno tenue (mismo tipo que el rig limpio, pero con menos energía y
    # solo 2 focos en vez de 4) para que el lado en sombra no quede negro puro.
    energy_per_radius_sq = 90 / (190.0 ** 2)
    fill_mult = rng.uniform(0.6, 1.4)
    energy = energy_per_radius_sq * radius ** 2 * fill_mult
    for i, (x, y, z) in enumerate([(1, -1, 0.5), (-1, 1, -0.4)]):
        v = center + Vector((x, y, z)).normalized() * radius * 2.2
        light_data = bpy.data.lights.new(f"fill_{i}", type="AREA")
        light_data.energy = energy
        light_data.size = radius * 1.5
        light_obj = bpy.data.objects.new(f"fill_{i}", light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = v
        direction = (center - v).normalized()
        light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    return {
        "sun_energy": round(sun_energy, 4),
        "sun_angle_deg": round(sun_angle_deg, 4),
        "sun_color_temp_preset": list(base_color),
        "fill_energy_mult": round(fill_mult, 4),
    }


def bbox_bounds(obj):
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def add_shadow_catcher(obj, center, radius):
    # Plano de suelo bajo el hueso, grande (12x el radio) para que la sombra
    # nunca se corte en el encuadre. `is_shadow_catcher=True` + el
    # `film_transparent=True` de la escena hace que Cycles deje el plano
    # invisible salvo por la sombra real que cae sobre él (alpha parcial en
    # esa zona) — es la sombra de contacto que compone bien sobre cualquier
    # fondo real en vez de un recorte limpio sin apoyo visual.
    _, _, _, _, min_z, _ = bbox_bounds(obj)
    bpy.ops.mesh.primitive_plane_add(size=radius * 12, location=(center.x, center.y, min_z))
    plane = bpy.context.active_object
    plane.is_shadow_catcher = True
    plane.name = "shadow_catcher"
    return plane


def clear_scene_lights_and_world():
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world)


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
    # Sin fondo "quemado" en el PNG: el hueso + su sombra de contacto salen
    # con alpha parcial/total y el resto a alpha=0, listos para componer en
    # Python sobre una foto de fondo real (ver compose_noisy.py).
    scene.render.film_transparent = True
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
    center_on_mass(obj)
    keep_original_material = has_real_texture(obj)
    if not keep_original_material:
        obj.data.materials.clear()

    center, radius = framing_target(obj)

    add_shadow_catcher(obj, center, radius)

    fov_half = math.radians(args.fov_deg) / 2
    distance = radius / (args.fill * math.tan(fov_half))
    cam = make_camera(args.fov_deg, clip_end=distance * 3)

    positions = fibonacci_sphere(args.n_views, distance)
    # 2026-10-31: `args.seed or hash(...)` trataba --seed 0 como "sin
    # semilla" (0 es falsy) y caía siempre al hash(prefix), no reproducible
    # entre procesos (PYTHONHASHSEED no fijado). Ver mismo fix en
    # render_bone_lab.py -- el flag --seed nunca tuvo efecto en la práctica.
    base_seed = args.seed if args.seed is not None else abs(hash(args.prefix)) % (2**31)

    for i, offset in enumerate(positions):
        rng = random.Random(base_seed + i)
        clear_scene_lights_and_world()
        world_meta = setup_world(rng)
        light_meta = add_lights(center, radius, rng)
        material_meta = {"keep_original_texture": keep_original_material}
        if not keep_original_material:
            obj.data.materials.clear()
            mat, mat_meta = make_material(rng)
            obj.data.materials.append(mat)
            material_meta.update(mat_meta)

        cam.location = center + offset
        # Encuadre no siempre centrado: desplaza el punto al que apunta la
        # cámara (no la posición) un poco respecto al centro real de la bbox.
        jitter = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.5, 0.5)))
        point_at(cam, center + jitter * radius * 0.15)
        out_path = out_dir / f"{args.prefix}_view{i:03d}.png"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)

        write_render_metadata(out_path, {
            "recipe": "noisy_composited_source",
            "camera_strategy": "fibonacci_sphere",
            "prefix": args.prefix, "view_index": i, "base_seed": base_seed,
            "fov_deg": args.fov_deg, "fill": args.fill,
            "world": world_meta, "light": light_meta, "material": material_meta,
        })

    print(f"RENDER_DONE views={args.n_views} out={out_dir}")


if __name__ == "__main__":
    main()
