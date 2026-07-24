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
  - Fondo: color sólido aleatorio o degradado de dos tonos, de una paleta
    amplia (no solo blanco).
  - Luz: una fuente direccional (sol) con ángulo/intensidad/temporatura de
    color aleatorios por vista (shadows reales, sombreado desigual a
    propósito) + una luz de relleno tenue para que el lado oscuro no quede
    en negro absoluto.
  - Material: tono de "hueso" aleatorio dentro de un rango plausible (blanco,
    beige, marrón, grisáceo), no siempre el mismo tono fijo.

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


def make_material(rng):
    mat = bpy.data.materials.new("bone_matte_noisy")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    color = rng.choice(BONE_COLOR_PALETTE)
    # pequeño jitter para no repetir literalmente el mismo tono en dos vistas
    jitter = lambda c: max(0.0, min(1.0, c + rng.uniform(-0.04, 0.04)))
    bsdf.inputs["Base Color"].default_value = (*[jitter(c) for c in color], 1.0)
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.35, 0.75)
    return mat


def setup_world(rng):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg = nodes["Background"]

    if rng.random() < 0.5:
        # fondo sólido
        color = rng.choice(BACKGROUND_PALETTE)
        bg.inputs[0].default_value = (*color, 1.0)
    else:
        # degradado de dos tonos (justo el tipo de fondo que rompió la
        # segmentación en producción — se incluye a propósito)
        c1 = rng.choice(BACKGROUND_PALETTE)
        c2 = rng.choice(BACKGROUND_PALETTE)
        grad = nodes.new("ShaderNodeTexGradient")
        grad.gradient_type = rng.choice(["LINEAR", "SPHERICAL"])
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

    bg.inputs[1].default_value = rng.uniform(0.5, 1.3)


def add_lights(center, radius, rng):
    # Luz clave direccional (sol): ángulo y potencia aleatorios por vista, para
    # generar sombreado desigual real sobre la superficie del hueso — a
    # propósito, es justo lo que faltaba en el rig limpio.
    sun_data = bpy.data.lights.new("key_sun", type="SUN")
    sun_data.energy = rng.uniform(1.5, 4.5)
    sun_data.angle = math.radians(rng.uniform(2, 15))  # sombras más o menos duras
    warmth = rng.uniform(-0.15, 0.15)
    sun_data.color = (1.0, 1.0 - warmth * 0.5, 1.0 - warmth)
    sun_obj = bpy.data.objects.new("key_sun", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (
        rng.uniform(0.2, 1.3), rng.uniform(0, math.tau), rng.uniform(0, math.tau)
    )

    # Relleno tenue (mismo tipo que el rig limpio, pero con menos energía y
    # solo 2 focos en vez de 4) para que el lado en sombra no quede negro puro.
    energy_per_radius_sq = 90 / (190.0 ** 2)
    energy = energy_per_radius_sq * radius ** 2 * rng.uniform(0.6, 1.4)
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
    scene.render.film_transparent = False
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
    obj.data.materials.clear()

    center, radius = framing_target(obj)

    fov_half = math.radians(args.fov_deg) / 2
    distance = radius / (args.fill * math.tan(fov_half))
    cam = make_camera(args.fov_deg, clip_end=distance * 3)

    positions = fibonacci_sphere(args.n_views, distance)
    base_seed = args.seed or abs(hash(args.prefix)) % (2**31)

    for i, offset in enumerate(positions):
        rng = random.Random(base_seed + i)
        clear_scene_lights_and_world()
        setup_world(rng)
        add_lights(center, radius, rng)
        obj.data.materials.clear()
        obj.data.materials.append(make_material(rng))

        cam.location = center + offset
        point_at(cam, center)
        scene.render.filepath = str(out_dir / f"{args.prefix}_view{i:03d}.png")
        bpy.ops.render.render(write_still=True)

    print(f"RENDER_DONE views={args.n_views} out={out_dir}")


if __name__ == "__main__":
    main()
