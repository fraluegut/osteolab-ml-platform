import blenderproc as bproc  # noqa: E402 -- DEBE ser el primer import (lo exige blenderproc run)

"""Reemplaza a render_bone_noisy.py: mismo propósito (renders "ruidosos" para
cerrar el hueco sim-to-real) pero sobre BlenderProc en vez de bpy a pelo, para
aprovechar tres cosas que bpy a pelo no daba gratis:
  1. Máscara de segmentación EXACTA desde el alpha del render (film
     transparente + umbral >250 = silueta real del hueso, sin la sombra de
     contacto de alpha parcial) -- en vez de que extract_features() tenga que
     adivinar el contorno sobre el render ya compuesto (falla en huesos
     pequeños/lejanos de bajo contraste, ~12% de renders limpios descartados
     por esto). Se guarda junto a cada vista como "<prefix>_view###_mask.png".
  2. HDRI real (14 HDRIs CC0 de Poly Haven en base_datos_osea/hdris/, mezcla
     de estudio fotográfico + exteriores nublados) en vez de un World de
     color sólido/degradado -- reflejos e iluminación ambiente reales.
  3. Focal length variable (24-70mm) en vez de un FOV fijo.

Conserva el criterio de has_real_texture() (commit 5dbca0d, sesión
2026-07-26): si la malla trae textura de imagen o color de vértice real, se
deja tal cual; si no, se aplica suciedad PROCEDURAL (ruido de Blender
modulando roughness + tinte, no una descarga de texturas PBR externas --
decisión de tiempo/fiabilidad para esta migración, ver HISTORIAL.md) en vez
del color plano anterior.

Uso:
    blenderproc run --custom-blender-path /opt/blender \\
        base_datos_osea/blender/render_bone_bproc.py -- \\
        --mesh data/meshes/.../malla.glb --out-dir renders_noisy_bproc \\
        --prefix homo_sapiens_femur_xxx --n-views 10
"""

import argparse
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HDRI_DIR = Path(__file__).resolve().parents[1] / "hdris"

BONE_COLOR_PALETTE = [
    (0.85, 0.82, 0.75), (0.62, 0.54, 0.42), (0.70, 0.60, 0.45),
    (0.45, 0.38, 0.30), (0.55, 0.52, 0.48), (0.75, 0.65, 0.50),
    (0.35, 0.28, 0.22), (0.80, 0.75, 0.68),
]


def parse_args():
    # blenderproc run ya separa sus propios argumentos de los del script
    # (ver command_line.py: sys.argv = sys.argv[:1] + unknown_args) -- no
    # queda un "--" literal que buscar, a diferencia de bpy a pelo.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--fill", type=float, default=0.72)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def has_real_texture(obj) -> bool:
    """Mismo criterio que render_bone_noisy.py (commit 5dbca0d). El glTF trae
    nodos Empty en la jerarquía (grupos/transform) además de las mallas
    reales -- `blender_obj.data` es None para esos, se ignoran."""
    if obj.blender_obj.type != "MESH" or obj.blender_obj.data is None:
        return False
    mesh = obj.blender_obj.data
    if mesh.color_attributes and len(mesh.color_attributes) > 0:
        return True
    for mat in mesh.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                return True
    return False


def apply_procedural_dirt(obj, rng):
    """Sustituto SIN descargas externas de la suciedad/desgaste de la
    propuesta original: ruido de Blender (Musgrave-like Noise Texture)
    modulando roughness y un tinte oscuro localizado, en vez de un único
    color/roughness plano -- se nota "sucio en parches", no uniforme."""
    mesh = obj.blender_obj.data
    mesh.materials.clear()
    mat = bpy.data.materials.new("bone_dirty")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    base_color = rng.choice(BONE_COLOR_PALETTE)
    jitter = lambda c: max(0.0, min(1.0, c + rng.uniform(-0.04, 0.04)))
    base_rgba = (*[jitter(c) for c in base_color], 1.0)

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = rng.uniform(8.0, 25.0)
    noise.inputs["Detail"].default_value = rng.uniform(2.0, 6.0)
    noise.inputs["Roughness"].default_value = rng.uniform(0.4, 0.8)

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = rng.uniform(0.35, 0.55)
    ramp.color_ramp.elements[0].color = (0.05, 0.04, 0.03, 1.0)  # mancha oscura
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = base_rgba
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    mix_color = nodes.new("ShaderNodeMixRGB")
    mix_color.blend_type = "MIX"
    mix_color.inputs["Fac"].default_value = rng.uniform(0.15, 0.45)  # cuánta suciedad se ve
    mix_color.inputs["Color1"].default_value = base_rgba
    links.new(ramp.outputs["Color"], mix_color.inputs["Color2"])
    links.new(mix_color.outputs["Color"], bsdf.inputs["Base Color"])

    rough_ramp = nodes.new("ShaderNodeValToRGB")
    rough_ramp.color_ramp.elements[0].position = 0.4
    rough_ramp.color_ramp.elements[0].color = (0.35, 0.35, 0.35, 1.0)
    rough_ramp.color_ramp.elements[1].position = 1.0
    rough_ramp.color_ramp.elements[1].color = (0.85, 0.85, 0.85, 1.0)
    links.new(noise.outputs["Fac"], rough_ramp.inputs["Fac"])
    links.new(rough_ramp.outputs["Color"], bsdf.inputs["Roughness"])

    mesh.materials.append(mat)


def fibonacci_sphere(n, radius):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(max(n - 1, 1))) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x, z = math.cos(theta) * r, math.sin(theta) * r
        points.append(Vector((x, y, z)) * radius)
    return points


def look_at_matrix(cam_location, target):
    direction = (target - cam_location)
    rot_quat = direction.to_track_quat("-Z", "Y")
    mat = rot_quat.to_matrix().to_4x4()
    mat.translation = cam_location
    return mat


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bproc.init()

    objs = bproc.loader.load_obj(args.mesh)
    if not objs:
        print("NO_OBJECTS_LOADED")
        return
    # NO se fusionan las sub-mallas del glTF en un único objeto:
    # bpy.ops.object.join() exige un contexto de UI que blenderproc (headless,
    # sin view_layer "real") no da de forma fiable incluso con
    # `context.temp_override` -- en vez de pelear con eso, se opera sobre la
    # LISTA de objetos: bbox/centroide agregados, material aplicado a cada
    # uno. El resultado visual es idéntico (todos se renderizan igual).
    mesh_objs = [o for o in objs if o.blender_obj.type == "MESH" and o.blender_obj.data is not None]
    if not mesh_objs:
        print("NO_MESH_OBJECTS")
        return
    raw_mesh_objs = [o.blender_obj for o in mesh_objs]
    keep_original = any(has_real_texture(o) for o in mesh_objs)

    # Centro/bbox en espacio de MUNDO -- no se tocan las locations de los
    # objetos (evita el bug de mover padres Y sus hijos del glTF a la vez,
    # que desplazaba el objeto por partida doble a través de la jerarquía).
    # La cámara apunta directamente a este centro en espacio de mundo.
    all_verts_world = []
    for o in raw_mesh_objs:
        mat_world = o.matrix_world
        all_verts_world.extend(mat_world @ v.co for v in o.data.vertices)
    verts = np.array([list(v) for v in all_verts_world])
    center = Vector(verts.mean(axis=0).tolist())

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    radius = float(np.linalg.norm(bbox_max - bbox_min)) / 2.0

    rng_material = random.Random(args.seed or abs(hash(args.prefix)) % (2**31))
    if not keep_original:
        for o in mesh_objs:
            apply_procedural_dirt(o, rng_material)

    # Suelo shadow-catcher para sombra de contacto real (igual que antes)
    bpy.ops.mesh.primitive_plane_add(size=radius * 12, location=(center.x, center.y, bbox_min[2]))
    plane = bpy.context.active_object
    plane.is_shadow_catcher = True

    hdri_files = sorted(HDRI_DIR.glob("*.hdr"))

    scene = bpy.context.scene
    scene.render.film_transparent = True
    bproc.camera.set_resolution(args.resolution, args.resolution)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_output_format(enable_transparency=True)

    base_seed = args.seed or abs(hash(args.prefix)) % (2**31)
    fill = args.fill
    positions = fibonacci_sphere(args.n_views, 1.0)  # direcciones unitarias, distancia por vista según focal

    for i, direction in enumerate(positions):
        rng = random.Random(base_seed + i)

        # HDRI real por vista (variedad de iluminación/reflejos)
        if hdri_files:
            bproc.world.set_world_background_hdr_img(str(rng.choice(hdri_files)))

        # Óptica variable: focal length 24-70mm en vez de FOV fijo. Se fija
        # ANTES de calcular la distancia (usa la misma sensor_width que
        # blenderproc asume por defecto para sensor_fit AUTO/HORIZONTAL) y
        # se aplica vía la API de blenderproc, no tocando `.data.lens` a
        # mano -- blenderproc gestiona los intrínsecos con su propia matriz
        # K y pisa asignaciones directas al render.
        focal_mm = rng.uniform(24, 70)
        sensor_width = 36.0  # sensor full-frame estándar (default de Blender para cámaras nuevas)
        bproc.camera.set_intrinsics_from_blender_params(lens=focal_mm, lens_unit="MILLIMETERS")
        bpy.context.scene.camera.data.sensor_width = sensor_width
        fov_half = math.atan(sensor_width / (2 * focal_mm))
        distance = radius / (fill * math.tan(fov_half))

        cam_location = center + direction * distance
        jitter = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.5, 0.5)))
        look_target = center + jitter * radius * 0.15
        cam2world = look_at_matrix(cam_location, look_target)
        bproc.camera.add_camera_pose(cam2world, frame=i)

        data = bproc.renderer.render()
        rgba = np.asarray(data["colors"][0])  # HxWx4, uint8

        out_path = out_dir / f"{args.prefix}_view{i:03d}.png"
        mask_path = out_dir / f"{args.prefix}_view{i:03d}_mask.png"
        from PIL import Image
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        # Máscara GT: alpha>250 = hueso sólido (excluye la sombra de
        # contacto, que queda con alpha parcial) -- exacta desde la
        # geometría, no adivinada sobre el render ya compuesto.
        mask = (rgba[:, :, 3] > 250).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(mask_path)

        bproc.utility.reset_keyframes()

    print(f"RENDER_DONE views={args.n_views} out={out_dir}")


if __name__ == "__main__":
    main()
