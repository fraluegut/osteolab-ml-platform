"""Script de Blender (requiere `--background --python`, no se importa como
módulo normal): carga una escena completa (FBX/glTF/OBJ/blend) SIN fusionar
objetos — a diferencia de `base_datos_osea/blender/render_bone.py`, que
importa un único hueso y junta las piezas si vienen repartidas en varios
objetos, aquí cada objeto de la escena debe quedar separado, porque cada uno
puede ser un hueso distinto.

Renderiza UNA miniatura por objeto (no las 24 vistas de entrenamiento — esto
es solo para que un humano reconozca el hueso en "Ampliar Base de Datos" /
"Identificación Manual"; el multi-vista de verdad se genera después, si el
hueso se acepta, con `render_bone.py` sobre el objeto ya separado).

Uso:
    blender --background --python shared/bone_ingest/blender_split_render.py -- \\
        --scene <path> --out-dir <dir> --resolution 256
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--fov-deg", type=float, default=40.0)
    p.add_argument("--fill", type=float, default=0.72)
    p.add_argument("--min-verts", type=int, default=8,
                    help="objetos con menos vértices que esto se ignoran (helpers/ruido, no huesos)")
    return p.parse_args(argv)


def import_scene(path: Path):
    ext = path.suffix.lower()
    before = set(bpy.data.objects)
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"formato de escena no soportado: {ext}")

    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]

    # aplanar parent+transform (igual que render_bone.py) para que
    # obj.data.vertices/obj.dimensions queden en espacio de mundo correcto —
    # SIN join(), a propósito: aquí cada objeto se queda separado.
    bpy.ops.object.select_all(action="DESELECT")
    for o in imported:
        o.select_set(True)
    if imported:
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return imported


def make_matte_material():
    mat = bpy.data.materials.new("bone_matte")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.62, 0.54, 0.42, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    return mat


def setup_world(strength=0.9):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = strength


def add_fill_lights(center, radius):
    # misma calibración que render_bone.py (energía ~ radius^2, ver ahí el porqué).
    energy_per_radius_sq = 450 / (190.0 ** 2)
    energy = energy_per_radius_sq * max(radius, 1e-4) ** 2
    offsets = [(1, 1, 1), (-1, -1, 1), (-1, 1, -0.3), (1, -1, -0.3)]
    lights = []
    for i, (x, y, z) in enumerate(offsets):
        v = center + Vector((x, y, z)).normalized() * radius * 2.2
        light_data = bpy.data.lights.new(f"fill_{i}", type="AREA")
        light_data.energy = energy
        light_data.size = max(radius * 1.5, 0.01)
        light_obj = bpy.data.objects.new(f"fill_{i}", light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = v
        direction = (center - v).normalized()
        light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        lights.append(light_obj)
    return lights


def make_camera(fov_deg):
    cam_data = bpy.data.cameras.new("bone_cam")
    cam_data.lens_unit = "FOV"
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = math.radians(fov_deg)
    cam_data.clip_start = 0.001
    cam_data.clip_end = 1e6
    cam_obj = bpy.data.objects.new("bone_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def point_at(obj, target):
    direction = (target - obj.location).normalized()
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def bbox_center_radius(obj):
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    center = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
    radius = max((Vector((x, y, z)) - center).length for x, y, z in zip(xs, ys, zs))
    return center, radius


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name).strip("_") or "unnamed"


def main():
    args = parse_args()
    scene_path = Path(args.scene)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type in ("CUDA", "OPTIX")
        scene.cycles.device = "GPU"

    objects = import_scene(scene_path)
    for obj in objects:
        obj.hide_render = True

    setup_world()
    mat = make_matte_material()
    cam = make_camera(args.fov_deg)

    items = []
    seen_names: dict[str, int] = {}
    for obj in objects:
        n_verts = len(obj.data.vertices)
        if n_verts < args.min_verts:
            continue

        safe = sanitize(obj.name)
        count = seen_names.get(safe, 0)
        seen_names[safe] = count + 1
        filename = f"{safe}.png" if count == 0 else f"{safe}_{count}.png"

        item = {"object_name": obj.name, "vertex_count": n_verts, "thumbnail": filename}
        lights = []
        try:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            center, radius = bbox_center_radius(obj)
            item["dimensions_m"] = [round(d, 4) for d in obj.dimensions]

            lights = add_fill_lights(center, radius)
            fov_half = math.radians(args.fov_deg) / 2
            distance = radius / (args.fill * math.tan(fov_half)) if radius > 1e-9 else 1.0
            cam.location = center + Vector((0.6, -1.0, 0.5)).normalized() * distance
            point_at(cam, center)

            obj.hide_render = False
            scene.render.filepath = str(out_dir / filename)
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            item["error"] = str(exc)
        finally:
            obj.hide_render = True
            for light in lights:
                bpy.data.objects.remove(light, do_unlink=True)
        items.append(item)

    manifest_path = out_dir / "parts.json"
    manifest_path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"SPLIT_RENDER_DONE n_objects={len(objects)} n_items={len(items)} out={out_dir}")


if __name__ == "__main__":
    main()
