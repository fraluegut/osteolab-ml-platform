import blenderproc as bproc  # noqa: E402 -- primer import, lo exige blenderproc run

"""Prueba aislada del material procedural multicapa (bone_material.py) sobre
una sola malla, sin tocar el pipeline de producción (render_bone_bproc.py).

Reutiliza el mismo setup mínimo de cámara/HDRI que render_bone_bproc.py para
que el resultado sea comparable, pero no tiene cascada de vistas ni
guarda máscara -- es solo para validar visualmente el material antes de
integrarlo.

Uso:
    blenderproc run --custom-blender-path /opt/blender \\
        base_datos_osea/blender/test_bone_material.py -- \\
        --mesh <ruta.glb> --out <salida.png> --seed 42
"""

import argparse
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("render_bone_bproc", Path(__file__).resolve().parent / "render_bone_bproc.py")
_rbp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_rbp)
apply_procedural_dirt = _rbp.apply_procedural_dirt  # noqa: E402

HDRI_DIR = Path(__file__).resolve().parents[1] / "hdris"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--hdri", default=None)
    return p.parse_args(argv)


def look_at_matrix(cam_location, target):
    direction = target - cam_location
    rot_quat = direction.to_track_quat("-Z", "Y")
    mat = rot_quat.to_matrix().to_4x4()
    mat.translation = cam_location
    return mat


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bproc.init()

    objs = bproc.loader.load_obj(args.mesh)
    mesh_objs = [o for o in objs if o.blender_obj.type == "MESH" and o.blender_obj.data is not None]
    if not mesh_objs:
        print("NO_MESH_OBJECTS")
        return

    raw_mesh_objs = [o.blender_obj for o in mesh_objs]
    all_verts_world = []
    for o in raw_mesh_objs:
        mat_world = o.matrix_world
        all_verts_world.extend(mat_world @ v.co for v in o.data.vertices)
    verts = np.array([list(v) for v in all_verts_world])
    center = Vector(verts.mean(axis=0).tolist())
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    radius = float(np.linalg.norm(bbox_max - bbox_min)) / 2.0

    rng = random.Random(args.seed)
    for o in mesh_objs:
        apply_procedural_dirt(o, rng)

    bpy.ops.mesh.primitive_plane_add(size=radius * 12, location=(center.x, center.y, bbox_min[2]))
    plane = bpy.context.active_object
    plane.is_shadow_catcher = True

    hdri_files = sorted(HDRI_DIR.glob("*.hdr"))
    scene = bpy.context.scene
    scene.render.film_transparent = True
    bproc.camera.set_resolution(args.resolution, args.resolution)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_output_format(enable_transparency=True)

    if args.hdri:
        bproc.world.set_world_background_hdr_img(str(HDRI_DIR / args.hdri))
    elif hdri_files:
        bproc.world.set_world_background_hdr_img(str(rng.choice(hdri_files)))

    # Misma fórmula que render_bone_bproc.py: distancia desde el FOV real
    # (focal + sensor), no un factor inventado -- si no, el encuadre no es
    # representativo de lo que de verdad genera el pipeline.
    focal_mm = 50.0
    sensor_width = 36.0
    bproc.camera.set_intrinsics_from_blender_params(lens=focal_mm, lens_unit="MILLIMETERS")
    bpy.context.scene.camera.data.sensor_width = sensor_width
    fov_half = math.atan(sensor_width / (2 * focal_mm))
    fill = 0.72
    distance = radius / (fill * math.tan(fov_half))
    cam_location = center + Vector((0.3, -1.0, 0.4)).normalized() * distance
    cam_pose = look_at_matrix(cam_location, center)
    bproc.camera.add_camera_pose(cam_pose)

    data = bproc.renderer.render()
    bproc.writer.write_hdf5(str(out_path.parent / "_tmp_hdf5"), data)

    # write_hdf5 no da PNG directo -- se extrae del array ya en memoria.
    from PIL import Image
    colors = data["colors"][0]
    Image.fromarray(colors.astype(np.uint8)).save(out_path)
    print(f"RENDER_OK {out_path}")


if __name__ == "__main__":
    main()
