"""Script de Blender: importa una escena (fbx/glb/gltf/obj/blend, posiblemente
multi-objeto), funde TODO en un único objeto (igual que `render_bone_lab.py::
import_mesh`, necesario para tener un solo objeto de colisión por hueso) y lo
exporta como `.glb`. Complementa a `export_object_glb.py` (que exporta un
objeto SUELTO de una escena multi-hueso, sin fundir) para el caso contrario:
una escena de un solo espécimen que puede traer varias piezas (ej. cráneo +
mandíbula separados) que aquí SÍ deben fundirse en una sola malla.

Uso:
    blender --background --python shared/bone_ingest/convert_join_glb.py -- \\
        --scene <path> --out <ruta/malla.glb>
"""
import argparse
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
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
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif ext == ".blend":
        with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"formato de escena no soportado: {ext}")
    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not imported:
        raise SystemExit("sin objetos de malla en la escena")

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


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = import_scene(Path(args.scene))

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_path), export_format="GLB", use_selection=True, export_apply=True,
    )
    print(f"CONVERT_DONE out={out_path}")


if __name__ == "__main__":
    main()
