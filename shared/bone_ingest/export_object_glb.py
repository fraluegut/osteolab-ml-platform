"""Script de Blender (requiere `--background --python`): extrae UN objeto
concreto de una escena multi-objeto (misma lógica de import que
`blender_split_render.py`, sin fusionar nada) y lo exporta como .glb
independiente -- el paso que faltaba entre "aceptado en la cola de ingesta"
(`data/bone_ingest/queue.json`) y tener una malla usable por
`render_lab_batch.py` en `data/meshes/`.

Uso:
    blender --background --python shared/bone_ingest/export_object_glb.py -- \\
        --scene <path> --object-name l_talus --out <ruta/malla.glb>
"""
import argparse
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--object-name", required=True)
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
    else:
        raise ValueError(f"formato de escena no soportado: {ext}")
    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in imported:
        o.select_set(True)
    if imported:
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return imported


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objects = import_scene(Path(args.scene))

    match = next((o for o in objects if o.name == args.object_name), None)
    if match is None:
        available = ", ".join(sorted(o.name for o in objects))
        raise SystemExit(f"objeto '{args.object_name}' no encontrado. Disponibles: {available}")

    bpy.ops.object.select_all(action="DESELECT")
    match.select_set(True)
    bpy.context.view_layer.objects.active = match

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_path), export_format="GLB", use_selection=True,
        export_apply=True,
    )
    print(f"EXPORT_DONE object={args.object_name} out={out_path}")


if __name__ == "__main__":
    main()
