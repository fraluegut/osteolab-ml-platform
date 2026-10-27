import bpy, json, sys
src, dst = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.context.view_layer.objects.active = meshes[0]
for o in meshes: o.select_set(True)
if len(meshes) > 1: bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
n = len(obj.data.polygons)
mod = obj.modifiers.new("d", "DECIMATE"); mod.ratio = min(1.0, 2500 / n)
bpy.ops.object.modifier_apply(modifier="d")
me = obj.data; mw = obj.matrix_world
json.dump({"v": [list(mw @ v.co) for v in me.vertices], "f": [list(p.vertices) for p in me.polygons], "orig_faces": n}, open(dst, "w"))
