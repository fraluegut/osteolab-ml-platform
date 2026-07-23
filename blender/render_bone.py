"""Rig de captura multi-vista para un hueso: centra la malla en su centro de masa,
coloca una cámara por punto de una esfera de Fibonacci alrededor y renderiza cada
vista sobre fondo blanco con iluminación suave omnidireccional.

Pensado para alimentar `src/cv_extractor/extract.py` de osteolab-ml-platform (Hu
moments, ratios de forma, etc. se calculan ahí, sobre estas imágenes 2D — este
script solo genera las fotos, no mide nada).

Se ejecuta con Blender, no con python suelto (necesita el módulo `bpy`):

    blender --background --python blender/render_bone.py -- \\
        --mesh data/meshes/homo_sapiens/femur/000031310/.../Human5-000031310.stl \\
        --out-dir renders --prefix homo_sapiens_femur_000031310 --n-views 24
"""

import argparse
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", required=True, help="prefijo de fichero, p.ej. homo_sapiens_femur_000031310")
    p.add_argument("--n-views", type=int, default=24)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT"])
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--fov-deg", type=float, default=40.0)
    p.add_argument("--fill", type=float, default=0.72, help="fracción del encuadre que ocupa el hueso (margen = 1-fill)")
    p.add_argument("--view-transform", default="Standard", choices=["Standard", "AgX", "Filmic", "Raw"],
                    help="AgX (por defecto de Blender 4.x) desatura y lava el contraste; Standard da imagenes nítidas mejores para segmentación")
    return p.parse_args(argv)


def import_mesh(path):
    ext = path.suffix.lower()
    before = set(bpy.data.objects)
    if ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"formato no soportado: {ext}")
    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]

    # El importador glTF anida cada malla bajo Empties con escala/traslación no
    # triviales (Sketchfab exporta con su propio sistema de unidades) — sin esto,
    # los vértices en obj.data quedan en espacio local del objeto, no en el mundo,
    # y todo el centrado/encuadre posterior (que lee obj.data.vertices) sale mal
    # (malla invisible o fuera de cuadro). Aplanar parent+transform lo corrige
    # para cualquier formato, no solo glTF.
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


def volume_centroid_local(obj):
    # bpy.ops.object.origin_set falla en modo --background sin contexto de ventana
    # (no lanza error, simplemente no hace nada) — se calcula a mano el centroide de
    # volumen (asumiendo densidad uniforme) por descomposición en tetraedros desde el
    # origen local, el mismo método que usa internamente ORIGIN_CENTER_OF_VOLUME.
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    total_volume = 0.0
    weighted_sum = Vector((0.0, 0.0, 0.0))
    for face in bm.faces:
        v0, v1, v2 = (v.co for v in face.verts)
        signed_vol = v0.dot(v1.cross(v2)) / 6.0
        centroid = (v0 + v1 + v2) / 4.0
        total_volume += signed_vol
        weighted_sum += centroid * signed_vol
    bm.free()
    if abs(total_volume) < 1e-9:
        raise ValueError("malla sin volumen cerrado detectable (¿normales invertidas o no-manifold?)")
    return weighted_sum / total_volume


def center_on_mass(obj):
    centroid_local = volume_centroid_local(obj)
    obj.data.transform(Matrix.Translation(-centroid_local))
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()


def framing_target(obj):
    """Centro y radio de encuadre — DISTINTOS del centro de masa a propósito.

    El hueso se posiciona en su centro de masa (correcto físicamente), pero para
    huesos alargados y asimétricos (fémur: cabeza pesada + diáfisis larga y fina)
    el centro de masa NO es el centro visual — apuntar la cámara ahí deja el hueso
    pegado a un borde del cuadro en vez de centrado (verificado empíricamente: un
    encuadre basado en el centro de masa sólo llenaba ~47% del cuadro y muy
    descentrado; basado en el centro de la caja delimitadora, ~72% bien centrado,
    tal como se pedía). Por eso cámaras y luces orbitan sobre el centro de la bbox,
    no sobre el origen del objeto.
    """
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    center = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
    radius = max((Vector((x, y, z)) - center).length for x, y, z in zip(xs, ys, zs))
    return center, radius


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
    # 4 luces de área grandes y suaves en disposición tetraédrica: dan relieve sin
    # sombras duras, y son fijas en el espacio (no dependen del ángulo de cámara).
    # La potencia se escala con radius^2: la distancia y el tamaño de la luz ya
    # escalan con radius (linealmente), así que la irradiancia sobre la superficie
    # cae como 1/radius^2 si no se compensa — sin esto, un hueso pequeño (p.ej. un
    # sesamoideo de unos mm) sale completamente sobreexpuesto frente a uno grande
    # como el fémur, con la misma potencia fija.
    energy_per_radius_sq = 450 / (190.0 ** 2)  # calibrado visualmente sobre el fémur (radius≈190)
    energy = energy_per_radius_sq * radius ** 2
    offsets = [(1, 1, 1), (-1, -1, 1), (-1, 1, -0.3), (1, -1, -0.3)]
    for i, (x, y, z) in enumerate(offsets):
        v = center + Vector((x, y, z)).normalized() * radius * 2.2
        light_data = bpy.data.lights.new(f"fill_{i}", type="AREA")
        light_data.energy = energy
        light_data.size = radius * 1.5
        light_obj = bpy.data.objects.new(f"fill_{i}", light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = v
        direction = (center - v).normalized()
        light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def fibonacci_sphere(n, radius):
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(max(n - 1, 1))) * 2
        r_xy = math.sqrt(max(0.0, 1 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * r_xy
        z = math.sin(theta) * r_xy
        points.append(Vector((x, y, z)) * radius)
    return points


def make_camera(fov_deg, clip_end):
    cam_data = bpy.data.cameras.new("bone_cam")
    cam_data.lens_unit = "FOV"
    # sensor_fit='VERTICAL' fija sin ambigüedad qué eje controla `angle`: con
    # 'AUTO' y sensor 36x24 (3:2), un render cuadrado puede acabar aplicando el
    # FOV al eje horizontal y dejando el vertical más estrecho de lo esperado.
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = math.radians(fov_deg)
    # las mallas vienen en mm con distancias de cámara de hasta varios miles de
    # unidades: el clip_end por defecto (1000) las recorta y deja el render en blanco.
    cam_data.clip_start = 0.01
    cam_data.clip_end = clip_end
    cam_obj = bpy.data.objects.new("bone_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def point_at(obj, target):
    direction = (target - obj.location).normalized()
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()
    mesh_path = Path(args.mesh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = args.view_transform
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type in ("CUDA", "OPTIX")
        scene.cycles.device = "GPU"

    obj = import_mesh(mesh_path)
    center_on_mass(obj)  # posición física: centro de masa del hueso en el origen
    obj.data.materials.clear()
    obj.data.materials.append(make_matte_material())

    center, radius = framing_target(obj)  # encuadre: centro de la bbox (ver framing_target)
    setup_world()
    add_fill_lights(center, radius)

    fov_half = math.radians(args.fov_deg) / 2
    distance = radius / (args.fill * math.tan(fov_half))
    cam = make_camera(args.fov_deg, clip_end=distance * 3)

    for i, offset in enumerate(fibonacci_sphere(args.n_views, distance)):
        cam.location = center + offset
        point_at(cam, center)
        scene.render.filepath = str(out_dir / f"{args.prefix}_view{i:03d}.png")
        bpy.ops.render.render(write_still=True)

    print(f"RENDER_DONE views={args.n_views} out={out_dir}")


if __name__ == "__main__":
    main()
