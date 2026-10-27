"""Versión condensada e ILUSTRATIVA (no es el pipeline real, ver
render_bone_bproc.py) de los conceptos que se repiten en cada render:
1. la bola (fibonacci_sphere) que reparte posiciones de cámara,
2. centrar la vista en el objeto,
3. mover la cámara a cada vértice de la bola mirando siempre al centro,
4. cambiar fondo y luces entre vista y vista.
"""
import math
import random

import bpy
from mathutils import Vector


# 1) LA BOLA: puntos bien repartidos en una esfera (no al azar, que se
#    amontona en los polos) -- cada punto es una posición de cámara.
def fibonacci_sphere(n_points, radius=1.0):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))          # ángulo dorado
    for i in range(n_points):
        y = 1 - (i / (n_points - 1)) * 2             # de +1 a -1
        r = math.sqrt(1 - y * y)
        theta = phi * i
        x, z = math.cos(theta) * r, math.sin(theta) * r
        points.append(Vector((x, y, z)) * radius)
    return points


# 2) CENTRAR LA VISTA: el centro del objeto en espacio de MUNDO, no su
#    location (si el objeto es un hijo de otro, moverlo lo desplaza doble).
def object_center(obj):
    verts_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs, ys, zs = zip(*[(v.x, v.y, v.z) for v in verts_world])
    return Vector(((min(xs) + max(xs)) / 2,
                   (min(ys) + max(ys)) / 2,
                   (min(zs) + max(zs)) / 2))


# 3) MOVER LA CÁMARA A CADA VÉRTICE DE LA BOLA, MIRANDO AL CENTRO
def point_camera_at(cam, cam_location, target):
    cam.location = cam_location
    direction = target - cam_location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# 4) CAMBIAR FONDO Y LUCES ENTRE VISTA Y VISTA
FONDOS = [(0.05, 0.05, 0.07), (0.45, 0.42, 0.38), (0.5, 0.48, 0.44)]


def cambiar_ambiente(world, luz, i):
    world.node_tree.nodes["Background"].inputs[0].default_value = (*FONDOS[i % len(FONDOS)], 1.0)
    luz.data.energy = random.uniform(400, 1500)      # variedad de iluminación


# --- recorrido completo: una vista por cada punto de la bola ---
obj = bpy.context.object
cam = bpy.data.objects["Camera"]
world = bpy.context.scene.world

center = object_center(obj)
for i, direction in enumerate(fibonacci_sphere(n_points=12, radius=3.0)):
    point_camera_at(cam, center + direction, center)
    cambiar_ambiente(world, bpy.data.objects["Light"], i)
    bpy.ops.render.render(write_still=True)
