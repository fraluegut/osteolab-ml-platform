import blenderproc as bproc

bproc.init()
objs = bproc.loader.load_obj("base_datos_osea/data/meshes/homo_sapiens/femur/sketchfab_a9c1f1a88b104c3fbfe975fa10b31b31/human_femur.glb")
print("LOADED", len(objs), "objects")
print("BPROC_TEST_OK")
