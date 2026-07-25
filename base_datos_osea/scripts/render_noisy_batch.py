"""Lanza `blender/render_bone_noisy.py` sobre todos los especímenes de
`data/meshes/`, uno por proceso Blender (`--background`), escribiendo cada
uno en `renders_noisy/<especie>/<hueso>/<especimen>/` — la misma estructura
que esperan `build_dataset.py` (repo padre) y `scripts/compose_noisy.py`.

Un proceso Blender por espécimen (no todo el lote en un solo proceso) para
que un fallo aislado en una malla no tire el resto del lote, y para que la
memoria de Cycles se libere entre especímenes.

Nota (2026-07-24): un intento anterior de este mismo script tenía un bug de
índices al extraer especie/hueso/especimen de la ruta con `cut -d/ -f2,3`,
que no contaba con el prefijo `data/meshes/` de la ruta real y escribió una
carpeta corrupta — aquí se usa `Path.relative_to(MESHES_DIR).parts[:3]`, sin
manipulación de texto por posición, para que ese bug no se pueda repetir.

Uso:
    python scripts/render_noisy_batch.py
    python scripts/render_noisy_batch.py --only homo_sapiens/femur
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MESHES_DIR = REPO_ROOT / "data" / "meshes"
BLENDER = Path("/opt/blender/blender")
RENDER_SCRIPT = REPO_ROOT / "blender" / "render_bone_noisy.py"
MESH_EXTS = {".glb", ".gltf", ".stl", ".ply", ".obj"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "renders_noisy")
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--only", default=None,
                    help="procesa solo especímenes cuya ruta relativa (especie/hueso/especimen) empiece por este prefijo")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def find_specimen_dirs():
    return sorted(d for d in MESHES_DIR.glob("*/*/*") if d.is_dir())


def find_mesh_file(specimen_dir):
    # La malla puede estar directamente en el directorio del espécimen
    # (especímenes de Sketchfab) o anidada un nivel más (especímenes de
    # MorphoSource, bajo una carpeta "Media .../") — de ahí el rglob en vez
    # de listar solo el nivel superior.
    candidates = [p for p in specimen_dir.rglob("*") if p.suffix.lower() in MESH_EXTS]
    if len(candidates) != 1:
        raise SystemExit(f"Esperaba exactamente 1 malla en {specimen_dir}, encontré {len(candidates)}: {candidates}")
    return candidates[0]


def main():
    args = parse_args()
    specimen_dirs = find_specimen_dirs()
    if args.only:
        specimen_dirs = [d for d in specimen_dirs if str(d.relative_to(MESHES_DIR)).startswith(args.only)]

    print(f"{len(specimen_dirs)} especímenes a procesar", flush=True)
    failures = []
    for i, specimen_dir in enumerate(specimen_dirs):
        species, bone, specimen = specimen_dir.relative_to(MESHES_DIR).parts[:3]
        out_dir = args.out_dir / species / bone / specimen
        prefix = f"{species}_{bone}_{specimen}"

        if args.skip_existing and out_dir.is_dir() and any(out_dir.glob(f"{prefix}_view*.png")):
            print(f"[{i + 1}/{len(specimen_dirs)}] {species}/{bone}/{specimen} ya existe, salto", flush=True)
            continue

        mesh_path = find_mesh_file(specimen_dir)
        print(f"[{i + 1}/{len(specimen_dirs)}] {species}/{bone}/{specimen} <- {mesh_path.name}", flush=True)
        cmd = [
            str(BLENDER), "--background", "--python", str(RENDER_SCRIPT), "--",
            "--mesh", str(mesh_path), "--out-dir", str(out_dir), "--prefix", prefix,
            "--n-views", str(args.n_views), "--samples", str(args.samples),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or "RENDER_DONE" not in result.stdout:
            print(f"  FALLO: {specimen_dir}", flush=True)
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            failures.append(str(specimen_dir))

    print(f"\nBATCH_DONE total={len(specimen_dirs)} fallos={len(failures)}", flush=True)
    if failures:
        print("Especímenes fallidos:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
