"""Lanza `blender/render_bone_lab.py` (escena de laboratorio: mesa + física +
luces + material en 2 capas) sobre todos los especímenes de `data/meshes/`,
uno por proceso Blender -- mismo patrón que `render_noisy_batch.py` (un
proceso por espécimen, para que un fallo aislado no tire el lote y para que
la memoria de Cycles se libere entre especímenes).

Escribe en `renders_lab/<especie>/<hueso>/<especimen>/`, misma convención de
carpetas que `renders_noisy/`.

Uso:
    python scripts/render_lab_batch.py
    python scripts/render_lab_batch.py --only homo_sapiens/femur
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # base_datos_osea/, pese al nombre -- este script vive en base_datos_osea/scripts/
MESHES_DIR = REPO_ROOT / "data" / "meshes"
BLENDER = Path("/opt/blender/blender")
RENDER_SCRIPT = REPO_ROOT / "blender" / "render_bone_lab.py"
MESH_EXTS = {".glb", ".gltf", ".stl", ".ply", ".obj"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "renders_lab")
    p.add_argument("--n-drops", type=int, default=2)
    p.add_argument("--views-per-drop", type=int, default=3)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--drop-frames", type=int, default=100)
    p.add_argument("--light-preset", default="calido_suave")
    p.add_argument("--only", default=None,
                    help="procesa solo especímenes cuya ruta relativa (especie/hueso/especimen) empiece por este prefijo")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def find_specimen_dirs():
    return sorted(d for d in MESHES_DIR.glob("*/*/*") if d.is_dir())


def find_mesh_file(specimen_dir):
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

        if args.skip_existing and out_dir.is_dir() and any(out_dir.glob(f"{prefix}_drop*_view*.png")):
            print(f"[{i + 1}/{len(specimen_dirs)}] {species}/{bone}/{specimen} ya existe, salto", flush=True)
            continue

        mesh_path = find_mesh_file(specimen_dir)
        print(f"[{i + 1}/{len(specimen_dirs)}] {species}/{bone}/{specimen} <- {mesh_path.name}", flush=True)
        cmd = [
            str(BLENDER), "--background", "--python", str(RENDER_SCRIPT), "--",
            "--mesh", str(mesh_path), "--out-dir", str(out_dir), "--prefix", prefix,
            "--n-drops", str(args.n_drops), "--views-per-drop", str(args.views_per_drop),
            "--samples", str(args.samples), "--drop-frames", str(args.drop_frames),
            "--light-preset", args.light_preset,
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
