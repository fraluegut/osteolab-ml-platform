#!/usr/bin/env bash
# Driver de lote para render_bone_bproc.py -- mismo criterio de especímenes
# que render_noisy_batch.py (bpy a pelo), pero N_VIEWS reducido (6 en vez de
# 10) para que el lote completo (122 especímenes) quepa en un tiempo
# razonable dentro de esta migración -- decisión de tiempo, no de calidad;
# ver HISTORIAL.md 2026-07-26 (migración BlenderProc).
set -uo pipefail
cd "$(dirname "$0")/../.."

MESHES_DIR="base_datos_osea/data/meshes"
OUT_ROOT="base_datos_osea/renders_noisy_bproc"
N_VIEWS=6
LOG="/tmp/render_noisy_bproc.log"
> "$LOG"

mapfile -t SPECIMEN_DIRS < <(find base_datos_osea/renders_noisy -mindepth 3 -maxdepth 3 -type d | sed 's#base_datos_osea/renders_noisy/##' | sort -u)

total=${#SPECIMEN_DIRS[@]}
i=0
for rel in "${SPECIMEN_DIRS[@]}"; do
  i=$((i+1))
  species=$(echo "$rel" | cut -d/ -f1)
  bone=$(echo "$rel" | cut -d/ -f2)
  specimen=$(echo "$rel" | cut -d/ -f3)
  mesh_dir="$MESHES_DIR/$rel"
  mesh_file=$(find "$mesh_dir" -maxdepth 1 -iname "*.glb" -o -iname "*.gltf" -o -iname "*.ply" -o -iname "*.stl" -o -iname "*.obj" 2>/dev/null | head -1)
  if [ -z "$mesh_file" ]; then
    echo "[$i/$total] SIN MALLA: $rel" | tee -a "$LOG"
    continue
  fi
  prefix="${species}_${bone}_${specimen}"
  out_dir="$OUT_ROOT/$species/$bone/$specimen"
  mkdir -p "$out_dir"
  echo "[$i/$total] $rel" | tee -a "$LOG"
  timeout 180 .venv/bin/blenderproc run --custom-blender-path /opt/blender \
    base_datos_osea/blender/render_bone_bproc.py -- \
    --mesh "$mesh_file" --out-dir "$out_dir" --prefix "$prefix" --n-views "$N_VIEWS" --samples 48 \
    >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$i/$total] FALLO: $rel" | tee -a "$LOG"
  fi
done

echo "BATCH_DONE total=$total" | tee -a "$LOG"
