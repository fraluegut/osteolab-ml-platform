#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG="/tmp/render_sample9.log"
> "$LOG"

SPECIMENS=(
  "gorilla_gorilla/femur/sketchfab_3fbe6a09153b4a0cb38d15005726d8ff"
  "alouatta_palliata/mandible/nmnhvz_7235149"
  "alouatta_palliata/cranium/nmnhvz_7283884"
  "homo_sapiens/vertebra/sketchfab_7c5e902085294a6fa9fda5cc775e0e17"
  "homo_sapiens/metacarpal_1/sketchfab_153fe961b2cc40dd85357f33ca97d7c6"
  "homo_sapiens/scapula/sketchfab_da38f448a06749daa09859d767971088"
  "homo_sapiens/pelvis/sketchfab_10f950afa4284019a1d744ac14b6cf90"
  "homo_sapiens/rib/sketchfab_cb2f2ea263024913afc76e418c766ea9"
  "homo_sapiens/sacrum/sketchfab_49aafdac520046ab97afc33a26d47d0c"
)

i=0
for rel in "${SPECIMENS[@]}"; do
  i=$((i+1))
  species=$(echo "$rel" | cut -d/ -f1)
  bone=$(echo "$rel" | cut -d/ -f2)
  specimen=$(echo "$rel" | cut -d/ -f3)
  mesh_dir="base_datos_osea/data/meshes/$rel"
  mesh_file=$(find "$mesh_dir" -maxdepth 1 -iname "*.glb" -o -iname "*.gltf" -o -iname "*.ply" -o -iname "*.stl" -o -iname "*.obj" 2>/dev/null | head -1)
  if [ -z "$mesh_file" ]; then
    echo "[$i/9] SIN MALLA: $rel" | tee -a "$LOG"; continue
  fi
  out_dir="base_datos_osea/renders_noisy_bproc/$species/$bone/$specimen"
  mkdir -p "$out_dir"
  echo "[$i/9] $rel -> $mesh_file" | tee -a "$LOG"
  timeout 180 .venv/bin/blenderproc run --custom-blender-path /opt/blender \
    base_datos_osea/blender/render_bone_bproc.py -- \
    --mesh "$mesh_file" --out-dir "$out_dir" --prefix "${species}_${bone}_${specimen}" \
    --n-views 6 --samples 48 >> "$LOG" 2>&1
  echo "[$i/9] listo (exit $?)" | tee -a "$LOG"
done
echo "SAMPLE9_DONE" | tee -a "$LOG"
