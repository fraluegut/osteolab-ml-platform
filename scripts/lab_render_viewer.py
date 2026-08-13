"""Visor de pruebas de render_bone_lab.py — galería de vistas + máscara,
agrupadas por espécimen/caída, para juzgar la escena nueva sin abrir
ficheros a mano.

Uso:
    streamlit run scripts/lab_render_viewer.py --server.port 8505 -- \\
        --dir base_datos_osea/renders_lab_test
"""
import re
import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Render lab — pruebas", page_icon="🦴", layout="wide")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
default_dir = "base_datos_osea/renders_lab_test"
if "--dir" in argv:
    default_dir = argv[argv.index("--dir") + 1]

st.sidebar.title("🦴 Render lab — pruebas")
dir_input = st.sidebar.text_input("Carpeta de renders", value=default_dir)
render_dir = BASE_DIR / dir_input

if not render_dir.exists():
    st.error(f"No existe: {render_dir}")
    st.stop()

show_masks = st.sidebar.checkbox("Mostrar máscara junto a cada vista", value=False)

all_pngs = sorted(render_dir.glob("*.png"))
color_pngs = [p for p in all_pngs if not p.stem.endswith("_mask") and "_alpha_tmp" not in p.stem]

PATTERN = re.compile(r"^(?P<prefix>.+)_drop(?P<drop>\d+)_view(?P<view>\d+)$")
groups: dict[str, list[Path]] = {}
for p in color_pngs:
    m = PATTERN.match(p.stem)
    prefix = m.group("prefix") if m else p.stem
    groups.setdefault(prefix, []).append(p)

if not groups:
    st.info("Carpeta vacía o sin renders todavía.")
    st.stop()

specimen = st.sidebar.selectbox("Espécimen", sorted(groups.keys()))
st.caption(f"Carpeta: `{render_dir}` — {len(color_pngs)} vistas totales, {len(groups)} especímenes")

files = sorted(groups[specimen])
cols_per_row = 2 if show_masks else 3
for i in range(0, len(files), cols_per_row):
    row_files = files[i: i + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, path in zip(cols, row_files):
        with col:
            st.image(str(path), caption=path.stem, use_container_width=True)
            if show_masks:
                mask_path = path.parent / f"{path.stem}_mask.png"
                if mask_path.exists():
                    st.image(str(mask_path), caption="máscara", use_container_width=True)
