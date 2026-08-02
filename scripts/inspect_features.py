"""Inspector visual de `extract_features()`: para una imagen elegida a mano
(foto real del banco, render limpio o render con ruido), compara los TRES
backends de segmentación (`classic`, `fastsam`, `grounded_sam`) lado a lado
— mismo extractor que usa el pipeline de clasificación/cascada, para poder
juzgar a ojo qué superficie mide cada uno y si la medida tiene sentido antes
de confiar en ella para diseñar una puerta. Ver HISTORIAL.md 2026-07-26 para
los casos que motivaron añadir `grounded_sam`.

Uso:
    streamlit run scripts/inspect_features.py --server.port 8503
"""
import base64
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # streamlit no añade la raíz del repo por defecto

from shared.cv_extractor.extract import extract_features  # noqa: E402

REAL_DIR = BASE_DIR / "data/real_world_benchmark"
REAL_MANIFEST = REAL_DIR / "manifest.csv"
CLEAN_CSV = BASE_DIR / "data/processed/bone_geometric_features.csv"
NOISY_CSV = BASE_DIR / "data/processed/bone_geometric_features_noisy.csv"

BACKENDS = ["classic", "fastsam", "grounded_sam"]
BACKEND_LABELS = {
    "classic": "classic (Otsu+adaptativo)",
    "fastsam": "fastsam (punto central)",
    "grounded_sam": "grounded_sam (texto \"bone\")",
}

st.set_page_config(page_title="Inspector de features geométricas", layout="wide")
st.title("Inspector de features geométricas")
st.caption("Comparando classic vs fastsam vs grounded_sam sobre la misma imagen")

SOURCES = {
    "Fotos reales (banco de evaluación)": "real",
    "Renders limpios (entrenamiento)": "clean",
    "Renders con ruido compuesto": "noisy",
}


@st.cache_data
def load_real_manifest():
    return pd.read_csv(REAL_MANIFEST)


@st.cache_data
def load_render_paths(csv_path: str):
    return pd.read_csv(csv_path, usecols=["bone_group", "specimen", "view", "source_path"])


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


@st.cache_data(show_spinner=False)
def run_backend(img_bytes: bytes, backend: str) -> dict:
    try:
        return extract_features(BytesIO(img_bytes), backend=backend)
    except Exception as exc:  # un backend roto no debe tumbar la comparación de los otros
        return {"found": False, "error": str(exc)}


with st.sidebar:
    source_label = st.radio("Origen de la imagen", list(SOURCES.keys()))
    source = SOURCES[source_label]

    if source == "real":
        manifest = load_real_manifest()
        group = st.selectbox("Grupo óseo", sorted(manifest["bone_group"].unique()))
        sub = manifest[manifest["bone_group"] == group].reset_index(drop=True)
        idx = st.selectbox("Foto", range(len(sub)), format_func=lambda i: sub.loc[i, "filename"])
        row = sub.loc[idx]
        img_path = REAL_DIR / row["bone_group"] / row["filename"]
    else:
        csv_path = str(CLEAN_CSV if source == "clean" else NOISY_CSV)
        paths_df = load_render_paths(csv_path)
        group = st.selectbox("Grupo óseo", sorted(paths_df["bone_group"].unique()))
        sub_group = paths_df[paths_df["bone_group"] == group]
        specimen = st.selectbox("Espécimen", sorted(sub_group["specimen"].unique()))
        sub = sub_group[sub_group["specimen"] == specimen].reset_index(drop=True)
        idx = st.selectbox("Vista", range(len(sub)), format_func=lambda i: sub.loc[i, "view"])
        row = sub.loc[idx]
        img_path = resolve_path(row["source_path"])

    extra_meta = row.to_dict()

if not img_path.exists():
    st.error(f"No existe el fichero: {img_path}")
    st.stop()

st.caption(str(img_path))
img_bytes = img_path.read_bytes()

st.image(Image.open(BytesIO(img_bytes)), caption="Original", width=350)

st.subheader("Superficie segmentada por cada backend")
results = {}
cols = st.columns(3)
for backend, col in zip(BACKENDS, cols):
    with col:
        st.markdown(f"**{BACKEND_LABELS[backend]}**")
        with st.spinner(f"Corriendo {backend}..."):
            result = run_backend(img_bytes, backend)
        results[backend] = result
        if result.get("error"):
            st.error(f"Error: {result['error']}")
        elif not result.get("found"):
            st.warning("No se detectó ningún contorno (`found=False`)")
        else:
            mask_img = Image.open(BytesIO(base64.b64decode(result["mask_overlay_base64"])))
            st.image(mask_img, caption=f"área={result['area_ratio']*100:.1f}%")

st.subheader("Tabla comparativa de features")
rows = {}
for backend, result in results.items():
    if not result.get("found"):
        continue
    flat = dict(result)
    hu = flat.pop("hu_moments", {})
    flat.pop("width_profile", None)
    flat.pop("annotated_image_base64", None)
    flat.pop("mask_overlay_base64", None)
    flat = {**flat, **hu}
    for nested_key in ("image_size", "bounding_box_px", "rotated_box_px"):
        nested = flat.pop(nested_key, {})
        for k, v in nested.items():
            flat[f"{nested_key}.{k}"] = v
    rows[BACKEND_LABELS[backend]] = flat

if rows:
    df_compare = pd.DataFrame(rows).sort_index()
    st.dataframe(df_compare, use_container_width=True, height=600)
else:
    st.info("Ningún backend encontró un contorno en esta imagen.")

st.subheader("Detalle por backend (contorno + bounding box + perfil de anchura)")
detail_backend = st.selectbox(
    "Backend a inspeccionar en detalle", [b for b in BACKENDS if results[b].get("found")],
    format_func=lambda b: BACKEND_LABELS[b],
)
if detail_backend:
    detail = results[detail_backend]
    ann_img = Image.open(BytesIO(base64.b64decode(detail["annotated_image_base64"])))
    st.image(ann_img, caption=BACKEND_LABELS[detail_backend], width=500)
    width_profile = detail.get("width_profile")
    if width_profile:
        st.line_chart(pd.Series(width_profile, name="ancho (px)"))

with st.expander("Metadatos de la fila de origen (CSV / manifest)"):
    st.json({k: (None if pd.isna(v) else v) for k, v in extra_meta.items()})
