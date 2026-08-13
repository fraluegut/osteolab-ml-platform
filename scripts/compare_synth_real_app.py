"""Comparador visual sintético vs real, lado a lado, por grupo morfológico.

Pensado para juzgar a ojo qué tan lejos está el estilo de render sintético
del real (motivación: igualar apariencia) y, aparte, qué ve realmente la CNN
tras la segmentación (fondo gris/negro plano).

Uso:
    streamlit run scripts/compare_synth_real_app.py --server.port 8504
"""
import random
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
COMBINED_PATH = BASE_DIR / "data/processed/bone_geometric_features_combined.csv"
ACCEPTED_DIR = BASE_DIR / "data/real_photos_accepted"
ACCEPTED_MANIFEST = ACCEPTED_DIR / "manifest.csv"
REAL_SEGMENTED_DIR = BASE_DIR / "data/segmented_crops/real_photos"

st.set_page_config(page_title="Sintético vs real", page_icon="🦴", layout="wide")


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


RENDER_TYPE_LABELS = {
    "renders": "renders (silueta plana, sin sombreado)",
    "renders_noisy_composited": "renders_noisy_composited (fondo textura aleatoria)",
    "renders_noisy_bproc": "renders_noisy_bproc (BlenderProc, fotorealista)",
}


@st.cache_data
def load_synthetic() -> pd.DataFrame:
    df = pd.read_csv(COMBINED_PATH, usecols=["bone_group", "species", "source_path", "segmented_path"])
    df = df[df["species"] != "real"].copy()
    df["render_type"] = df["source_path"].str.extract(r"/(renders(?:_noisy_\w+)?)/")
    return df


@st.cache_data
def load_real() -> pd.DataFrame:
    df = pd.read_csv(ACCEPTED_MANIFEST, dtype=str)
    df["raw_path"] = df.apply(lambda r: str(ACCEPTED_DIR / r["bone_group"] / r["filename"]), axis=1)
    df["segmented_path"] = df.apply(
        lambda r: str(REAL_SEGMENTED_DIR / r["bone_group"] / (Path(r["filename"]).stem + ".png")), axis=1
    )
    return df


synthetic_df = load_synthetic()
real_df = load_real()

st.sidebar.title("🦴 Sintético vs real")
groups = sorted(set(synthetic_df["bone_group"]) & set(real_df["bone_group"]))
group_choice = st.sidebar.selectbox("Grupo morfológico", groups)
view = st.sidebar.radio("Vista", ["Crudo (estilo/render)", "Segmentado (lo que ve la CNN)"])
n_samples = st.sidebar.slider("Fotos por lado", 3, 15, 9)

render_types = sorted(synthetic_df["render_type"].dropna().unique())
render_type_choice = st.sidebar.multiselect(
    "Sub-pipeline sintético a incluir",
    render_types,
    default=render_types,
    format_func=lambda t: RENDER_TYPE_LABELS.get(t, t),
)

if "shuffle_seed" not in st.session_state:
    st.session_state.shuffle_seed = 0
if st.sidebar.button("🔀 Nueva muestra"):
    st.session_state.shuffle_seed += 1

rng = random.Random(st.session_state.shuffle_seed)


def sample_rows(df: pd.DataFrame, group: str, col: str, n: int, caption_col: str | None = None) -> list[tuple]:
    sub = df[df["bone_group"] == group]
    sub = sub[sub[col].notna()]
    sub = sub[sub[col].map(lambda p: resolve(p).exists())]
    records = list(zip(sub[col], sub[caption_col] if caption_col else [None] * len(sub)))
    if len(records) <= n:
        return records
    return rng.sample(records, n)


def render_grid(records: list[tuple], cols: int = 3, width: int = 220) -> None:
    if not records:
        st.info("Sin imágenes disponibles para este filtro.")
        return
    rows = [records[i : i + cols] for i in range(0, len(records), cols)]
    for row in rows:
        columns = st.columns(cols)
        for col, (path, caption) in zip(columns, row):
            with col:
                st.image(str(resolve(path)), width=width, caption=caption)


synth_col_name = "source_path" if view.startswith("Crudo") else "segmented_path"
real_col_name = "raw_path" if view.startswith("Crudo") else "segmented_path"

synth_filtered = synthetic_df[synthetic_df["render_type"].isin(render_type_choice)]

st.caption(f"Grupo: **{group_choice}** — vista: **{view}**")
col_synth, col_real = st.columns(2)
with col_synth:
    st.subheader(f"Sintético ({len(synth_filtered[synth_filtered.bone_group == group_choice])} disponibles)")
    render_grid(sample_rows(synth_filtered, group_choice, synth_col_name, n_samples, caption_col="render_type"))
with col_real:
    st.subheader(f"Real ({len(real_df[real_df.bone_group == group_choice])} disponibles)")
    render_grid(sample_rows(real_df, group_choice, real_col_name, n_samples, caption_col="split"))
