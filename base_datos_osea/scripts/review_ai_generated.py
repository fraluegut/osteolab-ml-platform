"""Revisor manual de las imágenes generadas por `generate_ai_variants.py`:
muestra el render original y la versión "fotorrealizada" lado a lado, una a
una, con dos botones (Aceptar/Rechazar) — igual que un swipe de aceptar o
descartar. La decisión se guarda en el momento en `manifest.csv`, así que se
puede cerrar y retomar la revisión cuando se quiera sin perder lo ya hecho.

Uso:
    streamlit run base_datos_osea/scripts/review_ai_generated.py --server.port 8502

(puerto distinto del 8501 de la UI de producción, para poder tener las dos
abiertas a la vez sin conflicto)
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = BASE_DIR / "data/ai_generated_review/manifest.csv"
ACCEPTED_DIR = BASE_DIR / "data/ai_generated_accepted"

st.set_page_config(page_title="Revisor de fotos generadas por IA", layout="wide")


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        st.error(f"No existe {MANIFEST_PATH}. Generar candidatos primero con "
                 "`python -m base_datos_osea.scripts.generate_ai_variants`.")
        st.stop()
    return pd.read_csv(MANIFEST_PATH)


def save_manifest(df: pd.DataFrame):
    df.to_csv(MANIFEST_PATH, index=False)


def accept(df: pd.DataFrame, idx: int):
    row = df.loc[idx]
    dest_dir = ACCEPTED_DIR / row["bone_group"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(row["generated_path"]).name
    Image.open(row["generated_path"]).convert("RGB").save(dest_path, quality=92)
    df.loc[idx, "status"] = "accepted"
    df.loc[idx, "accepted_path"] = str(dest_path)
    save_manifest(df)


def reject(df: pd.DataFrame, idx: int):
    df.loc[idx, "status"] = "rejected"
    save_manifest(df)


def main():
    df = load_manifest()
    if "accepted_path" not in df.columns:
        df["accepted_path"] = ""

    pending = df[df["status"] == "pending"]
    n_total = len(df)
    n_accepted = (df["status"] == "accepted").sum()
    n_rejected = (df["status"] == "rejected").sum()
    n_pending = len(pending)

    st.title("Revisor de fotos generadas por IA")
    st.caption(f"{n_total} candidatos totales — {n_accepted} aceptados, {n_rejected} rechazados, "
               f"{n_pending} pendientes")
    st.progress(0 if n_total == 0 else (n_accepted + n_rejected) / n_total)

    if pending.empty:
        st.success("No quedan candidatos pendientes de revisar.")
        st.info("Generar más con `python -m base_datos_osea.scripts.generate_ai_variants "
                "--n-per-group N` (usa `--seed` distinto o se saltará lo ya generado).")
        return

    idx = pending.index[0]
    row = df.loc[idx]

    st.subheader(f"Grupo: `{row['bone_group']}`  ·  espécimen: `{row['specimen']}`")
    col1, col2 = st.columns(2)
    with col1:
        st.image(row["source_path"], caption="Original (render 3D limpio)", use_container_width=True)
    with col2:
        st.image(row["generated_path"], caption="Generado por IA (candidato)", use_container_width=True)

    st.write("")
    col_reject, col_accept = st.columns(2)
    with col_reject:
        if st.button("⬅️ Rechazar", use_container_width=True, type="secondary"):
            reject(df, idx)
            st.rerun()
    with col_accept:
        if st.button("Aceptar ➡️", use_container_width=True, type="primary"):
            accept(df, idx)
            st.rerun()

    with st.expander("Detalles del candidato"):
        st.write(f"- Prompt: {row['prompt']}")
        st.write(f"- Strength: {row['strength']}")
        st.write(f"- Fuente: `{row['source_path']}`")
        st.write(f"- Generado: `{row['generated_path']}`")


if __name__ == "__main__":
    main()
