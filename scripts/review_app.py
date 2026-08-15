"""App de revisión estilo Tinder para el backlog de fotos candidatas
(`data/real_photos_review/manifest.csv`, generado por `search_openverse.py`).

Muestra una foto pendiente a la vez con sus metadatos y persiste la decisión
al instante (nada de estado propio: la cola se recalcula leyendo el CSV en
cada carga, así que es reanudable sin perder progreso). Tres acciones:
- Aceptar: mueve el archivo a `data/real_photos_accepted/<grupo>/` y añade
  la fila a su manifest (con el split train/test elegido).
- Descartar: el archivo se queda donde está, solo cambia `status`.
- Duda: exige nota, pasa a una cola aparte ("Revisar dudas") para resolver
  después con más calma.

Uso:
    streamlit run scripts/review_app.py --server.port 8502
"""
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEW_DIR = BASE_DIR / "data/real_photos_review"
REVIEW_MANIFEST = REVIEW_DIR / "manifest.csv"
ACCEPTED_DIR = BASE_DIR / "data/real_photos_accepted"
ACCEPTED_MANIFEST = ACCEPTED_DIR / "manifest.csv"
ACCEPTED_COLUMNS = ["bone_group", "filename", "source_url", "license", "notes", "split"]

st.set_page_config(page_title="Revisión de fotos candidatas", page_icon="🦴", layout="wide")


def load_review() -> pd.DataFrame:
    return pd.read_csv(REVIEW_MANIFEST, dtype=str).fillna("")


def load_accepted() -> pd.DataFrame:
    if ACCEPTED_MANIFEST.exists():
        return pd.read_csv(ACCEPTED_MANIFEST, dtype=str).fillna("")
    return pd.DataFrame(columns=ACCEPTED_COLUMNS)


def merged_license(row: pd.Series) -> str:
    creator = row.get("creator", "").strip()
    license_ = row.get("license", "").strip()
    return f"{license_} - {creator}" if creator else license_


def set_review_status(review_df: pd.DataFrame, idx, new_status: str, note: str) -> None:
    review_df.loc[idx, "status"] = new_status
    review_df.loc[idx, "notes"] = note
    review_df.to_csv(REVIEW_MANIFEST, index=False)


def inject_keyboard_shortcuts() -> None:
    # Botones de Streamlit viven en el documento padre (este bloque se
    # renderiza en un iframe aparte) — mismo origen (localhost), así que
    # window.parent.document es accesible sin trucos de CORS.
    components.html(
        """
        <script>
        const doc = window.parent.document;
        function clickByLabel(fragment) {
            const buttons = doc.querySelectorAll('button');
            for (const b of buttons) {
                if (b.innerText.includes(fragment)) { b.click(); return; }
            }
        }
        doc.addEventListener('keydown', function(e) {
            const tag = (doc.activeElement && doc.activeElement.tagName) || '';
            if (tag === 'TEXTAREA' || tag === 'INPUT') return;
            if (e.key === 'ArrowRight') clickByLabel('Aceptar');
            if (e.key === 'ArrowLeft') clickByLabel('Descartar');
            if (e.key === 'ArrowDown') clickByLabel('Duda');
        });
        </script>
        """,
        height=0,
    )


review_df = load_review()

st.sidebar.title("🦴 Revisión de fotos candidatas")
mode = st.sidebar.radio("Cola", ["Pendientes", "Revisar dudas"])
target_status = "pending" if mode == "Pendientes" else "flagged"

queue_all = review_df[review_df["status"] == target_status].sort_values(["bone_group", "filename"])
groups = ["Todos"] + sorted(queue_all["bone_group"].unique().tolist())
group_choice = st.sidebar.selectbox("Grupo", groups)
queue = queue_all if group_choice == "Todos" else queue_all[queue_all["bone_group"] == group_choice]

search_text = st.sidebar.text_input(
    "Buscar (nombre o nota)",
    help="Filtra por texto en el nombre de archivo o la nota — ej. 'wolf', 'Canis', 'cow' para "
         "encontrar solo las candidatas de una búsqueda concreta.",
)
if search_text.strip():
    needle = search_text.strip().lower()
    mask = (queue["filename"].str.lower().str.contains(needle, regex=False)
            | queue["notes"].str.lower().str.contains(needle, regex=False))
    queue = queue[mask]

st.sidebar.markdown("---")
counts = review_df["status"].value_counts()
for status in ["pending", "accepted", "rejected", "flagged"]:
    st.sidebar.caption(f"{status}: {int(counts.get(status, 0))}")

if queue.empty:
    st.success(f"Cola '{mode}' vacía para el filtro elegido. 🎉")
    st.stop()

current = queue.iloc[0]
st.sidebar.markdown(f"**{len(queue)}** foto(s) en esta cola")

st.subheader(current["bone_group"])
st.markdown(
    f"<div style='word-break:break-all;color:gray;font-size:0.9rem;'>{current['filename']}</div>",
    unsafe_allow_html=True,
)
st.caption(f"{len(queue)} restantes en '{mode}' / grupo '{group_choice}'")

img_path = REVIEW_DIR / current["bone_group"] / current["filename"]
col_img, col_meta = st.columns([3, 2])
with col_img:
    if img_path.exists():
        st.image(str(img_path), width=440)
    else:
        st.error(f"No se encuentra el archivo: {img_path}")

with col_meta:
    st.markdown(f"**Fuente**: [{current['source_url']}]({current['source_url']})")
    st.markdown(f"**Licencia**: {current['license']}")
    st.markdown(f"**Autor**: {current.get('creator', '')}")

    note = st.text_area(
        "Nota (obligatoria para 'Duda'; si aceptas, queda como nota del hueso en el manifest)",
        value=current.get("notes", ""),
        key=f"note_{current['filename']}",
        height=90,
    )

    accepted_df = load_accepted()
    group_accepted = accepted_df[accepted_df["bone_group"] == current["bone_group"]]
    train_n = int((group_accepted["split"] == "train").sum())
    test_n = int((group_accepted["split"] == "test").sum())
    split_choice = st.radio(
        f"Split — grupo ya tiene {train_n} train / {test_n} test",
        ["train", "test"],
        horizontal=True,
        key=f"split_{current['filename']}",
    )

    col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✅ Aceptar (→)", use_container_width=True):
        dest_dir = ACCEPTED_DIR / current["bone_group"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / current["filename"]
        if img_path.exists():
            shutil.move(str(img_path), str(dest_path))
        new_row = {
            "bone_group": current["bone_group"],
            "filename": current["filename"],
            "source_url": current["source_url"],
            "license": merged_license(current),
            "notes": note,
            "split": split_choice,
        }
        acc_df = pd.concat([load_accepted(), pd.DataFrame([new_row])], ignore_index=True)
        acc_df.to_csv(ACCEPTED_MANIFEST, index=False)
        set_review_status(review_df, current.name, "accepted", note)
        st.rerun()

with col2:
    if st.button("❌ Descartar (←)", use_container_width=True):
        set_review_status(review_df, current.name, "rejected", note)
        st.rerun()

with col3:
    if st.button("❓ Duda (↓)", use_container_width=True):
        if not note.strip():
            st.warning("Escribe la duda en la nota antes de marcarla.")
        else:
            set_review_status(review_df, current.name, "flagged", note)
            st.rerun()

inject_keyboard_shortcuts()
