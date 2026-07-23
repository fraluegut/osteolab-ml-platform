import base64
import hashlib
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# URL base de la API FastAPI; se puede sobreescribir con la variable de entorno OSTEOLAB_API_URL
API_URL = os.getenv("OSTEOLAB_API_URL", "http://localhost:8000")

# Configuración de la página: título e ícono de la pestaña del navegador, en
# layout ancho para que cada paso tenga sitio de sobra (nada de columnas
# estrechas apretando pasos que en realidad son secuenciales).
st.set_page_config(page_title="OsteoLab - Clasificador de huesos", page_icon="🦴", layout="wide")


def _reset_state():
    st.session_state.stage = "idle"  # idle -> clip_checked -> processed
    st.session_state.clip_result = None
    st.session_state.classify_result = None


if "file_id" not in st.session_state:
    st.session_state.file_id = None
    _reset_state()

# La imagen se sube y se previsualiza en la barra lateral: así el área
# principal queda libre para mostrar los pasos del pipeline a todo lo ancho.
with st.sidebar:
    st.title("🦴 OsteoLab")
    st.caption(f"API: `{API_URL}`")
    uploaded_file = st.file_uploader("Sube una imagen de un hueso", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Imagen subida", width="stretch")

st.header("Clasificador de huesos")

if uploaded_file is None:
    st.info("Sube una imagen desde la barra lateral para empezar.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_id = hashlib.md5(file_bytes).hexdigest()
if file_id != st.session_state.file_id:
    # Nueva imagen: se descarta cualquier resultado de la anterior
    st.session_state.file_id = file_id
    _reset_state()


def _file_payload():
    return {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}


@st.cache_data(ttl=3600)
def _load_reference_points(api_url: str) -> pd.DataFrame:
    """Las ~1300 vistas usadas para entrenar, en coordenadas PCA — no cambian
    entre peticiones (solo si se re-entrena), así que se piden una vez por
    hora como mucho, no en cada rerender de Streamlit."""
    resp = requests.get(f"{api_url}/pca/reference", timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["points"])


# --- Paso 1: filtro CLIP zero-shot (endpoint /filter/clip), disparado a demanda ---
with st.container(border=True):
    st.subheader("Paso 1 · Filtro CLIP (zero-shot): ¿es un hueso?")

    if st.session_state.stage == "idle":
        if st.button("▶️ Procesar", type="primary", key="procesar_btn"):
            with st.spinner("Analizando con CLIP..."):
                resp = requests.post(f"{API_URL}/filter/clip", files=_file_payload(), timeout=30)
            if resp.ok:
                st.session_state.clip_result = resp.json()
                st.session_state.stage = "clip_checked"
                st.rerun()
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
    else:
        clip_result = st.session_state.clip_result
        c1, c2 = st.columns([1, 2])
        c1.metric(
            "Resultado CLIP",
            "Sí, es hueso" if clip_result["is_bone"] else "No es un hueso",
            f"{clip_result['confidence'] * 100:.1f}% de confianza",
        )

        with c2:
            if not clip_result["is_bone"]:
                st.warning(
                    "Tras analizarlo, no creemos que la imagen contenga un hueso. "
                    "Prueba con otra imagen."
                )
            else:
                st.success(
                    "Tras analizarlo, consideramos que en la imagen aparece un hueso. "
                    "Vamos a continuar procesándolo."
                )
                suggestion = clip_result["suggestion"]
                st.caption(
                    f"Pista orientativa de CLIP sobre el tipo: **{suggestion['type']}** "
                    f"({suggestion['confidence'] * 100:.1f}%) — se guarda como una variable "
                    "más del análisis, no como resultado definitivo."
                )

# --- Disparo del Paso 2: medir con OpenCV + clasificar (un único endpoint) ---
if st.session_state.stage == "clip_checked" and st.session_state.clip_result["is_bone"]:
    if st.session_state.classify_result is None:
        with st.spinner("Midiendo con OpenCV y comparando contra lo ya conocido..."):
            resp = requests.post(f"{API_URL}/classify", files=_file_payload(), timeout=30)
        if not resp.ok:
            st.error(f"Error {resp.status_code}: {resp.text}")
            st.stop()
        st.session_state.classify_result = resp.json()

# --- Paso 2: medidas OpenCV + Paso 3: clasificación y posición en el espacio de features ---
if st.session_state.classify_result is not None:
    result = st.session_state.classify_result
    features = result["cv_features"]

    with st.container(border=True):
        st.subheader("Paso 2 · Localización y medidas con OpenCV")

        if not features["found"]:
            st.warning("OpenCV no consiguió aislar una región clara de hueso en la imagen.")
        else:
            # Las dos imágenes van lado a lado (no apiladas) y con un ancho
            # limitado, para que una foto en vertical no obligue a bajar mucho.
            img_a, img_b = st.columns(2)
            with img_a:
                if features.get("annotated_image_base64"):
                    st.image(
                        base64.b64decode(features["annotated_image_base64"]),
                        caption="Medidas en píxeles pegadas a cada lado del hueso detectado",
                        width=340,
                    )
            with img_b:
                if features.get("mask_overlay_base64"):
                    st.image(
                        base64.b64decode(features["mask_overlay_base64"]),
                        caption="Región que OpenCV identifica como hueso (para verificar la segmentación)",
                        width=340,
                    )

            # Métricas en rejillas compactas debajo de las imágenes, en vez
            # de una columna larga al lado.
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Ancho", f"{features['bounding_box_px']['width']}px")
            m2.metric("Alto", f"{features['bounding_box_px']['height']}px")
            m3.metric("% ocupado", f"{features['area_ratio'] * 100:.1f}%")
            m4.metric("Aspecto", features["aspect_ratio"])
            m5.metric("Extensión", features["extent"])
            m6.metric("Solidez", features["solidity"])

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Elongación", features["elongation"])
            n2.metric("Ratio elipse", features["ellipse_ratio"])
            n3.metric("Circularidad", features["circularity"])
            n4.metric("Convexidad", features["convexity"])

            rb = features["rotated_box_px"]
            st.caption(
                f"Rectángulo rotado (orientación real): {rb['width']}×{rb['height']} px, "
                f"{rb['angle_deg']}°"
            )

            st.caption(
                f"Anchura a lo largo del hueso (20 cortes) — media {features['mean_width']}px, "
                f"mín {features['min_width']}px, máx {features['max_width']}px, "
                f"desv. {features['std_width']}px · curvatura media {features['curvature_mean']}, "
                f"máx {features['curvature_max']}"
            )
            st.line_chart(features["width_profile"], height=140)

            with st.expander("Ver todas las medidas (JSON, incluye momentos de Hu)"):
                st.json(features)

    if features["found"]:
        with st.container(border=True):
            st.subheader("Paso 3 · ¿Qué tipo de hueso es?")
            st.caption(
                "El modelo compara las medidas de arriba contra las de ~1300 vistas de huesos ya "
                "identificados (misma especie o no) y da una probabilidad por grupo morfológico — "
                "cráneo, hueso largo, vértebra... no por hueso exacto ni por especie: con 1-2 "
                "especímenes reales por hueso concreto el modelo no puede generalizar más allá de "
                "\"reconozco este objeto\", así que se agrupa por forma general (ver HISTORIAL.md "
                "en base_datos_osea para el porqué)."
            )

            probs = result["probabilities"]
            probs_sorted = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
            top_group, top_prob = next(iter(probs_sorted.items()))

            c1, c2 = st.columns([1, 2])
            c1.metric("Grupo más probable", top_group, f"{top_prob * 100:.1f}%")
            with c2:
                st.bar_chart(probs_sorted)

            st.markdown("#### Dónde cae esta imagen respecto a lo ya conocido")
            st.caption(
                "Proyección PCA (2 componentes) de las mismas medidas geométricas — cada punto de "
                "fondo es una vista de un hueso ya identificado, coloreada por grupo; la estrella "
                "negra es la imagen que acabas de subir."
            )
            try:
                reference_df = _load_reference_points(API_URL)
                fig = px.scatter(
                    reference_df, x="pca_x", y="pca_y", color="bone_group",
                    hover_data=["species", "bone", "specimen"],
                    opacity=0.55,
                )
                fig.add_scatter(
                    x=[result["pca_point"]["x"]], y=[result["pca_point"]["y"]],
                    mode="markers", marker=dict(symbol="star", size=22, color="black",
                                                 line=dict(width=1, color="white")),
                    name="Tu imagen",
                )
                fig.update_layout(height=520, legend_title_text="Grupo morfológico")
                st.plotly_chart(fig, width="stretch")
            except requests.RequestException as e:
                st.error(f"No se pudieron cargar los puntos de referencia: {e}")

    if st.button("Analizar otra imagen"):
        _reset_state()
        st.rerun()
