import base64
import hashlib
import os

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
    st.session_state.cv_result = None
    st.session_state.extra_result = None


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

# --- Contexto opcional del usuario + disparo del Paso 2 (OpenCV) ---
if st.session_state.stage == "clip_checked" and st.session_state.clip_result["is_bone"]:
    with st.container(border=True):
        st.subheader("¿Tienes datos del hueso?")
        st.caption("Es opcional: si los rellenas se usarán, si no, se ignoran.")
        with st.form("context_form"):
            c1, c2, c3 = st.columns(3)
            largo_cm = c1.text_input("Largo (cm)")
            ancho_cm = c2.text_input("Ancho (cm)")
            profundo_cm = c3.text_input("Profundidad (cm)")

            c4, c5, c6 = st.columns(3)
            tipo_hueso = c4.text_input("Tipo de hueso (si lo sabes)")
            peso_g = c5.text_input("Peso (g)")
            color = c6.text_input("Color / tonalidad")

            c7, c8 = st.columns(2)
            estado_conservacion = c7.selectbox(
                "Estado de conservación",
                ["", "Completo", "Fragmentado", "Erosionado", "Desconocido"],
            )
            procedencia = c8.text_input("Procedencia / lugar del hallazgo")
            notas = st.text_area("Notas adicionales")
            submitted = st.form_submit_button("Continuar análisis", type="primary")

        if submitted:
            context_values = {
                "largo_cm": largo_cm,
                "ancho_cm": ancho_cm,
                "profundo_cm": profundo_cm,
                "tipo_hueso": tipo_hueso,
                "peso_g": peso_g,
                "color": color,
                "estado_conservacion": estado_conservacion,
                "procedencia": procedencia,
                "notas": notas,
            }
            data = {k: v for k, v in context_values.items() if v}

            with st.spinner("Localizando el hueso con OpenCV..."):
                cv_resp = requests.post(
                    f"{API_URL}/features/extract", files=_file_payload(), data=data, timeout=30
                )
            if not cv_resp.ok:
                st.error(f"Error {cv_resp.status_code}: {cv_resp.text}")
            else:
                st.session_state.cv_result = cv_resp.json()

                # Resultados adicionales (filtro ResNet18 entrenado + clasificador de
                # tipo), calculados una sola vez aquí para no repetir llamadas en cada
                # rerender de Streamlit.
                extra = {}
                filter_resp = requests.post(
                    f"{API_URL}/filter/is-bone", files=_file_payload(), timeout=30
                )
                if filter_resp.ok:
                    extra["filter"] = filter_resp.json()
                    if extra["filter"]["is_bone"]:
                        predict_resp = requests.post(
                            f"{API_URL}/predict", files=_file_payload(), timeout=30
                        )
                        if predict_resp.ok:
                            extra["predict"] = predict_resp.json()
                        else:
                            extra["predict_error"] = f"{predict_resp.status_code}: {predict_resp.text}"
                else:
                    extra["filter_error"] = f"{filter_resp.status_code}: {filter_resp.text}"

                st.session_state.extra_result = extra
                st.session_state.stage = "processed"
                st.rerun()

# --- Paso 2: resultado de OpenCV + resultados adicionales ---
if st.session_state.stage == "processed":
    cv_result = st.session_state.cv_result
    features = cv_result["cv_features"]

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

            # Métricas en una rejilla compacta debajo de las imágenes, en vez
            # de una columna larga al lado.
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Ancho", f"{features['bounding_box_px']['width']}px")
            m2.metric("Alto", f"{features['bounding_box_px']['height']}px")
            m3.metric("% ocupado", f"{features['area_ratio'] * 100:.1f}%")
            m4.metric("Aspecto", features["aspect_ratio"])
            m5.metric("Extensión", features["extent"])
            m6.metric("Solidez", features["solidity"])
            rb = features["rotated_box_px"]
            st.caption(
                f"Rectángulo rotado (orientación real): {rb['width']}×{rb['height']} px, "
                f"{rb['angle_deg']}°"
            )

            with st.expander("Ver todas las medidas (JSON)"):
                st.json(features)

        if cv_result["context"]:
            st.write("Contexto que aportaste:")
            st.json(cv_result["context"])

    with st.expander("Resultados adicionales (filtro ResNet18 entrenado + clasificador de tipo)"):
        extra = st.session_state.extra_result
        if "filter_error" in extra:
            st.error(f"Error {extra['filter_error']}")
        else:
            filter_result = extra["filter"]
            st.metric(
                "Filtro ResNet18",
                "Sí, es hueso" if filter_result["is_bone"] else "No es un hueso",
                f"{filter_result['confidence'] * 100:.1f}% de confianza",
            )
            if "predict" in extra:
                result = extra["predict"]
                st.metric("Predicción", result["prediction"])
                st.bar_chart(result["probabilities"])
                st.json(result["probabilities"])
            elif "predict_error" in extra:
                st.error(f"Error {extra['predict_error']}")

    if st.button("Analizar otra imagen"):
        _reset_state()
        st.rerun()
