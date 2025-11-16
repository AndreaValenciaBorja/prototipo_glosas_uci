import streamlit as st
import pandas as pd
import joblib
import json
from io import BytesIO
from PIL import Image
import base64

# =========================
# CONFIGURACIÓN INICIAL
# =========================
st.set_page_config(page_title="Predicción de Glosas en UCI")

# =========================
# FUNCIÓN PARA CENTRAR IMAGEN (LOGO)
# =========================
def center_image(path, width=180):
    with open(path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-bottom:10px;">
            <img src="data:image/png;base64,{img_base64}" width="{width}">
        </div>
        """,
        unsafe_allow_html=True
    )

# Logo centrado
try:
    center_image("logo_fvl.png", width=180)
except Exception:
    st.write("")

# =========================
# TÍTULOS CENTRADOS
# =========================
st.markdown(
    """
    <style>
    h1, h2, h3{
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Predicción de glosas por estancia no pertinente en UCI")

# =========================
# CARGA DE MODELO & SCALER
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("modelo_rf_glosa.pkl")
    scaler = joblib.load("scaler_robust_glosa.pkl")
    with open("features_glosa.json") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_artifacts()

# =========================
# SECCIÓN 1 - CARGA ARCHIVO
# =========================
st.subheader("🗂️ Cargar archivo Excel")
uploaded_file = st.file_uploader(
    "Subir archivo (.xlsx) con los pacientes de UCI",
    type=["xlsx"]
)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        # 1) Verificar columnas
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(
                "Faltan estas columnas necesarias para el modelo:\n\n"
                + ", ".join(missing)
            )
        else:
            st.info("Predicción de riesgo de glosa en curso...")

            # 2) Ordenar columnas como las espera el modelo
            X = df[features]

            # 3) Escalar
            X_scaled = scaler.transform(X)

            # 4) Predicciones
            probas = model.predict_proba(X_scaled)[:, 1]
            preds = (probas >= 0.5).astype(int)

            # 5) Construir DataFrame resultado
            df_result = df.copy()
            df_result["prob_glosa"] = probas
            df_result["pred_glosa"] = preds

            # Pequeño resumen (opcional)
            n_total = len(df_result)
            n_glosa = int((df_result["pred_glosa"] == 1).sum())
            st.write(f"Total de pacientes procesados: **{n_total}**")
            st.write(f"Clasificados como glosa: **{n_glosa}** ({n_glosa/n_total:.1%})")

            # 6) Generar Excel en memoria
            buffer = BytesIO()
            df_result.to_excel(buffer, index=False)
            buffer.seek(0)

            # =========================
            # SECCIÓN 2 - DESCARGA
            # =========================
            st.subheader("📥 Descargar resultados")

            # Centrar botón
            st.markdown(
                "<div style='display:flex; justify-content:center;'>",
                unsafe_allow_html=True
            )

            st.download_button(
                label="Descargar Excel con predicciones",
                data=buffer,
                file_name="predicciones_glosas_uci.xlsx",
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                )
            )

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Ocurrió un error al procesar el archivo.")
        st.exception(e)

else:
    st.info("Por favor, cargue un archivo Excel para continuar.")
