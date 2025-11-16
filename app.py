import streamlit as st
import pandas as pd
import joblib
import json
from io import BytesIO
from PIL import Image  

# =========================
# CONFIGURACIÓN INICIAL
# =========================
st.set_page_config(page_title="Predicción de Glosas en UCI")

# =========================
# LOGO FUNDACIÓN
# =========================
try:
    logo = Image.open("logo_fvl.png")  
    st.image(logo, width=180)           
except:
    st.write(" ")  # si no encuentra el logo, no se rompe

# =========================
# TÍTULO Y DESCRIPCIÓN
# =========================
st.title("Prototipo: Predicción de glosas por estancia no pertinente en UCI")
st.write("""
Esta herramienta recibe un archivo Excel con pacientes de UCI y 
devuelve un archivo con la probabilidad estimada de glosa por pertinencia de estancia.
""")

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
st.subheader("1️⃣ Cargar archivo Excel")
uploaded_file = st.file_uploader("Subir archivo (.xlsx) con los pacientes de UCI", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df.head())

    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Faltan estas columnas necesarias para el modelo: {missing}")

    else:
        st.subheader("2️⃣ Predicción de riesgo de glosa")

        X = df[features]
        X_scaled = scaler.transform(X)

        probas = model.predict_proba(X_scaled)[:, 1]
        preds = (probas >= 0.5).astype(int)

        df_result = df.copy()
        df_result["prob_glosa"] = probas
        df_result["pred_glosa"] = preds

        st.write("Tabla con las predicciones (primeras filas):")
        st.dataframe(df_result.head())

        buffer = BytesIO()
        df_result.to_excel(buffer, index=False)
        buffer.seek(0)

        st.subheader("3️⃣ Descargar resultados")
        st.download_button(
            label="📥 Descargar Excel con predicciones",
            data=buffer,
            file_name="predicciones_glosas_uci.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Por favor, cargue un archivo Excel para continuar.")
