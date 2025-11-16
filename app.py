import streamlit as st
import pandas as pd
import joblib
import json
from io import BytesIO
from PIL import Image

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Predicción de Glosas en UCI - FVL",
    page_icon="🩺",
    layout="wide",
)

# =========================
# ESTILOS (COLORES FVL)
# =========================
st.markdown(
    """
    <style>
    /* Fondo general en tono beige */
    [data-testid="stAppViewContainer"] {
        background-color: #F5F2EA;
    }

    /* Contenedor central más angosto y limpio */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Botones tipo FVL (verde vivo y redondeados) */
    .stButton > button {
        background-color: #69BE28;
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #4E9B1E;
    }

    /* Títulos estilo FVL */
    .fvl-title {
        font-size: 28px;
        font-weight: 700;
        color: #005F3B;
        margin-bottom: 0.2rem;
    }
    .fvl-subtitle {
        font-size: 15px;
        color: #555555;
        margin-top: 0.1rem;
    }

    /* Tarjetas de información tipo métricas */
    .fvl-card {
        background-color: white;
        border-radius: 16px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    .fvl-card-number {
        font-size: 24px;
        font-weight: 700;
        color: #007A3D;
        margin-bottom: 0.2rem;
    }
    .fvl-card-label {
        font-size: 13px;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# CARGAR ARTEFACTOS DEL MODELO
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
# ENCABEZADO TIPO FVL
# =========================
with st.container():
    col_logo, col_text = st.columns([1, 3])

    with col_logo:
        try:
            logo = Image.open("logo_fvl.png")
            st.image(logo, use_column_width=True)
        except Exception:
            st.write("")  # si no encuentra el logo, no rompe

    with col_text:
        st.markdown(
            '<p class="fvl-title">Predicción de glosas por estancia no pertinente en UCI</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="fvl-subtitle">Prototipo de apoyo a la auditoría médica en la Fundación Valle del Lili</p>',
            unsafe_allow_html=True
        )

st.markdown("---")

# =========================
# BARRA LATERAL (UMBRAL)
# =========================
st.sidebar.header("Configuración del modelo")
umbral = st.sidebar.slider(
    "Umbral de clasificación de glosa",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Valores por encima del umbral se clasifican como 'Glosa'."
)

st.sidebar.info(
    f"Con el umbral actual ({umbral:.2f}), los casos con probabilidad "
    "igual o superior se marcarán como Glosa (1)."
)

# =========================
# TARJETAS RESUMEN (PLACEHOLDER)
# =========================
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="fvl-card">'
            '<div class="fvl-card-number">UCI Adultos</div>'
            '<div class="fvl-card-label">Servicio objetivo del modelo</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="fvl-card">'
            '<div class="fvl-card-number">2022–2024</div>'
            '<div class="fvl-card-label">Periodo de datos analizados</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="fvl-card">'
            '<div class="fvl-card-number">Random Forest</div>'
            '<div class="fvl-card-label">Modelo seleccionado para el prototipo</div>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("")

# =========================
# SECCIÓN PRINCIPAL: CARGA Y PREDICCIÓN
# =========================
st.subheader("1️⃣ Cargar archivo de pacientes de UCI")

st.write(
    "Suba un archivo Excel con las variables clínicas y administrativas requeridas "
    "para cada episodio de UCI. El sistema aplicará el escalado y el modelo entrenado "
    "para estimar la probabilidad de glosa por pertinencia de estancia."
)

uploaded_file = st.file_uploader(
    "Subir archivo (.xlsx)", 
    type=["xlsx"], 
    help="Use el archivo de ejemplo generado desde SAP o desde el notebook."
)

if uploaded_file is not None:
    # Leer archivo
    df = pd.read_excel(uploaded_file)
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df.head())

    # Verificar columnas requeridas
    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(
            "Faltan columnas necesarias para el modelo:\n\n"
            + ", ".join(missing)
        )
    else:
        st.subheader("2️⃣ Predicción de riesgo de glosa")

        # Ordenar columnas como las espera el modelo
        X = df[features]

        # Escalado
        X_scaled = scaler.transform(X)

        # Probabilidades y predicciones según umbral
        probas = model.predict_proba(X_scaled)[:, 1]
        preds = (probas >= umbral).astype(int)

        # Construir DataFrame de salida
        df_result = df.copy()
        df_result["prob_glosa"] = probabilidad
        df_result["pred_glosa"] = predicción

        # Resumen numérico
        n_total = len(df_result)
        n_glosa = int((df_result["pred_glosa"] == 1).sum())
        st.write(f"**Total de registros procesados:** {n_total}")
        st.write(f"**Casos clasificados como glosa (1):** {n_glosa} ({n_glosa/n_total:.1%})")

        st.write("Tabla con las predicciones (primeras filas):")
        st.dataframe(df_result.head())

        # Preparar archivo para descarga
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
