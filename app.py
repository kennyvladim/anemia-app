# =============================================================================
# APP STREAMLIT — Predicción de Anemia Infantil (ENDES 2024 Perú)
# =============================================================================
# Autor      : Kenny Mallqui
# Descripción: Interfaz web para predecir el riesgo de anemia en niños
#              menores de 5 años usando un modelo Random Forest entrenado
#              con datos de la ENDES 2024 (INEI Perú).
# Modelo     : Pipeline(SimpleImputer → StandardScaler → RandomForestClassifier)
# AUC-ROC    : ~0.72–0.75 (conjunto de prueba)
# Referencia : Yimer et al. (2025); ENDES 2024 — INEI Perú
#
# ESTRUCTURA DE ARCHIVOS REQUERIDA:
#   ├── app.py                  ← este archivo
#   ├── modelo_anemia_rf.pkl    ← pipeline serializado con joblib
#   ├── predictores.json        ← lista ordenada de 31 nombres de predictores
#   └── requirements.txt        ← dependencias del proyecto
# =============================================================================

import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ── Configuración de página ───────────────────────────────────────────────────
# layout="wide" aprovecha el ancho completo para mostrar los 31 campos del
# formulario de manera cómoda sin scroll horizontal.
st.set_page_config(
    page_title="Predicción de Anemia Infantil · ENDES 2024",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS personalizados ────────────────────────────────────────────────
# Se inyectan estilos inline para:
#   - Tarjetas de resultado con color semántico (rojo=alto, verde=bajo)
#   - Títulos de sección con línea inferior roja
#   - Texto de probabilidad de gran tamaño para lectura rápida
#   - Sidebar con fondo rosado suave acorde al tema de salud
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #C0392B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    .result-card-alto {
        background: linear-gradient(135deg, #FDEDEC, #FADBD8);
        border-left: 6px solid #E74C3C;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .result-card-bajo {
        background: linear-gradient(135deg, #EAFAF1, #D5F5E3);
        border-left: 6px solid #27AE60;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .prob-text {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #2C3E50;
        border-bottom: 2px solid #E74C3C;
        padding-bottom: 4px;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .disclaimer {
        font-size: 0.78rem;
        color: #95A5A6;
        font-style: italic;
        margin-top: 1rem;
    }
    [data-testid="stSidebar"] {
        background-color: #FDF2F2;
    }
</style>
""", unsafe_allow_html=True)


# ── Carga del modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """
    Carga el pipeline de ML y la lista de predictores desde disco.

    Usa @st.cache_resource para que el modelo se cargue una sola vez
    por sesión del servidor, evitando re-lecturas innecesarias del .pkl
    en cada interacción del usuario.

    Returns
    -------
    modelo : sklearn.pipeline.Pipeline
        Pipeline con pasos: SimpleImputer → StandardScaler → RandomForestClassifier.
        Entrenado con scikit-learn 1.6.1 sobre ENDES 2024.
    predictores : list of str
        Lista ordenada con los 31 nombres de columnas en el orden exacto
        en que el modelo fue entrenado. El orden es crítico para predict_proba.

    Raises
    ------
    FileNotFoundError
        Si modelo_anemia_rf.pkl o predictores.json no se encuentran
        en el directorio de trabajo.
    """
    modelo = joblib.load("modelo_anemia_rf.pkl")
    with open("predictores.json") as f:
        predictores = json.load(f)
    return modelo, predictores


# Intento de carga — si falla, la app muestra error en lugar de romperse
try:
    modelo, PREDICTORES = cargar_modelo()
    modelo_cargado = True
except FileNotFoundError:
    modelo_cargado = False


# ── Sidebar informativo ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩸 Sobre esta app")
    st.markdown("""
    Predice el **riesgo de anemia** en niños menores de 5 años
    usando datos de la **ENDES 2024 (Perú)**.

    **Modelo:** Random Forest  
    **Datos:** ENDES 2024 — INEI Perú  
    **Referencia:** Yimer et al. (2025)

    ---
    """)
    # Indicador visual del estado del modelo
    if modelo_cargado:
        st.success(f"✅ Modelo cargado  \n{len(PREDICTORES)} predictores activos")
    else:
        st.error("❌ `modelo_anemia_rf.pkl` no encontrado.  \nSube el archivo al repo.")

    st.markdown("""
    ---
    <div class='disclaimer'>
    ⚠️ Esta herramienta es de apoyo investigativo/educativo.
    No reemplaza el diagnóstico clínico profesional.
    </div>
    """, unsafe_allow_html=True)


# ── Encabezado principal ──────────────────────────────────────────────────────
st.markdown("<div class='main-header'>🩸 Predicción de Anemia Infantil</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>ENDES 2024 · Perú · Modelo Random Forest · 31 predictores</div>", unsafe_allow_html=True)

# Detener ejecución si el modelo no está disponible
if not modelo_cargado:
    st.error("⚠️ El modelo no está disponible. Sube `modelo_anemia_rf.pkl` y `predictores.json` al repositorio.")
    st.stop()


# =============================================================================
# FORMULARIO DE INGRESO DE DATOS
# =============================================================================
# El formulario está organizado en 7 bloques temáticos siguiendo el orden
# lógico de la encuesta ENDES y la ficha clínica pediátrica:
#   1. Niño          — edad, sexo, estado nutricional
#   2. Hogar         — altitud, área, región, riqueza
#   3. Agua/Saneam.  — agua, desagüe, combustible
#   4. Madre         — educación, trabajo, seguro, hemoglobina, prenatal
#   5. Gestación     — hierro en embarazo, peso al nacer
#   6. Salud infant. — morbilidad reciente, vitamina A, antiparasitarios
#   7. Suplementación— SIS niño, hierro jarabe, micronutrientes, CRED
# =============================================================================
st.markdown("### 📋 Datos del niño y su entorno")
st.caption("Completa los 31 campos y presiona **Predecir** para obtener el resultado.")

with st.form("form_prediccion"):

    # ── BLOQUE 1: NIÑO ────────────────────────────────────────────────────────
    # Variables antropométricas y demográficas del niño índice.
    # stunting y wasting se derivan de z-scores OMS; son fuertes predictores
    # de anemia por la relación entre desnutrición crónica y déficit de hierro.
    st.markdown("<div class='section-title'>👶 Características del niño</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        edad_meses = st.number_input(
            "Edad (meses)", min_value=0, max_value=59, value=18,
            help="Edad del niño en meses completos (0–59).")
    with c2:
        sexo_nino = st.selectbox(
            "Sexo", options=[(1, "Masculino"), (0, "Femenino")],
            format_func=lambda x: x[1],
            help="1 = Masculino, 0 = Femenino")
        sexo_nino = sexo_nino[0]
    with c3:
        stunting = st.selectbox(
            "Desnutrición crónica (stunting)",
            options=[(1, "Sí (z-talla < −2)"), (0, "No")],
            format_func=lambda x: x[1],
            help="Z-score talla/edad < −2.0 = desnutrición crónica.")
        stunting = stunting[0]
    with c4:
        wasting = st.selectbox(
            "Desnutrición aguda (wasting)",
            options=[(1, "Sí (z-peso/talla < −2)"), (0, "No")],
            format_func=lambda x: x[1],
            help="Z-score peso/talla < −2.0 = desnutrición aguda.")
        wasting = wasting[0]

    # ── BLOQUE 2: HOGAR ───────────────────────────────────────────────────────
    # Variables del contexto socioeconómico y geográfico del hogar.
    # La altitud es crítica en Perú: a mayor altitud se ajusta el punto de
    # corte de hemoglobina, aumentando la prevalencia real de anemia.
    st.markdown("<div class='section-title'>🏠 Características del hogar</div>", unsafe_allow_html=True)
    c5, c6, c7, c8, c9 = st.columns(5)
    with c5:
        altitud_msnm = st.number_input(
            "Altitud (msnm)", min_value=0, max_value=5000, value=200,
            help="Altitud del hogar sobre el nivel del mar.")
    with c6:
        area_urbana = st.selectbox(
            "Área de residencia",
            options=[(1, "Urbano"), (0, "Rural")],
            format_func=lambda x: x[1])
        area_urbana = area_urbana[0]
    with c7:
        region = st.number_input(
            "Región (código ENDES)",
            min_value=1, max_value=25, value=15,
            help="Código de región ENDES (1–25). Lima = 15.")
    with c8:
        nro_ninos_h = st.number_input(
            "N.° niños <5 en el hogar",
            min_value=0, max_value=15, value=1,
            help="Total de menores de 5 años en el mismo hogar.")
    with c9:
        indice_riqueza = st.selectbox(
            "Índice de riqueza",
            options=[(1,"Muy pobre"),(2,"Pobre"),(3,"Medio"),(4,"Rico"),(5,"Muy rico")],
            format_func=lambda x: x[1], index=1)
        indice_riqueza = indice_riqueza[0]

    # ── BLOQUE 3: AGUA Y SANEAMIENTO ─────────────────────────────────────────
    # Determinantes ambientales de salud. El acceso a agua segura y
    # saneamiento reduce la carga de infecciones intestinales que
    # interfieren con la absorción de hierro.
    st.markdown("<div class='section-title'>💧 Agua, saneamiento y combustible</div>", unsafe_allow_html=True)
    c10, c11, c12 = st.columns(3)
    with c10:
        agua_mejorada = st.selectbox(
            "Fuente de agua mejorada",
            options=[(1, "Sí (red pública, pileta, etc.)"), (0, "No")],
            format_func=lambda x: x[1])
        agua_mejorada = agua_mejorada[0]
    with c11:
        san_mejorado = st.selectbox(
            "Saneamiento mejorado",
            options=[(1, "Sí (inodoro conectado a red)"), (0, "No")],
            format_func=lambda x: x[1])
        san_mejorado = san_mejorado[0]
    with c12:
        comb_limpio = st.selectbox(
            "Combustible limpio para cocinar",
            options=[(1, "Sí (gas, electricidad, solar)"), (0, "No (leña, bosta, etc.)")],
            format_func=lambda x: x[1])
        comb_limpio = comb_limpio[0]

    # ── BLOQUE 4: MADRE ───────────────────────────────────────────────────────
    # Características maternas. La hemoglobina materna es un predictor
    # proxy de la reserva de hierro transferida al feto durante la gestación.
    # Se registra como valor continuo × 10 (ej. 130 = 13.0 g/dL).
    st.markdown("<div class='section-title'>👩 Características de la madre</div>", unsafe_allow_html=True)
    c13, c14, c15, c16 = st.columns(4)
    with c13:
        educ_madre = st.selectbox(
            "Nivel educativo de la madre",
            options=[(0,"Sin educación"),(1,"Primaria"),(2,"Secundaria"),(3,"Superior")],
            format_func=lambda x: x[1], index=2)
        educ_madre = educ_madre[0]
    with c14:
        madre_trabaja = st.selectbox(
            "Madre trabaja actualmente",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        madre_trabaja = madre_trabaja[0]
    with c15:
        madre_sis = st.selectbox(
            "Madre afiliada al SIS",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        madre_sis = madre_sis[0]
    with c16:
        edad_madre = st.number_input(
            "Edad de la madre (años)",
            min_value=14, max_value=49, value=28,
            help="Edad de la madre en años cumplidos.")

    c17, c18, c19 = st.columns(3)
    with c17:
        hb_materna = st.number_input(
            "Hemoglobina materna (g/dL × 10)",
            min_value=70, max_value=200, value=130,
            help="HA56: hemoglobina materna continua. Rango plausible 70–200. "
                 "Ejemplo: 130 = 13.0 g/dL. Si no se midió, dejar en 130 (mediana nacional).")
    with c18:
        ctrl_prenatal_temprano = st.selectbox(
            "Control prenatal en 1.er trimestre",
            options=[(1, "Sí (meses 1–3)"), (0, "No")],
            format_func=lambda x: x[1])
        ctrl_prenatal_temprano = ctrl_prenatal_temprano[0]
    with c19:
        sin_ctrl_prenatal = st.selectbox(
            "Sin ningún control prenatal",
            options=[(1, "Sí (sin control)"), (0, "No (tuvo control)")],
            format_func=lambda x: x[1])
        sin_ctrl_prenatal = sin_ctrl_prenatal[0]

    # ── BLOQUE 5: GESTACIÓN Y NACIMIENTO ─────────────────────────────────────
    # El suplemento de hierro en el embarazo protege las reservas fetales.
    # El bajo peso al nacer (<2500 g) se asocia con reservas de hierro
    # insuficientes al momento del nacimiento.
    st.markdown("<div class='section-title'>🤰 Gestación y nacimiento</div>", unsafe_allow_html=True)
    c20, c21 = st.columns(2)
    with c20:
        hierro_emb = st.selectbox(
            "Tomó hierro durante el embarazo",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        hierro_emb = hierro_emb[0]
    with c21:
        bajo_peso_nacer = st.selectbox(
            "Bajo peso al nacer",
            options=[(1, "Sí (pequeño o muy pequeño)"), (0, "No (normal o grande)")],
            format_func=lambda x: x[1])
        bajo_peso_nacer = bajo_peso_nacer[0]

    # ── BLOQUE 6: SALUD INFANTIL ──────────────────────────────────────────────
    # Morbilidad reciente (últimas 2 semanas). La diarrea e IRA aumentan
    # las pérdidas de hierro y reducen su absorción intestinal.
    # vitamina_a y antiparasit son intervenciones MINSA que modulan el riesgo.
    st.markdown("<div class='section-title'>🏥 Salud infantil reciente (últimas 2 semanas)</div>", unsafe_allow_html=True)
    c22, c23, c24, c25, c26 = st.columns(5)
    with c22:
        diarrea = st.selectbox(
            "Diarrea",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        diarrea = diarrea[0]
    with c23:
        tos = st.selectbox(
            "Tos",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        tos = tos[0]
    with c24:
        ira = st.selectbox(
            "IRA (infección respiratoria aguda)",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1],
            help="Solo aplica si hubo tos. Si no hubo tos, seleccionar No.")
        ira = ira[0]
    with c25:
        vitamina_a = st.selectbox(
            "Recibió vitamina A",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        vitamina_a = vitamina_a[0]
    with c26:
        antiparasit = st.selectbox(
            "Recibió antiparasitario",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        antiparasit = antiparasit[0]

    # ── BLOQUE 7: SEGURO Y SUPLEMENTACIÓN ────────────────────────────────────
    # Variables de intervención del sistema de salud peruano.
    # hierro_7d y micronut_minsa capturan la adherencia reciente a la
    # suplementación preventiva del MINSA (Estrategia Nacional de Anemia).
    # hb_materna_falta es un indicador de dato faltante imputado en el modelo.
    st.markdown("<div class='section-title'>💊 Seguro y suplementación (Perú)</div>", unsafe_allow_html=True)
    c27, c28, c29, c30, c31 = st.columns(5)
    with c27:
        sis_nino = st.selectbox(
            "Niño afiliado al SIS",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1])
        sis_nino = sis_nino[0]
    with c28:
        hb_materna_falta = st.selectbox(
            "Hemoglobina materna no medida",
            options=[(1, "Dato ausente"), (0, "Dato disponible")],
            format_func=lambda x: x[1],
            help="Marcar '1' si no se midió la hemoglobina de la madre.")
        hb_materna_falta = hb_materna_falta[0]
    with c29:
        hierro_7d = st.selectbox(
            "Hierro jarabe (últimos 7 días)",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1],
            help="S465EA: suplemento hierro jarabe en los últimos 7 días.")
        hierro_7d = hierro_7d[0]
    with c30:
        micronut_minsa = st.selectbox(
            "Micronutrientes MINSA (7 días)",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1],
            help="S465EB: micronutrientes MINSA en los últimos 7 días.")
        micronut_minsa = micronut_minsa[0]
    with c31:
        cred = st.selectbox(
            "Control CRED al día",
            options=[(1, "Sí"), (0, "No")],
            format_func=lambda x: x[1],
            help="S466: el niño tiene sus controles CRED al día.")
        cred = cred[0]

    # ── BOTÓN DE PREDICCIÓN ───────────────────────────────────────────────────
    st.markdown("---")
    submitted = st.form_submit_button(
        "🔍 Predecir riesgo de anemia",
        use_container_width=True,
        type="primary")


# =============================================================================
# PREDICCIÓN Y VISUALIZACIÓN DEL RESULTADO
# =============================================================================
if submitted:
    # ── 1. Construcción del vector de entrada ─────────────────────────────────
    # Se arma un diccionario con los 31 valores capturados en el formulario.
    # Luego se reordena según PREDICTORES (orden exacto de entrenamiento)
    # para garantizar que cada columna coincida con lo que el modelo espera.
    # Un orden incorrecto produciría predicciones silenciosamente erróneas.
    valores = {
        'edad_meses':              edad_meses,
        'sexo_nino':               sexo_nino,
        'stunting':                stunting,
        'wasting':                 wasting,
        'altitud_msnm':            altitud_msnm,
        'area_urbana':             area_urbana,
        'region':                  region,
        'nro_ninos_h':             nro_ninos_h,
        'indice_riqueza':          indice_riqueza,
        'agua_mejorada':           agua_mejorada,
        'san_mejorado':            san_mejorado,
        'comb_limpio':             comb_limpio,
        'educ_madre':              educ_madre,
        'madre_trabaja':           madre_trabaja,
        'madre_sis':               madre_sis,
        'edad_madre':              edad_madre,
        'hb_materna':              hb_materna,
        'ctrl_prenatal_temprano':  ctrl_prenatal_temprano,
        'sin_ctrl_prenatal':       sin_ctrl_prenatal,
        'hierro_emb':              hierro_emb,
        'bajo_peso_nacer':         bajo_peso_nacer,
        'diarrea':                 diarrea,
        'tos':                     tos,
        'ira':                     ira,
        'vitamina_a':              vitamina_a,
        'antiparasit':             antiparasit,
        'sis_nino':                sis_nino,
        'hierro_7d':               hierro_7d,
        'micronut_minsa':          micronut_minsa,
        'cred':                    cred,
        'hb_materna_falta':        hb_materna_falta,
    }

    # DataFrame con una sola fila, columnas en el orden exacto del modelo
    X_input = pd.DataFrame([[valores[p] for p in PREDICTORES]], columns=PREDICTORES)

    # ── 2. Inferencia ─────────────────────────────────────────────────────────
    # predict_proba devuelve [[prob_clase_0, prob_clase_1]]
    # Se toma el índice [0, 1] = probabilidad de anemia (clase positiva)
    proba = modelo.predict_proba(X_input)[0, 1]
    prob_pct = proba * 100
    clase = "ALTO" if proba >= 0.5 else "BAJO"   # umbral estándar 50%

    # ── 3. Visualización del resultado ────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Resultado de la predicción")

    col_res, col_gauge = st.columns([1.5, 1])

    with col_res:
        # Tarjeta de resultado con color semántico:
        #   Rojo  → Riesgo ALTO  (proba ≥ 0.50)
        #   Verde → Riesgo BAJO  (proba < 0.50)
        if clase == "ALTO":
            st.markdown(f"""
            <div class='result-card-alto'>
                <div class='result-title'>🔴 Riesgo ALTO de anemia</div>
                <div class='prob-text'>{prob_pct:.1f}%</div>
                <p>Probabilidad de que el niño presente anemia según los predictores ingresados.</p>
                <p><strong>Recomendación:</strong> Derivar para hemograma de confirmación y evaluación nutricional.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-bajo'>
                <div class='result-title'>🟢 Riesgo BAJO de anemia</div>
                <div class='prob-text'>{prob_pct:.1f}%</div>
                <p>Probabilidad de que el niño presente anemia según los predictores ingresados.</p>
                <p><strong>Recomendación:</strong> Mantener controles CRED y suplementación preventiva.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        # Medidor visual de probabilidad con barra de progreso nativa de Streamlit
        st.markdown("#### Probabilidad de anemia")
        st.metric(
            label="Probabilidad",
            value=f"{prob_pct:.1f}%",
            delta=f"{'↑ Por encima' if proba >= 0.5 else '↓ Por debajo'} del umbral (50%)"
        )
        st.progress(float(proba))
        st.caption("Umbral de clasificación: 50%")

    # ── 4. Tabla de factores de riesgo presentes ──────────────────────────────
    # Se revisan manualmente 14 factores clínicamente relevantes.
    # Esta sección es interpretativa, no sale directamente del modelo,
    # pero ayuda al usuario a entender qué variables están contribuyendo
    # al riesgo en el caso específico ingresado.
    with st.expander("🔎 Ver factores de riesgo identificados en este caso"):
        factores_riesgo = {
            "Desnutrición crónica (stunting)":           stunting == 1,
            "Desnutrición aguda (wasting)":              wasting == 1,
            "Bajo peso al nacer":                        bajo_peso_nacer == 1,
            "Sin control prenatal":                      sin_ctrl_prenatal == 1,
            "Sin suplemento de hierro en embarazo":      hierro_emb == 0,
            "Sin micronutrientes MINSA (últimos 7 días)":micronut_minsa == 0,
            "Sin hierro jarabe (últimos 7 días)":        hierro_7d == 0,
            "Agua no mejorada":                          agua_mejorada == 0,
            "Saneamiento no mejorado":                   san_mejorado == 0,
            "Combustible no limpio":                     comb_limpio == 0,
            "Diarrea reciente":                          diarrea == 1,
            "IRA reciente":                              ira == 1,
            "Hemoglobina materna baja (<110 g/dL×10)":  hb_materna < 110,
            "Hogar muy pobre":                           indice_riqueza == 1,
        }

        # Filtrar solo los factores presentes en este caso
        presentes = {k: v for k, v in factores_riesgo.items() if v}

        if presentes:
            df_factores = pd.DataFrame(
                {"Factor de riesgo": list(presentes.keys()),
                 "Presente": ["⚠️ Sí"] * len(presentes)}
            )
            st.dataframe(df_factores, use_container_width=True, hide_index=True)
        else:
            st.success("No se identificaron factores de riesgo críticos en este caso.")

    # ── 5. Disclaimer final ───────────────────────────────────────────────────
    st.markdown("""
    <div class='disclaimer'>
    ⚠️ Esta predicción es de carácter investigativo/educativo basada en ENDES 2024.
    No reemplaza el diagnóstico clínico. AUC del modelo en test: ~72–75%.
    </div>
    """, unsafe_allow_html=True)
