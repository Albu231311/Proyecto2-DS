"""
Página de inicio de la aplicación.
Correr desde la raíz del proyecto:  streamlit run app/Inicio.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.estilo import (
    CONDICIONES, NIVELES, NOMBRES_MODELOS, ORDEN_CLASES, PALETA,
    aplicar_layout, chip_clase, configurar_pagina, tarjeta,
)

configurar_pagina("Inicio", "🩻")

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"
PAGES = Path(__file__).resolve().parent / "pages"


@st.cache_data
def cargar_resultados() -> dict:
    with open(PROC / "resultados.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_splits() -> pd.DataFrame:
    return pd.read_csv(PROC / "splits.csv")


resultados = cargar_resultados()
meta = resultados["meta"]
splits = cargar_splits()

# ── Encabezado ──
st.title("Clasificación degenerativa de la columna lumbar")
st.markdown(
    "Clasificación automática de la **severidad** de condiciones degenerativas "
    "de la columna lumbar a partir de resonancias magnéticas, con datos de la "
    "competencia **RSNA 2024 Lumbar Spine Degenerative Classification**."
)

# ── Cifras principales ──
n_estudios = len(splits)
n_muestras = meta["n_train"] + meta["n_val"] + meta["n_test"]
dist_test = meta["distribucion_test"]
pct_severe = dist_test["Severe"] / sum(dist_test.values())

mejor = max(resultados["modelos"], key=lambda m: m["f1_macro"])
nombre_mejor = NOMBRES_MODELOS.get(mejor["modelo"], mejor["modelo"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    tarjeta("Estudios (pacientes)", f"{n_estudios:,}", "Resonancias completas")
with c2:
    tarjeta("Muestras etiquetadas", f"{n_muestras:,}", "Condición × nivel con coordenada")
with c3:
    tarjeta("Casos Severe en test", f"{pct_severe:.1%}", f"{dist_test['Severe']:,} de {sum(dist_test.values()):,}")
with c4:
    tarjeta("Mejor modelo (F1 macro)", f"{mejor['f1_macro']:.3f}", nombre_mejor)

st.divider()

# ── El problema ──
col_texto, col_graf = st.columns([3, 2], gap="large")

with col_texto:
    st.subheader("¿Qué se clasifica?")
    st.markdown(
        f"Para cada estudio se evalúan **{len(CONDICIONES)} condiciones** en "
        f"**{len(NIVELES)} niveles vertebrales** ({NIVELES[0]} a {NIVELES[-1]}), "
        f"es decir **{len(CONDICIONES) * len(NIVELES)} diagnósticos por paciente**. "
        "Cada uno se clasifica en una de tres clases de severidad:"
    )

    descripciones = {
        "Normal/Mild": "El espacio por donde pasan el saco dural o la raíz nerviosa está preservado.",
        "Moderate":    "Estrechamiento parcial, con compresión leve del saco dural o de la raíz.",
        "Severe":      "Estrechamiento marcado, con compresión severa de las estructuras nerviosas.",
    }
    for clase in ORDEN_CLASES:
        st.markdown(f"{chip_clase(clase)} &nbsp; {descripciones[clase]}", unsafe_allow_html=True)

    st.markdown("")
    with st.expander("Condiciones evaluadas"):
        st.markdown(
            "- **Spinal Canal Stenosis:** estrechamiento del canal central.\n"
            "- **Neural Foraminal Narrowing (izq. / der.):** estrechamiento del "
            "foramen por donde sale la raíz nerviosa.\n"
            "- **Subarticular Stenosis (izq. / der.):** estrechamiento del receso "
            "lateral, justo antes del foramen."
        )
        st.caption("Secuencias de resonancia usadas: Sagital T1, Sagital T2/STIR y Axial T2.")

with col_graf:
    st.subheader("Un problema desbalanceado")
    df_dist = pd.DataFrame({"Severidad": list(dist_test.keys()), "Muestras": list(dist_test.values())})
    fig = px.bar(
        df_dist, x="Muestras", y="Severidad", orientation="h", text="Muestras",
        color="Severidad", color_discrete_map=PALETA,
        category_orders={"Severidad": ORDEN_CLASES[::-1]},
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(aplicar_layout(fig, "Distribución de clases en test", alto=300), use_container_width=True)
    st.caption(
        f"El {dist_test['Normal/Mild'] / sum(dist_test.values()):.0%} de las muestras es Normal/Mild. "
        "Por eso los modelos se comparan con F1 macro, balanced accuracy y recall de Severe, "
        "y no con la exactitud."
    )

st.divider()

# ── Navegación ──
st.subheader("¿Qué puedes hacer en la aplicación?")

secciones = [
    ("1_Explorar_datos.py", "🔎", "Explorar datos",
     "Distribución de la severidad por condición y nivel, coordenadas de las anotaciones "
     "y relaciones entre condiciones."),
    ("2_Predecir.py", "🧠", "Predecir",
     "Sube una imagen DICOM, indica la condición y el nivel, y compara la predicción de cada modelo."),
    ("3_Rendimiento.py", "📊", "Rendimiento",
     "Métricas de cada modelo en el conjunto de prueba, matrices de confusión y tiempos."),
]

cols = st.columns(len(secciones))
for col, (archivo, icono, titulo, texto) in zip(cols, secciones):
    with col:
        with st.container(border=True):
            st.markdown(f"#### {icono} {titulo}")
            st.markdown(texto)
            if (PAGES / archivo).exists():
                st.page_link(f"pages/{archivo}", label=f"Ir a {titulo}", icon="➡️")
            else:
                st.caption("En desarrollo")

# ── Metodología ──
with st.expander("¿Cómo se entrenaron los modelos?"):
    st.markdown(
        f"- **Partición:** {meta['split']}. Al separar por estudio, ningún paciente "
        "aparece en más de un conjunto.\n"
        "- **Modelos sobre metadatos:** Árbol de Decisión, Random Forest y Gradient Boosting, "
        "usando condición, nivel, tipo de serie, número de instancia y coordenadas.\n"
        f"- **Modelo de imagen:** {meta['backbone']}, con un recorte de "
        f"{meta['crop_size']}×{meta['crop_size']} píxeles alrededor de la coordenada anotada, "
        "más la condición y el nivel.\n"
        "- **Línea base:** clase mayoritaria (siempre predice Normal/Mild)."
    )

st.caption("CC3084 Data Science · Universidad del Valle de Guatemala · 2026")
