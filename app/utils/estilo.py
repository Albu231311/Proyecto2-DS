"""
Estilo compartido de la aplicación: paleta de colores, nombres legibles
y helpers para que todas las páginas se vean igual.

Uso en cualquier página:
    from utils.estilo import PALETA, ORDEN_CLASES, configurar_pagina, aplicar_layout
    configurar_pagina("Explorar datos", "🔎")
    fig = px.bar(df, x="severidad", color="severidad", color_discrete_map=PALETA)
    st.plotly_chart(aplicar_layout(fig), use_container_width=True)
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Clases de severidad ──
# Escala semántica: azul (frío, tranquilo) → ámbar (advertencia) → rojo oscuro (alerta).
# Son los mismos colores de las figuras estáticas de los notebooks.
PALETA = {
    "Normal/Mild": "#4C72B0",
    "Moderate":    "#DD8452",
    "Severe":      "#8B0000",
}
ORDEN_CLASES = ["Normal/Mild", "Moderate", "Severe"]

# ── Modelos ──
# Claves tal como aparecen en data/processed/resultados.json
NOMBRES_MODELOS = {
    "Clase_Mayoritaria": "Clase mayoritaria",
    "Arbol_Decision":    "Árbol de Decisión",
    "Random_Forest":     "Random Forest",
    "Gradient_Boosting": "Gradient Boosting",
    "ResNet50_imagen":   "ResNet50 (imagen)",
}
# Colores que no se confunden con los de las clases. La línea base va en gris
# y la ResNet50 (mejor modelo) en el color principal de la app.
COLORES_MODELOS = {
    "Clase mayoritaria": "#A0A8B3",
    "Árbol de Decisión": "#8AA05B",
    "Random Forest":     "#2A9D8F",
    "Gradient Boosting": "#7B6CB0",
    "ResNet50 (imagen)": "#2E4057",
}

# ── Variables del dataset ──
CONDICIONES = [
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]
NIVELES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

# ── Colores de la interfaz (iguales a .streamlit/config.toml) ──
COLOR_PRINCIPAL = "#2E4057"
COLOR_TEXTO     = "#1F2933"
COLOR_SECUNDARIO = "#6B7785"
COLOR_FONDO_ALT = "#F2F4F7"

# ── Plantilla de Plotly ──
pio.templates["proyecto2"] = go.layout.Template(
    layout=dict(
        font=dict(family="sans-serif", size=13, color=COLOR_TEXTO),
        colorway=list(COLORES_MODELOS.values()),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(x=0.0, xanchor="left", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(font_size=12),
    )
)
pio.templates.default = "plotly_white+proyecto2"

_CSS = f"""
<style>
.tarjeta {{
    background: {COLOR_FONDO_ALT};
    border-left: 4px solid {COLOR_PRINCIPAL};
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
}}
.tarjeta .titulo {{ font-size: 0.85rem; color: {COLOR_SECUNDARIO}; margin: 0; }}
.tarjeta .valor  {{ font-size: 1.6rem; font-weight: 700; color: {COLOR_TEXTO}; margin: 0; }}
.tarjeta .ayuda  {{ font-size: 0.8rem; color: {COLOR_SECUNDARIO}; margin: 0; }}
.chip {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    color: white;
    font-size: 0.85rem;
    font-weight: 600;
}}
</style>
"""


def configurar_pagina(titulo: str, icono: str = "🩻") -> None:
    """Llamar al inicio de cada página, antes de cualquier otro st.*"""
    st.set_page_config(
        page_title=f"{titulo} · Columna Lumbar RSNA",
        page_icon=icono,
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def aplicar_layout(fig: go.Figure, titulo: str | None = None, alto: int = 420) -> go.Figure:
    """Deja una figura de Plotly con el estilo común del proyecto."""
    fig.update_layout(template="plotly_white+proyecto2", height=alto)
    if titulo:
        fig.update_layout(title_text=titulo)
    return fig


def tarjeta(titulo: str, valor, ayuda: str = "") -> None:
    """Tarjeta con una cifra destacada (por ejemplo en la página de Inicio)."""
    ayuda_html = f'<p class="ayuda">{ayuda}</p>' if ayuda else ""
    st.markdown(
        f'<div class="tarjeta"><p class="titulo">{titulo}</p>'
        f'<p class="valor">{valor}</p>{ayuda_html}</div>',
        unsafe_allow_html=True,
    )


def chip_clase(clase: str) -> str:
    """HTML de una etiqueta con el color de la clase. Usar con st.markdown(..., unsafe_allow_html=True)."""
    color = PALETA.get(clase, COLOR_SECUNDARIO)
    return f'<span class="chip" style="background:{color}">{clase}</span>'
