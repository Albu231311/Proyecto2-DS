"""
Página Predecir: el usuario da un DICOM, la condición, el nivel y la coordenada,
y cada modelo entrenado clasifica la severidad. El preprocesamiento es automático.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw

from utils.estilo import (PALETA, ORDEN_CLASES, NOMBRES_MODELOS, COLORES_MODELOS,
                          CONDICIONES, NIVELES, configurar_pagina, aplicar_layout,
                          tarjeta, chip_clase)
from utils import modelos as md
from utils import preprocesamiento as pp

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # sin el componente se usan solo los campos numéricos
    streamlit_image_coordinates = None

configurar_pagina("Predecir", "🧠")
RAIZ = Path(__file__).resolve().parents[2]

DIR_EJEMPLO = RAIZ / "data" / "44036939"
COLOR_MARCA = "#3FD0E0"  # cian: se ve sobre la resonancia y no se confunde con las clases
ANCHO_IMAGEN = 480


# ════════════════════════════ Carga con caché ════════════════════════════

@st.cache_data
def cargar_metricas() -> dict:
    """Métricas de test por modelo (resultados.json) + tiempo por muestra en test."""
    with open(RAIZ / "data" / "processed" / "resultados.json", encoding="utf-8") as f:
        res = json.load(f)
    with open(RAIZ / "data" / "processed" / "resultados_metadatos.json", encoding="utf-8") as f:
        res_meta = json.load(f)
    # resultados.json solo trae el tiempo de la ResNet50; el de los demás está en resultados_metadatos.json
    tiempos = {m["modelo"]: m["t_infer_ms_per_sample"]
               for m in res_meta["modelos"] + res["modelos"] if m.get("t_infer_ms_per_sample") is not None}
    return {
        m["modelo"]: {
            "f1_macro": m["f1_macro"],
            "recall_severe": m["recall_severe"],
            "balanced_accuracy": m["balanced_accuracy"],
            "t_test_ms": tiempos.get(m["modelo"]),
        }
        for m in res["modelos"]
    }


@st.cache_data
def series_ejemplo() -> dict[str, dict]:
    """Series del estudio de ejemplo (id de carpeta → tipo detectado y cortes)."""
    series = {}
    for carpeta in sorted(p for p in DIR_EJEMPLO.iterdir() if p.is_dir()):
        cortes = sorted(int(f.stem) for f in carpeta.glob("*.dcm"))
        if not cortes:
            continue
        _, dcm = pp.leer_dicom(carpeta / f"{cortes[0]}.dcm")
        series[carpeta.name] = {"tipo": pp.inferir_serie(dcm), "cortes": cortes}
    return series


@st.cache_data(max_entries=64)
def cargar_dicom(contenido: bytes) -> tuple[np.ndarray, str, int | None]:
    """Imagen normalizada, tipo de serie sugerido y número de instancia del DICOM."""
    img, dcm = pp.leer_dicom(contenido)
    instancia = dcm.get("InstanceNumber")
    return img, pp.inferir_serie(dcm), int(instancia) if instancia is not None else None


def imagen_marcada(img: np.ndarray, x: float, y: float) -> Image.Image:
    """Resonancia en RGB con una cruz en (x, y) y el cuadro de 96×96 que recorta la CNN."""
    rgb = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    dibujo = ImageDraw.Draw(rgb)
    grosor = max(1, round(max(img.shape) / 256))
    mitad = pp.CROP_SIZE // 2
    cx, cy = int(x), int(y)
    dibujo.rectangle([cx - mitad, cy - mitad, cx + mitad, cy + mitad], outline=COLOR_MARCA, width=grosor)
    brazo = 6 * grosor
    dibujo.line([x - brazo, y, x + brazo, y], fill=COLOR_MARCA, width=grosor)
    dibujo.line([x, y - brazo, x, y + brazo], fill=COLOR_MARCA, width=grosor)
    return rgb


def al_hacer_clic() -> None:
    """Pasa el clic (en píxeles de pantalla) a píxeles del DICOM. Corre antes de dibujar los campos."""
    clic = st.session_state.get("clic_imagen")
    if not clic:
        return
    alto, ancho = st.session_state["dims_imagen"]
    escala = ancho / clic["width"]
    st.session_state["coord_x"] = float(np.clip(round(clic["x"] * escala, 1), 0, ancho - 1))
    st.session_state["coord_y"] = float(np.clip(round(clic["y"] * escala, 1), 0, alto - 1))


# ════════════════════════════ Encabezado ════════════════════════════

st.title("Predecir la severidad")
st.markdown(
    "Elige una resonancia, indica la **condición**, el **nivel vertebral** y el **punto** "
    "donde está la estructura a evaluar. Cada modelo entrenado da su clasificación; "
    "la imagen y los datos se preparan automáticamente igual que en el entrenamiento."
)

disponibles, faltantes = md.modelos_disponibles()
if faltantes:
    nombres = ", ".join(NOMBRES_MODELOS[c] for c in faltantes)
    st.warning(
        f"No se encontró **{nombres}** en `models/`. La página funciona con los demás modelos. "
        "Para usarlo, descárgalo como indica el README (sección *Modelo Random Forest*)."
    )

metricas = cargar_metricas()

# ════════════════════════════ 1. Entrada ════════════════════════════

st.subheader("1. Entrada")
col_datos, col_imagen = st.columns([2, 3], gap="large")

with col_datos:
    origen = st.radio("Imagen", ["Ejemplo del repositorio", "Subir un DICOM"], horizontal=True)

    if origen == "Ejemplo del repositorio":
        series = series_ejemplo()
        ids = list(series)
        id_serie = st.selectbox(
            "Serie del estudio 44036939",
            ids,
            index=next((i for i, s in enumerate(ids) if series[s]["tipo"] == "Sagittal T2/STIR"), 0),
            format_func=lambda s: f"{series[s]['tipo']} · {len(series[s]['cortes'])} cortes",
        )
        cortes = series[id_serie]["cortes"]
        corte = st.select_slider("Corte (número de instancia)", cortes, value=cortes[len(cortes) // 2])
        contenido = (DIR_EJEMPLO / id_serie / f"{corte}.dcm").read_bytes()
    else:
        archivo = st.file_uploader("Archivo DICOM (.dcm)", type=["dcm", "dicom"])
        if archivo is None:
            st.info("Sube un archivo DICOM de una resonancia lumbar para continuar.")
            st.stop()
        contenido = archivo.getvalue()

    try:
        img, serie_detectada, instancia_dcm = cargar_dicom(contenido)
    except Exception as e:
        st.error(f"No se pudo leer el archivo como DICOM: {e}")
        st.stop()

    alto, ancho = img.shape
    # Si cambia el tamaño de la imagen, la coordenada vuelve al centro
    if st.session_state.get("dims_imagen") != (alto, ancho):
        st.session_state["dims_imagen"] = (alto, ancho)
        st.session_state["coord_x"] = float(ancho // 2)
        st.session_state["coord_y"] = float(alto // 2)

    if origen == "Ejemplo del repositorio":
        instancia = corte
    else:
        instancia = st.number_input("Número de instancia (corte)", min_value=1,
                                    value=instancia_dcm or 1, step=1,
                                    help="Se lee del DICOM; puedes corregirlo.")

    serie = st.selectbox("Tipo de serie", pp.SERIES, index=pp.SERIES.index(serie_detectada),
                         help="Se detecta con la orientación del corte del DICOM.")
    condicion = st.selectbox("Condición", CONDICIONES)
    nivel = st.selectbox("Nivel vertebral", NIVELES, index=NIVELES.index("L4/L5"))

    c_x, c_y = st.columns(2)
    x = c_x.number_input("Coordenada x (px)", 0.0, float(ancho - 1), step=1.0, format="%.1f", key="coord_x")
    y = c_y.number_input("Coordenada y (px)", 0.0, float(alto - 1), step=1.0, format="%.1f", key="coord_y")

    esperada = pp.serie_esperada(condicion)
    if serie != esperada:
        st.warning(
            f"En el entrenamiento, *{condicion}* se anotó siempre en series **{esperada}**. "
            f"Con una serie {serie} los modelos están fuera de lo que aprendieron."
        )

with col_imagen:
    marcada = imagen_marcada(img, x, y)
    if streamlit_image_coordinates is not None:
        st.caption("Haz clic sobre la imagen para mover el punto.")
        streamlit_image_coordinates(marcada, width=ANCHO_IMAGEN, key="clic_imagen",
                                    on_click=al_hacer_clic, cursor="crosshair")
    else:
        st.image(marcada, width=ANCHO_IMAGEN)
    st.caption(f"{serie} · corte {instancia} · {ancho}×{alto} px. "
               f"El cuadro cian es la región de {pp.CROP_SIZE}×{pp.CROP_SIZE} px que recorta la CNN.")

# ════════════════════════════ Preprocesamiento ════════════════════════════

recorte = pp.recortar(img, x, y)
x_meta = pp.entrada_metadatos(condicion, nivel, serie, int(instancia), x, y)
x_img = pp.tensor_imagen(recorte)
x_meta_cnn = pp.tensor_meta_cnn(condicion, nivel)

# ════════════════════════════ 2. Modelos ════════════════════════════

st.subheader("2. Modelos")
seleccion = st.multiselect(
    "Modelos a comparar (uno, varios o todos)",
    disponibles, default=disponibles, format_func=lambda c: NOMBRES_MODELOS[c],
)
if not seleccion:
    st.info("Elige al menos un modelo.")
    st.stop()

filas = []
for clave in seleccion:
    proba, ms = md.predecir(clave, md.cargar_modelo(clave), x_meta, x_img, x_meta_cnn)
    clase = ORDEN_CLASES[int(proba.argmax())]
    filas.append({"clave": clave, "Modelo": NOMBRES_MODELOS[clave], "clase": clase,
                  "confianza": float(proba.max()), "ms": ms, **dict(zip(ORDEN_CLASES, proba))})
res = pd.DataFrame(filas)

# ════════════════════════════ 3. Resultados ════════════════════════════

st.subheader("3. Resultado")

votos = res["clase"].value_counts()
if len(res) > 1:
    if len(votos) == 1:
        st.markdown(f"Los {len(res)} modelos coinciden: {chip_clase(votos.index[0])}", unsafe_allow_html=True)
    else:
        resumen = " · ".join(f"{chip_clase(c)} {votos[c]}" for c in ORDEN_CLASES if c in votos)
        st.markdown(f"Los modelos no coinciden: {resumen}", unsafe_allow_html=True)

columnas = st.columns(len(res))
for col, fila in zip(columnas, res.itertuples()):
    with col, st.container(border=True):
        st.markdown(f"**{fila.Modelo}**")
        st.markdown(chip_clase(fila.clase), unsafe_allow_html=True)
        st.caption(f"Probabilidad {fila.confianza:.0%} · {fila.ms:.2f} ms")

col_proba, col_recorte = st.columns([3, 1], gap="large")

with col_proba:
    largo = res.melt(id_vars=["Modelo", "clase"], value_vars=ORDEN_CLASES,
                     var_name="Severidad", value_name="Probabilidad")
    # Solo se rotula la clase que gana en cada modelo
    largo["etiqueta"] = np.where(largo["Severidad"] == largo["clase"],
                                 largo["Probabilidad"].map("{:.0%}".format), "")
    fig = px.bar(
        largo, x="Modelo", y="Probabilidad", color="Severidad", barmode="group",
        text="etiqueta", color_discrete_map=PALETA,
        category_orders={"Severidad": ORDEN_CLASES, "Modelo": list(res["Modelo"])},
        custom_data=["Severidad"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}<br>%{customdata[0]}: %{y:.1%}<extra></extra>")
    fig.update_layout(yaxis=dict(range=[0, 1.1], tickformat=".0%", title=None),
                      xaxis_title=None, bargap=0.25, bargroupgap=0.08, legend_title_text=None)
    st.plotly_chart(aplicar_layout(fig, "Probabilidad de cada clase según el modelo"),
                    use_container_width=True)

with col_recorte:
    st.markdown("**Lo que ve la CNN**")
    vista = Image.fromarray((recorte * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    st.image(vista, width=224)
    st.caption(f"Recorte de {pp.CROP_SIZE}×{pp.CROP_SIZE} px alrededor del punto, "
               "ampliado a 224×224 como entra a la ResNet50.")

st.info(
    "**Por qué los modelos pueden no coincidir.** El Árbol de Decisión, el Random Forest y el "
    "Gradient Boosting **no ven la imagen**: solo usan la condición, el nivel, el tipo de serie, "
    "el número de corte y la coordenada. La ResNet50 sí mira el recorte. Además, todos se "
    "entrenaron con puntos marcados por radiólogos, así que un punto lejos de la estructura "
    "da resultados poco fiables. La clase mayoritaria siempre responde Normal/Mild.",
    icon="ℹ️",
)

# ════════════════════════════ 4. Eficiencia ════════════════════════════

st.subheader("4. Eficiencia: respuesta contra rendimiento")
st.markdown(
    "Tiempo que tardó cada modelo en responder **esta** predicción, junto con lo bien que "
    "le fue en el conjunto de prueba (14,612 casos). Un modelo eficiente responde rápido "
    "**y** tiene F1 macro alto."
)

res["F1 macro (test)"] = res["clave"].map(lambda c: metricas[c]["f1_macro"])
res["Recall Severe (test)"] = res["clave"].map(lambda c: metricas[c]["recall_severe"])
res["Balanced acc. (test)"] = res["clave"].map(lambda c: metricas[c]["balanced_accuracy"])
res["ms en test"] = res["clave"].map(lambda c: metricas[c]["t_test_ms"])

tabla = res[["Modelo", "clase", "confianza", "ms", "F1 macro (test)",
             "Recall Severe (test)", "Balanced acc. (test)", "ms en test"]].rename(
    columns={"clase": "Predicción", "confianza": "Probabilidad", "ms": "Tiempo aquí (ms)",
             "ms en test": "Tiempo en test (ms/muestra)"})
st.dataframe(
    tabla, hide_index=True, use_container_width=True,
    column_config={
        "Probabilidad": st.column_config.NumberColumn(format="percent"),
        "Tiempo aquí (ms)": st.column_config.NumberColumn(format="%.3f"),
        "F1 macro (test)": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
        "Recall Severe (test)": st.column_config.NumberColumn(format="%.3f"),
        "Balanced acc. (test)": st.column_config.NumberColumn(format="%.3f"),
        "Tiempo en test (ms/muestra)": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption(
    "*Tiempo aquí*: una sola muestra en esta computadora (CPU). *Tiempo en test*: promedio por "
    "muestra al predecir todo el conjunto de prueba por lotes; la ResNet50 se midió en una GPU T4."
)

disp = res.assign(ms_graf=res["ms"].clip(lower=0.001))
# Los modelos lentos quedan a la derecha: su nombre va a la izquierda del punto para no salirse
disp["posicion"] = np.where(disp["ms_graf"] >= 1, "middle left", "middle right")
log_ms = np.log10(disp["ms_graf"])
fig = px.scatter(
    disp, x="ms_graf", y="F1 macro (test)", color="Modelo", text="Modelo",
    color_discrete_map=COLORES_MODELOS, log_x=True,
    custom_data=["ms", "Recall Severe (test)"],
)
for traza in fig.data:
    traza.textposition = disp.loc[disp["Modelo"] == traza.name, "posicion"].iloc[0]
fig.update_traces(marker=dict(size=14, line=dict(width=2, color="white")),
                  hovertemplate="<b>%{text}</b><br>Tiempo: %{customdata[0]:.3f} ms"
                                "<br>F1 macro: %{y:.3f}<br>Recall Severe: %{customdata[1]:.3f}"
                                "<extra></extra>")
fig.update_layout(xaxis=dict(title="Tiempo de esta predicción (ms, escala log)",
                             range=[log_ms.min() - 0.6, log_ms.max() + 0.6]),
                  yaxis=dict(title="F1 macro en test", range=[0.2, 0.55]),
                  showlegend=False)
st.plotly_chart(aplicar_layout(fig, "Más arriba = mejor; más a la izquierda = más rápido", alto=380),
                use_container_width=True)

# ════════════════════════════ Detalle del preprocesamiento ════════════════════════════

with st.expander("¿Cómo se preparó tu entrada?"):
    st.markdown(
        "- **Imagen:** se normaliza la intensidad con los percentiles 1 y 99 del corte, se recorta "
        f"una ventana de {pp.CROP_SIZE}×{pp.CROP_SIZE} px centrada en el punto, se amplía a 224×224, "
        "se copia a 3 canales y se escala a [-1, 1].\n"
        "- **Metadatos:** condición, nivel, tipo de serie y grupo se pasan a one-hot; la coordenada se "
        "normaliza con los percentiles 1 y 99 de todas las anotaciones del entrenamiento.\n"
        "- **CNN:** además del recorte recibe la condición y el nivel en one-hot (10 valores)."
    )
    st.dataframe(
        pd.DataFrame(x_meta, columns=pp.FEATURE_COLS).T.rename(columns={0: "valor"}),
        use_container_width=True,
    )
