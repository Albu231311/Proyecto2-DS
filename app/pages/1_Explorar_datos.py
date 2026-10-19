"""
Página Explorar datos: distribución de la severidad, ubicación de las anotaciones,
relaciones entre condiciones y las variables que usan los modelos.
Cada sección cierra con el hallazgo del EDA y cómo influye en los modelos.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.estilo import (PALETA, ORDEN_CLASES, NOMBRES_MODELOS, COLORES_MODELOS,
                          CONDICIONES, NIVELES, configurar_pagina, aplicar_layout,
                          tarjeta, chip_clase, COLOR_PRINCIPAL, COLOR_FONDO_ALT)
from utils import modelos as md
from utils import preprocesamiento as pp

configurar_pagina("Explorar datos", "🔎")
RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "data" / "processed"

# Los CSV procesados usan spinal_canal_stenosis y l1_l2; en la app se muestran como en estilo.py
CONDICION_CSV = {c.lower().replace(" ", "_"): c for c in CONDICIONES}
NIVEL_CSV = {n.lower().replace("/", "_"): n for n in NIVELES}
SERIES = ["Sagittal T2/STIR", "Sagittal T1", "Axial T2"]
GRUPOS = ["canal", "foraminal", "subarticular"]
CORTO = {c: c.replace(" Neural", "").replace(" Narrowing", "").replace(" Stenosis", "") for c in CONDICIONES}

# Colores de grupo y de serie: el mismo tono para cada grupo y la serie donde se anota
# (canal ↔ Sagittal T2/STIR, foraminal ↔ Sagittal T1, subarticular ↔ Axial T2).
# Tres tonos validados para daltonismo que no se confunden con los de severidad.
TONOS = ["#6250d6", "#008300", "#e87ba4"]
COLORES_GRUPO = dict(zip(GRUPOS, TONOS))
COLORES_SERIE = dict(zip(SERIES, TONOS))

# Escalas secuenciales de un solo tono (claro → oscuro)
ESCALA_SEVERE = [[0, "#F7F2F2"], [1, PALETA["Severe"]]]
ESCALA_PRINCIPAL = [[0, COLOR_FONDO_ALT], [1, COLOR_PRINCIPAL]]


# ════════════════════════════ Carga con caché ════════════════════════════

@st.cache_data
def cargar_largo() -> pd.DataFrame:
    """Una fila por estudio × condición × nivel con su severidad (sin nulos)."""
    df = pd.read_csv(PROC / "train_formato_largo.csv")
    df["condicion"] = df["condicion"].map(CONDICION_CSV)
    df["nivel"] = df["nivel"].map(NIVEL_CSV)
    return df


@st.cache_data
def cargar_ancho() -> pd.DataFrame:
    """Una fila por estudio; columnas condición_nivel con severidad 0/1/2 (con nulos)."""
    return pd.read_csv(PROC / "train_severidad_numerica.csv").set_index("study_id")


@st.cache_data
def cargar_nulos() -> pd.DataFrame:
    df = pd.read_csv(PROC / "tabla_nulos_condicion_nivel.csv")
    df["condicion"] = df["condicion"].map(CONDICION_CSV)
    return df.set_index("condicion").rename(columns=NIVEL_CSV).loc[CONDICIONES, NIVELES]


@st.cache_data
def cargar_datos_modelo() -> pd.DataFrame:
    """Las mismas filas y variables con que se entrenaron los modelos de metadatos."""
    df = pp.datos_entrenamiento()
    x = pd.DataFrame(pp.features_metadatos(df), columns=pp.FEATURE_COLS, index=df.index)
    df["grupo"] = df["condition"].map(pp._grupo)
    df["x_norm"] = x["x_norm"]
    df["y_norm"] = x["y_norm"]
    splits = pd.read_csv(PROC / "splits.csv").set_index("study_id")["split"]
    df["split"] = df["study_id"].map(splits)
    return df.rename(columns={"condition": "condicion", "level": "nivel",
                              "series_description": "serie"})


@st.cache_data
def importancias(clave: str) -> pd.Series:
    modelo = md.cargar_modelo(clave)
    return pd.Series(modelo.feature_importances_, index=pp.FEATURE_COLS).sort_values(ascending=False)


def hallazgo(texto: str, impacto: str) -> None:
    """Recuadro con un hallazgo del EDA y su efecto en los modelos."""
    with st.container(border=True):
        st.markdown(f"**Hallazgo.** {texto}")
        st.markdown(f"**Qué significa para los modelos.** {impacto}")


def perfil(df: pd.DataFrame, por: str, orden: list[str]) -> pd.DataFrame:
    """Porcentaje de cada clase de severidad dentro de cada categoría de `por`."""
    tabla = (df.groupby([por, "severidad"]).size().rename("n").reset_index())
    tabla["pct"] = tabla["n"] / tabla.groupby(por)["n"].transform("sum")
    tabla[por] = pd.Categorical(tabla[por], orden, ordered=True)
    return tabla.sort_values(por)


def barras_perfil(tabla: pd.DataFrame, por: str, orden: list[str], horizontal: bool, titulo: str):
    ejes = dict(x="pct", y=por) if horizontal else dict(x=por, y="pct")
    fig = px.bar(
        tabla, **ejes, color="severidad", orientation="h" if horizontal else "v",
        color_discrete_map=PALETA, custom_data=["n"],
        category_orders={"severidad": ORDEN_CLASES, por: orden[::-1] if horizontal else orden},
    )
    valor = "%{x:.1%}" if horizontal else "%{y:.1%}"
    fig.update_traces(marker_line=dict(width=1, color="white"),
                      hovertemplate=f"%{{fullData.name}}: {valor}<br>%{{customdata[0]:,}} etiquetas<extra></extra>")
    eje_pct = dict(tickformat=".0%", title=None, range=[0, 1.001], dtick=0.25)
    fig.update_layout(legend_title_text=None, bargap=0.3,
                      **({"xaxis": eje_pct, "yaxis_title": None} if horizontal
                         else {"yaxis": eje_pct, "xaxis_title": None}))
    return aplicar_layout(fig, titulo, alto=360)


# ════════════════════════════ Encabezado ════════════════════════════

largo = cargar_largo()
ancho = cargar_ancho()

st.title("Explorar datos")
st.markdown(
    "Los datos limpios del reto: **una etiqueta de severidad por estudio, condición y nivel**, "
    "y la coordenada donde el radiólogo marcó cada hallazgo. Usa los filtros para enfocarte en "
    "una parte del dataset; al final de cada pestaña está lo que encontramos y cómo afecta a los modelos."
)

pct_clase = largo["severidad"].value_counts(normalize=True)
por_nivel = largo.groupby("nivel")["severidad"].value_counts(normalize=True).unstack()
nivel_peor = por_nivel["Severe"].idxmax()

c1, c2, c3, c4 = st.columns(4)
with c1:
    tarjeta("Etiquetas de severidad", f"{len(largo):,}", f"{largo['study_id'].nunique():,} estudios")
with c2:
    tarjeta("Normal/Mild", f"{pct_clase['Normal/Mild']:.1%}", "La clase mayoritaria")
with c3:
    tarjeta("Severe", f"{pct_clase['Severe']:.1%}", "La clase que más importa clínicamente")
with c4:
    tarjeta("Nivel más afectado", nivel_peor, f"{por_nivel.loc[nivel_peor, 'Severe']:.1%} de casos Severe")

# ════════════════════════════ Filtros ════════════════════════════

f1, f2, f3 = st.columns([5, 3, 3])
filtro_cond = f1.multiselect("Condición", CONDICIONES, default=CONDICIONES, format_func=CORTO.get)
filtro_nivel = f2.multiselect("Nivel", NIVELES, default=NIVELES)
filtro_sev = f3.multiselect("Severidad", ORDEN_CLASES, default=ORDEN_CLASES)
if not (filtro_cond and filtro_nivel and filtro_sev):
    st.info("Elige al menos una condición, un nivel y una severidad.")
    st.stop()

largo_f = largo[largo["condicion"].isin(filtro_cond) & largo["nivel"].isin(filtro_nivel)
                & largo["severidad"].isin(filtro_sev)]
st.caption(f"{len(largo_f):,} de {len(largo):,} etiquetas con los filtros actuales.")

tab_sev, tab_coord, tab_rel, tab_var = st.tabs(
    ["Severidad", "Ubicación de las anotaciones", "Relaciones entre condiciones", "Variables de los modelos"]
)

# ════════════════════════════ 1. Severidad ════════════════════════════

with tab_sev:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        conteo = largo_f["severidad"].value_counts().reindex(ORDEN_CLASES).dropna().rename("n").reset_index()
        conteo["pct"] = conteo["n"] / conteo["n"].sum()
        fig = px.bar(conteo, x="n", y="severidad", orientation="h", color="severidad",
                     color_discrete_map=PALETA, custom_data=["pct"],
                     category_orders={"severidad": ORDEN_CLASES[::-1]})
        fig.update_traces(texttemplate="%{x:,} (%{customdata[0]:.1%})", textposition="outside",
                          cliponaxis=False, hovertemplate="%{y}: %{x:,} etiquetas<extra></extra>")
        fig.update_layout(showlegend=False, yaxis_title=None,
                          xaxis=dict(title="Etiquetas", range=[0, conteo["n"].max() * 1.3]))
        st.plotly_chart(aplicar_layout(fig, "Distribución de la severidad", alto=360),
                        use_container_width=True)

    with col_b:
        st.plotly_chart(
            barras_perfil(perfil(largo_f, "nivel", NIVELES), "nivel", NIVELES, horizontal=False,
                          titulo="La severidad aumenta al bajar por la columna"),
            use_container_width=True)

    col_c, col_d = st.columns(2, gap="large")

    with col_c:
        st.plotly_chart(
            barras_perfil(perfil(largo_f, "condicion", CONDICIONES), "condicion", CONDICIONES,
                          horizontal=True, titulo="Perfil de severidad por condición"),
            use_container_width=True)

    with col_d:
        medida = st.radio("Mapa de calor", ["% Severe", "Severidad promedio (0 a 2)", "Etiquetas"],
                          horizontal=True, label_visibility="collapsed")
        grupos = largo_f.groupby(["condicion", "nivel"])
        if medida == "% Severe":
            valores, escala, formato = grupos["severidad"].apply(lambda s: (s == "Severe").mean()), ESCALA_SEVERE, ".1%"
        elif medida == "Etiquetas":
            valores, escala, formato = grupos.size(), ESCALA_PRINCIPAL, ","
        else:
            valores, escala, formato = grupos["severidad_num"].mean(), ESCALA_SEVERE, ".2f"
        matriz = valores.unstack().reindex(index=[c for c in CONDICIONES if c in filtro_cond],
                                           columns=[n for n in NIVELES if n in filtro_nivel])
        fig = px.imshow(matriz, text_auto=formato, color_continuous_scale=escala, aspect="auto",
                        labels=dict(x="Nivel", y="Condición", color=medida))
        fig.update_layout(xaxis_title=None, yaxis_title=None,
                          coloraxis_colorbar=dict(title=None, tickformat=".0%" if formato == ".1%" else None))
        st.plotly_chart(aplicar_layout(fig, f"{medida} por condición y nivel", alto=330),
                        use_container_width=True)

    canal = largo[largo["condicion"] == "Spinal Canal Stenosis"].groupby("nivel")["severidad_num"].mean()
    hallazgo(
        f"Las clases están muy desbalanceadas: **{pct_clase['Normal/Mild']:.1%}** de las etiquetas son "
        f"Normal/Mild y solo **{pct_clase['Severe']:.1%}** son Severe. Además hay un gradiente claro a lo "
        f"largo de la columna: los casos Severe pasan de **{por_nivel.loc['L1/L2', 'Severe']:.1%}** en "
        f"L1/L2 a **{por_nivel.loc['L4/L5', 'Severe']:.1%}** en L4/L5, el nivel con más carga mecánica. "
        f"La estenosis del canal es la excepción en L5/S1: cae de {canal['L4/L5']:.2f} a "
        f"{canal['L5/S1']:.2f} de severidad promedio, mientras las demás condiciones siguen altas.",
        "Un modelo que siempre responde Normal/Mild acierta el 77% de las veces sin aprender nada; por "
        "eso se entrenó con pesos de clase y se compara con **F1 macro** y **recall de Severe**, no con "
        "la exactitud (ver la página Rendimiento). El gradiente hace que el **nivel** sea una variable "
        "útil: el Árbol de Decisión la usa como su variable más importante.",
    )

    with st.expander("Calidad de los datos: etiquetas faltantes en train.csv"):
        nulos = cargar_nulos()
        fig = px.imshow(nulos, text_auto=".2f", color_continuous_scale=ESCALA_PRINCIPAL, aspect="auto",
                        labels=dict(color="% faltante"))
        fig.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_colorbar=dict(title="%"))
        st.plotly_chart(aplicar_layout(fig, "Porcentaje de etiquetas faltantes", alto=300),
                        use_container_width=True)
        st.markdown(
            "Los faltantes se concentran en **Subarticular Stenosis** en los niveles altos (hasta 8.3% en "
            "L1/L2). No se imputaron, porque inventar un diagnóstico introduce un sesgo clínico: esas "
            "filas simplemente no se usan para entrenar."
        )

# ════════════════════════════ 2. Ubicación de las anotaciones ════════════════════════════

with tab_coord:
    datos = cargar_datos_modelo()
    datos_f = datos[datos["condicion"].isin(filtro_cond) & datos["nivel"].isin(filtro_nivel)
                    & datos["severidad"].isin(filtro_sev)]

    color_por = st.radio("Colorear por", ["Grupo de condición", "Tipo de serie", "Severidad"], horizontal=True)
    columna, mapa, orden = {
        "Grupo de condición": ("grupo", COLORES_GRUPO, GRUPOS),
        "Tipo de serie": ("serie", COLORES_SERIE, SERIES),
        "Severidad": ("severidad", PALETA, ORDEN_CLASES),
    }[color_por]

    # Una muestra fija basta para ver los grupos y evita depender de WebGL en el navegador
    MAX_PUNTOS = 8000
    muestra = datos_f.sample(min(MAX_PUNTOS, len(datos_f)), random_state=42)

    col_disp, col_caja = st.columns([3, 2], gap="large")
    with col_disp:
        # Severe al final para que quede encima de las otras clases
        datos_graf = muestra.sort_values(columna, key=lambda s: s.map({v: i for i, v in enumerate(orden)}))
        fig = px.scatter(
            datos_graf, x="x", y="y", color=columna, render_mode="svg", opacity=0.5,
            color_discrete_map=mapa, category_orders={columna: orden},
            hover_data={"condicion": True, "nivel": True, "serie": True, "severidad": True,
                        "x": ":.1f", "y": ":.1f"},
        )
        fig.update_traces(marker=dict(size=4))
        # En una imagen el eje y crece hacia abajo
        fig.update_layout(yaxis=dict(autorange="reversed", title="y (px)"), xaxis_title="x (px)",
                          legend_title_text=None)
        st.plotly_chart(aplicar_layout(fig, "Dónde se marcó cada anotación en la imagen", alto=480),
                        use_container_width=True)
        if len(datos_f) > MAX_PUNTOS:
            st.caption(f"Muestra aleatoria de {MAX_PUNTOS:,} de {len(datos_f):,} anotaciones. "
                       "Los diagramas de caja usan todas.")

    with col_caja:
        eje = st.radio("Coordenada", ["x", "y"], horizontal=True)
        fig = px.box(datos_f, x=eje, y=columna, color=columna, color_discrete_map=mapa,
                     category_orders={columna: orden}, points=False)
        fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title=f"{eje} (px)")
        st.plotly_chart(aplicar_layout(fig, f"Distribución de {eje}", alto=430), use_container_width=True)

    # Cifras con todos los datos, para que el texto no cambie con los filtros
    sagital = datos[datos["grupo"].isin(["canal", "foraminal"])].groupby("severidad")["y"].median()
    subart = datos[datos["grupo"] == "subarticular"].groupby("condicion")["x"].median()
    hallazgo(
        "Cada condición se anotó casi siempre en un solo tipo de serie: el canal en **Sagittal T2/STIR**, "
        "el foramen en **Sagittal T1** y el receso subarticular en **Axial T2** (por eso colorear por "
        "grupo o por serie da casi el mismo dibujo). En los cortes sagitales, **y** crece al bajar por la "
        f"columna, así que los casos Severe quedan más abajo: mediana de y = {sagital['Severe']:.0f} px "
        f"contra {sagital['Normal/Mild']:.0f} px en Normal/Mild. En los cortes axiales, izquierda y derecha "
        f"se separan en **x** (mediana de {subart['Left Subarticular Stenosis']:.0f} px contra "
        f"{subart['Right Subarticular Stenosis']:.0f} px). Los valores atípicos (1.5% en x, 1.8% en y) "
        "aparecen en las tres series y se deben a imágenes de otro tamaño, así que se conservaron.",
        "La coordenada **y** funciona como un sustituto del nivel vertebral: lleva consigo el gradiente "
        "de severidad de la pestaña anterior. Por eso las coordenadas son las variables más usadas por el "
        "Random Forest (pestaña *Variables de los modelos*). Pero la coordenada dice **dónde** está la "
        "estructura, no **cómo** se ve; para ver el tejido hace falta la imagen, que es lo que aporta la "
        "ResNet50.",
    )

# ════════════════════════════ 3. Relaciones entre condiciones ════════════════════════════

with tab_rel:
    st.caption("Esta pestaña usa los filtros de condición y nivel; el de severidad no aplica, "
               "porque se analiza a cada paciente completo.")
    columnas = {c: [f"{k}_{n.lower().replace('/', '_')}" for n in filtro_nivel]
                for k, c in CONDICION_CSV.items() if c in filtro_cond}

    col_corr, col_comor = st.columns(2, gap="large")

    with col_corr:
        if len(columnas) < 2:
            st.info("Elige al menos dos condiciones para ver su correlación.")
        else:
            puntaje = pd.DataFrame({c: ancho[cols].mean(axis=1) for c, cols in columnas.items()})
            corr = puntaje.corr()
            fig = px.imshow(corr, text_auto=".2f", zmin=0, zmax=1, aspect="auto",
                            color_continuous_scale=ESCALA_PRINCIPAL)
            fig.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_colorbar=dict(title="r"))
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(aplicar_layout(fig, "Correlación de la severidad promedio por paciente", alto=460),
                            use_container_width=True)

    todas = [c for cols in columnas.values() for c in cols]
    n_severe = (ancho[todas] == 2).sum(axis=1)
    with col_comor:
        conteo = n_severe.value_counts().sort_index().rename("estudios").reset_index()
        conteo.columns = ["n_severe", "estudios"]
        conteo["pct"] = conteo["estudios"] / conteo["estudios"].sum()
        fig = px.bar(conteo, x="n_severe", y="estudios", custom_data=["pct"],
                     color_discrete_sequence=[PALETA["Severe"]])
        fig.update_traces(hovertemplate="%{x} etiquetas Severe: %{y:,} estudios "
                                        "(%{customdata[0]:.1%})<extra></extra>")
        fig.update_layout(xaxis=dict(title=f"Etiquetas Severe por estudio (de {len(todas)})", dtick=1),
                          yaxis_title="Estudios", bargap=0.15)
        st.plotly_chart(aplicar_layout(fig, "Comorbilidad: casos Severe en el mismo paciente", alto=460),
                        use_container_width=True)

    sin_severe = (n_severe == 0).mean()
    cinco_o_mas = (n_severe >= 5).mean()
    hallazgo(
        "Las cinco condiciones están correlacionadas entre sí (de 0.43 a 0.66 con todos los datos): un "
        "paciente con más degeneración en una condición tiende a tenerla también en las demás. La más "
        "fuerte es entre la estenosis del canal y la subarticular, que son estructuras vecinas. Con los "
        f"filtros actuales, **{sin_severe:.1%}** de los estudios no tiene ninguna etiqueta Severe, pero "
        f"**{cinco_o_mas:.1%}** tiene 5 o más al mismo tiempo.",
        "Todos los modelos clasifican cada etiqueta por separado, sin saber cómo están las demás "
        "condiciones del mismo paciente. Se pierde esa información: un modelo que viera el estudio "
        "completo podría usar la severidad de los niveles vecinos. Por la misma razón el split se hizo "
        "por paciente (`study_id`), para que un paciente no aparezca en train y en test a la vez.",
    )

# ════════════════════════════ 4. Variables de los modelos ════════════════════════════

with tab_var:
    st.markdown(
        f"Los modelos de metadatos reciben **{len(pp.FEATURE_COLS)} columnas** por etiqueta. La ResNet50 "
        "recibe además el recorte de 96×96 px de la imagen alrededor de la coordenada, junto con la "
        "condición y el nivel."
    )
    variables = pd.DataFrame([
        ("Condición", "condition_* (5)", "Categórica, one-hot", "Cuál de las 5 condiciones se evalúa."),
        ("Nivel", "level_* (5)", "Categórica, one-hot", "Disco vertebral, de L1/L2 a L5/S1."),
        ("Tipo de serie", "series_description_* (3)", "Categórica, one-hot",
         "Secuencia de resonancia donde se anotó: Sagittal T1, Sagittal T2/STIR o Axial T2."),
        ("Grupo de condición", "grupo_* (3)", "Categórica, one-hot",
         "Canal, foraminal o subarticular (une izquierda y derecha)."),
        ("Número de instancia", "instance_number", "Numérica", "Corte de la serie donde se marcó el punto."),
        ("Coordenadas", "x, y", "Numérica (px)", "Posición del punto en la imagen."),
        ("Coordenadas normalizadas", "x_norm, y_norm", "Numérica (0 a 1)",
         "x, y recortadas a los percentiles 1 y 99 y llevadas a 0–1."),
    ], columns=["Variable", "Columnas", "Tipo", "Qué representa"])
    st.dataframe(variables, hide_index=True, use_container_width=True)

    datos = cargar_datos_modelo()
    datos_f = datos[datos["condicion"].isin(filtro_cond) & datos["nivel"].isin(filtro_nivel)
                    & datos["severidad"].isin(filtro_sev)]

    col_dist, col_imp = st.columns(2, gap="large")

    with col_dist:
        opciones = {
            "Condición": ("condicion", CONDICIONES), "Nivel": ("nivel", NIVELES),
            "Tipo de serie": ("serie", SERIES), "Grupo de condición": ("grupo", ["canal", "foraminal", "subarticular"]),
            "Número de instancia": ("instance_number", None), "x": ("x", None), "y": ("y", None),
            "x_norm": ("x_norm", None), "y_norm": ("y_norm", None),
        }
        elegida = st.selectbox("Variable", list(opciones))
        columna, orden = opciones[elegida]
        if orden is not None:
            presentes = [o for o in orden if o in set(datos_f[columna])]
            fig = barras_perfil(perfil(datos_f, columna, presentes), columna, presentes, horizontal=True,
                                titulo=f"Severidad según {elegida.lower()}")
        else:
            valores = datos_f[columna]
            hasta = valores.quantile(0.99)
            fig = px.histogram(datos_f[valores <= hasta], x=columna, color="severidad", nbins=50,
                               histnorm="percent", barmode="overlay", opacity=0.55,
                               color_discrete_map=PALETA, category_orders={"severidad": ORDEN_CLASES})
            fig.update_layout(yaxis_title="% de cada clase", xaxis_title=elegida, legend_title_text=None)
            fig = aplicar_layout(fig, f"Distribución de {elegida} por severidad", alto=360)
        st.plotly_chart(fig, use_container_width=True)
        if orden is None:
            st.caption("Cada clase suma 100%, para comparar su forma aunque Severe tenga pocas etiquetas. "
                       "Se omite el 1% de valores más altos.")

    with col_imp:
        _, faltantes = md.modelos_disponibles()
        con_importancia = [c for c in ["Random_Forest", "Arbol_Decision"] if c not in faltantes]
        if "Random_Forest" in faltantes:
            st.image(str(RAIZ / "figures" / "fig4_importancia_variables_rf.png"),
                     caption="Importancia de variables del Random Forest (figura del notebook). "
                             "Descarga el modelo para verla interactiva.")
        clave = st.selectbox("Modelo", con_importancia, format_func=lambda c: NOMBRES_MODELOS[c])
        imp = importancias(clave).head(12).rename("importancia").reset_index()
        imp.columns = ["columna", "importancia"]
        imp["tipo"] = np.where(imp["columna"].isin(["x", "y", "x_norm", "y_norm", "instance_number"]),
                               "Posición", "Categórica")
        imp["nombre"] = (imp["columna"].str.replace("condition_", "Condición: ")
                         .str.replace("level_", "Nivel: ").str.replace("series_description_", "Serie: ")
                         .str.replace("grupo_", "Grupo: "))
        fig = px.bar(imp, x="importancia", y="nombre", color="tipo", orientation="h",
                     color_discrete_map={"Posición": COLOR_PRINCIPAL, "Categórica": "#A0A8B3"},
                     category_orders={"nombre": list(imp["nombre"])})
        fig.update_traces(hovertemplate="%{y}: %{x:.3f}<extra></extra>")
        fig.update_layout(yaxis_title=None, xaxis=dict(title="Importancia (reducción de Gini)", tickformat=".0%"),
                          legend_title_text=None)
        st.plotly_chart(aplicar_layout(fig, f"Variables más importantes · {NOMBRES_MODELOS[clave]}", alto=420),
                        use_container_width=True)

    imp_rf = importancias("Random_Forest") if "Random_Forest" in con_importancia else None
    imp_dt = importancias("Arbol_Decision")
    posicion = ["x", "y", "x_norm", "y_norm", "instance_number"]
    texto_rf = (f"En el Random Forest, las coordenadas y el número de corte suman "
                f"**{imp_rf[posicion].sum():.0%}** de la importancia" if imp_rf is not None
                else "En el Random Forest dominan las coordenadas y el número de corte")
    niveles_dt = imp_dt[[c for c in imp_dt.index if c.startswith("level_")]].sum()
    hallazgo(
        f"{texto_rf}, mientras que en el Árbol de Decisión el **nivel** pesa "
        f"**{niveles_dt:.0%}**. Son dos formas de usar los mismos datos: el árbol, con pocas divisiones, "
        "se queda con el gradiente por nivel que vimos en la pestaña *Severidad*; el bosque, con miles de "
        "divisiones, aprovecha las variables continuas (la importancia de Gini, además, tiende a favorecer "
        "variables con muchos valores distintos).",
        "Ninguna de estas variables describe el tejido: todas dicen **qué** se evalúa y **dónde**. Eso pone "
        "un techo a los modelos de metadatos (F1 macro entre 0.39 y 0.43) y explica por qué la ResNet50, "
        "que ve la imagen, los supera con claridad (F1 macro de 0.667 y kappa de 0.65).",
    )
