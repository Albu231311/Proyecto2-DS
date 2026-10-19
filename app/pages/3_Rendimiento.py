"""
Página Rendimiento: métricas de cada modelo en el conjunto de prueba, con gráficas
interactivas que el usuario puede mostrar u ocultar.

Combina dos archivos:
- resultados_metadatos.json: los 4 modelos de metadatos con todo el detalle (y validación).
- resultados.json: la ResNet50 y sus curvas de entrenamiento (history_cnn).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.estilo import (PALETA, ORDEN_CLASES, NOMBRES_MODELOS, COLORES_MODELOS,
                          CONDICIONES, NIVELES, configurar_pagina, aplicar_layout,
                          tarjeta, chip_clase, COLOR_PRINCIPAL, COLOR_FONDO_ALT)

configurar_pagina("Rendimiento", "📊")
RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "data" / "processed"

METRICAS = {
    "exactitud": "Exactitud",
    "f1_macro": "F1 macro",
    "balanced_accuracy": "Balanced accuracy",
    "recall_severe": "Recall Severe",
    "kappa_quadratic": "Kappa cuadrático",
}
COLOR_SECUNDARIO_GRAF = "#A0A8B3"  # gris de la línea base, para la serie de apoyo
ESCALA_PRINCIPAL = [[0, COLOR_FONDO_ALT], [1, COLOR_PRINCIPAL]]
EPOCAS_CONGELADO = 2  # en el notebook: 2 épocas con el backbone congelado, luego fine-tuning


# ════════════════════════════ Carga con caché ════════════════════════════

@st.cache_data
def cargar_resultados() -> tuple[dict, dict, dict]:
    """(modelos por clave en el orden de NOMBRES_MODELOS, history_cnn, meta)."""
    with open(PROC / "resultados.json", encoding="utf-8") as f:
        res = json.load(f)
    with open(PROC / "resultados_metadatos.json", encoding="utf-8") as f:
        res_meta = json.load(f)

    todos = {m["modelo"]: m for m in res_meta["modelos"]}
    # De resultados.json solo se toma la ResNet50: los demás ahí vienen resumidos
    todos["ResNet50_imagen"] = next(m for m in res["modelos"] if m["modelo"] == "ResNet50_imagen")

    # Validación de la CNN: la época que eligió la parada temprana (mejor F1 macro en validación)
    hist = res["history_cnn"]
    mejor = int(np.argmax(hist["val_f1"]))
    todos["ResNet50_imagen"]["val"] = {"f1_macro": hist["val_f1"][mejor],
                                       "recall_severe": hist["val_recall_severe"][mejor]}
    modelos = {c: todos[c] for c in NOMBRES_MODELOS if c in todos}
    return modelos, hist, res["meta"]


@st.cache_data
def estudios_test() -> int:
    return int((pd.read_csv(PROC / "splits.csv")["split"] == "test").sum())


def hallazgo(texto: str, impacto: str) -> None:
    """Recuadro con lo que muestra la gráfica y su relación con el EDA."""
    with st.container(border=True):
        st.markdown(f"**Qué se ve.** {texto}")
        st.markdown(f"**Por qué pasa.** {impacto}")


def tabla_metricas(modelos: dict, claves: list[str]) -> pd.DataFrame:
    filas = []
    for c in claves:
        m = modelos[c]
        filas.append({"Modelo": NOMBRES_MODELOS[c], **{v: m[k] for k, v in METRICAS.items()},
                      "Log-loss RSNA": m.get("log_loss_rsna")})
    return pd.DataFrame(filas)


# ════════════════════════════ Encabezado ════════════════════════════

modelos, hist, meta = cargar_resultados()
dist_test = meta["distribucion_test"]
n_test = sum(dist_test.values())

st.title("Rendimiento de los modelos")
st.markdown(
    f"Resultados en el **conjunto de prueba**: {n_test:,} etiquetas de {estudios_test():,} pacientes "
    "que ningún modelo vio al entrenar (split por `study_id`). Elige qué modelos comparar y qué "
    "gráficas mostrar."
)

mejor_f1 = max(modelos, key=lambda c: modelos[c]["f1_macro"])
mejor_sev = max(modelos, key=lambda c: modelos[c]["recall_severe"])
mejor_exa = max(modelos, key=lambda c: modelos[c]["exactitud"])

c1, c2, c3 = st.columns(3)
with c1:
    tarjeta("Mejor F1 macro", f"{modelos[mejor_f1]['f1_macro']:.3f}", NOMBRES_MODELOS[mejor_f1])
with c2:
    tarjeta("Mejor recall de Severe", f"{modelos[mejor_sev]['recall_severe']:.3f}", NOMBRES_MODELOS[mejor_sev])
with c3:
    tarjeta("Mayor exactitud", f"{modelos[mejor_exa]['exactitud']:.3f}",
            f"{NOMBRES_MODELOS[mejor_exa]}, con F1 macro de solo {modelos[mejor_exa]['f1_macro']:.3f}")

# ════════════════════════════ Controles ════════════════════════════

seleccion = st.multiselect("Modelos a comparar (uno, varios o todos)", list(modelos),
                           default=list(modelos), format_func=lambda c: NOMBRES_MODELOS[c])
if not seleccion:
    st.info("Elige al menos un modelo.")
    st.stop()

with st.container(border=True):
    st.markdown("**Gráficas visibles**")
    t1, t2, t3 = st.columns(3)
    ver_metricas = t1.toggle("Métricas", value=True)
    ver_confusion = t2.toggle("Matriz de confusión", value=True)
    ver_f1_clase = t3.toggle("F1 por clase", value=True)
    ver_tiempo = t1.toggle("Tiempo vs. rendimiento", value=True)
    ver_curvas = t2.toggle("Curvas de la CNN", value=True)
    ver_validacion = t3.toggle("Validación vs. test", value=True)
    st.caption("En cada gráfica también puedes ocultar un modelo o una clase haciendo clic en la leyenda.")

st.dataframe(
    tabla_metricas(modelos, seleccion), hide_index=True, use_container_width=True,
    column_config={
        **{v: st.column_config.ProgressColumn(v, format="%.3f", min_value=0, max_value=1)
           for v in METRICAS.values()},
        "Log-loss RSNA": st.column_config.NumberColumn(format="%.3f", help="Menor es mejor"),
    },
)

nombres_sel = [NOMBRES_MODELOS[c] for c in seleccion]

# ════════════════════════════ 1. Comparación de métricas ════════════════════════════

if ver_metricas:
    st.subheader("Comparación de métricas")
    elegidas = st.multiselect("Métricas", list(METRICAS.values()), default=list(METRICAS.values()))
    largo = tabla_metricas(modelos, seleccion).melt(id_vars="Modelo", value_vars=elegidas,
                                                    var_name="Métrica", value_name="Valor")
    fig = px.bar(largo, x="Métrica", y="Valor", color="Modelo", barmode="group",
                 color_discrete_map=COLORES_MODELOS,
                 category_orders={"Métrica": elegidas, "Modelo": nombres_sel})
    fig.update_traces(marker_line=dict(width=1, color="white"),
                      hovertemplate="%{fullData.name}<br>%{x}: %{y:.3f}<extra></extra>")
    fig.update_layout(yaxis=dict(range=[0, 1], title=None), xaxis_title=None,
                      legend_title_text=None, bargap=0.2)
    st.plotly_chart(aplicar_layout(fig, "Métricas en test por modelo"), use_container_width=True)

    may = modelos["Clase_Mayoritaria"]
    pct_normal = dist_test["Normal/Mild"] / n_test
    hallazgo(
        f"La **clase mayoritaria** tiene la mayor exactitud ({may['exactitud']:.1%}), pero su F1 macro es "
        f"{may['f1_macro']:.3f} y su recall de Severe y su kappa son **0**: nunca detecta un caso Severe. "
        "Los demás modelos bajan la exactitud a cambio de encontrar casos Moderate y Severe; la ResNet50 "
        "es la mejor en las cuatro métricas que toman en cuenta el desbalance.",
        f"Es el desbalance que se vio en Explorar datos: el {pct_normal:.0%} de las etiquetas de test son "
        "Normal/Mild, así que responder siempre Normal/Mild ya da 77% de exactitud. Por eso la exactitud "
        "no sirve para comparar aquí: F1 macro y balanced accuracy le dan el mismo peso a cada clase, y "
        "el recall de Severe mide justo los casos que no se pueden pasar por alto.",
    )

# ════════════════════════════ 2. Matriz de confusión y F1 por clase ════════════════════════════

if ver_confusion or ver_f1_clase:
    st.subheader("Errores por clase")
    col_cm, col_f1 = st.columns(2, gap="large") if (ver_confusion and ver_f1_clase) else (st.container(), st.container())

    if ver_confusion:
        with col_cm:
            c_a, c_b = st.columns(2)
            clave_cm = c_a.selectbox("Modelo", seleccion, format_func=lambda c: NOMBRES_MODELOS[c],
                                     index=seleccion.index("ResNet50_imagen") if "ResNet50_imagen" in seleccion else 0)
            modo = c_b.radio("Mostrar", ["% de cada clase real", "Conteos"], horizontal=True)
            cm = np.array(modelos[clave_cm]["confusion_matrix"], dtype=float)
            if modo == "Conteos":
                valores, formato = cm, ","
            else:
                valores, formato = cm / cm.sum(axis=1, keepdims=True), ".1%"
            fig = px.imshow(valores, x=ORDEN_CLASES, y=ORDEN_CLASES, text_auto=formato,
                            color_continuous_scale=ESCALA_PRINCIPAL, aspect="auto",
                            labels=dict(x="Predicho", y="Real", color=""))
            fig.update_traces(customdata=cm, hovertemplate="Real %{y} → predicho %{x}<br>"
                                                           "%{customdata:,.0f} etiquetas<extra></extra>")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(aplicar_layout(fig, f"Matriz de confusión · {NOMBRES_MODELOS[clave_cm]}", alto=400),
                            use_container_width=True)

    if ver_f1_clase:
        with col_f1:
            filas = [{"Modelo": NOMBRES_MODELOS[c], "Clase": clase, "F1": f1}
                     for c in seleccion for clase, f1 in zip(ORDEN_CLASES, modelos[c]["f1_per_class"])]
            fig = px.bar(pd.DataFrame(filas), x="Modelo", y="F1", color="Clase", barmode="group",
                         color_discrete_map=PALETA, category_orders={"Clase": ORDEN_CLASES, "Modelo": nombres_sel})
            fig.update_traces(marker_line=dict(width=1, color="white"),
                              hovertemplate="%{x}<br>F1 %{fullData.name}: %{y:.3f}<extra></extra>")
            fig.update_layout(yaxis=dict(range=[0, 1], title="F1"), xaxis_title=None, legend_title_text=None)
            st.plotly_chart(aplicar_layout(fig, "F1 por clase", alto=470), use_container_width=True)

    cnn = np.array(modelos["ResNet50_imagen"]["confusion_matrix"], dtype=float)
    rf = np.array(modelos["Random_Forest"]["confusion_matrix"], dtype=float)
    prec_sev_cnn = cnn[2, 2] / cnn[:, 2].sum()
    hallazgo(
        "En todos los modelos el F1 de **Severe** es el más bajo y el de Normal/Mild el más alto. El "
        f"Random Forest predice Normal/Mild en el {rf[:, 0].sum() / rf.sum():.0%} de los casos y deja "
        f"pasar el {1 - rf[2, 2] / rf[2].sum():.0%} de los Severe. La ResNet50 hace lo contrario: encuentra "
        f"el {cnn[2, 2] / cnn[2].sum():.0%} de los Severe, pero solo el {prec_sev_cnn:.0%} de lo que marca "
        "como Severe lo es de verdad.",
        "Con solo 6% de casos Severe, cada modelo elige en qué equivocarse. El entrenamiento con pesos "
        "de clase (y, en la CNN, el muestreo balanceado) empuja a los modelos a buscar Severe, lo que "
        "sube el recall pero también las falsas alarmas. Para un sistema de apoyo clínico es preferible "
        "una falsa alarma que un caso grave sin detectar.",
    )

# ════════════════════════════ 3. Tiempo contra rendimiento ════════════════════════════

if ver_tiempo:
    st.subheader("Tiempo contra rendimiento")
    tipo = st.radio("Tiempo", ["Entrenamiento", "Inferencia por muestra"], horizontal=True)
    campo, unidad = ("t_train_s", "s") if tipo == "Entrenamiento" else ("t_infer_ms_per_sample", "ms")
    filas = [{"Modelo": NOMBRES_MODELOS[c], "tiempo": modelos[c].get(campo), "F1 macro": modelos[c]["f1_macro"]}
             for c in seleccion]
    datos = pd.DataFrame(filas)
    sin_tiempo = datos[~(datos["tiempo"] > 0)]["Modelo"].tolist()
    datos = datos[datos["tiempo"] > 0]
    if datos.empty:
        st.info("Ninguno de los modelos elegidos tiene ese tiempo registrado.")
    else:
        log_t = np.log10(datos["tiempo"])
        datos["posicion"] = np.where(log_t > log_t.median(), "middle left", "middle right")
        fig = px.scatter(datos, x="tiempo", y="F1 macro", color="Modelo", text="Modelo", log_x=True,
                         color_discrete_map=COLORES_MODELOS)
        for traza in fig.data:
            traza.textposition = datos.loc[datos["Modelo"] == traza.name, "posicion"].iloc[0]
        fig.update_traces(marker=dict(size=14, line=dict(width=2, color="white")),
                          hovertemplate=f"<b>%{{text}}</b><br>{tipo}: %{{x:.4g}} {unidad}"
                                        "<br>F1 macro: %{y:.3f}<extra></extra>")
        fig.update_layout(showlegend=False,
                          xaxis=dict(title=f"{tipo} ({unidad}, escala log)",
                                     range=[log_t.min() - 0.7, log_t.max() + 0.7]),
                          yaxis=dict(title="F1 macro en test", range=[0.2, 0.55]))
        st.plotly_chart(aplicar_layout(fig, "Más arriba = mejor; más a la izquierda = más barato", alto=400),
                        use_container_width=True)
        if sin_tiempo:
            st.caption(f"Sin tiempo registrado: {', '.join(sin_tiempo)} (no se entrena).")

    cnn, rf = modelos["ResNet50_imagen"], modelos["Random_Forest"]
    hallazgo(
        f"La ResNet50 tardó **{cnn['t_train_s'] / 60:.0f} minutos** en entrenar en una GPU T4; el Random "
        f"Forest, **{rf['t_train_s']:.1f} s** en CPU, unas {cnn['t_train_s'] / rf['t_train_s']:,.0f} veces "
        f"menos. En inferencia, la CNN tarda {cnn['t_infer_ms_per_sample']:.1f} ms por muestra y el Random "
        f"Forest {rf['t_infer_ms_per_sample'] * 1000:.0f} µs.",
        f"El costo extra de la CNN compra una mejora pequeña en F1 macro ({rf['f1_macro']:.3f} → "
        f"{cnn['f1_macro']:.3f}) y una grande en recall de Severe ({rf['recall_severe']:.3f} → "
        f"{cnn['recall_severe']:.3f}) y balanced accuracy. Los modelos de metadatos son baratos, pero "
        "tienen un techo porque no ven la imagen (ver *Variables de los modelos* en Explorar datos).",
    )

# ════════════════════════════ 4. Curvas de entrenamiento de la CNN ════════════════════════════

if ver_curvas:
    st.subheader("Curvas de entrenamiento de la ResNet50")
    if "ResNet50_imagen" not in seleccion:
        st.info("Agrega la ResNet50 a los modelos seleccionados para ver sus curvas.")
    else:
        epocas = list(range(1, len(hist["train_loss"]) + 1))
        col_loss, col_val = st.columns(2, gap="large")

        def linea_descongelar(fig: go.Figure) -> go.Figure:
            fig.add_vline(x=EPOCAS_CONGELADO + 0.5, line_dash="dash", line_color=COLOR_SECUNDARIO_GRAF,
                          annotation_text="Se descongela el backbone", annotation_position="top right",
                          annotation_font_size=11)
            return fig

        with col_loss:
            fig = go.Figure([
                go.Scatter(x=epocas, y=hist["train_loss"], name="Entrenamiento", mode="lines+markers",
                           line=dict(color=COLOR_SECUNDARIO_GRAF, width=2)),
                go.Scatter(x=epocas, y=hist["val_loss"], name="Validación", mode="lines+markers",
                           line=dict(color=COLOR_PRINCIPAL, width=2)),
            ])
            fig.update_traces(hovertemplate="Época %{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>")
            fig.update_layout(xaxis=dict(title="Época", dtick=1), yaxis=dict(title="Pérdida (cross-entropy)", rangemode="tozero"))
            st.plotly_chart(aplicar_layout(linea_descongelar(fig), "Pérdida", alto=380), use_container_width=True)

        with col_val:
            fig = go.Figure([
                go.Scatter(x=epocas, y=hist["val_f1"], name="F1 macro", mode="lines+markers",
                           line=dict(color=COLOR_PRINCIPAL, width=2)),
                go.Scatter(x=epocas, y=hist["val_recall_severe"], name="Recall Severe", mode="lines+markers",
                           line=dict(color=PALETA["Severe"], width=2)),
            ])
            fig.update_traces(hovertemplate="Época %{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>")
            fig.update_layout(xaxis=dict(title="Época", dtick=1), yaxis=dict(title=None, range=[0, 1.05]))
            st.plotly_chart(aplicar_layout(linea_descongelar(fig), "Métricas en validación", alto=380),
                            use_container_width=True)

        min_val = int(np.argmin(hist["val_loss"])) + 1
        hallazgo(
            f"Con el backbone congelado (épocas 1 y {EPOCAS_CONGELADO}) la red casi solo predice Severe: "
            f"recall de Severe cerca de 1 y F1 macro bajo. Al descongelarlo, el F1 macro sube hasta "
            f"{max(hist['val_f1']):.3f}. La pérdida de validación es mínima en la época {min_val} y después "
            "sube mientras la de entrenamiento sigue bajando.",
            "El entrenamiento usa muestreo balanceado (ve tantos Severe como Normal/Mild), pero la "
            "validación tiene la distribución real, con 77% Normal/Mild. Al inicio la red sobreestima "
            "Severe; luego aprende a distinguir las clases y empieza a sobreajustar, lo que se ve en la "
            "pérdida de validación. La parada temprana eligió el modelo por F1 macro, que siguió subiendo "
            "hasta la última época.",
        )

# ════════════════════════════ 5. Validación contra test ════════════════════════════

if ver_validacion:
    st.subheader("Validación contra test")
    metrica_vt = st.radio("Métrica", ["F1 macro", "Recall Severe"], horizontal=True, key="metrica_vt")
    campo = "f1_macro" if metrica_vt == "F1 macro" else "recall_severe"
    filas = [{"Modelo": NOMBRES_MODELOS[c], "Conjunto": conjunto, "Valor": valor}
             for c in seleccion
             for conjunto, valor in [("Validación", modelos[c]["val"][campo]), ("Test", modelos[c][campo])]]
    datos = pd.DataFrame(filas)
    fig = px.bar(datos, x="Modelo", y="Valor", color="Conjunto", barmode="group",
                 color_discrete_map={"Validación": COLOR_SECUNDARIO_GRAF, "Test": COLOR_PRINCIPAL},
                 category_orders={"Modelo": nombres_sel})
    fig.update_traces(marker_line=dict(width=1, color="white"),
                      hovertemplate="%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>")
    fig.update_layout(yaxis=dict(range=[0, 1], title=metrica_vt), xaxis_title=None, legend_title_text=None)
    st.plotly_chart(aplicar_layout(fig, f"{metrica_vt} en validación y en test", alto=380),
                    use_container_width=True)

    dif = max(abs(modelos[c]["val"]["f1_macro"] - modelos[c]["f1_macro"]) for c in modelos)
    hallazgo(
        f"El F1 macro de cada modelo cambia como máximo **{dif:.3f}** entre validación y test. La "
        "validación de la ResNet50 es la de la época que eligió la parada temprana.",
        "Como el split se hizo por paciente, validación y test son pacientes nuevos para el modelo, y las "
        "métricas se mantienen: los hiperparámetros elegidos en validación no quedaron ajustados a esos "
        "pacientes en particular, y lo que se reporta en test es una estimación honesta.",
    )
