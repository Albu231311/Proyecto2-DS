# Proyecto 2 — Clasificación degenerativa de la columna lumbar (RSNA 2024)

**CC3084 Data Science · Universidad del Valle de Guatemala · Semestre II 2026**

Clasificación de la severidad (**Normal/Mild, Moderate, Severe**) de 5 condiciones degenerativas de la columna lumbar en 5 niveles vertebrales (L1/L2 a L5/S1), a partir de resonancias magnéticas en formato DICOM. El reto está basado en la competencia de Kaggle [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification).

Condiciones evaluadas:
- Spinal Canal Stenosis
- Left / Right Neural Foraminal Narrowing
- Left / Right Subarticular Stenosis

## Estructura del repositorio

```
Proyecto2-DS/
├── Proyecto2_Limpieza_Preprocesamiento.ipynb   # Limpieza, preprocesamiento y análisis exploratorio
├── Proyecto2_Modelos_Baseline.ipynb            # Modelos sobre metadatos (Árbol, Random Forest, Gradient Boosting)
├── Modelo_CNN_ResNet50.ipynb                   # CNN ResNet50 sobre recortes de imagen DICOM
├── Analisis.pdf                                # Investigación previa (contexto médico)
├── data/
│   ├── train.csv, train_label_coordinates.csv, train_series_descriptions.csv
│   ├── 44036939/                               # Estudio DICOM de ejemplo
│   └── processed/                              # Datos limpios, splits y métricas (resultados.json)
├── figures/                                    # Figuras estáticas de resultados
├── models/                                     # Modelos entrenados
├── app/                                        # Aplicación en Streamlit (en desarrollo)
└── requirements.txt
```

## Instalación

Se recomienda Python 3.11 o superior.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> `scikit-learn` está fijo en **1.9.0** porque es la versión con la que se guardaron los modelos `.joblib`. Con otra versión pueden no cargar o dar resultados distintos.

### Modelo Random Forest (descarga manual)

`models/random_forest_metadatos.joblib` pesa **286 MB** y no cabe en GitHub (límite de 100 MB), así que no está en el repositorio. Para usarlo:

1. Descargarlo de Google Drive: https://drive.google.com/file/d/18lkQVsv5I2L7PJRFrLbypYy65_I9_n56/view?usp=sharing
2. Guardarlo en `models/random_forest_metadatos.joblib`.

El archivo ya está en `.gitignore`, así que no se va a subir por error.

| Modelo | Archivo | En el repositorio |
|---|---|---|
| Árbol de Decisión | `models/arbol_decision_metadatos.joblib` | Sí |
| Gradient Boosting | `models/gradient_boosting_metadatos.joblib` | Sí |
| ResNet50 | `models/resnet50_best.pt` | Sí |
| Random Forest | `models/random_forest_metadatos.joblib` | No (Google Drive) |

## Cómo correr

**Notebooks:** abrirlos en orden: limpieza → baseline → CNN. La CNN necesita GPU y el dataset completo de imágenes de la competencia; se entrenó en Kaggle con una Tesla T4 (~70 minutos).

**Aplicación:**
```bash
streamlit run app/Inicio.py
```

## Metodología resumida

- **Split:** 35 / 35 / 30 por `study_id` (690 / 691 / 593 estudios), semilla 42. Al separar por estudio, ningún paciente aparece en más de un conjunto. El conjunto de prueba amplio deja 929 casos Severe para evaluar la clase minoritaria.
- **Modelos sobre metadatos:** condición, nivel, tipo de serie, número de instancia y coordenadas (x, y).
- **CNN:** ResNet50 preentrenada en ImageNet. Usa un recorte de 96×96 alrededor de la coordenada anotada, más la condición y el nivel. Se entrenó primero con el backbone congelado y luego con fine-tuning.
- **Métricas:** por el desbalance de clases se priorizan F1 macro, balanced accuracy, recall de Severe y kappa cuadrático, en lugar de la exactitud.

## Resultados en test

| Modelo | Exactitud | F1 macro | Balanced acc. | Recall Severe | Kappa cuadrático |
|---|---|---|---|---|---|
| Clase mayoritaria | 0.770 | 0.290 | 0.333 | 0.000 | 0.000 |
| Árbol de Decisión | 0.505 | 0.393 | 0.512 | **0.657** | 0.197 |
| Gradient Boosting | 0.540 | 0.415 | 0.513 | 0.539 | 0.242 |
| Random Forest | 0.655 | 0.430 | 0.444 | 0.222 | 0.247 |
| **ResNet50 (imagen)** | 0.768 | **0.667** | **0.714** | 0.582 | **0.652** |

La clase mayoritaria tiene la mayor exactitud solo porque el 77% de las muestras de test son Normal/Mild. La ResNet50 es el mejor modelo en F1 macro, balanced accuracy y kappa cuadrático, y es el único que mantiene la exactitud cerca de la línea base sin dejar de detectar las clases minoritarias. El Árbol de Decisión tiene el mayor recall de Severe, pero a costa de muchas falsas alarmas. Los detalles completos están en `data/processed/resultados.json` y en `figures/`.
