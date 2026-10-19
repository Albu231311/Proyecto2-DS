"""
Carga de los modelos entrenados y predicción para una sola entrada.

- Árbol de Decisión, Random Forest y Gradient Boosting: .joblib sobre metadatos.
- ResNet50: models/resnet50_best.pt (imagen + condición + nivel), en CPU.
- Clase mayoritaria: siempre predice Normal/Mild (DummyClassifier "most_frequent").

Si falta algún archivo (por ejemplo el Random Forest, que se descarga de Drive),
ese modelo simplemente no aparece y los demás siguen funcionando.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as tv_models

RAIZ = Path(__file__).resolve().parents[2]
MODELS = RAIZ / "models"

# Clave (como en resultados.json) → archivo. El orden es el que se muestra en la app.
ARCHIVOS = {
    "Clase_Mayoritaria": None,
    "Arbol_Decision":    "arbol_decision_metadatos.joblib",
    "Random_Forest":     "random_forest_metadatos.joblib",
    "Gradient_Boosting": "gradient_boosting_metadatos.joblib",
    "ResNet50_imagen":   "resnet50_best.pt",
}
MODELOS_IMAGEN = {"ResNet50_imagen"}


class RSNAResNet(nn.Module):
    """Misma arquitectura que en Modelo_CNN_ResNet50.ipynb: ResNet50 → 2048 + 10 → 512 → 128 → 3."""

    def __init__(self, n_classes=3, n_meta=10, dropout=0.4):
        super().__init__()
        # Sin pesos de ImageNet: el state_dict guardado los reemplaza todos
        backbone = tv_models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 + n_meta, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, n_classes),
        )

    def forward(self, img, meta):
        feats = self.backbone(img).squeeze(-1).squeeze(-1)
        x = torch.cat([feats, meta], dim=1)
        return self.head(x)


class ClaseMayoritaria:
    """Equivale al DummyClassifier(strategy="most_frequent") del notebook: siempre Normal/Mild."""

    def predict_proba(self, X):
        proba = np.zeros((len(X), 3))
        proba[:, 0] = 1.0
        return proba


def ruta_modelo(clave: str) -> Path | None:
    archivo = ARCHIVOS[clave]
    return MODELS / archivo if archivo else None


def modelos_disponibles() -> tuple[list[str], list[str]]:
    """(claves que se pueden usar, claves cuyo archivo no está en models/)."""
    disponibles, faltantes = [], []
    for clave in ARCHIVOS:
        ruta = ruta_modelo(clave)
        (disponibles if ruta is None or ruta.exists() else faltantes).append(clave)
    return disponibles, faltantes


@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo(clave: str):
    ruta = ruta_modelo(clave)
    if ruta is None:
        return ClaseMayoritaria()
    if clave in MODELOS_IMAGEN:
        modelo = RSNAResNet()
        modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
        modelo.eval()
        # Una pasada en vacío para que el primer tiempo medido no incluya la inicialización
        with torch.no_grad():
            modelo(torch.zeros(1, 3, 224, 224), torch.zeros(1, 10))
        return modelo
    modelo = joblib.load(ruta)
    if hasattr(modelo, "n_jobs"):
        # Para una sola muestra, repartir 500 árboles entre hilos solo agrega espera
        modelo.n_jobs = 1
    # Gradient Boosting tarda ~1 s en su primera predicción; se hace aquí y no al medir
    modelo.predict_proba(np.zeros((1, modelo.n_features_in_)))
    return modelo


def predecir(clave: str, modelo, x_meta: np.ndarray,
             img: torch.Tensor | None = None, meta_cnn: torch.Tensor | None = None) -> tuple[np.ndarray, float]:
    """Devuelve (probabilidades de las 3 clases, milisegundos que tardó)."""
    t0 = time.perf_counter()
    if clave in MODELOS_IMAGEN:
        with torch.no_grad():
            proba = torch.softmax(modelo(img, meta_cnn), dim=1)[0].numpy()
    else:
        proba = modelo.predict_proba(x_meta)[0]
    return np.asarray(proba, dtype=float), (time.perf_counter() - t0) * 1000
