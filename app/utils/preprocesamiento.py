"""
Preprocesamiento de una entrada nueva, igual al que se usó para entrenar.
El usuario solo da un DICOM, la condición, el nivel y la coordenada; todo lo
demás (one-hot, normalización, recorte, tensor) se hace aquí sin que lo note.

- Metadatos: reproduce make_features() de Proyecto2_Modelos_Baseline.ipynb.
- Imagen: reproduce leer_dicom_como_array(), recortar() y transform_val de
  Modelo_CNN_ResNet50.ipynb.

Los percentiles de x, y para x_norm / y_norm se calcularon una sola vez con los
mismos datos del notebook y están en models/normalizacion.json. Para volver a
generarlos (desde la raíz del proyecto):
    python app/utils/preprocesamiento.py
"""

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as T

RAIZ = Path(__file__).resolve().parents[2]
RUTA_NORMALIZACION = RAIZ / "models" / "normalizacion.json"
RUTA_RESULTADOS_META = RAIZ / "data" / "processed" / "resultados_metadatos.json"

CLASES = ["Normal/Mild", "Moderate", "Severe"]
SEV_MAP = {c: i for i, c in enumerate(CLASES)}
CROP_SIZE = 96
SERIES = ["Axial T2", "Sagittal T1", "Sagittal T2/STIR"]
# En el entrenamiento cada grupo de condiciones se anotó (casi) siempre en una sola serie
SERIE_POR_GRUPO = {"canal": "Sagittal T2/STIR", "foraminal": "Sagittal T1", "subarticular": "Axial T2"}

# ── Codificación de la CNN (igual que en el notebook) ──
# COND_ENCODE salía de sorted(df_clean["condition"].unique()), o sea orden alfabético.
COND_ENCODE = {c: i for i, c in enumerate(sorted([
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]))}
LEVEL_ENCODE = {"L1/L2": 0, "L2/L3": 1, "L3/L4": 2, "L4/L5": 3, "L5/S1": 4}

# transform_val del notebook (sin aumentación)
TRANSFORM_CNN = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ════════════════════════════ Metadatos ════════════════════════════

def _cargar_feature_cols() -> list[str]:
    with open(RUTA_RESULTADOS_META, encoding="utf-8") as f:
        return json.load(f)["meta"]["features"]


FEATURE_COLS = _cargar_feature_cols()


def _grupo(condicion: str) -> str:
    if "Spinal Canal" in condicion:
        return "canal"
    if "Foraminal" in condicion:
        return "foraminal"
    return "subarticular"


def serie_esperada(condicion: str) -> str:
    """Serie en la que se anotó esta condición en los datos de entrenamiento."""
    return SERIE_POR_GRUPO[_grupo(condicion)]


def datos_entrenamiento(raiz: Path = RAIZ) -> pd.DataFrame:
    """df_clean del notebook: coordenadas + serie + severidad, sin filas sin etiqueta."""
    train = pd.read_csv(raiz / "data" / "train.csv")
    coords = pd.read_csv(raiz / "data" / "train_label_coordinates.csv")
    series = pd.read_csv(raiz / "data" / "train_series_descriptions.csv")

    df = coords.merge(series, on=["study_id", "series_id"], how="left")
    largo = train.melt(id_vars="study_id", var_name="columna", value_name="severidad")
    df["columna"] = (
        df["condition"].str.lower().str.replace(" ", "_")
        + "_" + df["level"].str.lower().str.replace("/", "_")
    )
    df = df.merge(largo, on=["study_id", "columna"], how="left").drop(columns="columna")
    df["severidad_num"] = df["severidad"].map(SEV_MAP)
    return df.dropna(subset=["severidad"]).reset_index(drop=True)


def calcular_normalizacion(df: pd.DataFrame) -> dict:
    """Percentiles 1 y 99 de x, y. En el notebook se calcularon sobre df_clean completo."""
    x1, x99 = np.percentile(df["x"], [1, 99])
    y1, y99 = np.percentile(df["y"], [1, 99])
    return {"x1": float(x1), "x99": float(x99), "y1": float(y1), "y99": float(y99)}


def cargar_normalizacion() -> dict:
    with open(RUTA_NORMALIZACION, encoding="utf-8") as f:
        return json.load(f)


def features_metadatos(df: pd.DataFrame, norm: dict | None = None) -> np.ndarray:
    """
    Convierte filas con condition, level, series_description, instance_number, x, y
    en la matriz que esperan los modelos de metadatos (columnas en el orden de FEATURE_COLS).
    """
    norm = norm or cargar_normalizacion()
    d = pd.DataFrame(index=df.index)

    for col in FEATURE_COLS:
        for prefijo, origen in [("condition_", "condition"), ("level_", "level"),
                                ("series_description_", "series_description")]:
            if col.startswith(prefijo):
                d[col] = (df[origen] == col[len(prefijo):]).astype(float)
        if col.startswith("grupo_"):
            d[col] = (df["condition"].map(_grupo) == col[len("grupo_"):]).astype(float)

    d["instance_number"] = df["instance_number"].astype(float)
    d["x"] = df["x"].astype(float)
    d["y"] = df["y"].astype(float)
    d["x_norm"] = (d["x"].clip(norm["x1"], norm["x99"]) - norm["x1"]) / (norm["x99"] - norm["x1"] + 1e-9)
    d["y_norm"] = (d["y"].clip(norm["y1"], norm["y99"]) - norm["y1"]) / (norm["y99"] - norm["y1"] + 1e-9)

    # Los modelos se entrenaron con .values, sin nombres de columna
    return d[FEATURE_COLS].to_numpy(dtype=float)


def entrada_metadatos(condicion: str, nivel: str, serie: str,
                      instance_number: int, x: float, y: float) -> np.ndarray:
    """Matriz (1, 21) para una sola entrada del usuario."""
    fila = pd.DataFrame([{
        "condition": condicion, "level": nivel, "series_description": serie,
        "instance_number": instance_number, "x": x, "y": y,
    }])
    return features_metadatos(fila)


# ════════════════════════════ Imagen ════════════════════════════

def leer_dicom(fuente) -> tuple[np.ndarray, pydicom.Dataset]:
    """
    Lee un DICOM (ruta, bytes o archivo subido) y devuelve la imagen en [0, 1]
    normalizada con los percentiles 1-99, igual que leer_dicom_como_array().
    """
    if isinstance(fuente, (bytes, bytearray)):
        fuente = BytesIO(fuente)
    dcm = pydicom.dcmread(fuente, force=True)
    img = dcm.pixel_array.astype(np.float32)
    if img.ndim == 3:  # DICOM multi-frame: se toma el corte central
        img = img[img.shape[0] // 2]
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-7)
    return img, dcm


def recortar(img: np.ndarray, cx: float, cy: float, size: int = CROP_SIZE) -> np.ndarray:
    """Ventana cuadrada centrada en (cx, cy); lo que se sale del borde queda en ceros."""
    H, W = img.shape
    half = size // 2
    x1, x2 = int(cx) - half, int(cx) + half
    y1, y2 = int(cy) - half, int(cy) + half

    sx1, sx2 = max(0, x1), min(W, x2)
    sy1, sy2 = max(0, y1), min(H, y2)
    dx1 = sx1 - x1
    dy1 = sy1 - y1

    crop = np.zeros((size, size), dtype=np.float32)
    if sx2 > sx1 and sy2 > sy1:
        crop[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = img[sy1:sy2, sx1:sx2]
    return crop


def tensor_imagen(crop: np.ndarray) -> torch.Tensor:
    """Recorte → 3 canales uint8 → 224×224 → normalizado a [-1, 1]. Devuelve (1, 3, 224, 224)."""
    crop_uint8 = (crop * 255).astype(np.uint8)
    crop_3ch = np.stack([crop_uint8] * 3, axis=-1)
    return TRANSFORM_CNN(crop_3ch).unsqueeze(0)


def tensor_meta_cnn(condicion: str, nivel: str) -> torch.Tensor:
    """One-hot de condición (5) + nivel (5) para la CNN. Devuelve (1, 10)."""
    cond_oh = torch.zeros(5)
    level_oh = torch.zeros(5)
    cond_oh[COND_ENCODE[condicion]] = 1.0
    level_oh[LEVEL_ENCODE[nivel]] = 1.0
    return torch.cat([cond_oh, level_oh]).unsqueeze(0)


def inferir_serie(dcm: pydicom.Dataset) -> str:
    """
    Sugiere el tipo de serie (como en train_series_descriptions.csv) a partir de
    la orientación del corte y la descripción del DICOM. El usuario puede cambiarlo.
    """
    desc = str(dcm.get("SeriesDescription", "")).upper()
    for serie in SERIES:
        if serie.upper() in desc:
            return serie

    orient = dcm.get("ImageOrientationPatient")
    if orient is not None and len(orient) == 6:
        normal = np.abs(np.cross(np.array(orient[:3], float), np.array(orient[3:], float)))
        if normal.argmax() == 2:  # normal a lo largo del eje cabeza-pies → corte axial
            return "Axial T2"
    if "T1" in desc:
        return "Sagittal T1"
    return "Sagittal T2/STIR"


# ════════════════════════════ Script ════════════════════════════

if __name__ == "__main__":
    norm = calcular_normalizacion(datos_entrenamiento())
    RUTA_NORMALIZACION.write_text(json.dumps(norm, indent=2), encoding="utf-8")
    print(f"Guardado {RUTA_NORMALIZACION}: {norm}")
