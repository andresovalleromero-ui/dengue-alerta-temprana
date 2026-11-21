"""
Interfaz de usuario (Streamlit) del sistema de alerta temprana de dengue.

No se conecta en vivo a SIVIGILA ni a Google Earth Engine: toda la serie
histórica con la que se construyó y evaluó el modelo (hasta dic. 2025) está
empaquetada dentro de la propia aplicación (carpeta data/), tal como se
definió con el experto del proyecto. El usuario elige un municipio y un mes
dentro de ese rango, y ve qué hubiera predicho cada modelo para ese mes.

Correr localmente:
    streamlit run app/main.py
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.pipeline import cargar_datos, prob_exceso  # noqa: E402

CFG = load_config()
RUTA_DATOS = CFG["rutas"]["datos"]
RUTA_MODELOS = CFG["rutas"]["modelos"]
NOMBRES_MUNICIPIO = CFG["municipios"]
ARCHIVO_DE_MODELO = CFG["modelos_archivo"]

st.set_page_config(page_title=CFG["app"]["titulo"], layout="wide")


def municipios_disponibles() -> dict:
    """Municipios para los que ya hay modelos entrenados en models/.
    Se detectan automáticamente: agregar un municipio nuevo es solo
    entrenar y guardar sus modelos, no tocar este código."""
    base = Path(RUTA_MODELOS)
    disponibles = {}
    if base.exists():
        for carpeta in sorted(base.iterdir()):
            if carpeta.is_dir() and (carpeta / "metadata.json").exists():
                disponibles[carpeta.name] = NOMBRES_MUNICIPIO.get(carpeta.name, carpeta.name)
    return disponibles


@st.cache_data(show_spinner=False)
def cargar_tabla(cod_mun: str):
    return cargar_datos(RUTA_DATOS, cod_mun)


@st.cache_resource(show_spinner=False)
def cargar_modelos(cod_mun: str):
    carpeta = Path(RUTA_MODELOS) / cod_mun
    modelos = {}
    for nombre, archivo in ARCHIVO_DE_MODELO.items():
        ruta_archivo = carpeta / archivo
        if ruta_archivo.exists():
            modelos[nombre] = joblib.load(ruta_archivo)
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    return modelos, metadata


def predecir(nombre_modelo: str, modelo, fila: pd.Series, predictores: list[str]):
    """Probabilidad de exceso (0-1) de un modelo ya entrenado para una fila
    de la tabla de modelado (un mes concreto)."""
    # fila viene de una sola fila de la tabla completa (incluye columnas no
    # numéricas como 'municipio' o 'mes_objetivo'), así que al aislar solo
    # los predictores hay que forzar tipo numérico explícitamente: si no,
    # pandas conserva dtype "object" y XGBoost lo rechaza.
    if nombre_modelo == "Binomial negativa":
        res, esc = modelo
        fila_df = fila.to_frame().T  # necesita también 'umbral_obj', no solo los predictores
        fila_df[predictores + ["umbral_obj"]] = fila_df[predictores + ["umbral_obj"]].astype(float)
        prob, _ = prob_exceso(res, esc, fila_df, predictores)
        return float(prob[0])
    X = fila[predictores].to_frame().T.astype(float)
    return float(modelo.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.title(CFG["app"]["titulo"])
st.caption(
    "Proyecto final MAIA (Universidad de los Andes, grupo BIOMAC) — predicción de "
    "exceso de casos de dengue con 2 meses de anticipación respecto al canal endémico."
)

municipios = municipios_disponibles()
if not municipios:
    st.error(
        "No hay modelos entrenados todavía en la carpeta `models/`. "
        "Corre `python scripts/train_and_export.py --cod-mun 73001` primero."
    )
    st.stop()

col_izq, col_der = st.columns([1, 2], gap="large")

with col_izq:
    st.subheader("1. Elige municipio y mes")
    cod_mun = st.selectbox(
        "Municipio",
        options=list(municipios.keys()),
        format_func=lambda c: municipios[c],
        help="Lista de municipios para los que ya hay un modelo entrenado.",
    )

    tabla, nivel2 = cargar_tabla(cod_mun)
    modelos, metadata = cargar_modelos(cod_mun)

    fechas_disponibles = sorted(tabla["mes_objetivo"].dt.date.unique())
    fecha = st.select_slider(
        "Mes a predecir",
        options=fechas_disponibles,
        value=fechas_disponibles[-1],
        help=(
            "La aplicación no se conecta en vivo a SIVIGILA ni a fuentes de clima: "
            "solo puede predecir dentro del rango de datos históricos ya cargados "
            "(hasta diciembre de 2025)."
        ),
    )

    st.caption(
        f"Datos disponibles de {fechas_disponibles[0]:%b %Y} a "
        f"{fechas_disponibles[-1]:%b %Y} ({len(fechas_disponibles)} meses)."
    )

fila = tabla.loc[tabla["mes_objetivo"].dt.date == fecha].iloc[0]
predictores = metadata["predictores"]

with col_der:
    st.subheader(f"2. Casos y clima — {municipios[cod_mun]}")
    serie = nivel2.reset_index()
    serie_reciente = serie[serie["mes"].dt.date <= fecha].tail(24).set_index("mes")

    g1, g2 = st.columns(2)
    with g1:
        st.line_chart(serie_reciente[["casos", "umbral"]])
        st.caption("Casos mensuales vs. umbral del canal endémico (últimos 24 meses)")
    with g2:
        st.line_chart(serie_reciente[["lluvia_mm", "temp_c"]])
        st.caption("Lluvia (mm) y temperatura (°C) mensual")

st.divider()
st.subheader(f"3. Predicción de exceso para {fecha:%B %Y}")

cols_modelos = st.columns(len(modelos))
for (nombre, modelo), col in zip(modelos.items(), cols_modelos):
    prob = predecir(nombre, modelo, fila, predictores)
    umbral_decision = metadata["info"][nombre]["umbral"]
    alerta = prob >= umbral_decision
    with col:
        st.metric(nombre, f"{prob:.0%}", help=f"Umbral de decisión: {umbral_decision:.0%}")
        if alerta:
            st.error("⚠ Exceso probable")
        else:
            st.success("Sin exceso esperado")

exceso_real = int(fila["exceso"])
if fecha <= fechas_disponibles[-3]:  # hay dato real de "exceso" ya conocido para comparar
    st.caption(
        f"Dato real para este mes: {'hubo exceso' if exceso_real else 'no hubo exceso'} "
        "(incluido aquí solo como referencia histórica, no como parte de la predicción)."
    )

with st.expander("¿Qué modelo es cada uno y con qué datos se entrenó?"):
    st.markdown(
        f"""
Los tres modelos se entrenaron con 16 variables por mes (lluvia, temperatura y casos
de los 4 meses anteriores, estacionalidad, el índice climático ONI y el estado del
mes previo), sobre datos mensuales de **{municipios[cod_mun]}** desde 2012 hasta 2025,
evaluados siempre en un conjunto de prueba separado (2023–2025) que el modelo nunca vio
durante el entrenamiento.

- **Regresión logística** — modelo lineal regularizado (scikit-learn); la opción más
  simple e interpretable, usada como referencia.
- **XGBoost** — árboles de decisión potenciados (gradient boosting); puede capturar
  relaciones no lineales entre las variables.
- **Binomial negativa** — en vez de clasificar directamente, predice el número de
  casos esperado y calcula la probabilidad de que supere el umbral del canal endémico
  (statsmodels); aprovecha que los casos de dengue son datos de conteo.

Las métricas completas de cada modelo sobre el conjunto de prueba están en
`models/{cod_mun}/metadata.json`, y el código de entrenamiento en `src/pipeline.py`.
Próximamente se incorporará una red neuronal recurrente (RNN) como un cuarto modelo,
siguiendo la jerarquía de complejidad recomendada por el experto del proyecto.
        """
    )
