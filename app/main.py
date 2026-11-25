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

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO_UNIANDES = ASSETS / "logo_uniandes.png"
LOGO_MAIA = ASSETS / "logo_maia.png"
IMG_MOSQUITO = ASSETS / "mosquito.jpg"

st.set_page_config(page_title=CFG["app"]["titulo"], layout="wide")

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .app-hero {
        background: linear-gradient(90deg, #00263A 0%, #00496B 55%, #2E7D63 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.1rem;
    }
    .app-hero h1 {
        color: #FFFFFF !important;
        font-size: 2.0rem;
        font-weight: 700;
        margin: 0 0 .35rem 0;
        line-height: 1.2;
    }
    .app-hero p {
        color: #DCEFE6;
        font-size: 0.98rem;
        margin: 0;
    }
    h2, h3 {
        color: #00394F;
        border-left: 5px solid #2E7D63;
        padding-left: 0.6rem;
    }
    .credito-foto {
        font-size: 0.75rem;
        color: #6b7280;
        text-align: center;
        margin-top: -0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
# Encabezado (logos + título)
# ---------------------------------------------------------------------------
col_logo_izq, col_hero, col_logo_der = st.columns([1, 5, 1], vertical_alignment="center")

with col_logo_izq:
    if LOGO_UNIANDES.exists():
        st.image(str(LOGO_UNIANDES), use_container_width=True)

with col_hero:
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{CFG["app"]["titulo"]}</h1>
            <p>Proyecto final MAIA (Universidad de los Andes, grupo BIOMAC) — predicción de
            exceso de casos de dengue con 2 meses de anticipación respecto al canal endémico.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_logo_der:
    if LOGO_MAIA.exists():
        st.image(str(LOGO_MAIA), use_container_width=True)

# ---------------------------------------------------------------------------
# Panel plegable de instrucciones
# ---------------------------------------------------------------------------
with st.expander("📖 Cómo leer esta interfaz (haz clic para desplegar)", expanded=False):
    st.markdown(
        """
Esta aplicación **no se conecta en vivo** a SIVIGILA, al clima satelital ni a ninguna
otra fuente: todo lo que ves viene de la serie histórica (hasta diciembre de 2025) ya
empaquetada con el modelo. Así se usa:

1. **Elige un municipio y un mes** en el panel de la izquierda (sección 1). Solo
   aparecen los municipios para los que ya hay un modelo entrenado.
2. **Revisa el contexto** en la sección 2: casos mensuales frente al umbral del canal
   endémico, y el clima (lluvia y temperatura) de los últimos 24 meses hasta el mes
   elegido.
3. **Lee la predicción** en la sección 3: la probabilidad de que el mes elegido tenga
   un *exceso* de casos (por encima del canal endémico), según cada uno de los tres
   modelos. Si la probabilidad supera el umbral de decisión del modelo, se marca
   como "⚠ Exceso probable".
4. El dato real de ese mes (cuando ya se conoce) aparece solo como referencia
   histórica — la predicción no lo usa.

El botón **"Cargar datos de internet"** en la barra lateral es solo informativo por
ahora: la idea es, a futuro, poder traer datos en vivo de SIVIGILA/GEE para otros
municipios, pero eso requiere permisos adicionales que el equipo todavía no tiene.
        """
    )

# ---------------------------------------------------------------------------
# Barra lateral: fuente de datos en vivo (decorativo) + crédito de la foto
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Fuentes de datos")
    if st.button("🌐 Cargar datos de internet (SIVIGILA, GEE...)", use_container_width=True):
        st.warning(
            "Se necesitan permisos adicionales para traer datos en vivo de otros "
            "municipios (SIVIGILA, Google Earth Engine, NOAA). Por ahora la aplicación "
            "solo trabaja con la serie histórica ya cargada. Si tu municipio no aparece "
            "en la lista, contacta al equipo del proyecto para solicitar acceso."
        )

    st.divider()
    if IMG_MOSQUITO.exists():
        st.image(str(IMG_MOSQUITO), use_container_width=True)
        st.markdown('<p class="credito-foto">Foto: National Geographic</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Interfaz principal
# ---------------------------------------------------------------------------
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
    opciones = list(municipios.keys())
    default = CFG["app"]["municipio_default"]
    cod_mun = st.selectbox(
        "Municipio",
        options=opciones,
        index=opciones.index(default) if default in opciones else 0,
        format_func=lambda c: municipios[c],
        help="Lista de municipios para los que ya hay un modelo entrenado.",
    )

    if cod_mun == "76001":
        st.info(
            "Cali se incluye de forma **provisional**: XGBoost no anticipó ninguno de los "
            "2 inicios de brote del conjunto de prueba, aunque sí detecta bien los meses "
            "en exceso en general. Interpreta sus alertas con esa salvedad.",
            icon="ℹ️",
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

with st.expander("¿Qué modelo es cada uno?"):
    st.markdown(
        f"""
Los tres modelos aprenden del comportamiento histórico del dengue en **{municipios[cod_mun]}**:
cuántos casos hubo cada mes entre 2012 y 2025, cómo venía la lluvia y la temperatura, la época
del año, y si los meses recientes ya estaban en exceso. Para comprobar qué tan bien funcionan,
cada uno se puso a prueba solo con los datos más nuevos (2023 a 2025), que nunca había visto
antes — así se mide qué tan bien predice meses nuevos, no qué tan bien memorizó el pasado.

- **Regresión logística** — el más simple de los tres; sirve como punto de comparación.
- **XGBoost** — busca patrones más complejos entre las variables; es el modelo principal
  de esta aplicación.
- **Binomial negativa** — en vez de responder directamente "sí" o "no" hay exceso, primero
  estima cuántos casos habrá ese mes y de ahí calcula qué tan probable es que se pase del
  umbral del canal endémico.

Más adelante se planea sumar un cuarto modelo, una red neuronal, para ver si mejora estas
predicciones.
        """
    )

st.divider()
st.caption("Foto del mosquito: © National Geographic. Usada únicamente con fines ilustrativos y educativos.")
