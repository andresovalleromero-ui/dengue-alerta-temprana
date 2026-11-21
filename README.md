# Sistema de alerta temprana de dengue — Proyecto Final MAIA

## Descripción del problema y la solución

El dengue en Colombia se mueve en picos epidémicos difíciles de anticipar con los
mecanismos de vigilancia actuales. Este proyecto construye un sistema que, con **dos
meses de anticipación**, predice si un municipio va camino a un **exceso de casos**
respecto a su comportamiento histórico (su "canal endémico"), para que esa ventana de
tiempo se pueda aprovechar en la respuesta institucional (fumigación, eliminación de
criaderos, preparación hospitalaria).

El problema se plantea como una clasificación binaria mensual (exceso / no exceso),
usando como entrada clima (lluvia, temperatura), el índice climático ONI (El Niño /
La Niña) y la propia historia de casos de los 4 meses anteriores. Se compararon tres
modelos — regresión logística, XGBoost y binomial negativa — sobre datos mensuales de
Ibagué (2012–2025), evaluados siempre en un conjunto de prueba que nunca vieron
durante el entrenamiento (2023–2025). Un cuarto modelo (red neuronal recurrente) y la
réplica a otros municipios están en desarrollo.

La aplicación (`app/main.py`, Streamlit) deja ver, para cualquier mes ya disponible en
los datos históricos, qué hubiera predicho cada modelo, junto con las gráficas de
casos y clima de ese periodo.

## Enlaces

- Aplicación desplegada: _(pendiente — se agrega aquí la URL una vez esté en línea)_
- Dataset: los CSV de entrada están en [`data/`](data/) (ver su propio
  [README](data/README.md)); el origen de los datos es SIVIGILA (INS), Google Earth
  Engine (TerraClimate para temperatura, CHIRPS para precipitación) y NOAA (índice ONI).

## Instrucciones de uso (local)

### 1. Requisitos

- Python 3.11+
- Dependencias listadas en [`requirements.txt`](requirements.txt)

### 2. Instalación

```bash
pip install -r requirements.txt
```

### 3. (Opcional) Reentrenar los modelos

Los modelos de Ibagué ya vienen entrenados en `models/73001/`. Para reentrenarlos (o
entrenar un municipio nuevo una vez tenga sus datos en `data/`):

```bash
python scripts/train_and_export.py --cod-mun 73001 --ruta data --salida models
```

**Entrada esperada:** `data/tabla_modelado_{cod_mun}.csv` y
`data/serie_nivel2_{cod_mun}.csv` (ver [`data/README.md`](data/README.md)).
**Salida:** `models/{cod_mun}/*.joblib` (un archivo por modelo) y
`models/{cod_mun}/metadata.json` con los predictores usados y las métricas en prueba.

### 4. Correr la aplicación

```bash
streamlit run app/main.py
```

Abre `http://localhost:8501`. Ejemplo de uso: eliges un municipio (ej. Ibagué) y un
mes dentro del rango disponible (ej. octubre de 2024); la aplicación muestra las
gráficas de casos/clima de los meses anteriores y, para cada modelo, la probabilidad
de exceso (ej. "Binomial negativa: 92%, ⚠ Exceso probable") junto con el umbral de
decisión usado.

## Reproducir el despliegue (Docker)

```bash
docker build -t dengue-alerta-temprana .
docker run -p 8501:8501 dengue-alerta-temprana
```

La imagen incluye el código, los modelos ya entrenados y los datos históricos
necesarios — no depende de nada externo en tiempo de ejecución.

## Parametrización

El comportamiento de la aplicación se ajusta mediante [`config.yaml`](config.yaml)
(municipio por defecto, rutas de datos y de modelos) o, al desplegar, mediante
variables de entorno del mismo nombre en mayúsculas, sin tocar el código, por ejemplo:

```bash
docker run -p 8501:8501 -e MUNICIPIO_DEFAULT=50001 dengue-alerta-temprana
```

## Consideraciones del modelo

- **Tamaño:** los tres modelos entrenados de un municipio pesan en total menos de
  120 KB (regresión logística ~2 KB, XGBoost ~53 KB, binomial negativa ~64 KB) —
  no se requiere GPU ni hardware especializado.
- **Entrenamiento:** menos de 1 segundo por municipio en una máquina normal (la
  grilla de hiperparámetros se dejó deliberadamente pequeña, dado el tamaño de la
  muestra: unos 80 meses de entrenamiento).
- **Inferencia:** inmediata (milisegundos por predicción); los modelos se cargan una
  sola vez al iniciar la aplicación (`st.cache_resource`) y nunca se reentrenan en
  producción.
- **Credenciales:** no se requieren. La aplicación no se conecta a ningún servicio
  externo (SIVIGILA, Google Earth Engine, NOAA) en tiempo de ejecución — todos los
  datos históricos con los que se entrenó y evalúa el modelo están empaquetados en
  `data/`.

## Estructura del repositorio

```
notebooks/   — desarrollo original del pipeline (extracción, canal endémico, modelos)
src/         — pipeline de modelado reutilizable (pipeline.py) y configuración (config.py)
scripts/     — entrenamiento y exportación de modelos (train_and_export.py)
app/         — interfaz de usuario (Streamlit, main.py)
data/        — CSV de entrada por municipio (ver su README)
models/      — modelos entrenados, serializados, por municipio
config.yaml  — parámetros ajustables sin tocar el código
Dockerfile   — empaqueta todo lo anterior para desplegarlo igual en cualquier entorno
```

## Estado del proyecto

- [x] Pipeline de datos y canal endémico (Ibagué).
- [x] Comparación de 3 modelos sobre datos reales (Ibagué): regresión logística,
  XGBoost, binomial negativa.
- [x] Interfaz de usuario y Dockerfile.
- [ ] Red neuronal recurrente como cuarto modelo (jerarquía de complejidad avanzada).
- [ ] Réplica a Villavicencio y Montería.
- [ ] Despliegue en AWS con URL pública.
