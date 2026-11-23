# Sistema de alerta temprana de dengue — Proyecto Final MAIA

## Descripción del problema y la solución

El dengue en Colombia se mueve en picos epidémicos difíciles de anticipar con los
mecanismos de vigilancia actuales. Este proyecto construye un sistema que, con **dos
meses de anticipación**, predice si un municipio va camino a un **exceso de casos**
respecto a su comportamiento histórico (su "canal endémico"), para que esa ventana de
tiempo se pueda aprovechar en la respuesta institucional (fumigación, eliminación de
criaderos, preparación hospitalaria).

El problema se plantea como una clasificación binaria mensual (exceso / no exceso), en dos niveles:

- **Por municipio:** Ibagué (2012–2025) como caso base, replicado en Villavicencio y Cali. Entradas: clima local (lluvia, temperatura), el índice ONI (El Niño / La Niña) y los casos de los 4 meses anteriores. Se compararon persistencia, regresión logística, XGBoost, binomial negativa y una red neuronal simple (MLP), evaluados en un periodo de prueba que nunca vieron al entrenar (2023–2025; en Cali, 2023–2024). Montería no se pudo evaluar: su canal endémico no da etiqueta desde 2022. Los resultados de Cali son provisionales porque el clima extraído para Cali no es representativo.
- **Panel de 39 municipios:** un modelo global de Machine Learning (LightGBM) comparado con redes neuronales (GRU, LSTM y TCN), con ONI y casos pero sin clima local, validado año por año de 2016 a 2025. Detalle en la sección "Comparación ML vs Deep Learning en 39 municipios".

La aplicación (`app/main.py`, Streamlit) muestra Ibagué y Villavicencio: para cualquier mes ya disponible en los datos históricos, qué hubiera predicho cada modelo local (regresión logística, XGBoost y binomial negativa), junto con las gráficas de casos y clima de ese periodo.


## Enlaces

- Aplicación desplegada: https://dengue-alerta-temprana-production.up.railway.app
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

Los modelos de Ibagué y Villavicencio ya vienen entrenados en `models/73001/` y `models/50001/`. Para reentrenarlos (o
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
notebooks/   — pipeline (01–03), comparación ML vs DL en 39 municipios (05) y réplica a
               Villavicencio y Montería (06), modelo final LightGBM (07) y Cali (08); ver notebooks/README.md
src/         — pipeline de modelado reutilizable (pipeline.py) y configuración (config.py)
scripts/     — entrenamiento y exportación de modelos (train_and_export.py)
app/         — interfaz de usuario (Streamlit, main.py)
data/        — CSV de entrada por municipio (ver su README)
models/      — modelos entrenados, serializados, por municipio (los que usa la app)
models_global/ — modelo global LightGBM de 39 municipios (no lo usa la app)
resultados/  — predicciones y métricas de los notebooks 05, 06 y 08
config.yaml  — parámetros ajustables sin tocar el código
Dockerfile   — empaqueta todo lo anterior para desplegarlo igual en cualquier entorno
```

## Comparación ML vs Deep Learning en 39 municipios (notebook 05)

Panel de 39 municipios con canal endémico estable, mensual, con ONI y sin clima local. Validación año por año (2016–2025), cada modelo entrenado solo con datos anteriores: 4.507 meses de prueba y 335 inicios de brote. Resultados en `resultados/global_05/`.

| Modelo | F1 | PR-AUC | Inicios anticipados | Falsas alarmas* |
|---|---|---|---|---|
| Persistencia (referencia) | 0,742 | 0,625 | 0 % | 10,8 % |
| LightGBM (ML) | 0,756 | 0,837 | 46,6 % | 15,5 % |
| GRU (1 capa) | 0,762 | 0,855 | 36,7 % | 12,4 % |
| LSTM (2 capas) | 0,748 | 0,824 | 47,5 % | 17,1 % |
| TCN | 0,757 | 0,842 | 40,0 % | 15,5 % |

\* Meses sin exceso en los que el modelo emite alerta.

- La mejor red (GRU) queda apenas por encima de LightGBM (F1 +0,006; PR-AUC +0,018). La diferencia no es significativa en la prueba por municipio-año (p ajustado 0,33). Más capas (LSTM, TCN) no mejoran.
- Frente a la persistencia el F1 casi no cambia; la ganancia está en la PR-AUC (+0,21) y en anticipar inicios de brote (37–48 % contra 0 %), a cambio de más falsas alarmas.
- Se elige LightGBM: rinde como las redes, entrena en segundos y se puede explicar por variable.

Las conclusiones se mantienen sin 2020–2022 y sin 2025.

### Modelos por municipio (notebooks 03, 06 y 08)

F1 en la prueba 2023–2025, con clima local. El LightGBM global, sin clima local, evaluado en los mismos meses:

| Modelo | Ibagué | Villavicencio | Cali* |
|---|---|---|---|
| Persistencia | 0,89 | 0,82 | 0,93 |
| Regresión logística | 0,80 | 0,65 | 0,95 |
| XGBoost | 0,78 | 0,92 | 0,87 |
| Binomial negativa | 0,92 | 0,92 | 0,92 |
| MLP | 0,76 | 0,75 | 0,93 |
| **LightGBM global** | **0,92** | **0,92** | **0,95** |
| Inicios de brote en la prueba | 2 | 5 | 2 |

El modelo global iguala al mejor modelo de cada municipio sin usar clima local. Con 2 a 5 inicios de brote por municipio, estas pruebas no alcanzan para elegir modelo; para eso está el panel de 39 municipios. La app usa los modelos locales de Ibagué y Villavicencio (`models/`). Tablas en `resultados/locales/`.

\* Cali es provisional: prueba 2023–2024 con 21 de 24 meses en exceso (alertar siempre ya da F1 0,93), y el clima extraído de Google Earth Engine no es representativo (6.000–8.000 mm/año y 19,6 °C, frente a 1.850 mm/año y 24,2 °C según la Alcaldía).

**Limitaciones:** el desempeño varía entre municipios (F1 del modelo global de 0,49 a 0,94). El canal endémico se calibró en Ibagué y falla con tendencias crecientes: en Montería los casos pasan de 388 (2012) a 4.880 (2025), casi todos los años cuentan como epidémicos y desde 2022 no hay umbral. El criterio de municipios estables usa toda la serie 2007–2025. La validación usa datos consolidados, sin el retraso de notificación. El umbral elegido por F1 favorece la inercia de los brotes.
