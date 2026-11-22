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
durante el entrenamiento (2023–2025). La comparación con redes recurrentes en 39 municipios y la réplica a
Villavicencio y Montería están en la sección "Comparación ML vs Deep Learning en 39 municipios".

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
               Villavicencio y Montería (06); ver notebooks/README.md
src/         — pipeline de modelado reutilizable (pipeline.py) y configuración (config.py)
scripts/     — entrenamiento y exportación de modelos (train_and_export.py)
app/         — interfaz de usuario (Streamlit, main.py)
data/        — CSV de entrada por municipio (ver su README)
models/      — modelos entrenados, serializados, por municipio (los que usa la app)
models_global/ — modelo global LightGBM de 39 municipios (no lo usa la app)
resultados/  — predicciones y métricas de los notebooks 05 y 06
config.yaml  — parámetros ajustables sin tocar el código
Dockerfile   — empaqueta todo lo anterior para desplegarlo igual en cualquier entorno
```

## Comparación ML vs Deep Learning en 39 municipios (notebooks 05 y 06)

Para saber si un modelo de Deep Learning le gana al mejor modelo de ML se armó un panel de **39 municipios** con canal endémico estable (menos del 20 % de los meses con menos de 5 casos), mensual, con ONI y sin clima local. Validación con ventana creciente: un pliegue por año de 2016 a 2025, cada modelo entrenado solo con datos anteriores. 4.507 meses de prueba y 335 inicios de brote. Las métricas se recalcularon desde las predicciones guardadas (sección 8 del 06) y están en `resultados/global_05/`.

| Modelo | F1 | PR-AUC | Inicios anticipados | Falsas alarmas* | Brier |
|---|---|---|---|---|---|
| Persistencia | 0,742 | 0,625 | 0 % | 10,8 % | 0,151 |
| Binomial negativa local | 0,739 | 0,844 | 52,2 % | 16,4 % | 0,100 |
| LightGBM global | 0,756 | 0,837 | 46,6 % | 15,5 % | 0,104 |
| GRU global (1 capa) | 0,762 | 0,855 | 36,7 % | 12,4 % | 0,102 |
| LSTM global (2 capas) | 0,748 | 0,824 | 47,5 % | 17,1 % | 0,112 |
| TCN global | 0,757 | 0,842 | 40,0 % | 15,5 % | 0,105 |

\* Proporción de meses sin exceso en los que el modelo emite alerta.

Pruebas pareadas (bootstrap por bloques municipio-año, 2.000 réplicas; Wilcoxon por municipio-año con p ajustado por Holm):

| Comparación | Dif. F1 (IC 95 %) | Dif. PR-AUC (IC 95 %) | p Holm PR-AUC |
|---|---|---|---|
| LightGBM − GRU | −0,006 (−0,018 a 0,007) | −0,018 (−0,028 a −0,008) | 0,33 |
| LSTM 2 capas − GRU | −0,014 (−0,028 a 0,000) | −0,030 (−0,039 a −0,023) | < 0,001 |
| TCN − GRU | −0,005 (−0,017 a 0,007) | −0,013 (−0,020 a −0,006) | < 0,001 |
| LightGBM − Persistencia | +0,014 (−0,005 a 0,031) | +0,212 (0,190 a 0,234) | < 0,001 |
| GRU − Persistencia | +0,020 (0,003 a 0,035) | +0,230 (0,208 a 0,253) | < 0,001 |

Lectura:

1. **Más profundidad no ayuda:** la LSTM de 2 capas y la TCN quedan por debajo de la GRU de 1 capa.
2. **GRU y LightGBM empatan en F1.** La GRU tiene 0,018 más de PR-AUC y gana en 27 de 39 municipios, pero la diferencia por municipio-año no es significativa (p Holm 0,33).
3. **Frente a la persistencia** el F1 casi no cambia; la ganancia está en la PR-AUC (+0,21 a +0,23) y en anticipar inicios de brote (37–52 % contra 0 %), a cambio de más falsas alarmas.
4. **La binomial negativa local da mejores probabilidades** (Brier mejor que LightGBM en ~60 % de los municipio-años, p Holm 0,011).

Las conclusiones se mantienen sin 2020–2022 y sin 2025.

**Modelo global entrenado (`models_global/lightgbm/`).** LightGBM reentrenado con todos los datos (39 municipios, 2012–2025, 6.045 filas) y el código del 05. Se eligió sobre la GRU porque empatan en F1, la ventaja de la GRU no es significativa y LightGBM entrena en segundos y se explica por variable. `num_leaves` = 7 (moda de los pliegues), 221 árboles y umbral de alerta 0,425 (medianas); detalle en `metadata.json`. No tiene test propio: su desempeño esperado es el de la validación (F1 0,756, PR-AUC 0,837). La variable que más pesa es la de casos del mes origen relativos al umbral (`rel_0`). Todavía no está conectado a la app.

```python
import json, lightgbm as lgb
meta = json.load(open('models_global/lightgbm/metadata.json'))
modelo = lgb.Booster(model_file='models_global/lightgbm/lightgbm_global_final.txt')
# X: tabla construida como en la sección 7 del 05 (origen = mes objetivo − 2), mun_id según meta['mun_id']
alerta = modelo.predict(X[meta['variables']]) >= meta['umbral_decision']
```

**Villavicencio (50001)**, notebook 03 con clima local, test 2023–2025 (36 meses, 5 inicios): XGBoost y binomial negativa F1 0,92 y 5 de 5 inicios; persistencia 0,82; logística 0,65; MLP 0,75. El LightGBM global sin clima local da lo mismo en esos meses (F1 0,92, 5 de 5): **el clima local no agrega** sobre la dinámica de los casos. Tablas en `resultados/locales/`; los modelos que usa la app están en `models/50001/` (mismas métricas de test).

**Montería (23001): el canal endémico no funciona.** Los casos pasan de 388 (2012) a 4.880 (2025). Con la regla "año epidémico = más de 2 × la mediana histórica" casi todos los años cuentan como epidémicos; desde 2022 no quedan 4 años de referencia, no hay umbral y no hay etiqueta (test vacío). En menor medida pasa en Ocaña, Cali, Tuluá y Aguachica.

Limitaciones: el canal se calibró en Ibagué y falla con tendencias crecientes; el criterio de municipios estables usa toda la serie 2007–2025; el backtest usa datos consolidados (sin retraso de notificación); el umbral por F1 premia la inercia de los brotes.

## Estado del proyecto

- [x] Pipeline de datos y canal endémico (Ibagué).
- [x] Comparación de 3 modelos sobre datos reales (Ibagué): regresión logística,
  XGBoost, binomial negativa.
- [x] Interfaz de usuario y Dockerfile.
- [x] Red neuronal recurrente como cuarto modelo (jerarquía de complejidad avanzada):
  GRU, LSTM de 2 capas y TCN contra LightGBM en 39 municipios (notebook 05). No están en la app.
- [x] Réplica a Villavicencio y Montería (notebook 06). Villavicencio ya está en la app
  (`data/*_50001.csv` y `models/50001/`, entrenado con `scripts/train_and_export.py`). Montería
  no: el canal endémico no da etiqueta desde 2022 y no queda conjunto de prueba.
- [ ] Despliegue en AWS con URL pública.
