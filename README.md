# Sistema de alerta temprana de dengue — Proyecto Final MAIA

Pipeline de predicción de exceso de casos de dengue (clasificación binaria mensual,
2 meses de anticipación) para municipios colombianos, comparando persistencia,
regresión logística, XGBoost y binomial negativa (próximamente: RNN).

## Estructura

- `notebooks/` — desarrollo original del pipeline (extracción SIVIGILA, canal endémico
  y clima, comparación de modelos). Se conservan como registro, pero el código que usa
  la aplicación vive en `src/`.
- `src/pipeline.py` — el pipeline de modelado, parametrizado por código de municipio
  (DANE), listo para reusarse con cualquier municipio que tenga sus datos en `data/`.
- `scripts/train_and_export.py` — entrena los modelos de un municipio y los deja
  serializados en `models/{cod_mun}/`.
- `data/` — CSV de entrada por municipio (no versionados, ver `data/README.md`).
- `models/` — modelos ya entrenados, serializados (joblib + metadata.json con las
  métricas de test). La app los carga de aquí, sin reentrenar.
- `app/` — interfaz (pendiente de desarrollo).

## Cómo correr el entrenamiento

```bash
pip install -r requirements.txt
python scripts/train_and_export.py --cod-mun 73001 --ruta data --salida models
```

Esto deja en `models/73001/` los tres modelos entrenables (`logistica.joblib`,
`xgboost.joblib`, `binomial_negativa.joblib`) y `metadata.json` con los predictores
usados y las métricas obtenidas en el conjunto de prueba.

Para otro municipio, basta con poner sus CSV en `data/` y repetir el comando con su
código DANE.

## Estado del proyecto

- [x] Pipeline de datos y canal endémico (Ibagué).
- [x] Comparación de 4 modelos sobre datos reales (Ibagué).
- [ ] RNN como quinto modelo (jerarquía de complejidad avanzada).
- [ ] Réplica a Villavicencio y Montería.
- [ ] Interfaz de usuario (`app/`).
- [ ] Despliegue en AWS.
