# Notebooks

| Notebook | Qué hace |
|---|---|
| `01_extraccion_sivigila_dengue` | Limpieza de SIVIGILA y serie mensual por municipio (versión original). |
| `02_clima_canal_endemico_ibague` | Clima y canal endémico de Ibagué (versión original). |
| `03_modelos_ibague` | Modelos locales de Ibagué (versión original). |
| `01_extraccion_sivigila_dengue_1` | Versión más reciente del 01 (nov-2026); genera `serie_mensual_municipios.csv` y `casos_limpios.parquet`, que usa el 05. |
| `02_clima_canal_endemico_ibague_3` | 02 parametrizado por municipio (`COD_MUN`, `MUNICIPIO`, `CLIMA_DIR`) con `METODO_CANAL = 'bortman_sin_epidemicos'`. Lo usa el 06. |
| `03_modelos_ibague_1` | 03 con el MLP (sección 6b) que reporta el informe de avance. Lo usa el 06. |
| `05_dl_profundo_dengue` | Comparación global en 39 municipios, ventana creciente 2016–2025: persistencia, binomial negativa local, LightGBM global, GRU, LSTM de 2 capas y TCN, con pruebas pareadas (Holm). Corrida del 20-nov-2026 con ONI. |
| `06_villavicencio_monteria` | Corre copias del 02_3 y el 03_1 en Villavicencio y Montería, compara con el 05 y recalcula las métricas del 05 desde sus predicciones. |
| `ejecutados/` | Copias que genera el 06: 02 de Villavicencio y Montería, 03 de Villavicencio. No hay 03 de Montería: su test queda vacío. |

Todos leen los datos desde `MyDrive/Datos/` en Colab (acceso directo a la carpeta compartida del equipo). Los microdatos de SIVIGILA no están en el repositorio.
