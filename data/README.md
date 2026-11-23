# data/

CSV con los datos históricos usados para entrenar y para alimentar la aplicación
(no se versionan por separado en un servicio externo: van directo en el repositorio,
porque son pequeños — unos 70 KB en total para un municipio — y no contienen
información personal, solo conteos mensuales agregados).

Para cada municipio hay dos archivos:

- `tabla_modelado_{COD_MUN}.csv` — una fila por mes, con los 16 predictores ya
  calculados (lluvia, temperatura, casos y su rezago de hasta 4 meses, estacionalidad,
  ONI) y la etiqueta de exceso.
- `serie_nivel2_{COD_MUN}.csv` — la serie mensual de casos, clima y el umbral del
  canal endémico, usada para las gráficas de la interfaz.

`COD_MUN` es el código DANE del municipio (`73001` = Ibagué). Estos archivos los
produce el notebook `02_clima_canal_endemico_ibague.ipynb` (o su equivalente para
otro municipio).

La aplicación lee estos archivos directamente (no se conecta en vivo a SIVIGILA ni a
Google Earth Engine), así que solo puede predecir sobre los meses que ya estén aquí.

Cali (`76001`) y Montería (`23001`) solo tienen `serie_nivel2`: los usa el notebook `09_figuras_documento` para las figuras del documento final; no están en la aplicación.
