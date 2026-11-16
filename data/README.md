# data/

Esta carpeta no se versiona en git (ver `.gitignore`): los CSV de entrada se quedan locales.

Para cada municipio que quieras entrenar, pon aquí:

- `tabla_modelado_{COD_MUN}.csv`
- `serie_nivel2_{COD_MUN}.csv`

Estos archivos los produce el notebook `02_clima_canal_endemico_ibague.ipynb` (o su
equivalente para otro municipio). `COD_MUN` es el código DANE (por ejemplo `73001` para
Ibagué).

En este entregable ya vienen puestos los de Ibagué, para que puedas probar el script de
entrenamiento de inmediato.
