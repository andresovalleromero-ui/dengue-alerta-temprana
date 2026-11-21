"""
Carga la configuración de la app desde config.yaml, permitiendo que cualquier
valor se sobreescriba con una variable de entorno (mismo nombre, en
mayúsculas). Así se puede ajustar el comportamiento del sistema (municipio
por defecto, rutas de datos/modelos) sin tocar el código — ver config.yaml.
"""
import os
from pathlib import Path

import yaml

_RUTA_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(ruta_config: str = None) -> dict:
    ruta = Path(ruta_config) if ruta_config else _RUTA_CONFIG
    with open(ruta, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Overrides simples por variable de entorno, para los parámetros más
    # comunes al desplegar (ej. MUNICIPIO_DEFAULT=50001, RUTA_DATOS=/mnt/data)
    cfg["app"]["municipio_default"] = os.environ.get(
        "MUNICIPIO_DEFAULT", cfg["app"]["municipio_default"]
    )
    cfg["rutas"]["datos"] = os.environ.get("RUTA_DATOS", cfg["rutas"]["datos"])
    cfg["rutas"]["modelos"] = os.environ.get("RUTA_MODELOS", cfg["rutas"]["modelos"])
    return cfg
