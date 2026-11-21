"""
Entrena los modelos de un municipio y los deja serializados en models/,
listos para que la app los cargue sin reentrenar.

Uso:
    python scripts/train_and_export.py --cod-mun 73001 --ruta data

Requiere que data/ tenga tabla_modelado_{cod_mun}.csv y serie_nivel2_{cod_mun}.csv.
"""
import argparse
import json
import os
import sys
import time

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline import entrenar_todos  # noqa: E402

NOMBRE_ARCHIVO = {
    'Logística': 'logistica.joblib',
    'XGBoost': 'xgboost.joblib',
    'Binomial negativa': 'binomial_negativa.joblib',
}


def main(cod_mun: str, ruta: str, salida: str):
    print(f'Entrenando modelos para municipio {cod_mun} con datos en {ruta}...')
    t0 = time.time()
    r = entrenar_todos(ruta, cod_mun)
    segundos_entrenamiento = round(time.time() - t0, 1)

    carpeta_salida = os.path.join(salida, cod_mun)
    os.makedirs(carpeta_salida, exist_ok=True)

    tamanos_kb = {}
    for nombre, modelo in r['modelos'].items():
        archivo = NOMBRE_ARCHIVO[nombre]
        ruta_archivo = os.path.join(carpeta_salida, archivo)
        joblib.dump(modelo, ruta_archivo)
        tamanos_kb[nombre] = round(os.path.getsize(ruta_archivo) / 1024, 1)
        print(f'  guardado: {archivo} ({tamanos_kb[nombre]} KB)')

    metadata = {
        'cod_mun': cod_mun,
        'predictores': r['predictores'],
        'ultimo': r['ultimo'],
        'info': r['info'],
        'resultados_test': {m: res['test'] for m, res in r['resultados'].items()},
        'tamanos_kb': tamanos_kb,
        'segundos_entrenamiento': segundos_entrenamiento,
    }
    with open(os.path.join(carpeta_salida, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print('  guardado: metadata.json')
    print(f'Listo en {segundos_entrenamiento}s.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cod-mun', default='73001', help='Codigo DANE del municipio (default: Ibague)')
    ap.add_argument('--ruta', default='data', help='Carpeta con los CSV de entrada')
    ap.add_argument('--salida', default='models', help='Carpeta donde se guardan los modelos entrenados')
    args = ap.parse_args()
    main(args.cod_mun, args.ruta, args.salida)
