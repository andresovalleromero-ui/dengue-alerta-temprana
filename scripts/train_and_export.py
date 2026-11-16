"""
Entrena los modelos de un municipio y los deja serializados en models/,
listos para que la app los cargue sin reentrenar.

Uso:
    python scripts/train_and_export.py --cod-mun 73001 --ruta data

Requiere que data/ tenga tabla_modelado_{cod_mun}.csv y serie_nivel2_{cod_mun}.csv
(no se versionan en git, ver data/README.md).
"""
import argparse
import json
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline import entrenar_todos  # noqa: E402

NOMBRE_ARCHIVO = {
    'Logística': 'logistica',
    'XGBoost': 'xgboost',
    'Binomial negativa': 'binomial_negativa',
}


def main(cod_mun: str, ruta: str, salida: str):
    print(f'Entrenando modelos para municipio {cod_mun} con datos en {ruta}...')
    r = entrenar_todos(ruta, cod_mun)

    os.makedirs(os.path.join(salida, cod_mun), exist_ok=True)

    for nombre, modelo in r['modelos'].items():
        archivo = NOMBRE_ARCHIVO[nombre]
        joblib.dump(modelo, os.path.join(salida, cod_mun, f'{archivo}.joblib'))
        print(f'  guardado: {archivo}.joblib')

    metadata = {
        'cod_mun': cod_mun,
        'predictores': r['predictores'],
        'ultimo': r['ultimo'],
        'info': r['info'],
        'resultados_test': {m: res['test'] for m, res in r['resultados'].items()},
    }
    with open(os.path.join(salida, cod_mun, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f'  guardado: metadata.json')
    print('Listo.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cod-mun', default='73001', help='Codigo DANE del municipio (default: Ibague)')
    ap.add_argument('--ruta', default='data', help='Carpeta con los CSV de entrada')
    ap.add_argument('--salida', default='models', help='Carpeta donde se guardan los modelos entrenados')
    args = ap.parse_args()
    main(args.cod_mun, args.ruta, args.salida)
