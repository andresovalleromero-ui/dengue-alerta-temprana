"""
Pipeline de modelado para el sistema de alerta temprana de dengue.

Parametrizado por municipio (codigo DANE, COD_MUN) para poder reutilizarse
cuando haya datos de Villavicencio y Monteria ademas de Ibague (73001).

Espera, para cada municipio, dos archivos (producidos por los notebooks
01 y 02 del pipeline):
    tabla_modelado_{COD_MUN}.csv
    serie_nivel2_{COD_MUN}.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import xgboost as xgb
import statsmodels.api as sm

GRILLA_C = [0.01, 0.03, 0.1, 0.3, 1, 3]
GRILLA_XGB = [
    dict(max_depth=d, n_estimators=n, learning_rate=0.05, subsample=0.8,
         colsample_bytree=0.8, min_child_weight=3, reg_lambda=5)
    for d in (1, 2, 3) for n in (50, 100, 200)
]
PENAL_NB = 0.5

NO_PREDICTORES = ('municipio', 'mes_objetivo', 'pandemia', 'exceso', 'split',
                   'casos_obj', 'umbral_obj')


def cargar_datos(ruta: str, cod_mun: str):
    """Carga tabla_modelado y serie_nivel2 para un municipio dado."""
    tabla = pd.read_csv(f'{ruta}/tabla_modelado_{cod_mun}.csv', parse_dates=['mes_objetivo'])
    nivel2 = pd.read_csv(f'{ruta}/serie_nivel2_{cod_mun}.csv', parse_dates=['mes']).set_index('mes')
    tabla['casos_obj'] = tabla['mes_objetivo'].map(nivel2['casos'])
    tabla['umbral_obj'] = tabla['mes_objetivo'].map(nivel2['umbral'])
    return tabla, nivel2


def particionar(tabla: pd.DataFrame):
    train, val, test = (tabla[tabla.split == s].copy() for s in ('train', 'val', 'test'))
    trval = pd.concat([train, val])
    return train, val, test, trval


def _predictores(tabla: pd.DataFrame):
    predictores = [c for c in tabla.columns if c not in NO_PREDICTORES]
    ultimo = [c for c in predictores if c.startswith('exceso_t-')][0]
    return predictores, ultimo


def evaluar(g, pred, ultimo, prob=None):
    y = g['exceso']
    pred = pd.Series(np.asarray(pred).astype(int), index=g.index)
    inicio = (y == 1) & (g[ultimo] == 0)
    return {
        'recall': recall_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'pr_auc': average_precision_score(y, prob) if prob is not None else np.nan,
        'recall_inicios': pred[inicio].mean() if inicio.any() else np.nan,
        'n_inicios': int(inicio.sum()),
    }


def elegir_umbral(y, prob):
    grilla = np.arange(0.05, 0.96, 0.05)
    f1s = [f1_score(y, prob >= u, zero_division=0) for u in grilla]
    return float(grilla[int(np.argmax(f1s))])


def _logistica(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000))


def ajustar_logistica(train, val, trval, test, cols, ultimo):
    pr = {C: average_precision_score(val.exceso, _logistica(C).fit(train[cols], train.exceso)
                                      .predict_proba(val[cols])[:, 1]) for C in GRILLA_C}
    C = max(pr, key=pr.get)
    p_val = _logistica(C).fit(train[cols], train.exceso).predict_proba(val[cols])[:, 1]
    u = elegir_umbral(val.exceso, p_val)
    modelo = _logistica(C).fit(trval[cols], trval.exceso)
    p_test = modelo.predict_proba(test[cols])[:, 1]
    res = {'val': evaluar(val, p_val >= u, ultimo, p_val), 'test': evaluar(test, p_test >= u, ultimo, p_test)}
    return res, p_test, modelo, {'C': C, 'umbral': u}


def _xgboost(params, g):
    peso = (g.exceso == 0).sum() / max(g.exceso.sum(), 1)
    return xgb.XGBClassifier(**params, scale_pos_weight=peso, eval_metric='logloss', random_state=42)


def ajustar_xgb(train, val, trval, test, cols, ultimo):
    pr = [average_precision_score(val.exceso, _xgboost(p, train).fit(train[cols], train.exceso)
                                   .predict_proba(val[cols])[:, 1]) for p in GRILLA_XGB]
    P = GRILLA_XGB[int(np.argmax(pr))]
    p_val = _xgboost(P, train).fit(train[cols], train.exceso).predict_proba(val[cols])[:, 1]
    u = elegir_umbral(val.exceso, p_val)
    modelo = _xgboost(P, trval).fit(trval[cols], trval.exceso)
    p_test = modelo.predict_proba(test[cols])[:, 1]
    res = {'val': evaluar(val, p_val >= u, ultimo, p_val), 'test': evaluar(test, p_test >= u, ultimo, p_test)}
    return res, p_test, modelo, {'max_depth': P['max_depth'], 'n_estimators': P['n_estimators'], 'umbral': u}


def _X(g, cols, esc):
    return sm.add_constant(pd.DataFrame(esc.transform(g[cols]), columns=cols, index=g.index), has_constant='add')


def _ajustar_bn(g, cols):
    esc = StandardScaler().fit(g[cols])
    penal = np.r_[0, np.full(len(cols), PENAL_NB), 0]
    res = sm.NegativeBinomial(g['casos_obj'], _X(g, cols, esc)).fit_regularized(alpha=penal, disp=0, maxiter=2000)
    return res, esc


def _prob_exceso(res, esc, g, cols):
    mu = res.predict(_X(g, cols, esc))
    n = 1 / res.params['alpha']
    return np.asarray(1 - stats.nbinom.cdf(np.floor(g['umbral_obj']), n, n / (n + mu))), mu


def ajustar_binomial(train, val, trval, test, cols, ultimo):
    p_val, _ = _prob_exceso(*_ajustar_bn(train, cols), val, cols)
    u = elegir_umbral(val.exceso, p_val)
    modelo, esc = _ajustar_bn(trval, cols)
    p_test, _ = _prob_exceso(modelo, esc, test, cols)
    res = {'val': evaluar(val, p_val >= u, ultimo, p_val), 'test': evaluar(test, p_test >= u, ultimo, p_test)}
    return res, p_test, (modelo, esc), {'umbral': u, 'alpha_dispersion': round(modelo.params['alpha'], 2)}


def entrenar_todos(ruta: str, cod_mun: str):
    """Carga los datos de un municipio y entrena los 4 modelos (persistencia,
    logistica, xgboost, binomial negativa). Devuelve resultados, probabilidades
    de test y los objetos de modelo ya ajustados (listos para serializar)."""
    tabla, nivel2 = cargar_datos(ruta, cod_mun)
    train, val, test, trval = particionar(tabla)
    predictores, ultimo = _predictores(tabla)

    resultados, probs_test, modelos, info = {}, {}, {}, {}

    resultados['Persistencia'] = {s: evaluar(g, g[ultimo], ultimo) for s, g in [('val', val), ('test', test)]}

    resultados['Logística'], probs_test['Logística'], modelos['Logística'], info['Logística'] = \
        ajustar_logistica(train, val, trval, test, predictores, ultimo)

    resultados['XGBoost'], probs_test['XGBoost'], modelos['XGBoost'], info['XGBoost'] = \
        ajustar_xgb(train, val, trval, test, predictores, ultimo)

    resultados['Binomial negativa'], probs_test['Binomial negativa'], modelos['Binomial negativa'], \
        info['Binomial negativa'] = ajustar_binomial(train, val, trval, test, predictores, ultimo)

    return {
        'cod_mun': cod_mun,
        'predictores': predictores,
        'ultimo': ultimo,
        'resultados': resultados,
        'probs_test': probs_test,
        'modelos': modelos,
        'info': info,
        'test': test,
    }
