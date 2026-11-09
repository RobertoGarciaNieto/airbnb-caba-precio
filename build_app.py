"""Construye todo lo que la aplicación necesita a partir del CSV que produce el pipeline.

    python build_app.py [ruta_al_snapshot.csv]

Reproduce la Entrega 3 con la misma semilla (partición por ubicación, poda P1–P99 con train,
grilla del boosting, test una sola vez) y deja en app_data/:
  - modelo.joblib          el pipeline final entrenado con todo el train
  - publicaciones.parquet  cada publicación con su predicción (fuera de pliegue en train, modelo final en test)
  - metricas.json          todos los números que muestra la aplicación
Si cambian los datos, correr esto de nuevo actualiza la aplicación.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, KFold
from sklearn.tree import DecisionTreeRegressor

from transformadores import COLUMNAS, SEMILLA, armar

warnings.filterwarnings("ignore")
np.random.seed(SEMILLA)

RAIZ = Path(__file__).parent
RUTA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "data" / "snapshot=2026-07-24.csv"
SALIDA = RAIZ / "app_data"
SALIDA.mkdir(exist_ok=True)
SNAPSHOT = RUTA.stem.split("=")[-1]
OBJETIVO = "log_price"
K_ELEGIDO = 15
# Lo que dio el notebook de la Entrega 3: si el build no lo reproduce, avisa.
ESPERADO_TEST_MAE = 32959

t0 = time.time()
def paso(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def en_pesos(y_log):
    return np.exp(y_log)


SCORE_PESOS = make_scorer(lambda yt, yp: np.mean(np.abs(en_pesos(yt) - en_pesos(yp))), greater_is_better=False)


def resumen(y_log, p_log):
    y_real, p_real = en_pesos(np.asarray(y_log)), en_pesos(np.asarray(p_log))
    rel = np.abs(p_real - y_real) / y_real
    return {"MAE": float(np.mean(np.abs(y_real - p_real))), "err_rel_mediano": float(np.median(rel)),
            "err_rel_medio": float(np.mean(rel)), "R2_log": float(r2_score(y_log, p_log)),
            "R2_pesos": float(r2_score(y_real, p_real))}


def evaluar(modelo, X, y, cv, groups=None):
    """MAE en pesos por pliegue (media y desvío) + error relativo y R², como en la Entrega 3."""
    folds = []
    for a, b in cv.split(X, y, groups):
        m = clone(modelo).fit(X.iloc[a], y.iloc[a])
        folds.append(resumen(y.iloc[b], m.predict(X.iloc[b])))
    f = pd.DataFrame(folds)
    return {"MAE": float(f["MAE"].mean()), "desvio": float(f["MAE"].std(ddof=0)),
            "err_rel_mediano": float(f["err_rel_mediano"].mean()), "R2_log": float(f["R2_log"].mean()),
            "R2_pesos": float(f["R2_pesos"].mean())}


def separacion(x, y):
    x, y = x.dropna(), y.dropna()
    return float(abs(x.mean() - y.mean()) / np.sqrt((x.std() ** 2 + y.std() ** 2) / 2))


# ---------------------------------------------------------------- datos
paso(f"Leyendo {RUTA}")
df = pd.read_csv(RUTA, low_memory=False)
df[OBJETIVO] = np.log(df["price"])
df["grupo"] = df.groupby(["latitude", "longitude"]).ngroup()
tam = df["grupo"].value_counts()

M = {"snapshot": SNAPSHOT, "datos": {
    "filas": int(len(df)), "columnas": int(df.shape[1] - 2), "barrios": int(df["neighbourhood_cleansed"].nunique()),
    "precio_mediano": float(df["price"].median()), "precio_p99": float(df["price"].quantile(0.99)),
    "precio_max": float(df["price"].max()), "precio_min": float(df["price"].min()),
    "asimetria_precio": float(df["price"].skew()), "asimetria_log": float(df[OBJETIVO].skew()),
    "ubicaciones": int(tam.size), "filas_en_ubic_compartida": int(tam[tam > 1].sum()), "grupo_max": int(tam.max()),
    "desvio_log_dentro_ubic": float(df[df["grupo"].isin(tam[tam > 1].index)].groupby("grupo")[OBJETIVO].std().mean()),
    "desvio_log_total": float(df[OBJETIVO].std()),
    "nulos_bedrooms": int(df["bedrooms"].isna().sum()),
}}

# ---------------------------------------------------------------- partición y poda (igual que la Entrega 3)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEMILLA)
idx_tr, idx_te = next(gss.split(df, groups=df["grupo"]))
tr0, te0 = df.iloc[idx_tr], df.iloc[idx_te]
lo, hi = tr0[OBJETIVO].quantile([0.01, 0.99])
tr = tr0[tr0[OBJETIVO].between(lo, hi)].reset_index(drop=True)
te = te0[te0[OBJETIVO].between(lo, hi)].reset_index(drop=True)
M["particion"] = {"train_antes": len(tr0), "test_antes": len(te0), "train": len(tr), "test": len(te),
                  "limite_inf": float(np.exp(lo)), "limite_sup": float(np.exp(hi)),
                  "ubic_compartidas_tr_te": len(set(tr["grupo"]) & set(te["grupo"]))}
X_tr, y_tr, g_tr = tr[COLUMNAS], tr[OBJETIVO], tr["grupo"]
X_te, y_te = te[COLUMNAS], te[OBJETIVO]
CV = GroupKFold(n_splits=5)
paso(f"train {len(tr):,} | test {len(te):,} | rango ${np.exp(lo):,.0f}–${np.exp(hi):,.0f}")

# ---------------------------------------------------------------- validación al azar vs por ubicación
lin = armar(LinearRegression(), "lineal", k=K_ELEGIDO)
M["validacion"] = {
    "azar": evaluar(lin, X_tr, y_tr, KFold(5, shuffle=True, random_state=SEMILLA)),
    "ubicacion": evaluar(lin, X_tr, y_tr, CV, g_tr)}
paso("validación al azar vs por ubicación")

# ---------------------------------------------------------------- comparación de familias
candidatos = {
    "Piso (mediana)": armar(DummyRegressor(strategy="median"), "lineal"),
    "Regresión lineal": lin,
    "Árbol de decisión": armar(DecisionTreeRegressor(random_state=SEMILLA), "arbol"),
    "Bosque aleatorio": armar(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                    random_state=SEMILLA), "arbol"),
    "Gradient boosting": armar(HistGradientBoostingRegressor(random_state=SEMILLA), "arbol"),
}
M["comparacion"] = []
for nombre, pipe in candidatos.items():
    M["comparacion"].append({"modelo": nombre, **evaluar(pipe, X_tr, y_tr, CV, g_tr)})
    paso(f"CV {nombre}: ${M['comparacion'][-1]['MAE']:,.0f}")

# ---------------------------------------------------------------- grilla del elegido
BOOST = HistGradientBoostingRegressor(early_stopping=False, random_state=SEMILLA)
grilla = {"modelo__learning_rate": [0.05, 0.1], "modelo__max_iter": [300, 600],
          "modelo__max_leaf_nodes": [15, 31, 63], "modelo__min_samples_leaf": [20, 50]}
busqueda = GridSearchCV(armar(BOOST, "arbol", k=K_ELEGIDO), grilla, cv=CV, scoring=SCORE_PESOS, n_jobs=-1)
busqueda.fit(X_tr, y_tr, groups=g_tr)
res = pd.DataFrame(busqueda.cv_results_).sort_values("rank_test_score")
M["grilla"] = {
    "mejor": {k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()},
    "combinaciones": len(res),
    "top": [{**{k.replace("param_modelo__", ""): (r[k].item() if hasattr(r[k], "item") else r[k])
               for k in res if k.startswith("param_modelo")},
             "MAE": float(-r["mean_test_score"]), "desvio": float(r["std_test_score"])}
            for _, r in res.head(5).iterrows()]}
BOOST_AJ = clone(BOOST).set_params(**{k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()})
paso(f"grilla: {M['grilla']['mejor']} -> ${-busqueda.best_score_:,.0f}")

# ---------------------------------------------------------------- modelo final y test (una sola vez)
r_cv = evaluar(armar(BOOST_AJ, "arbol", k=K_ELEGIDO), X_tr, y_tr, CV, g_tr)
M["comparacion"].append({"modelo": "Gradient boosting ajustado", **r_cv})
final = armar(BOOST_AJ, "arbol", k=K_ELEGIDO).fit(X_tr, y_tr)
pred_te = final.predict(X_te)
piso_te = armar(DummyRegressor(strategy="median"), "lineal").fit(X_tr, y_tr).predict(X_te)
M["test"] = {"modelo": resumen(y_te, pred_te), "piso": resumen(y_te, piso_te)}
mae_te = M["test"]["modelo"]["MAE"]
paso(f"TEST: MAE ${mae_te:,.0f} (notebook: ${ESPERADO_TEST_MAE:,})")
if abs(mae_te - ESPERADO_TEST_MAE) > 1:
    print(f"  !! El MAE de test no coincide con el notebook (diferencia ${mae_te - ESPERADO_TEST_MAE:,.0f}). "
          "Revisar versiones de librerías o datos.")

# Importancia por permutación en test (en pesos)
imp = permutation_importance(final, X_te, y_te, scoring=SCORE_PESOS, n_repeats=5, random_state=SEMILLA, n_jobs=-1)
M["importancia"] = sorted([{"columna": c, "aumento_MAE": float(m), "desvio": float(s)}
                           for c, m, s in zip(COLUMNAS, imp.importances_mean, imp.importances_std)],
                          key=lambda d: -d["aumento_MAE"])
paso("importancia por permutación")

# ---------------------------------------------------------------- predicciones para explorar
# Train: fuera de pliegue (cada publicación predicha por un modelo que no la vio). Test: modelo final.
oof = np.zeros(len(tr))
for a, b in CV.split(X_tr, y_tr, g_tr):
    oof[b] = clone(final).fit(X_tr.iloc[a], y_tr.iloc[a]).predict(X_tr.iloc[b])
paso("predicciones fuera de pliegue")

COLS_APP = ["id", "price", "neighbourhood_cleansed", "room_type", "accommodates", "bedrooms", "bathrooms",
            "minimum_nights", "amenities_count", "has_pool", "has_ac", "has_elevator", "has_balcony_patio",
            "has_parking", "has_workspace", "latitude", "longitude"]
pub = pd.concat([tr[COLS_APP].assign(pred=en_pesos(oof), conjunto="train (fuera de pliegue)"),
                 te[COLS_APP].assign(pred=en_pesos(pred_te), conjunto="test")], ignore_index=True)
pub["error_rel"] = (pub["pred"] - pub["price"]) / pub["price"]   # con signo: + sobreestima, − subestima
pub["quintil"] = pd.qcut(pub["price"], 5, labels=["1 · más baratas", "2", "3", "4", "5 · más caras"]).astype(str)
pub.to_parquet(SALIDA / "publicaciones.parquet", index=False)

# U del error en test, por quintil de precio real
te_q = pd.DataFrame({"price": en_pesos(y_te), "pred": en_pesos(pred_te)})
te_q["sesgo"] = (te_q["pred"] - te_q["price"]) / te_q["price"]
te_q["abs_rel"] = te_q["sesgo"].abs()
te_q["quintil"] = pd.qcut(te_q["price"], 5, labels=["1 · más baratas", "2", "3", "4", "5 · más caras"])
M["quintiles"] = [{"quintil": str(q), "publicaciones": int(len(g)), "precio_mediano": float(g["price"].median()),
                   "err_rel_mediano": float(g["abs_rel"].median()), "sesgo_mediano": float(g["sesgo"].median()),
                   "sesgo_p25": float(g["sesgo"].quantile(0.25)), "sesgo_p75": float(g["sesgo"].quantile(0.75)),
                   "MAE": float((g["pred"] - g["price"]).abs().mean())}
                  for q, g in te_q.groupby("quintil", observed=True)]

# Rango de la predicción para "Probá el modelo": cuantiles de real/predicho (fuera de pliegue),
# por tramo de precio PREDICHO (al probar no se conoce el real).
pub["tramo_pred"], cortes = pd.qcut(pub["pred"], 5, retbins=True, labels=False)
M["intervalo"] = {"cortes": [float(c) for c in cortes],
                  "p10": [float(g.quantile(0.10)) for _, g in (pub["price"] / pub["pred"]).groupby(pub["tramo_pred"])],
                  "p90": [float(g.quantile(0.90)) for _, g in (pub["price"] / pub["pred"]).groupby(pub["tramo_pred"])]}

# ---------------------------------------------------------------- hipótesis (Entrega 2) sobre este snapshot
p01, p99 = df[OBJETIVO].quantile([0.01, 0.99])
ent = df[df["room_type"] == "Entire home/apt"]
ent2 = ent[(ent["accommodates"] == 2) & ent[OBJETIVO].between(p01, p99)]
filas = []
for barrio, g in ent.groupby("neighbourhood_cleansed"):
    g2 = ent2[ent2["neighbourhood_cleansed"] == barrio]
    c1, s1 = g.loc[g["has_pool"], OBJETIVO], g.loc[~g["has_pool"], OBJETIVO]
    c2, s2 = g2.loc[g2["has_pool"], OBJETIVO], g2.loc[~g2["has_pool"], OBJETIVO]
    if len(c1) >= 30 and len(s1) >= 30 and len(c2) >= 20 and len(s2) >= 20:
        filas.append({"barrio": barrio, "publicaciones": int(len(g)), "publicaciones_2p": int(len(g2)),
                      "sep_solo_barrio": separacion(c1, s1), "sep_barrio_y_tamano": separacion(c2, s2),
                      "mediana_con": float(g2.loc[g2["has_pool"], "price"].median()),
                      "mediana_sin": float(g2.loc[~g2["has_pool"], "price"].median())})
h4 = pd.DataFrame(filas)
M["h4"] = {"barrios": filas,
           "sep_global": separacion(ent.loc[ent["has_pool"], OBJETIVO], ent.loc[~ent["has_pool"], OBJETIVO]),
           "sep_solo_barrio": float(np.average(h4["sep_solo_barrio"], weights=h4["publicaciones"])),
           "sep_barrio_y_tamano": float(np.average(h4["sep_barrio_y_tamano"], weights=h4["publicaciones_2p"])),
           "barrios_sobre_corte": int((h4["sep_barrio_y_tamano"] >= 0.2).sum()),
           "tamano_con": {"huespedes": float(ent.loc[ent["has_pool"], "accommodates"].mean()),
                          "monoamb": float((ent.loc[ent["has_pool"], "bedrooms"].fillna(0) == 0).mean())},
           "tamano_sin": {"huespedes": float(ent.loc[~ent["has_pool"], "accommodates"].mean()),
                          "monoamb": float((ent.loc[~ent["has_pool"], "bedrooms"].fillna(0) == 0).mean())}}

sh = df[df[OBJETIVO].between(p01, p99)]
priv = sh[sh["room_type"] == "Private room"]
M["h2"] = {"sep_global": separacion(df.loc[df["host_is_superhost"] == "t", OBJETIVO],
                                    df.loc[df["host_is_superhost"] == "f", OBJETIVO]),
           "mediana_super": float(df.loc[df["host_is_superhost"] == "t", "price"].median()),
           "mediana_no_super": float(df.loc[df["host_is_superhost"] == "f", "price"].median()),
           "sep_hab_privada": separacion(priv.loc[priv["host_is_superhost"] == "t", OBJETIVO],
                                         priv.loc[priv["host_is_superhost"] == "f", OBJETIVO])}
h3 = sh[sh["accommodates"] == 2]
M["h3"] = {"sep": separacion(sh.loc[sh["room_type"] == "Entire home/apt", OBJETIVO],
                             sh.loc[sh["room_type"] == "Private room", OBJETIVO]),
           "mediana_entero_2p": float(h3.loc[h3["room_type"] == "Entire home/apt", "price"].median()),
           "mediana_privada_2p": float(h3.loc[h3["room_type"] == "Private room", "price"].median())}

# ---------------------------------------------------------------- referencia para validar casos nuevos
num_ref = ["accommodates", "bedrooms", "bathrooms", "minimum_nights", "amenities_count"]
X_ref = final.named_steps["reglas"].transform(X_tr)   # monoambientes = 0 dormitorios, como ve el modelo
M["referencia"] = {
    "rangos": {c: {"min": float(X_ref[c].min()), "p01": float(X_ref[c].quantile(0.01)),
                   "p99": float(X_ref[c].quantile(0.99)), "max": float(X_ref[c].max()),
                   "mediana": float(X_ref[c].median())} for c in num_ref},
    "barrios": {b: {"lat": float(g["latitude"].median()), "lon": float(g["longitude"].median()), "n": int(len(g))}
                for b, g in tr.groupby("neighbourhood_cleansed")},
    "room_types": tr["room_type"].value_counts().to_dict()}
M["histograma_precios"] = df["price"].round(0).tolist()   # todas las filas, incluidos los extremos

joblib.dump(final, SALIDA / "modelo.joblib", compress=3)
(SALIDA / "metricas.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
paso(f"Listo: {SALIDA}")
