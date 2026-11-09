"""Transformadores y armado del pipeline de la Entrega 3.

Copiados tal cual de modelo/entrega_3.ipynb. Viven en un módulo (y no en el notebook)
para que el modelo guardado con joblib se pueda cargar desde la aplicación.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEMILLA = 42

NUM_BASE = ["accommodates", "bathrooms", "bedrooms", "amenities_count", "minimum_nights", "latitude", "longitude"]
BOOL_BASE = ["has_ac", "has_pool", "has_elevator", "has_balcony_patio", "has_parking", "has_workspace", "bathrooms_shared"]
CAT_BASE = ["room_type", "neighbourhood_cleansed"]
DERIV_TAMANO = ["pers_x_dorm", "banos_x_pers", "monoambiente"]
COLUMNAS = NUM_BASE + BOOL_BASE + CAT_BASE


class ReglasDeDominio(BaseEstimator, TransformerMixin):
    """Completa nulos con reglas fijas del EDA. No aprende nada en fit, así que no puede fugar."""
    REGLAS = {"bedrooms": 0, "bathrooms": 1, "minimum_nights": 1, "bathrooms_shared": False}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, valor in self.REGLAS.items():
            X[col] = X[col].fillna(valor)
        X["bathrooms_shared"] = X["bathrooms_shared"].astype(bool)
        return X


class TamanoRelativo(BaseEstimator, TransformerMixin):
    """Cocientes de tamaño que un modelo lineal no puede construir solo."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["pers_x_dorm"] = X["accommodates"] / X["bedrooms"].clip(lower=1)
        X["banos_x_pers"] = X["bathrooms"] / X["accommodates"].clip(lower=1)
        X["monoambiente"] = (X["bedrooms"] == 0).astype(int)
        return X


class PiletaPorBarrio(BaseEstimator, TransformerMixin):
    """Interacción de H4 (Entrega 2): barrio + pileta, y tasa de pileta del barrio aprendida en train."""

    def fit(self, X, y=None):
        self.tasa_barrio_ = X.groupby("neighbourhood_cleansed")["has_pool"].mean().to_dict()
        self.tasa_global_ = float(X["has_pool"].mean())
        return self

    def transform(self, X):
        X = X.copy()
        X["pileta_barrio"] = X["neighbourhood_cleansed"].astype(str) + "_pool_" + X["has_pool"].astype(int).astype(str)
        X["tasa_pileta_barrio"] = X["neighbourhood_cleansed"].map(self.tasa_barrio_).fillna(self.tasa_global_)
        return X


class PrecioLocal(BaseEstimator, TransformerMixin):
    """Mediana de log_price de los k vecinos de train más cercanos (por latitud/longitud).

    En fit_transform calcula la columna FUERA DE PLIEGUE para que ninguna fila de train
    sea vecina de sí misma. En transform (al predecir nuevos deptos) usa todos los vecinos de train.
    """

    def __init__(self, k=15, fuera_de_pliegue=True, n_splits=5, random_state=SEMILLA):
        self.k, self.fuera_de_pliegue, self.n_splits, self.random_state = k, fuera_de_pliegue, n_splits, random_state

    def fit(self, X, y):
        self.coords_ = X[["latitude", "longitude"]].to_numpy()
        self.y_ = np.asarray(y)
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(self.coords_)
        return self

    def _mediana_vecinos(self, nn, y_ref, coords):
        _, idx = nn.kneighbors(coords)
        return np.median(y_ref[idx], axis=1)

    def transform(self, X):
        X = X.copy()
        X["precio_local"] = self._mediana_vecinos(self.nn_, self.y_, X[["latitude", "longitude"]].to_numpy())
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        if not self.fuera_de_pliegue:
            return self.transform(X)
        X = X.copy()
        col = np.zeros(len(X))
        for a, b in KFold(self.n_splits, shuffle=True, random_state=self.random_state).split(X):
            nn = NearestNeighbors(n_neighbors=self.k).fit(self.coords_[a])
            col[b] = self._mediana_vecinos(nn, self.y_[a], self.coords_[b])
        X["precio_local"] = col
        return X


def armar(modelo, familia, tamano=True, pileta_barrio=True, precio_local=True, k=15, fuera_de_pliegue=True):
    """Arma el pipeline completo. familia = 'lineal' (escala + One-Hot) o 'arbol' (sin escalar, categorías nativas)."""
    pasos = [("reglas", ReglasDeDominio())]
    num, cat = NUM_BASE + BOOL_BASE, list(CAT_BASE)
    if tamano:
        pasos.append(("tamano", TamanoRelativo()))
        num = num + DERIV_TAMANO
    if pileta_barrio:
        pasos.append(("pileta_barrio", PiletaPorBarrio()))
        num = num + ["tasa_pileta_barrio"]
        cat = cat + ["pileta_barrio"]
    if precio_local:
        pasos.append(("precio_local", PrecioLocal(k=k, fuera_de_pliegue=fuera_de_pliegue)))
        num = num + ["precio_local"]
    if familia == "lineal":
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), cat)])
    else:
        pre = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), cat)],
            verbose_feature_names_out=False)
        if isinstance(modelo, HistGradientBoostingRegressor):
            modelo = clone(modelo).set_params(categorical_features=cat)
    pasos += [("pre", pre.set_output(transform="pandas")), ("modelo", modelo)]
    return Pipeline(pasos)
