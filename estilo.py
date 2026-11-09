"""Un solo mapeo de colores y formatos para toda la aplicación.

Cada color significa una sola cosa en el tallo y en la copa:
  - tipo de habitación   -> ROOM_TYPE
  - pileta               -> PILETA
  - error del modelo     -> ERROR (violeta = subestima, naranja = sobreestima)
  - precio / conteos     -> NEUTRO
"""
import altair as alt

ROOM_TYPE = alt.Scale(
    domain=["Entire home/apt", "Private room", "Hotel room", "Shared room"],
    range=["#3b5b92", "#d1495b", "#edae49", "#8d8d8d"])

# Nombres en español de las columnas del modelo (se muestran junto al nombre original).
COLUMNA_ES = {"accommodates": "Huéspedes", "bedrooms": "Dormitorios", "bathrooms": "Baños",
              "minimum_nights": "Noches mínimas", "latitude": "Latitud", "longitude": "Longitud",
              "has_pool": "Pileta", "neighbourhood_cleansed": "Barrio", "room_type": "Tipo de alojamiento",
              "amenities_count": "Cantidad de comodidades", "has_ac": "Aire acondicionado",
              "has_elevator": "Ascensor", "has_balcony_patio": "Balcón o patio", "has_parking": "Cochera",
              "has_workspace": "Espacio de trabajo", "bathrooms_shared": "Baño compartido"}

# Nombres en español de los tipos de alojamiento de Airbnb (el modelo trabaja con los originales).
TIPO_ES = {"Entire home/apt": "Departamento o casa entera", "Private room": "Habitación privada",
           "Hotel room": "Habitación de hotel", "Shared room": "Habitación compartida"}

PILETA = alt.Scale(domain=["Sin pileta", "Con pileta"], range=["#b0b7bf", "#00a6a6"])

# Error relativo con signo: (predicho - real) / real. Recortado a ±60 % para que la escala no la dominen los extremos.
ERROR = alt.Scale(domain=[-0.6, 0, 0.6], range=["#5e3c99", "#d9d9d9", "#e66101"], clamp=True)

NEUTRO = "#6c7a89"
CLARO = "#cfd5db"
MARCA = "#111111"   # el caso que carga el usuario

# Ejes de precio en pesos con separador de miles argentino ($100.000)
EJE_PESOS = "'$' + replace(format(datum.value, ',.0f'), regexp(',', 'g'), '.')"
EJE_ENTERO = "replace(format(datum.value, ',.0f'), regexp(',', 'g'), '.')"
EJE_PESOS_LOG10 = "'$' + replace(format(pow(10, datum.value), ',.0f'), regexp(',', 'g'), '.')"


def pesos(x):
    return "$" + f"{x:,.0f}".replace(",", ".")


def pct(x, dec=1):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def num(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def entero(x):
    return f"{int(x):,}".replace(",", ".")


# Clases fijas para colorear Explorá.
# Precio: quintiles del precio real de toda CABA (los cortes se calculan en app.py con los datos).
COLORES_PRECIO = ["#d9f0a3", "#addd8e", "#41ab5d", "#1a7a3c", "#00441b"]
# Acierto: el corte de ±20 % es el error relativo mediano del modelo; mismos colores que el error del Tallo 3.
CORTES_ERROR = [-0.2, 0.2]
CLASES_ERROR = ["Pide de menos (más de 20 % abajo)", "Acierta (dentro de ±20 %)", "Pide de más (más de 20 % arriba)"]
COLORES_ERROR = ["#5e3c99", "#bdbdbd", "#e66101"]


def clases_precio(cortes):
    c = [pesos(x) for x in cortes]
    return [f"1 · hasta {c[0]}", f"2 · {c[0]}–{c[1]}", f"3 · {c[1]}–{c[2]}", f"4 · {c[2]}–{c[3]}", f"5 · más de {c[3]}"]


def expr_clases(campo, cortes, clases):
    """Expresión de Vega que asigna la clase según los cortes: c0 ? clase0 : c1 ? clase1 : ... : última."""
    partes = [f"{campo} < {c} ? '{k}'" for c, k in zip(cortes, clases)]
    return " : ".join(partes) + f" : '{clases[-1]}'"
