"""Mapa de fondo para el gráfico de Explorá, armado con capas de Altair sobre ejes de longitud/latitud.

Tres capas comparten las mismas escalas x = longitud, y = latitud:
  1. mosaicos de mapa base (mark_image), traídos del servidor de Esri según el zoom;
  2. límites de los barrios (mark_line) desde el GeoJSON de Inside Airbnb, guardado en el repo:
     si los mosaicos no cargan, el mapa sigue mostrando los barrios;
  3. las publicaciones (mark_circle), en app.py.
Se usan ejes cartesianos (y no una proyección) para que la selección por intervalo siga funcionando.
A la escala de CABA, latitud lineal y Mercator difieren en menos de un píxel.
"""
import json
import math
from pathlib import Path

import pandas as pd

GEOJSON = Path(__file__).parent / "data" / "geo" / "barrios_caba.geojson"
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
ATRIBUCION = "Mapa base: Esri, HERE, Garmin, © colaboradores de OpenStreetMap · Límites de barrios: Inside Airbnb"


def leer_barrios(paso_min=0.00015):
    """GeoJSON -> una fila por vértice (barrio, anillo, orden, lon, lat). Saltea vértices a menos de ~15 m."""
    g = json.loads(GEOJSON.read_text(encoding="utf-8"))
    filas, anillo = [], 0
    for f in g["features"]:
        barrio = f["properties"]["neighbourhood"]
        geom = f["geometry"]
        poligonos = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poligono in poligonos:
            for ring in poligono:
                ult = None
                for i, (lon, lat) in enumerate(ring):
                    if ult and i < len(ring) - 1 and abs(lon - ult[0]) + abs(lat - ult[1]) < paso_min:
                        continue
                    filas.append((barrio, anillo, len(filas), lon, lat))
                    ult = (lon, lat)
                anillo += 1
    return pd.DataFrame(filas, columns=["barrio", "anillo", "orden", "lon", "lat"])


def encuadre(lineas, barrios=None, margen=0.06):
    """Dominio [lon], [lat] que contiene los barrios pedidos (o toda CABA), con la proporción de un mapa real."""
    sub = lineas[lineas["barrio"].isin(barrios)] if barrios else lineas
    lon0, lon1, lat0, lat1 = sub["lon"].min(), sub["lon"].max(), sub["lat"].min(), sub["lat"].max()
    clat = (lat0 + lat1) / 2
    k = math.cos(math.radians(clat))          # un grado de longitud mide cos(lat) grados de latitud
    dx, dy = (lon1 - lon0) * (1 + 2 * margen), (lat1 - lat0) * (1 + 2 * margen)
    if dx * k > dy:
        dy = dx * k
    else:
        dx = dy / k
    clon = (lon0 + lon1) / 2
    return [clon - dx / 2, clon + dx / 2], [clat - dy / 2, clat + dy / 2]


def _tile_x(lon, z):
    return int((lon + 180) / 360 * 2 ** z)


def _tile_y(lat, z):
    r = math.radians(lat)
    return int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2 ** z)


def _lon(x, z):
    return x / 2 ** z * 360 - 180


def _lat(y, z):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2 ** z))))


def mosaicos(dom_lon, dom_lat, ancho_px=330):
    """Mosaicos que cubren el encuadre, con el zoom en que cada uno se ve a ~256 px o menos (nítido)."""
    dx = dom_lon[1] - dom_lon[0]
    z = min(17, max(10, math.ceil(math.log2(360 * ancho_px / (256 * dx)))))
    filas = []
    for x in range(_tile_x(dom_lon[0], z), _tile_x(dom_lon[1], z) + 1):
        for y in range(_tile_y(dom_lat[1], z), _tile_y(dom_lat[0], z) + 1):
            filas.append({"url": TILE_URL.format(z=z, x=x, y=y), "lon0": _lon(x, z), "lon1": _lon(x + 1, z),
                          "lat0": _lat(y + 1, z), "lat1": _lat(y, z)})
    return pd.DataFrame(filas)
