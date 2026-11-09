"""
Pipeline de ingesta — Proyecto Integrador (Entrega 1)
Grupo 5K09-06 · Fuente: Inside Airbnb, listings de Buenos Aires

Pregunta del proyecto: ¿a qué precio debería publicarse un alquiler
temporario en CABA, según las características de la propiedad?

Estructura: check_new_snapshot -> extract_bronze -> validate_bronze
           -> refine_silver -> validate_silver
Mismo patrón que fifa_ingest.py (TP1): la fuente se toca una sola vez por
snapshot (bronze append-only), y silver es lo que se defiende en la
Entrega 1. Se sumaron tres cosas del DAG del TP1:

  - retries en extract_bronze (segura porque la tarea es idempotente:
    si el archivo ya existe, no vuelve a pedir nada).
  - validate_bronze y validate_silver acumulan TODOS los problemas
    encontrados antes de fallar, en vez de frenar en el primero — así
    en una corrida ves todo lo que hay que arreglar, no uno por vez.
  - validate_silver corre sobre el dataset final (no solo sobre el
    bronze crudo) y calcula ahí mismo los 6 criterios medibles de la
    Entrega 1, dejando los números en el log de la tarea.

Orquestación autónoma:
  - schedule="@daily" con catchup=False: el DAG corre una vez al día.
  - check_new_snapshot scrapea el catálogo de Inside Airbnb y compara
    la fecha del snapshot más reciente con la Variable de Airflow
    airbnb_caba_latest_snapshot. Si no hay datos nuevos, el DAG se
    cortocircuita sin gastar recursos.

TODO antes de correrlo:
  - Sumar/sacar columnas de COLUMNAS_UTILES según qué variables
    finalmente uses para explicar el precio.
  - Si vuelve a fallar validate_bronze o validate_silver, revisar si
    cambió el volumen real de listados en un snapshot más nuevo y
    reajustar MIN_FILAS.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
from pathlib import Path

import pandas as pd
import pendulum
import requests
from airflow.sdk import DAG, task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from datetime import datetime

log = logging.getLogger(__name__)

# En Airflow 3, context["dag"].dag_id no es confiable: usar una constante.
DAG_ID = "airbnb_caba_ingest_V1"

PATH_BASE = Path("/usr/local/airflow/include/output/V1")  # include/ del proyecto Astro
# ~85% de las ~29.000 filas observadas en el snapshot 2026-07-24 (referencia
# tomada por Rober al armar el DAG). Umbral pensado para tolerar la
# fluctuación normal del mercado entre snapshots, no para exigir un piso
# arbitrario, pero sí para frenar el pipeline si un scrape roto devuelve
# un archivo truncado o una página de error. Se usa tanto en bronze (antes
# de deduplicar) como en silver (después) — en silver puede haber bajado
# algo si hubo duplicados reales.
MIN_FILAS = 25000

# Columnas candidatas para explicar el precio de un alquiler temporario.
# Recorta esta lista a medida que definan qué "características de la
# propiedad" van a usar en el modelo (Entrega 3).
COLUMNAS_UTILES = [
    "id",
    "price",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms_text",  # NO usar "bathrooms": viene deprecada y desalineada
                       # (multiplicada x10 respecto al valor real en este
                       # snapshot — "1 bath" -> 10, "1.5 baths" -> 15).
                       # Se parsea bathrooms_text más abajo en refine_silver.
    "bedrooms",
    "beds",
    "amenities",
    "minimum_nights",
    "availability_30",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "instant_bookable",
    "host_is_superhost",
]


def parse_bathrooms(texto) -> float | None:
    """Extrae la cantidad de baños del campo de texto ('1.5 baths',
    'Half-bath', '3 shared baths'). Es la fuente confiable: la columna
    numérica 'bathrooms' de este dataset viene deprecada/desalineada.
    """
    if pd.isna(texto):
        return None
    texto = str(texto).strip().lower()
    if texto.startswith("half"):
        return 0.5
    m = re.search(r"[\d.]+", texto)
    return float(m.group()) if m else None


def parse_bathrooms_shared(texto) -> bool | None:
    """True si el texto dice 'shared' (baño compartido con otros huéspedes
    u otras unidades, no privado). None si no hay dato.
    """
    if pd.isna(texto):
        return None
    return "shared" in str(texto).strip().lower()


def parse_amenities_list(texto) -> list[str]:
    """Parsea el campo 'amenities', que viene como string de un array JSON
    (ej. '["Wifi", "Air conditioning", ...]'). Devuelve una lista de strings,
    o vacía si no se pudo parsear.
    """
    if pd.isna(texto):
        return []
    try:
        items = json.loads(str(texto))
        return items if isinstance(items, list) else []
    except Exception:
        return []


def bronze_path(snapshot_date: str) -> Path:
    """Ruta de la capa bronze, particionada por fecha de snapshot.

    Igual criterio que en el TP1: escopar el glob a la ruta específica
    de este snapshot evita arrastrar archivos de corridas viejas.
    """
    return PATH_BASE / "bronze" / "airbnb_caba" / f"snapshot={snapshot_date}" / "listings.csv.gz"


def silver_path(snapshot_date: str) -> Path:
    return PATH_BASE / "silver" / "airbnb_caba" / f"snapshot={snapshot_date}.csv"


def listings_url(snapshot_date: str) -> str:
    return (
        "https://data.insideairbnb.com/argentina/"
        "ciudad-autónoma-de-buenos-aires/buenos-aires/"
        f"{snapshot_date}/data/listings.csv.gz"
    )


with DAG(
    dag_id=DAG_ID,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"snapshot_date": "2026-07-24"},  # fallback si se dispara manualmente
) as dag:

    @task.short_circuit
    def check_new_snapshot(
        url_catalogo: str = "https://insideairbnb.com/get-the-data/",
    ):
        """Scrapea el catálogo de Inside Airbnb, extrae la fecha del
        snapshot más reciente para Buenos Aires, y compara con la
        Variable de Airflow para decidir si hay datos nuevos.

        Devuelve la fecha (str) si hay un snapshot nuevo — viaja por
        XCom a extract_bronze. Devuelve False si no hay novedades y
        el DAG se cortocircuita sin gastar recursos.
        """
        response = requests.get(url_catalogo, timeout=60)
        response.raise_for_status()

        patron = r'href="([^"]+/buenos-aires/(\d{4}-\d{2}-\d{2})/data/listings\.csv\.gz)"'
        match = re.search(patron, response.text)

        if not match:
            log.warning("No se encontró un link de descarga para Buenos Aires en el catálogo")
            return False

        download_url = match.group(1)
        snapshot_date = match.group(2)
        log.info("Snapshot encontrado: %s (URL: %s)", snapshot_date, download_url)

        ultima_fecha = Variable.get("airbnb_caba_latest_snapshot", default_var=None)

        if snapshot_date == ultima_fecha:
            log.info(
                "Snapshot %s ya fue procesado (coincide con Variable). "
                "Cortocircuitando DAG.",
                snapshot_date,
            )
            return False

        Variable.set("airbnb_caba_latest_snapshot", snapshot_date)
        log.info(
            "Nuevo snapshot detectado: %s (anterior: %s). Continuando pipeline.",
            snapshot_date,
            ultima_fecha,
        )
        return snapshot_date

    @task(retries=2, retry_delay=pendulum.duration(seconds=30))
    def extract_bronze(snapshot_from_check=None, **context) -> str:
        """Capa bronze: baja el .csv.gz tal cual viene de la fuente.

        Append-only: si el snapshot ya está en disco, no se vuelve a
        pedir. La fuente se toca una vez por fecha, no una vez por
        corrida — mismo principio que con sofifa en el TP1.

        retries=2 es seguro acá porque la tarea es idempotente: si el
        archivo ya existe, ni siquiera intenta la request de nuevo.
        Un corte de red momentáneo no debería tirar abajo toda la
        corrida.

        snapshot_from_check recibe la fecha dinámica de
        check_new_snapshot vía XCom. Si no viene (trigger manual),
        usa context["params"]["snapshot_date"] como fallback.
        """
        snapshot_date = snapshot_from_check or context["params"]["snapshot_date"]
        destino = bronze_path(snapshot_date)
        destino.parent.mkdir(parents=True, exist_ok=True)

        if destino.exists():
            log.info("Snapshot %s ya en bronze, no se vuelve a descargar", snapshot_date)
            return str(destino)

        url = listings_url(snapshot_date)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        destino.write_bytes(resp.content)
        log.info("Descargado snapshot %s (%d bytes)", snapshot_date, len(resp.content))
        return str(destino)

    @task
    def validate_bronze(bronze_file: str) -> str:
        """Chequeo mínimo antes de gastar tiempo transformando: que el
        archivo abra como CSV, tenga las columnas que vamos a necesitar
        en refine_silver, y un volumen razonable.

        Acumula todos los problemas antes de fallar (patrón del TP1):
        si un día cambia el esquema de la fuente Y además viene con
        pocas filas, querés ver las dos cosas en el mismo log, no
        arreglar una y descubrir la otra recién en el reintento.
        """
        with gzip.open(bronze_file, "rb") as f:
            df = pd.read_csv(io.BytesIO(f.read()), low_memory=False)

        problemas = []

        if len(df) < MIN_FILAS:
            problemas.append(
                f"solo {len(df)} filas, se esperaban >= {MIN_FILAS} "
                "(posible scrape roto o snapshot equivocado)"
            )

        faltan = set(COLUMNAS_UTILES) - set(df.columns)
        if faltan:
            problemas.append(f"faltan columnas que usa refine_silver: {faltan}")

        if problemas:
            raise AirflowFailException(
                "Bronze inválido:\n  - " + "\n  - ".join(problemas)
            )

        log.info("Bronze validado: %d filas, %d columnas", *df.shape)
        return bronze_file

    @task
    def refine_silver(bronze_file: str, **context) -> str:
        """Capa silver: tipa, limpia y deja una fila = una publicación.

        - price viene como string ('$86,206.29'): se limpia a float.
          Los nulos de price se CONSERVAN (~6,5% del dataset) — es una
          decisión de modelado sacarlos o no, no de esta entrega. Ver
          la nota de validate_silver / la charla sobre availability_365.
        - se recorta a las columnas útiles para la pregunta del
          proyecto.
        - bathrooms_text se descompone en bathrooms (numérico) y
          bathrooms_shared (bool).
        - amenities se parsea a lista y se extraen amenities_count,
          has_parking, has_pool y has_elevator.
        - availability_30 y availability_90 se tipan a entero nullable
          (Int64: soporta NA sin romper, a diferencia de int nativo).
        - se deduplica por id (la clave primaria del dataset).
        """
        snapshot_date = context["params"]["snapshot_date"]

        with gzip.open(bronze_file, "rb") as f:
            df = pd.read_csv(io.BytesIO(f.read()), low_memory=False)

        df = df[COLUMNAS_UTILES].copy()

        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )

        # Limpieza de la variable objetivo: eliminar filas con price nulo.
        # Sin precio no hay target para el modelo — descartarlas acá
        # garantiza que silver siempre tenga un dataset entrenable.
        filas_antes_price = len(df)
        df = df.dropna(subset=["price"])
        filas_sin_precio = filas_antes_price - len(df)
        if filas_sin_precio > 0:
            log.warning(
                "Se descartaron %d filas con price nulo (%.1f%% del total)",
                filas_sin_precio,
                filas_sin_precio / filas_antes_price * 100,
            )

        # bathrooms_text es la fuente confiable (ver nota en COLUMNAS_UTILES);
        # se reemplaza por dos columnas: la cantidad numérica y si es
        # compartido (importa para el precio: no es lo mismo un baño
        # privado que uno "shared" con otras unidades).
        df["bathrooms"] = df["bathrooms_text"].apply(parse_bathrooms)
        df["bathrooms_shared"] = df["bathrooms_text"].apply(parse_bathrooms_shared)
        df = df.drop(columns=["bathrooms_text"])

        # Amenities: se preserva la columna original (lista de strings) y se
        # derivan variables atómicas de alto valor predictivo para el precio.
        # Se usa re.search sobre cada ítem de la lista parseada.
        amenities_series = df["amenities"].apply(parse_amenities_list)
        df["amenities_count"] = amenities_series.apply(len)

        df["has_ac"] = amenities_series.apply(
            lambda items: any(re.search(r"\bair conditioning\b|\bac\b|aire acondicionado", it, re.IGNORECASE) for it in items)
        )
        df["has_heating"] = amenities_series.apply(
            lambda items: any(re.search(r"\bheating\b|calefacci", it, re.IGNORECASE) for it in items)
        )
        df["has_pool"] = amenities_series.apply(
            lambda items: any(
                re.search(r"\bpool\b|pileta", it, re.IGNORECASE)
                and not re.search(r"pool table", it, re.IGNORECASE)
                for it in items
            )
        )
        df["has_gym"] = amenities_series.apply(
            lambda items: any(re.search(r"\bgym\b|fitness center|gimnasio", it, re.IGNORECASE) for it in items)
        )
        df["has_balcony_patio"] = amenities_series.apply(
            lambda items: any(re.search(r"balcony|patio|terrace|terraza|outdoor", it, re.IGNORECASE) for it in items)
        )
        df["has_washer"] = amenities_series.apply(
            lambda items: any(re.search(r"\bwasher\b|lavarropas", it, re.IGNORECASE) for it in items)
        )
        df["pets_allowed"] = amenities_series.apply(
            lambda items: any(re.search(r"pets allowed|pet friendly|mascotas", it, re.IGNORECASE) for it in items)
        )
        df["has_elevator"] = amenities_series.apply(
            lambda items: any(re.search(r"\belevator\b|ascensor", it, re.IGNORECASE) for it in items)
        )
        df["has_parking"] = amenities_series.apply(
            lambda items: any(re.search(r"\bparking\b|garage|cochera", it, re.IGNORECASE) for it in items)
        )
        df["has_workspace"] = amenities_series.apply(
            lambda items: any(re.search(r"dedicated workspace|escritorio", it, re.IGNORECASE) for it in items)
        )

        # Int64 (nullable) en vez de int: si availability_30/90 tuviera algún
        # nulo, `.astype(int)` común rompería el DAG.
        df["availability_30"] = df["availability_30"].astype("Int64")
        df["availability_90"] = df["availability_90"].astype("Int64")

        antes = len(df)
        df = df.drop_duplicates(subset="id")
        if len(df) != antes:
            log.warning("Se descartaron %d duplicados por id", antes - len(df))

        # columnas completamente vacías no aportan nada (criterio 6)
        vacias = df.columns[df.isna().all()]
        if len(vacias):
            log.warning("Columnas 100%% nulas descartadas: %s", list(vacias))
            df = df.drop(columns=vacias)

        destino = silver_path(snapshot_date)
        destino.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destino, index=False)
        log.info("Silver escrito en %s (%d filas, %d columnas)", destino, *df.shape)
        return str(destino)

    @task
    def validate_silver(silver_file: str) -> str:
        """Chequeo final sobre el dataset que se defiende en la Entrega 1.

        Calcula acá mismo los 6 criterios medibles de la consigna (clave
        sin duplicados, volumen, ancho, mezcla de tipos, nulos conocidos,
        columnas vacías) y los deja en el log de la tarea — son los
        números que hay que tener "a mano" el día de la entrega, ya
        generados por el pipeline y no calculados aparte en un notebook.

        Acumula todos los problemas antes de fallar, igual que
        validate_bronze.
        """
        df = pd.read_csv(silver_file, low_memory=False)

        problemas = []

        clave_unica = df["id"].is_unique
        if not clave_unica:
            repetidos = df["id"].duplicated().sum()
            problemas.append(f"la clave 'id' tiene {repetidos} repetidos")

        if len(df) < MIN_FILAS * 0.9:  # margen extra: acá ya se dedujo
            problemas.append(
                f"solo {len(df)} filas tras deduplicar, muy por debajo de lo esperado"
            )

        if df.shape[1] < 5:
            problemas.append(f"solo {df.shape[1]} columnas, se esperaban >= 5")

        vacias = df.columns[df.isna().all()]
        if len(vacias):
            problemas.append(f"quedaron columnas 100% nulas: {list(vacias)}")

        if problemas:
            raise AirflowFailException(
                "Silver inválido:\n  - " + "\n  - ".join(problemas)
            )

        nulos = df.isna().mean().sort_values(ascending=False)
        nulos_relevantes = nulos[nulos > 0]

        log.info("=== Chequeo de calidad — Entrega 1 ===")
        log.info("Clave 'id' sin duplicados: %s", clave_unica)
        log.info("Filas: %d", len(df))
        log.info("Columnas: %d", df.shape[1])
        log.info("Tipos de dato:\n%s", df.dtypes.value_counts().to_string())
        log.info("Columnas con nulos:\n%s", nulos_relevantes.to_string() or "(ninguna)")
        log.info("Columna objetivo: price")

        return silver_file

    # Flujo de ejecución: short-circuit → bronze → validación → silver → validación
    snapshot = check_new_snapshot()
    bronze = extract_bronze(snapshot_from_check=snapshot)
    validated_bronze = validate_bronze(bronze)
    silver = refine_silver(validated_bronze)
    validate_silver(silver)
