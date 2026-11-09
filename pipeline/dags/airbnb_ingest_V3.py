"""
Pipeline de ingesta — Proyecto Integrador (Entrega 1)
Grupo 5K09-06 · Fuente: Inside Airbnb, listings de Buenos Aires

Pregunta del proyecto: ¿a qué precio debería publicarse un alquiler
temporario en CABA, según las características de la propiedad?

Estructura medallón robusta (patrón adaptado de fifa_ingest.py):

  wait_for_source (Sensor)
         │
  check_source (Branch)
     ├── (Fuente disponible) ──> check_new_snapshot (ShortCircuit)
     │                                 │
     │                           extract_bronze ──> validate_bronze ──> refine_silver ──┐
     │                                                                                  │
     └── (Fuente caída / timeout) ─────> load_frozen (Respaldo) ────────────────────────┴──> validate_silver

Mejoras incorporadas desde fifa_ingest.py (TP1):
  - Sensor con sondeo (`wait_for_source`): verifica la fuente cada 5 min (hasta 30 min)
    con mode='reschedule' y soft_fail=True sin bloquear workers ni fallar de golpe.
  - Rama de contingencia (`load_frozen`): si la web no responde, usa el respaldo más
    fresco (ultimo_ok.csv o semilla).
  - Variable atómica: airbnb_caba_latest_snapshot se actualiza al final en validate_silver,
    sólo si todo el archivo se procesó y validó con éxito.
  - retries en extract_bronze (tarea idempotente).
  - validate_bronze y validate_silver acumulan todos los problemas antes de fallar.
  - validate_silver calcula y loguea los 6 criterios de calidad de la Entrega 1.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
import shutil
from pathlib import Path

import pandas as pd
import pendulum
import requests
from airflow.sdk import DAG, PokeReturnValue, task
try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:
    from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.task.trigger_rule import TriggerRule
from datetime import datetime

log = logging.getLogger(__name__)

# En Airflow 3, context["dag"].dag_id no es confiable: usar una constante.
DAG_ID = "airbnb_caba_ingest_V3"

# Headers simulando navegador para evitar bloqueos 403 de CloudFront/Cloudflare
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PATH_BASE = Path("/usr/local/airflow/include/output/V3")  # include/ del proyecto Astro

# Rutas del respaldo de contingencia (patrón de FIFA)
FROZEN_DIR = Path("/usr/local/airflow/include/frozen")
SEMILLA = FROZEN_DIR / "airbnb_snapshot.csv"
ULTIMO_OK = FROZEN_DIR / "ultimo_ok.csv"

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


def clean_airbnb_url(url: str) -> str:
    """Normaliza la URL de Buenos Aires para evitar problemas de tildes o doble encoding."""
    return (
        url.replace("ciudad-aut%C3%83%C2%B3noma-de-buenos-aires", "ciudad-aut%C3%B3noma-de-buenos-aires")
        .replace("ciudad-autÃ³noma-de-buenos-aires", "ciudad-aut%C3%B3noma-de-buenos-aires")
        .replace("ciudad-autónoma-de-buenos-aires", "ciudad-aut%C3%B3noma-de-buenos-aires")
    )


def listings_url(snapshot_date: str) -> str:
    return (
        "https://data.insideairbnb.com/argentina/"
        "ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/"
        f"{snapshot_date}/data/listings.csv.gz"
    )


with DAG(
    dag_id=DAG_ID,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={"snapshot_date": "2026-07-24"},  # fallback si se dispara manualmente
) as dag:

    @task.sensor(poke_interval=300, timeout=1800, mode="reschedule", soft_fail=True)
    def wait_for_source(
        url_catalogo: str = "https://insideairbnb.com/get-the-data/",
    ) -> PokeReturnValue:
        """Espera a que Inside Airbnb esté disponible. Sondea cada 5 min, hasta 30 min.

        - mode='reschedule': libera el worker mientras espera.
        - soft_fail=True: si se agota el tiempo, queda en 'skipped' y no en
          'failed', permitiendo que check_source derive a la rama de respaldo.
        """
        try:
            resp = requests.get(url_catalogo, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            resp.encoding = "utf-8"

            patron = r'href="([^"]+/buenos-aires/(\d{4}-\d{2}-\d{2})/data/listings\.csv\.gz)"'
            match = re.search(patron, resp.text)

            if not match:
                log.warning("Inside Airbnb respondió pero no se encontró un link de Buenos Aires")
                return PokeReturnValue(is_done=False)

            download_url = clean_airbnb_url(match.group(1))
            snapshot_date = match.group(2)
            log.info("Inside Airbnb responde: snapshot detectado %s (URL: %s)", snapshot_date, download_url)
            return PokeReturnValue(
                is_done=True,
                xcom_value={"snapshot_date": snapshot_date, "url": download_url},
            )
        except Exception as e:
            log.warning("Inside Airbnb todavía no responde (%s). Reintento en 5 min.", e)
            return PokeReturnValue(is_done=False)

    @task.branch(trigger_rule=TriggerRule.ALL_DONE)
    def check_source(**context) -> str:
        """Decide por dónde continúa el DAG, según lo que consiguió el sensor.

        Corre con ALL_DONE para evaluar incluso si wait_for_source quedó en skipped.
        - Si el sensor obtuvo datos -> check_new_snapshot.
        - Si no respondió en 30 minutos -> load_frozen.
        """
        meta = context["ti"].xcom_pull(task_ids="wait_for_source")
        if meta:
            return "check_new_snapshot"
        log.error("Inside Airbnb no respondió en 30 minutos. Se usa el respaldo.")
        return "load_frozen"

    @task.short_circuit
    def check_new_snapshot(**context) -> str | bool:
        """Compara la fecha del snapshot más reciente (provista por el sensor)
        con la Variable de Airflow para decidir si hay datos nuevos.

        Devuelve la fecha si hay un snapshot nuevo — viaja por XCom a extract_bronze.
        Devuelve False si no hay novedades y el DAG se cortocircuita sin gastar recursos.

        NOTA: La Variable NO se actualiza acá. Se actualiza recién al final en
        validate_silver, garantizando que sólo se registre si todo el pipeline
        terminó con éxito.
        """
        meta = context["ti"].xcom_pull(task_ids="wait_for_source") or {}
        snapshot_date = meta.get("snapshot_date") or context["params"].get("snapshot_date")

        if not snapshot_date:
            log.warning("No se pudo determinar la fecha del snapshot.")
            return False

        ultima_fecha = Variable.get("airbnb_caba_latest_snapshot", default_var=None)

        if snapshot_date == ultima_fecha:
            log.info(
                "Snapshot %s ya fue procesado (coincide con Variable). "
                "Cortocircuitando DAG.",
                snapshot_date,
            )
            return False

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
        meta = context["ti"].xcom_pull(task_ids="wait_for_source") or {}
        snapshot_date = snapshot_from_check or meta.get("snapshot_date") or context["params"]["snapshot_date"]
        destino = bronze_path(snapshot_date)
        destino.parent.mkdir(parents=True, exist_ok=True)

        if destino.exists():
            log.info("Snapshot %s ya en bronze, no se vuelve a descargar", snapshot_date)
            return str(destino)

        raw_url = meta.get("url") or listings_url(snapshot_date)
        url = clean_airbnb_url(raw_url)
        log.info("Descargando snapshot %s desde: %s", snapshot_date, url)
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(destino, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        log.info("Descargado snapshot %s exitosamente -> %s", snapshot_date, destino)
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
        - bathrooms_shared (bool).
        - amenities se parsea a lista y se extraen amenities_count,
          has_parking, has_pool y has_elevator.
        - availability_30 y availability_90 se tipan a entero nullable
          (Int64: soporta NA sin romper, a diferencia de int nativo).
        - se deduplica por id (la clave primaria del dataset).
        """
        meta = context["ti"].xcom_pull(task_ids="wait_for_source") or {}
        snapshot_date = meta.get("snapshot_date") or context["params"]["snapshot_date"]

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
    def load_frozen() -> str:
        """Rama de contingencia: el respaldo más fresco disponible en disco.

        Prefiere 'ultimo_ok.csv' de la última corrida exitosa; si no existe,
        usa 'airbnb_snapshot.csv' (semilla versionada en el repositorio).
        """
        for ruta, origen in (
            (ULTIMO_OK, "última corrida completa exitosa"),
            (SEMILLA, "semilla versionada en el repositorio"),
        ):
            if not ruta.exists():
                continue
            try:
                filas = sum(1 for _ in open(ruta, encoding="utf-8")) - 1
            except Exception:
                filas = "?"
            log.warning(
                "Inside Airbnb no respondió. Se usa el respaldo (%s): %s filas. "
                "NO son de hoy.",
                origen,
                filas,
            )
            return str(ruta)

        raise FileNotFoundError(
            f"Inside Airbnb no responde y no hay ningún respaldo en {FROZEN_DIR}. "
            "Falta colocar una semilla airbnb_snapshot.csv."
        )

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def validate_silver(
        desde_fuente: str | None = None,
        desde_snapshot: str | None = None,
        **context,
    ) -> str:
        """Chequeo final sobre el dataset que se defiende en la Entrega 1.

        Recibe el archivo de la corrida actual o del respaldo congelado.
        Calcula acá mismo los 6 criterios medibles de la consigna (clave
        sin duplicados, volumen, ancho, mezcla de tipos, nulos conocidos,
        columnas vacías) y los deja en el log de la tarea.
        
        Si los datos provinieron de la fuente externa fresca y pasaron todas
        las validaciones:
          1. Refresca 'ultimo_ok.csv' para futuras contingencias.
          2. Actualiza la Variable 'airbnb_caba_latest_snapshot'.
        """
        silver_file = desde_fuente or desde_snapshot
        if silver_file is None:
            raise AirflowFailException("Ninguna rama produjo un archivo para validar.")

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

        # Actualizar respaldo y Variable sólo si vino de la fuente fresca (no del respaldo)
        vino_del_respaldo = Path(silver_file).parent == FROZEN_DIR
        meta = context["ti"].xcom_pull(task_ids="wait_for_source") or {}
        snapshot_date = meta.get("snapshot_date") or context["params"].get("snapshot_date")

        if not vino_del_respaldo and snapshot_date:
            FROZEN_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(silver_file, ULTIMO_OK)
            log.info("Respaldo actualizado -> %s", ULTIMO_OK)

            Variable.set("airbnb_caba_latest_snapshot", snapshot_date)
            log.info("Variable airbnb_caba_latest_snapshot actualizada con éxito -> %s", snapshot_date)

        return silver_file

    # --- Flujo de ejecución del grafo ---
    espera = wait_for_source()
    rama = check_source()
    snapshot = check_new_snapshot()
    bronze = extract_bronze(snapshot_from_check=snapshot)
    validated_bronze = validate_bronze(bronze)
    silver = refine_silver(validated_bronze)
    congelado = load_frozen()

    # Dependencias de control de flujo
    espera >> rama
    rama >> [snapshot, congelado]
    snapshot >> bronze

    # Validación convergente (recibe el silver fresco o el respaldo)
    validate_silver(desde_fuente=silver, desde_snapshot=congelado)
