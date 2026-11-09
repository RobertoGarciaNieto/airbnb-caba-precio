# ¿A qué precio publicar un alquiler temporario en CABA?

Proyecto integrador de Ciencia de Datos (UTN FRM, 2026), grupo 5K09-06:
Canzio, Fernández Rossi, García Nieto, Giacomelli, Moyano y Rubiales.

**Aplicación:** https://precio-airbnb-caba.streamlit.app/

## La cadena completa

```
Inside Airbnb ──► pipeline/ (Airflow) ──► data/snapshot=AAAA-MM-DD.csv ──► build_app.py ──► app_data/ ──► app.py
                  Entrega 1                                           Entregas 2 y 3                  Entrega 4
```

| Carpeta / archivo | Qué es |
|---|---|
| `pipeline/` | DAG de Airflow (Astro) `airbnb_ingest_V3.py`: bronze → silver con validaciones y respaldo. |
| `analisis/entrega_2.ipynb` | Análisis exploratorio e hipótesis (H1–H4). |
| `modelo/entrega_3.ipynb` | Modelado: validación por ubicación, comparación, grilla, test. |
| `transformadores.py` | Los transformadores del pipeline del modelo (los mismos del notebook de la Entrega 3). |
| `build_app.py` | Toma el CSV, reentrena con la semilla 42 y genera `app_data/`. |
| `app.py` · `estilo.py` | Aplicación de Streamlit y paleta común de los gráficos (Altair). |
| `mapa.py` · `data/geo/` | Mapa de Explorá: mosaicos de Esri + límites de barrios (GeoJSON de Inside Airbnb, mismo snapshot). |

## Cómo se corre

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python build_app.py               # ~2 min: reproduce la Entrega 3 y escribe app_data/
streamlit run app.py              # abre la app en http://localhost:8501
```

`build_app.py` imprime el MAE de test y avisa si no coincide con el del notebook ($32.959).

## Si cambian los datos

1. Correr el pipeline (`cd pipeline && astro dev start`, disparar `airbnb_caba_ingest_V3`); deja el silver en
   `include/output/V3/silver/airbnb_caba/snapshot=AAAA-MM-DD.csv`.
2. Copiarlo a `data/` y correr `python build_app.py data/snapshot=AAAA-MM-DD.csv`.
3. Hacer commit de `app_data/`: Streamlit Cloud se actualiza solo con el push.

## Publicación

Streamlit Community Cloud → *New app* → este repositorio, rama `main`, archivo `app.py`,
*Advanced settings* → Python 3.12.
