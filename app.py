"""¿Cuánto cobrar por noche en CABA? — Entrega 4, grupo 5K09-06.

Corre con:  streamlit run app.py
Solo lee lo que generó build_app.py (app_data/); no entrena nada.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.isotonic import IsotonicRegression

import comodidades as CM
import estilo as E
import mapa as MP
from transformadores import COLUMNAS  # noqa: F401  (además, joblib necesita este módulo para cargar el modelo)

alt.data_transformers.disable_max_rows()
DATA = Path(__file__).parent / "app_data"
st.set_page_config(page_title="Precio Airbnb CABA", page_icon="🏠", layout="centered")


@st.cache_resource
def cargar_modelo():
    return joblib.load(DATA / "modelo.joblib")


@st.cache_data
def cargar_datos():
    m = json.loads((DATA / "metricas.json").read_text(encoding="utf-8"))
    pub = pd.read_parquet(DATA / "publicaciones.parquet")
    pub["precio_txt"] = pub["price"].map(E.pesos)
    pub["pred_txt"] = pub["pred"].map(E.pesos)
    pub["error_txt"] = pub["error_rel"].map(lambda x: ("+" if x > 0 else "") + E.pct(x, 0))
    return m, pub


M, PUB = cargar_datos()
BARRIOS_GEO = st.cache_data(MP.leer_barrios)()
CORTES_PRECIO = [round(q, -3) for q in PUB["price"].quantile([0.2, 0.4, 0.6, 0.8])]
CLASES_PRECIO = E.clases_precio(CORTES_PRECIO)
UMBRAL_PUNTOS = 4000   # más publicaciones que esto: el mapa agrupa en celdas (dibujar decenas de miles traba el navegador)
MOD = cargar_modelo()
T, D, P = M["test"], M["datos"], M["particion"]
Q = {q["quintil"]: q for q in M["quintiles"]}
centro = [Q[k]["err_rel_mediano"] for k in ("2", "3", "4")]


def grafico(ch):
    st.altair_chart(ch, width="stretch", theme=None)


# =============================================================== PORTADA
st.caption(f"Ciencia de Datos · UTN FRM · 2026 · Grupo 5K09-06 · datos de Inside Airbnb, snapshot {M['snapshot']}")
st.title("¿A qué precio publicar un alquiler temporario en CABA?")
st.markdown(
    f"#### Con solo la estructura y la ubicación de un departamento, el modelo sugiere su precio por noche "
    f"con un error mediano del **{E.pct(T['modelo']['err_rel_mediano'], 0)}**, y en el 60 % central del mercado "
    f"el error baja a **{E.pct(min(centro), 0)}–{E.pct(max(centro), 0)}**.")
c1, c2, c3 = st.columns(3)
c1.metric("Error relativo mediano (test)", E.pct(T["modelo"]["err_rel_mediano"]),
          f"{E.pct(T['modelo']['err_rel_mediano'] - T['piso']['err_rel_mediano'])} vs. el piso", delta_color="inverse")
c2.metric("Error promedio por noche (test)", E.pesos(T["modelo"]["MAE"]),
          f"−{E.pesos(T['piso']['MAE'] - T['modelo']['MAE'])} vs. el piso", delta_color="off")
c3.metric("Publicaciones analizadas", E.entero(D['filas']), f"{D['barrios']} barrios", delta_color="off")
st.caption("El piso es predecir siempre la mediana de train, sin mirar ninguna columna.")
st.divider()

# =============================================================== TALLO 1 · LOS DATOS
lo, hi = P["limite_inf"], P["limite_sup"]
st.caption("1 · LOS DATOS")
fuera = 1 - (P["train"] + P["test"]) / (P["train_antes"] + P["test_antes"])
st.header(f"Un {E.pct(fuera, 0)} de precios extremos —hasta ${E.num(D['precio_max'] / 1e6, 1)} millones la noche— "
          f"esconde un mercado que va de {E.pesos(lo)} a {E.pesos(hi)}")

precios = pd.DataFrame({"price": M["histograma_precios"]})
base = alt.Chart(precios).transform_calculate(
    log_precio="log(datum.price) / LN10",
    tramo=f"datum.price < {lo} || datum.price > {hi} ? 'Fuera del rango modelado' : 'Dentro del rango modelado'")
barras = base.transform_bin("bin_ini", "log_precio", bin=alt.Bin(maxbins=70)).mark_bar().encode(
    x=alt.X("bin_ini:Q", title="Precio por noche (escala logarítmica)", scale=alt.Scale(zero=False),
            axis=alt.Axis(values=[3, 4, 5, 6, 7, 8], labelExpr=E.EJE_PESOS_LOG10)),
    x2="bin_ini_end:Q",
    y=alt.Y("count():Q", title="Publicaciones (escala raíz cuadrada)", scale=alt.Scale(type="sqrt"),
            axis=alt.Axis(labelExpr=E.EJE_ENTERO)),
    color=alt.Color("tramo:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=["Dentro del rango modelado", "Fuera del rango modelado"],
                                    range=[E.NEUTRO, E.CLARO])),
    tooltip=[alt.Tooltip("count():Q", title="publicaciones")])
limites = pd.DataFrame({"x": np.log10([lo, D["precio_mediano"], hi]),
                        "txt": [f"P1 de train: {E.pesos(lo)}", f"mediana: {E.pesos(D['precio_mediano'])}",
                                f"P99 de train: {E.pesos(hi)}"],
                        "ypx": [12, 28, 44], "alinear": ["left", "left", "left"]})
reglas = alt.Chart(limites).mark_rule(strokeDash=[4, 3], color=E.MARCA).encode(x="x:Q")
textos = (alt.Chart(limites).transform_filter("datum.alinear == 'left'").mark_text(align="left", dx=4, fontSize=11)
          .encode(x="x:Q", y=alt.Y("ypx:Q", scale=None), text="txt:N")
          + alt.Chart(limites).transform_filter("datum.alinear == 'right'").mark_text(align="right", dx=-4, fontSize=11)
          .encode(x="x:Q", y=alt.Y("ypx:Q", scale=None), text="txt:N"))
grafico((barras + reglas + textos).properties(height=320))
st.markdown(
    f"**Qué mirar:** la mediana es {E.pesos(D['precio_mediano'])} y casi todo el mercado entra en la campana oscura. "
    f"Las barritas claras de los costados son publicaciones de {E.pesos(D['precio_min'])} o de "
    f"{E.pesos(D['precio_max'])} la noche: errores de carga que, sin la escala logarítmica, aplastan todo el gráfico "
    f"(asimetría {E.num(D['asimetria_precio'])}; con logaritmo, {E.num(D['asimetria_log'])}).")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**De dónde salen.** [Inside Airbnb](https://insideairbnb.com/get-the-data/), listados de Buenos Aires, snapshot del
**{M['snapshot']}**. Los baja un DAG de Apache Airflow (`pipeline/dags/airbnb_ingest_V3.py`) con estructura medallón:

1. `wait_for_source` (sensor, cada 5 min hasta 30 min) → `check_source`: si la fuente no responde, usa el último respaldo válido (`load_frozen`).
2. `extract_bronze`: descarga `listings.csv.gz` tal cual.
3. `validate_bronze` / `refine_silver`: limpia `price` (`"$86,206.29"` → número), reconstruye `bathrooms` desde
   `bathrooms_text` (la columna original viene multiplicada ×10), arma `amenities_count` y los `has_*`.
4. `validate_silver`: controla los criterios de calidad y recién ahí actualiza la variable del último snapshot.

**Qué es una fila.** Una publicación activa de alquiler temporario en CABA. Silver: **{E.entero(D['filas'])} filas**,
{D['columnas']} columnas, {D['barrios']} barrios.

**Qué se filtró, paso por paso.**

| Paso | Filas | Por qué |
|---|---|---|
| Bronze (crudo de Inside Airbnb) | ver log de `extract_bronze` | — |
| Sin `price` nulo | ver log de `refine_silver` | sin precio no hay objetivo |
| Sin duplicados de `id` | **{E.entero(D['filas'])}** | una publicación, una fila |
| Partición por ubicación (80/20) | train {E.entero(P['train_antes'])} · test {E.entero(P['test_antes'])} | una ubicación nunca queda de los dos lados |
| Poda P1–P99 calculada **solo con train** | train **{E.entero(P['train'])}** · test **{E.entero(P['test'])}** | precios fuera de {E.pesos(lo)}–{E.pesos(hi)} son errores de carga o casos únicos |

**Columnas descartadas por fuga** (no existen cuando se publica por primera vez): `number_of_reviews`,
`review_scores_rating`, `availability_30/90/365`. Medidas igual en la Entrega 3: bajarían el error, pero
reflejan la demanda *después* de haber puesto el precio.

**Nulos estructurales.** `bedrooms` tiene {E.entero(D['nulos_bedrooms'])} nulos: son monoambientes, se completan con 0.

**Cómo se construyó el gráfico.** Histograma de `price` sobre todas las filas de silver. Transformación:
`log10(price)` y bins de ese logaritmo; el eje se rotula de vuelta en pesos. El eje vertical usa raíz cuadrada para
que los pocos casos extremos se vean (en escala lineal miden menos de un píxel).
""")
st.divider()

# =============================================================== TALLO 2 · LO QUE ENCONTRAMOS
H4, H2, H3 = M["h4"], M["h2"], M["h3"]
st.caption("2 · LO QUE ENCONTRAMOS")
st.header("La pileta sí cobra un plus, pero solo aparece al comparar departamentos del mismo tamaño en el mismo barrio")

h4 = pd.DataFrame(H4["barrios"])
largo = h4.melt(id_vars=["barrio", "mediana_con", "mediana_sin", "publicaciones_2p"],
                value_vars=["sep_solo_barrio", "sep_barrio_y_tamano"], var_name="medida", value_name="sep")
largo["medida"] = largo["medida"].map({"sep_solo_barrio": "Mismo barrio (cualquier tamaño)",
                                       "sep_barrio_y_tamano": "Mismo barrio y 2 huéspedes"})
largo["premio"] = (largo["mediana_con"] / largo["mediana_sin"] - 1).map(lambda x: ("+" if x > 0 else "") + E.pct(x, 0))
orden = h4.sort_values("sep_barrio_y_tamano", ascending=False)["barrio"].tolist()
y = alt.Y("barrio:N", sort=orden, title=None)
cortes = pd.DataFrame({"x": [0.2, 0.8], "txt": ["0,2: de rojo a amarillo", "0,8: de amarillo a verde"]})
zonas = (alt.Chart(cortes).mark_rule(color="#999", strokeDash=[2, 2]).encode(x="x:Q")
         + alt.Chart(cortes).mark_text(align="left", dx=4, fontSize=11, color="#666", baseline="middle", fontStyle="italic").encode(
             x="x:Q", y=alt.value(len(h4) * 24 + 40 - 13), text="txt:N"))
linea = alt.Chart(h4).mark_rule(color=E.CLARO, strokeWidth=2).encode(y=y, x="sep_solo_barrio:Q", x2="sep_barrio_y_tamano:Q")
puntos = alt.Chart(largo).mark_circle(size=90, opacity=1).encode(
    y=y, x=alt.X("sep:Q", title="Separación con vs. sin pileta (desvíos de log precio)"),
    color=alt.Color("medida:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=["Mismo barrio (cualquier tamaño)", "Mismo barrio y 2 huéspedes"],
                                    range=[E.PILETA.range[0], E.PILETA.range[1]])),
    tooltip=["barrio", "medida", alt.Tooltip("sep:Q", format=".2f", title="separación"),
             alt.Tooltip("premio:N", title="plus de precio con pileta (2 huéspedes)"),
             alt.Tooltip("publicaciones_2p:Q", title="publicaciones de 2 huéspedes")])
grafico((zonas + linea + puntos).properties(height=24 * len(h4) + 40))
st.markdown(
    f"**Qué mirar:** cada fila es un barrio. El punto gris compara con y sin pileta mezclando tamaños: la mayoría "
    f"queda a la izquierda del corte de 0,2 (promedio {E.num(H4['sep_solo_barrio'])} en estos {len(h4)} barrios). El punto turquesa compara "
    f"solo departamentos para 2 huéspedes: casi todos saltan a la derecha (promedio "
    f"**{E.num(H4['sep_barrio_y_tamano'])}**, {H4['barrios_sobre_corte']} de {len(h4)} barrios sobre el corte). "
    f"El tamaño escondía el efecto: los departamentos con pileta son más chicos. Las líneas punteadas son los cortes "
    f"del semáforo de la cátedra: por debajo de 0,2 la diferencia es despreciable (rojo), sobre 0,8 es grande (verde).")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**Hipótesis que se sostuvo — H4, la pileta.** Solo departamentos enteros. A nivel ciudad la separación es
{E.num(H4['sep_global'])}; dentro de cada barrio, mezclando tamaños, cae a {E.num(H4['sep_solo_barrio'])} (rojo).
Pero los que tienen pileta son más chicos ({E.num(H4['tamano_con']['huespedes'])} huéspedes en promedio y
{E.pct(H4['tamano_con']['monoamb'], 0)} de monoambientes, contra {E.num(H4['tamano_sin']['huespedes'])} y
{E.pct(H4['tamano_sin']['monoamb'], 0)} sin pileta: torres nuevas con amenities). Fijando 2 huéspedes dentro del barrio la
separación sube a **{E.num(H4['sep_barrio_y_tamano'])}** (amarillo). Por eso el modelo lleva la columna
`PiletaPorBarrio`.

**Hipótesis que no se sostuvo — H2, el Superhost cobra más.** Mediana {E.pesos(H2['mediana_super'])} contra
{E.pesos(H2['mediana_no_super'])}, pero la separación es **{E.num(H2['sep_global'])}** (rojo) y dentro de habitaciones
privadas {E.num(H2['sep_hab_privada'])}. Los Superhosts tienen más departamentos enteros y en barrios caros; a igual
propiedad, el sello no permite cobrar más. En la Entrega 3, agregar `host_is_superhost` al modelo no bajó el error
más que el desvío entre pliegues.

**Apoyo — H3, tipo de habitación.** Departamento entero contra habitación privada: separación {E.num(H3['sep'])}
(verde). Con 2 huéspedes, {E.pesos(H3['mediana_entero_2p'])} contra {E.pesos(H3['mediana_privada_2p'])}.

**Cómo se construyó el gráfico.** *Separación* = diferencia de medias de `log(price)` dividida por el desvío
combinado de los dos grupos (con / sin pileta). Se calcula por barrio (mínimo 30 publicaciones por grupo; 20 cuando
se fija 2 huéspedes), y el promedio se pondera por cantidad de publicaciones. Para la medida con 2 huéspedes se
podan los precios fuera de P1–P99. Cortes del semáforo de la cátedra: 0,2 y 0,8.
""")
st.divider()

# =============================================================== TALLO 3 · EL MODELO
st.caption("3 · EL MODELO")
st.header(f"El modelo acierta en el centro del mercado y empuja los extremos hacia el medio: "
          f"sobreestima las más baratas y subestima las más caras")
qdf = pd.DataFrame(M["quintiles"])
qdf["etq"] = qdf.apply(lambda r: f"{'+' if r.sesgo_mediano > 0 else ''}{E.pct(r.sesgo_mediano, 0)}", axis=1)
qdf["med_txt"] = qdf["precio_mediano"].map(E.pesos)
xq = alt.X("quintil:N", title="Quintil de precio real (test)", axis=alt.Axis(labelAngle=0))
cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#999").encode(y="y:Q")
rango = alt.Chart(qdf).mark_rule(strokeWidth=3, color=E.CLARO).encode(x=xq, y="sesgo_p25:Q", y2="sesgo_p75:Q")
pto = alt.Chart(qdf).mark_circle(size=220, opacity=1, stroke="#555").encode(
    x=xq, y=alt.Y("sesgo_mediano:Q", title="Error con signo: (predicho − real) / real", axis=alt.Axis(format="+%")),
    color=alt.Color("sesgo_mediano:Q", scale=E.ERROR, legend=None),
    tooltip=[alt.Tooltip("quintil:N"), alt.Tooltip("med_txt:N", title="precio mediano"),
             alt.Tooltip("sesgo_mediano:Q", format="+.1%", title="error mediano con signo"),
             alt.Tooltip("err_rel_mediano:Q", format=".1%", title="error relativo mediano (sin signo)"),
             alt.Tooltip("publicaciones:Q")])
etq = alt.Chart(qdf).mark_text(dx=22, fontSize=12, fontWeight="bold").encode(x=xq, y="sesgo_mediano:Q", text="etq:N")
grafico((cero + rango + pto + etq).properties(height=320))
st.markdown(
    f"**Qué mirar:** el punto es el error mediano de cada quinto del mercado, y la barra gris va del cuartil 1 al 3. "
    f"En el centro el punto está pegado al cero. En el quinto más barato (mediana {E.pesos(Q['1 · más baratas']['precio_mediano'])}) "
    f"el modelo pide de más (naranja); en el más caro ({E.pesos(Q['5 · más caras']['precio_mediano'])}) pide de menos "
    f"(violeta). Lo que separa un departamento de lujo de uno común —vista, diseño, fotos— no está en las columnas.")

with st.expander("Cómo lo sabemos"):
    comp = pd.DataFrame(M["comparacion"])
    st.markdown(f"""
**Qué predice y con qué métrica.** Regresión sobre `log(price)` (por la asimetría de la sección 1), pero toda
métrica se calcula **en pesos**, volviendo con `exp()`. Métrica principal: **error absoluto medio (MAE) en pesos**,
porque se lee directo ("me equivoco en $X por noche") y no lo dominan unos pocos casos extremos. Se reportan además
el error relativo `|predicho − real| / real` y el R².

**El piso.** Predecir siempre la mediana de train: MAE {E.pesos(comp.iloc[0]['MAE'])} en validación,
{E.pesos(T['piso']['MAE'])} en test.

**Validación.** Test del 20 % separado **por ubicación** (`GroupShuffleSplit`): {E.entero(D['filas_en_ubic_compartida'])}
publicaciones comparten coordenadas exactas con otra (hasta {D['grupo_max']} en un mismo punto) y ahí el precio varía
menos (desvío {E.num(D['desvio_log_dentro_ubic'])} contra {E.num(D['desvio_log_total'])}). Dentro de train,
`GroupKFold` de 5 pliegues con el mismo criterio. Al azar vs. por ubicación (regresión lineal):
{E.pesos(M['validacion']['azar']['MAE'])} contra {E.pesos(M['validacion']['ubicacion']['MAE'])} — diferencia menor al
desvío, así que las ubicaciones repetidas no inflan el número, pero igual se valida por ubicación.

**Comparación (validación cruzada por ubicación, mismo pipeline para todos).**
""")
    tabla = comp.assign(**{
        "MAE en pesos": comp["MAE"].map(E.pesos), "± desvío": comp["desvio"].map(E.pesos),
        "error rel. mediano": comp["err_rel_mediano"].map(E.pct), "R² (log)": comp["R2_log"].map(E.num)})
    st.dataframe(tabla[["modelo", "MAE en pesos", "± desvío", "error rel. mediano", "R² (log)"]],
                 hide_index=True, width="stretch")
    g = M["grilla"]
    st.markdown(f"""
**El elegido y su ajuste.** Regla declarada antes de mirar: un modelo complejo entra solo si le gana al lineal por
más que el desvío. Boosting y bosque lo hacen y empatan entre sí; se elige **gradient boosting**
(`HistGradientBoostingRegressor`): más rápido, más liviano y maneja categorías y faltantes. Grilla de
{g['combinaciones']} combinaciones optimizando MAE en pesos: `learning_rate={g['mejor']['learning_rate']}`,
`max_iter={g['mejor']['max_iter']}`, `max_leaf_nodes={g['mejor']['max_leaf_nodes']}`,
`min_samples_leaf={g['mejor']['min_samples_leaf']}`. Las 5 mejores quedan dentro de
{E.pesos(g['top'][4]['MAE'] - g['top'][0]['MAE'])} entre sí: empatadas.

**Pipeline.** `ReglasDeDominio` (nulos con reglas del análisis) → `TamanoRelativo` (personas por dormitorio,
baños por persona, monoambiente) → `PiletaPorBarrio` (H4) → `PrecioLocal` (mediana de log precio de los 15 vecinos
de train más cercanos, calculada fuera de pliegue para que nadie sea vecino de sí mismo) → boosting.

**Test, una sola vez.** MAE **{E.pesos(T['modelo']['MAE'])}** (piso {E.pesos(T['piso']['MAE'])}), error relativo
mediano **{E.pct(T['modelo']['err_rel_mediano'])}**, R² {E.num(T['modelo']['R2_log'], 3)} en log y
{E.num(T['modelo']['R2_pesos'], 3)} en pesos. Validación y test coinciden: no hubo fuga.

**Dónde se equivoca.** En los extremos (gráfico de arriba). Error relativo mediano por quinto:
{" · ".join(f"{q['quintil'].split(' ')[0]}: {E.pct(q['err_rel_mediano'])}" for q in M['quintiles'])}.

**En qué se apoya.** Cuánto sube el MAE en test al mezclar al azar cada columna:
""")
    imp = pd.DataFrame(M["importancia"]).head(10)
    imp["columna"] = imp["columna"].map(lambda c: f"{E.COLUMNA_ES.get(c, c)} ({c})")
    imp["txt"] = imp["aumento_MAE"].map(lambda v: "+" + E.pesos(v))
    grafico(alt.Chart(imp).mark_bar(color=E.NEUTRO).encode(
        x=alt.X("aumento_MAE:Q", title="Aumento del MAE en pesos", axis=alt.Axis(labelExpr=E.EJE_PESOS)),
        y=alt.Y("columna:N", sort="-x", title=None),
        tooltip=["columna", alt.Tooltip("txt:N", title="aumento del MAE")]).properties(height=260))
st.divider()

# =============================================================== COPA · EXPLORÁ
st.caption("4 · EXPLORÁ")
st.header("Explorá las publicaciones y lo que dijo el modelo de cada una")
st.markdown("El mapa ubica cada publicación y el color responde una pregunta: **¿dónde están las caras y las baratas?** "
            "(precio real), **¿el modelo dibuja el mismo mapa?** (precio estimado) o **¿dónde se equivoca?** "
            "(acierto del modelo). A la derecha, esas mismas publicaciones con su precio real contra el sugerido. "
            "Arrastrá sobre el mapa para quedarte con una zona; doble clic borra la selección.")

@st.fragment
def explora():
    """Los controles de Explorá vuelven a correr solo esta sección, no toda la página."""
    f1, f2 = st.columns(2)
    barrios_sel = f1.multiselect("Barrios", sorted(PUB["neighbourhood_cleansed"].unique()),
                                 placeholder="Todos los barrios")
    tipos_sel = f2.multiselect("Tipo de alojamiento", list(E.ROOM_TYPE.domain), default=["Entire home/apt", "Private room"],
                               format_func=lambda t: E.TIPO_ES[t], placeholder="Todos los tipos")
    f3, f4 = st.columns(2)
    cap_sel = f3.slider("Capacidad (huéspedes)", 1, int(PUB["accommodates"].max()), (1, int(PUB["accommodates"].max())))
    color_por = f4.radio("Color", ["Precio real", "Precio estimado", "Acierto del modelo"], horizontal=True)

    vista = PUB[PUB["accommodates"].between(*cap_sel)]
    if barrios_sel:
        vista = vista[vista["neighbourhood_cleansed"].isin(barrios_sel)]
    if tipos_sel:
        vista = vista[vista["room_type"].isin(tipos_sel)]

    if vista.empty:
        st.info("No hay publicaciones con esos filtros.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Publicaciones", E.entero(len(vista)))
        k2.metric("Precio mediano", E.pesos(vista["price"].median()))
        k3.metric("Error relativo mediano", E.pct(vista["error_rel"].abs().median()))

        zona = alt.selection_interval(name="zona")
        # Color en pocas clases FIJAS (no cambian con los filtros), para que cada color responda una pregunta.
        if color_por == "Acierto del modelo":
            expr_clase = E.expr_clases("datum.error", E.CORTES_ERROR, E.CLASES_ERROR)
            color = alt.Color("clase:N", title="Acierto del modelo", sort=E.CLASES_ERROR,
                              scale=alt.Scale(domain=E.CLASES_ERROR, range=E.COLORES_ERROR),
                              legend=alt.Legend(orient="bottom", direction="vertical", titleLimit=400, labelLimit=300))
        else:
            campo = "datum.precio" if color_por == "Precio real" else "datum.pred"
            expr_clase = E.expr_clases(campo, CORTES_PRECIO, CLASES_PRECIO)
            color = alt.Color("clase:N", title=f"{color_por} por noche (quintiles del precio real de CABA)",
                              sort=CLASES_PRECIO, scale=alt.Scale(domain=CLASES_PRECIO, range=E.COLORES_PRECIO),
                              legend=alt.Legend(orient="bottom", direction="vertical", columns=1, titleLimit=400, labelLimit=300))
        detalle = [alt.Tooltip("barrio:N"), alt.Tooltip("tipo:N"), alt.Tooltip("huespedes:Q", title="huéspedes"),
                   alt.Tooltip("dormitorios:Q"), alt.Tooltip("pileta_txt:N", title="pileta"),
                   alt.Tooltip("precio_txt:N", title="precio real"),
                   alt.Tooltip("pred_txt:N", title="precio que sugiere el modelo"),
                   alt.Tooltip("error_txt:N", title="error"), alt.Tooltip("conjunto:N", title="predicción hecha en")]
        # Datos livianos: nombres cortos, números redondeados; los textos del tooltip los arma Vega.
        datos = pd.DataFrame({
            "barrio": vista["neighbourhood_cleansed"], "tipo": vista["room_type"].map(E.TIPO_ES), "huespedes": vista["accommodates"],
            "dormitorios": vista["bedrooms"].fillna(0).astype(int), "pileta": vista["has_pool"].astype(int),
            "precio": vista["price"].round(0), "pred": vista["pred"].round(0), "error": vista["error_rel"].round(3),
            "conjunto": vista["conjunto"].str.split(" ").str[0],
            "lon": vista["longitude"].round(5), "lat": vista["latitude"].round(5)})
        base = alt.Chart(datos).transform_calculate(
            precio_txt=E.EJE_PESOS.replace("datum.value", "datum.precio"),
            pred_txt=E.EJE_PESOS.replace("datum.value", "datum.pred"),
            error_txt="(datum.error > 0 ? '+' : '') + format(datum.error, '.0%')",
            pileta_txt="datum.pileta ? 'sí' : 'no'",
            lx="log(datum.precio) / LN10", ly="log(datum.pred) / LN10").transform_calculate(clase=expr_clase)
        # Zoom: el menú de barrios fija el dominio de las escalas x (longitud) e y (latitud).
        dom_lon, dom_lat = MP.encuadre(BARRIOS_GEO, barrios_sel)
        esc_lon = alt.Scale(domain=dom_lon, zero=False, nice=False)
        esc_lat = alt.Scale(domain=dom_lat, zero=False, nice=False)
        ex = alt.X("lon:Q", scale=esc_lon, axis=None)
        ey = alt.Y("lat:Q", scale=esc_lat, axis=None)
        fondo = alt.Chart(MP.mosaicos(dom_lon, dom_lat)).mark_image(aspect=False, clip=True).encode(
            x=alt.X("lon0:Q", scale=esc_lon, axis=None), x2="lon1:Q",
            y=alt.Y("lat1:Q", scale=esc_lat, axis=None), y2="lat0:Q", url="url:N")
        lineas = BARRIOS_GEO.assign(elegido=BARRIOS_GEO["barrio"].isin(barrios_sel))
        limites = alt.Chart(lineas).mark_line(clip=True, strokeWidth=0.8).encode(
            x=ex, y=ey, detail="anillo:N", order="orden:Q",
            color=alt.condition("datum.elegido", alt.value("#333333"), alt.value("#9aa3ad")))
        capas = [fondo, limites]
        if barrios_sel:
            nombres = (BARRIOS_GEO[BARRIOS_GEO["barrio"].isin(barrios_sel)].groupby("barrio")[["lon", "lat"]].mean()
                       .reset_index())
            capas.append(alt.Chart(nombres).mark_text(fontSize=12, fontWeight="bold", color="#333", clip=True,
                                                      stroke="white", strokeWidth=3, strokeOpacity=0.8)
                         .encode(x=ex, y=ey, text="barrio:N"))
            capas.append(alt.Chart(nombres).mark_text(fontSize=12, fontWeight="bold", color="#333", clip=True)
                         .encode(x=ex, y=ey, text="barrio:N"))
        if len(datos) > UMBRAL_PUNTOS:
            # Panorama: con muchas publicaciones, una grilla de celdas en vez de un punto por publicación.
            # Cada celda: cuántas hay y sus medianas de precio real, precio estimado y error.
            n_celdas = 70
            paso_lon = (dom_lon[1] - dom_lon[0]) / n_celdas
            paso_lat = (dom_lat[1] - dom_lat[0]) / n_celdas
            celdas = (alt.Chart(datos)
                      .transform_calculate(cx=f"floor(datum.lon / {paso_lon})", cy=f"floor(datum.lat / {paso_lat})")
                      .transform_aggregate(n="count()", precio="median(precio)", pred="median(pred)",
                                           error="median(error)", groupby=["cx", "cy"])
                      .transform_calculate(lon=f"(datum.cx + 0.5) * {paso_lon}", lat=f"(datum.cy + 0.5) * {paso_lat}",
                                           precio_txt=E.EJE_PESOS.replace("datum.value", "datum.precio"),
                                           pred_txt=E.EJE_PESOS.replace("datum.value", "datum.pred"),
                                           error_txt="(datum.error > 0 ? '+' : '') + format(datum.error, '.0%')")
                      .transform_calculate(clase=expr_clase)
                      .mark_square(size=(330 / n_celdas) ** 2 * 1.1, clip=True)
                      .encode(x=ex, y=ey, color=alt.condition(zona, color, alt.value(E.CLARO)),
                              # opacidad = cuántas publicaciones hay: las celdas con 1 o 2 casos casi no se ven
                              opacity=alt.Opacity("n:Q", scale=alt.Scale(domain=[1, 10], range=[0.25, 0.95], clamp=True),
                                                  legend=None),
                              tooltip=[alt.Tooltip("n:Q", title="publicaciones en la celda"),
                                       alt.Tooltip("precio_txt:N", title="precio real mediano"),
                                       alt.Tooltip("pred_txt:N", title="precio estimado mediano"),
                                       alt.Tooltip("error_txt:N", title="error mediano del modelo")])
                      .add_params(zona))
            capa_datos, titulo = celdas, "Panorama: cada cuadrito agrupa las publicaciones de esa zona"
        else:
            capa_datos = base.mark_circle(size=18, opacity=0.8, clip=True).encode(
                x=ex, y=ey, color=alt.condition(zona, color, alt.value(E.CLARO)), tooltip=detalle).add_params(zona)
            titulo = "Dónde está cada publicación"
        mapa = alt.layer(*capas[:2], capa_datos, *capas[2:]).properties(width=330, height=330, title=titulo)
        # Real vs. sugerido, en log10 de pesos. Muestra lo que dejan los filtros; la zona del mapa lo achica.
        lim = list(np.log10([datos[["precio", "pred"]].min().min() * 0.9, datos[["precio", "pred"]].max().max() * 1.1]))
        marcas = [float(v) for v in np.log10([20_000, 30_000, 50_000, 100_000, 200_000, 400_000, 800_000])
                  if lim[0] <= v <= lim[1]]
        sx = alt.Scale(domain=lim, zero=False, nice=False)
        eje = dict(labelExpr=E.EJE_PESOS_LOG10, values=marcas)
        # La diagonal (predicho = real) va en negro y arriba de los datos: ningún color de los datos es negro.
        diag = alt.Chart(pd.DataFrame({"v": lim})).mark_line(color=E.MARCA, strokeWidth=2, strokeDash=[6, 4]).encode(
            x=alt.X("v:Q", scale=sx), y=alt.Y("v:Q", scale=sx))
        if len(datos) > UMBRAL_PUNTOS:
            # Muchas publicaciones: grilla de densidad (cuántas caen en cada combinación real/sugerido).
            disp = (base.transform_filter(zona)
                    .transform_bin("bx", "lx", bin=alt.Bin(step=0.04)).transform_bin("by", "ly", bin=alt.Bin(step=0.04))
                    .transform_aggregate(n="count()", precio="median(precio)", pred="median(pred)",
                                         error="median(error)", groupby=["bx", "bx_end", "by", "by_end"])
                    .transform_calculate(clase=expr_clase,
                                         precio_txt=E.EJE_PESOS.replace("datum.value", "datum.precio"),
                                         pred_txt=E.EJE_PESOS.replace("datum.value", "datum.pred"))
                    .mark_rect().encode(
                        x=alt.X("bx:Q", title="Precio real", scale=sx, axis=alt.Axis(**eje)), x2="bx_end:Q",
                        y=alt.Y("by:Q", title="Precio que sugiere el modelo", scale=sx, axis=alt.Axis(**eje)),
                        y2="by_end:Q",
                        color=color,
                        opacity=alt.Opacity("n:Q", scale=alt.Scale(type="log", domain=[1, 50], range=[0.25, 1], clamp=True),
                                            legend=None),
                        tooltip=[alt.Tooltip("n:Q", title="publicaciones"),
                                 alt.Tooltip("precio_txt:N", title="precio real (mediana)"),
                                 alt.Tooltip("pred_txt:N", title="precio sugerido (mediana)")]))
        else:
            disp = base.transform_filter(zona).mark_circle(size=16, opacity=0.8).encode(
                x=alt.X("lx:Q", title="Precio real", scale=sx, axis=alt.Axis(**eje)),
                y=alt.Y("ly:Q", title="Precio que sugiere el modelo", scale=sx, axis=alt.Axis(**eje)),
                color=color, tooltip=detalle)
        grafico(alt.hconcat(mapa, (disp + diag).properties(width=330, height=330,
                                                           title="Real vs. sugerido (sobre la diagonal: pide de más)"))
                .resolve_scale(color="shared"))
        if len(datos) > UMBRAL_PUNTOS:
            st.info(f"Con más de {E.entero(UMBRAL_PUNTOS)} publicaciones el mapa las agrupa en cuadritos. "
                    "Elegí uno o más barrios (o achicá los filtros) para ver cada publicación con lo que dijo el modelo.",
                    icon="🔎")
        st.caption("En train, cada predicción la hizo un modelo que no había visto esa publicación (fuera de pliegue); "
                   "en test, el modelo final. Así ningún punto está predicho 'mirándose a sí mismo'. "
                   + MP.ATRIBUCION + ".")


explora()
st.divider()

# =============================================================== COPA · PROBÁ EL MODELO
st.caption("5 · PROBÁ EL MODELO")
st.header("Cargá un departamento y mirá qué precio sugiere el modelo")
st.markdown("El precio se recalcula con cada cambio. Pasá el mouse por el ícono ⓘ de cada campo para ver qué significa.")
R = M["referencia"]
rg = R["rangos"]
CLAVE_AM = {en: f"am_{i}" for i, (_, en, _, _) in enumerate(CM.TODAS)}
CLAVE_MOD = {col: f"mod_{col}" for col, *_ in CM.MODELO}
DEFECTO = {**{CLAVE_AM[en]: d for _, en, _, d in CM.TODAS}, **{CLAVE_MOD[c]: d for c, _, d, _ in CM.MODELO}}


def marcar(modo):
    """Botones: 'todas', 'ninguna' o 'tipicas' (las que tiene al menos el 45 % de las publicaciones)."""
    for clave, defecto in DEFECTO.items():
        st.session_state[clave] = modo == "todas" or (modo == "tipicas" and defecto)


# Cuántas publicaciones tienen cada cantidad de comodidades: pesos de la regresión isotónica.
PESO_AMEN = PUB["amenities_count"].value_counts()


def precio_monotono(caso):
    """Regla de la aplicación (el modelo no cambia): a igualdad de lo demás, más comodidades nunca dan un precio
    menor. Se calcula la curva del modelo de 0 al máximo de comodidades para este caso y se la reemplaza por la
    curva creciente más cercana (regresión isotónica), pesando cada cantidad por cuántas publicaciones la tienen:
    donde el modelo vio pocos casos, su ruido pesa poco."""
    grilla = np.arange(0, int(rg["amenities_count"]["max"]) + 1)
    filas = pd.concat([caso] * len(grilla), ignore_index=True).assign(amenities_count=grilla)
    log_p = MOD.predict(filas)
    pesos = PESO_AMEN.reindex(grilla, fill_value=0).to_numpy() + 1
    curva = IsotonicRegression(increasing=True).fit(grilla, log_p, sample_weight=pesos)
    amen = int(caso["amenities_count"].iloc[0])
    return float(np.exp(log_p[amen])), float(np.exp(curva.predict([amen])[0]))


@st.fragment
def proba():
    """Sin formulario: cada cambio recalcula al instante, y solo corre esta sección."""
    a1, a2 = st.columns(2)
    barrio = a1.selectbox("Barrio", sorted(R["barrios"], key=lambda b: -R["barrios"][b]["n"]),
                          help="Barrio de CABA. La ubicación del caso se toma como el centro del barrio; con ella el "
                               "modelo mira el precio de las 15 publicaciones más cercanas.")
    tipo = a2.selectbox("Tipo de alojamiento", list(R["room_types"]), format_func=lambda t: E.TIPO_ES[t],
                        help="Departamento o casa entera: el huésped tiene la unidad para él solo. "
                             "Habitación privada: un cuarto propio, compartiendo el resto con el anfitrión u otros. "
                             "Habitación de hotel: cuarto en un hotel u hostel. "
                             "Habitación compartida: comparte el cuarto con otras personas.")
    b1, b2, b3 = st.columns(3)
    huesp = b1.number_input("Huéspedes", int(rg["accommodates"]["min"]), int(rg["accommodates"]["max"]), 2,
                            help="Cantidad máxima de personas que pueden alojarse.")
    dorm = b2.number_input("Dormitorios", 0, int(rg["bedrooms"]["max"]), 1,
                           help="Ambientes cerrados para dormir. 0 = monoambiente: la cama está en el mismo ambiente "
                                "que el living.")
    noches = b3.number_input("Noches mínimas", int(max(rg["minimum_nights"]["min"], 1)),
                             int(rg["minimum_nights"]["max"]), 2,
                             help="Estadía mínima que exige la publicación. Con 28 o más es un alquiler mensual: en los "
                                  "datos esas publicaciones cobran bastante menos por noche.")
    c1, c2, _ = st.columns(3)
    banos = c1.number_input("Baños", 0.0, float(rg["bathrooms"]["max"]), 1.0, step=0.5, format="%.1f",
                            help="1 = baño completo: inodoro, lavatorio y ducha o bañera. 0,5 = medio baño o toilette: "
                                 "inodoro y lavatorio, sin ducha. Por ejemplo, 1,5 = un baño completo más un toilette.")
    uso_bano = c2.segmented_control("Uso del baño", ["Privado", "Compartido"], default="Privado",
                                    help="Privado: solo para el huésped. Compartido: se comparte con el anfitrión u "
                                         "otros huéspedes (típico de habitaciones privadas). Es una columna del modelo.")
    bano_comp = uso_bano == "Compartido"

    # ------------------------------------------------ comodidades
    for clave, defecto in DEFECTO.items():
        st.session_state.setdefault(clave, defecto)
    st.markdown("**Comodidades que usa el modelo**")
    cols = st.columns(3)
    for i, (col, nombre, _, ayuda) in enumerate(CM.MODELO):
        cols[i % 3].checkbox(nombre, key=CLAVE_MOD[col], help=ayuda)
    n_extra = sum(st.session_state[c] for c in CLAVE_AM.values())
    st.markdown(f"**Otras comodidades** · solo suman a la cantidad de comodidades · marcadas: **{n_extra}** de {len(CM.TODAS)}")
    k1, k2, k3, _ = st.columns([1, 1, 1, 2])
    k1.button("Típicas", on_click=marcar, args=("tipicas",), use_container_width=True,
              help="Marca las 31 que tiene al menos el 45 % de las publicaciones (una publicación típica lista 34). "
                   "Todas / Ninguna incluyen también las comodidades que usa el modelo.")
    k2.button("Todas", on_click=marcar, args=("todas",), use_container_width=True)
    k3.button("Ninguna", on_click=marcar, args=("ninguna",), use_container_width=True)
    for categoria, items in CM.CATEGORIAS.items():
        n_cat = sum(st.session_state[CLAVE_AM[en]] for en, _, _ in items)
        with st.expander(f"{categoria} · {n_cat} de {len(items)}"):
            cols = st.columns(3)
            for i, (en, es, _) in enumerate(items):
                cols[i % 3].checkbox(es, key=CLAVE_AM[en], help=CM.ayuda(en))

    # Igual que el pipeline: amenities_count = cantidad de comodidades (las del modelo + las otras).
    flags = {col: bool(st.session_state[CLAVE_MOD[col]]) for col, *_ in CM.MODELO}
    amen = sum(flags.values()) + sum(st.session_state[c] for c in CLAVE_AM.values())
    st.caption(f"Cantidad de comodidades que recibe el modelo: **{amen}** (una publicación típica lista 34).")

    avisos = []
    for nombre, valor, col in [("huéspedes", huesp, "accommodates"), ("dormitorios", dorm, "bedrooms"),
                               ("baños", banos, "bathrooms"), ("noches mínimas", noches, "minimum_nights"),
                               ("comodidades", amen, "amenities_count")]:
        if valor > rg[col]["p99"] or valor < rg[col]["p01"]:
            avisos.append(f"**{nombre} = {valor:g}** está fuera de lo que tiene el 98 % de las publicaciones de train "
                          f"({rg[col]['p01']:g} a {rg[col]['p99']:g}): el modelo vio muy pocos casos así.")
    if amen > rg["amenities_count"]["max"]:
        avisos.append(f"Ninguna publicación de train tiene más de {rg['amenities_count']['max']:g} comodidades: "
                      f"el modelo usa {rg['amenities_count']['max']:g}.")
        amen = int(rg["amenities_count"]["max"])
    if dorm > huesp:
        avisos.append("Hay más dormitorios que huéspedes: es una combinación rara en los datos.")
    if R["barrios"][barrio]["n"] < 30:
        avisos.append(f"En {barrio} hay solo {R['barrios'][barrio]['n']} publicaciones en train: la estimación es menos firme.")
    if R["room_types"][tipo] < 200:
        avisos.append(f"«{E.TIPO_ES[tipo]}» tiene solo {R['room_types'][tipo]} publicaciones en train.")

    caso = pd.DataFrame([{
        "accommodates": huesp, "bathrooms": banos, "bedrooms": dorm, "amenities_count": amen,
        "minimum_nights": noches, "latitude": R["barrios"][barrio]["lat"], "longitude": R["barrios"][barrio]["lon"],
        **flags, "bathrooms_shared": bano_comp, "room_type": tipo, "neighbourhood_cleansed": barrio}])
    precio_modelo, precio = precio_monotono(caso)
    I = M["intervalo"]
    tramo = int(np.clip(np.searchsorted(I["cortes"], precio) - 1, 0, len(I["p10"]) - 1))
    bajo, alto = precio * I["p10"][tramo], precio * I["p90"][tramo]

    st.divider()
    st.metric("Precio sugerido por noche", E.pesos(precio),
              help="Lo que predice el modelo con una regla de la aplicación: a igualdad de lo demás, más comodidades nunca "
                   "dan un precio menor (regresión isotónica sobre la curva del modelo). El modelo de la Entrega 3 no cambia.")
    if abs(precio / precio_modelo - 1) > 0.005:
        st.caption(f"El modelo solo da {E.pesos(precio_modelo)}. Su curva de precio según la cantidad de comodidades "
                   f"sube y baja donde vio pocos casos; la aplicación la reemplaza por la curva creciente más cercana, "
                   f"y para este caso eso da {E.pesos(precio)}.")
    st.markdown(f"8 de cada 10 publicaciones a las que el modelo les sugirió un precio parecido cobran entre "
                f"**{E.pesos(bajo)}** y **{E.pesos(alto)}**.")
    for a in avisos:
        st.warning(a, icon="⚠️")

    parecidas = PUB[(PUB["neighbourhood_cleansed"] == barrio) & (PUB["room_type"] == tipo)
                    & (PUB["accommodates"] == huesp)]
    if len(parecidas) >= 5:
        st.markdown(f"**Contra el mercado:** las {len(parecidas)} publicaciones de {barrio}, "
                    f"{E.TIPO_ES[tipo].lower()} y {huesp} huéspedes. La línea negra es lo que sugiere el modelo "
                    f"para tu caso; la banda gris, el rango de arriba.")
        p_ch = alt.Chart(parecidas).transform_calculate(log_precio="log(datum.price) / LN10").transform_bin(
            "bin_ini", "log_precio", bin=alt.Bin(maxbins=30)).mark_bar(
            color=E.ROOM_TYPE.range[list(E.ROOM_TYPE.domain).index(tipo)], opacity=0.8).encode(
            x=alt.X("bin_ini:Q", title="Precio por noche (escala logarítmica)", scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelExpr=E.EJE_PESOS_LOG10,
                                  values=list(np.log10([20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000])))),
            x2="bin_ini_end:Q",
            y=alt.Y("count():Q", title="Publicaciones"))
        banda = alt.Chart(pd.DataFrame({"a": [np.log10(bajo)], "b": [np.log10(alto)]})).mark_rect(
            color=E.MARCA, opacity=0.08).encode(x="a:Q", x2="b:Q")
        regla = alt.Chart(pd.DataFrame({"x": [np.log10(precio)]})).mark_rule(color=E.MARCA, strokeWidth=3).encode(x="x:Q")
        grafico((banda + p_ch + regla).properties(height=220))
    else:
        st.info(f"Hay menos de 5 publicaciones de {barrio}, {E.TIPO_ES[tipo].lower()} y {huesp} huéspedes para comparar.")
    st.caption("Los campos no dejan cargar valores fuera del mínimo y máximo vistos en train.")


proba()
