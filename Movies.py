# app.py (solo carga pickled ContentRecommender; no transforma ni entrena en runtime)
import streamlit as st
import pandas as pd
import altair as alt
import pickle
import os
import re
import html
import requests
from io import BytesIO
from urllib.parse import unquote_plus

st.set_page_config(layout="wide", page_title="Recomendador de Películas", initial_sidebar_state="expanded")

# -------------------------
# NAVBAR estilo IMDb (simple)
# -------------------------
NAVBAR_CSS = """
<style>
/* Body */
body {
  background-color: #0f1115;
  color: #ffffff;
}

/* Cabecera grande */
.imdb-navbar {
  background: linear-gradient(90deg, #0b0c0e 0%, #111214 100%);
  padding: 18px 24px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  display: flex;
  align-items: center;
  gap: 18px;
}
.imdb-logo {
  background: #f5c518;
  color: #0b0c0e;
  padding: 6px 10px;
  border-radius: 4px;
  font-weight: 800;
  font-size: 18px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
.imdb-title {
  font-size: 28px;
  font-weight: 700;
  color: #ffffff;
  margin: 0;
}
.imdb-sub {
  color: #cfcfcf;
  font-size: 13px;
  margin-left: 8px;
}

/* Sidebar tweaks */
[data-testid="stSidebar"] {
  background-color: #111217;
}
.stSlider > div > div > input {
  accent-color: #f5c518;
}

/* Card look for top table */
.top-table-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.5);
}
</style>
"""

st.markdown(NAVBAR_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="imdb-navbar">
      <div class="imdb-logo">IMDb</div>
      <div>
        <div class="imdb-title">Recomendador de Películas — Outliers Anónimos</div>
        <div class="imdb-sub">Pre pre Alfa</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,


)

# Estado para mostrar/ocultar el buscador
if "show_search" not in st.session_state:
    st.session_state["show_search"] = False
if "show_top_n" not in st.session_state:
    st.session_state["show_top_n"] = False

query_params = st.query_params
home_select_param = None
if "home_select" in query_params:
    param_val = query_params.get_all("home_select") if hasattr(query_params, "get_all") else query_params.get("home_select")
    if isinstance(param_val, list) and param_val:
        home_select_param = param_val[0]
    elif isinstance(param_val, str):
        home_select_param = param_val
    if home_select_param:
        decoded_title = unquote_plus(str(home_select_param))
        st.session_state["show_search"] = True
        st.session_state["movie_select_autocomplete"] = decoded_title
        st.session_state["show_top_n"] = False
        try:
            del st.query_params["home_select"]
        except Exception:
            pass


# --- Helpers & config ---
DEFAULT_MODEL_PATH = "best_model.pkl"   # <-- Debe contener la instancia ContentRecommender pickleda
TITLE_COL = "original_title"            # Ajusta si es necesario
OMDB_API_URL = "https://www.omdbapi.com/"
OMDB_API_KEY = "5af5a66d"
SPECIAL_POSTERS = {
    "inception": "https://m.media-amazon.com/images/M/MV5BMjExMjkwNTQ0Nl5BMl5BanBnXkFtZTcwNTY0OTk1Mw@@._V1_.jpg",
    "the wolverine": "https://monsterzeronj.wordpress.com/wp-content/uploads/2013/12/the_wolverine_posterus.jpg?w=584",
}

# Import recommender_module (solo para comprobaciones/typing)
import recommender_module as rm

@st.cache_data
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def fetch_omdb_poster(title: str):
    if not title:
        return None
    params = {
        "apikey": OMDB_API_KEY,
        "t": title,
        "type": "movie",
    }
    try:
        resp = requests.get(OMDB_API_URL, params=params, timeout=6)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if data.get("Response") != "True":
        return None
    poster_url = data.get("Poster")
    if poster_url and poster_url != "N/A":
        return poster_url
    return None

# ---------------------
# Cargar únicamente desde pickle (sin opción a construir o subir)
# ---------------------
# -----------------------
# Controles laterales (toggle exclusivos + rerun)
# -----------------------
# Botón para mostrar el buscador en la barra lateral (toggle, exclusivo)
search_button_clicked = st.sidebar.button("Explorar por título", use_container_width=True)
if search_button_clicked:
    # Toggle search state
    new_search = not st.session_state.get("show_search", False)
    st.session_state["show_search"] = new_search
    # Si activamos search, desactivamos Top N y limpiamos su slider
    if new_search:
        st.session_state["show_top_n"] = False
        st.session_state.pop("topn_slider", None)
    else:
        # si desactivamos search, limpiamos selección de película/ks
        st.session_state.pop("movie_select_autocomplete", None)
        st.session_state.pop("k_recs", None)
    # Forzar rerun para que la UI se actualice inmediatamente (solo una vista visible)
    st.rerun()


# contenedor donde renderizas los controles del buscador
search_container = st.sidebar.container()

# Botón para mostrar Top N (toggle, exclusivo)
top_button_clicked = st.sidebar.button("Top N mejores peliculas", use_container_width=True)
if top_button_clicked:
    # Toggle top_n state
    new_top = not st.session_state.get("show_top_n", False)
    st.session_state["show_top_n"] = new_top
    # Si activamos Top N, desactivamos search y limpiamos sus controles
    if new_top:
        st.session_state["show_search"] = False
        st.session_state.pop("movie_select_autocomplete", None)
        st.session_state.pop("k_recs", None)
    else:
        # si desactivamos Top N, limpiamos su slider
        st.session_state.pop("topn_slider", None)
    # Forzar rerun para que la UI se actualice inmediatamente (solo una vista visible)
    st.rerun()


# contenedor donde renderizas controles del top (si ya lo definiste más abajo puedes dejarlo)
top_container = st.sidebar.container()

model_info_container = st.sidebar.container()

with model_info_container:
    st.header("Origen de datos")
    st.markdown("**Modo de carga:** desde archivo pickle.")
recommender = None

if os.path.exists(DEFAULT_MODEL_PATH):
    try:
        with model_info_container:
            st.write(f"Usando `{DEFAULT_MODEL_PATH}`")
        loaded = load_pickle(DEFAULT_MODEL_PATH)
        # Aceptamos pickles que ya sean ContentRecommender o tengan la API esperada:
        if hasattr(loaded, "recommend_by_title") and hasattr(loaded, "items_df"):
            recommender = loaded
        else:
            with model_info_container:
                st.error(
                    "El pickle cargado NO parece ser una instancia preparada de `ContentRecommender`. "
                    "La app solo usa pickles que contengan el recommender ya inicializado (items_df, X, Xn, etc.). "
                    "Entrena offline y guarda `recomm = ContentRecommender(df, fitted_pipeline)` y picklea ese objeto."
                )
            recommender = None
    except Exception as e:
        with model_info_container:
            st.error(f"Error cargando pickle: {e}")
        recommender = None
else:
    with model_info_container:
        st.error(f"No existe `{DEFAULT_MODEL_PATH}` en el directorio. Coloca allí el pickle del recommender.")

# Si no hay recommender, detener (la app no intentará construir desde CSV ni pedir uploads)
if recommender is None:
    st.warning("No hay un recommender listo. Coloca `best_model.pkl` (pickle del ContentRecommender) en el directorio.")
    st.stop()

# si llegamos aquí, tenemos recommender
items_df = recommender.items_df.copy()
preferred_title_col = TITLE_COL
if preferred_title_col in items_df.columns:
    title_col = preferred_title_col
else:
    title_col = recommender.title_col if hasattr(recommender, "title_col") else TITLE_COL


if hasattr(recommender, "title_col"):
    recommender.title_col = title_col

if title_col not in items_df.columns:
    st.error(f"No encontr? la columna '{title_col}' en los datos cargados.")
    st.stop()


title_lower_lookup = items_df[title_col].astype(str).str.lower()

def get_item_row_by_title(title: str):
    if not title:
        return None
    normalized = title.strip().lower()
    matches = items_df.loc[title_lower_lookup == normalized]
    if matches.empty:
        return None
    return matches.iloc[0]

def resolve_poster_url(title: str, row: pd.Series | None = None):
    title_clean = (title or "").strip().lower()
    if not title_clean:
        return None
    if title_clean in SPECIAL_POSTERS:
        return SPECIAL_POSTERS[title_clean]
    if row is not None:
        foto_val = row.get("foto")
        if pd.notna(foto_val) and isinstance(foto_val, str) and foto_val.strip().lower().startswith("http"):
            return foto_val.strip()
    candidate_titles = []
    if row is not None:
        for cand in [title_col, "original_title", "title"]:
            if cand in row.index:
                val = row.get(cand)
                if isinstance(val, str) and val.strip():
                    candidate_titles.append(val.strip())
    if not candidate_titles:
        candidate_titles.append(title)
    for cand in candidate_titles:
        poster = fetch_omdb_poster(cand)
        if poster:
            return poster
    return None

def _extract_row_value(row: pd.Series, candidates: list[str]):
    for cand in candidates:
        if cand in row.index:
            value = row.get(cand)
            if isinstance(value, str):
                if value.strip() and value.strip().lower() != "nan":
                    return value.strip()
            elif pd.notna(value):
                return value
    return None

def _format_detail_value(field_key: str, value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return "—"
    if field_key == "avg_vote":
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
    if field_key == "duration":
        try:
            return f"{int(float(value))} min"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


def build_home_cards(limit: int = 12, scan_rows: int = 400, avg_col: str | None = None, votes_col: str | None = None):
    cards = []
    seen = set()
    df_candidates = items_df.copy()
    if avg_col and votes_col and avg_col in df_candidates.columns and votes_col in df_candidates.columns:
        df_candidates = df_candidates.dropna(subset=[avg_col, votes_col]).copy()
        df_candidates[votes_col] = pd.to_numeric(df_candidates[votes_col], errors="coerce").fillna(0).astype(float)
        df_candidates[avg_col] = pd.to_numeric(df_candidates[avg_col], errors="coerce").fillna(0.0).astype(float)
        vote_thresh = df_candidates[votes_col].quantile(0.80)
        avg_thresh = df_candidates[avg_col].quantile(0.75)
        filtered = df_candidates[
            (df_candidates[votes_col] >= vote_thresh) & (df_candidates[avg_col] >= avg_thresh)
        ]
        if not filtered.empty:
            df_candidates = filtered
    subset = df_candidates.sort_values(by=[votes_col, avg_col], ascending=[False, False], na_position="last") if votes_col and avg_col else df_candidates
    subset = subset.head(scan_rows)
    for _, row in subset.iterrows():
        title_value = str(row.get(title_col, "")).strip()
        if not title_value:
            continue
        key = title_value.lower()
        if key in seen:
            continue
        poster_url = resolve_poster_url(title_value, row)
        if poster_url:
            cards.append((title_value, poster_url))
            seen.add(key)
        if len(cards) >= limit:
            break
    return cards



# ---------------------
# Utilidades Top N — usa avg_vote y votes, mínimo 500 votes por requerimiento previo
# ---------------------
def _find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        try:
            pat = re.compile(cand, flags=re.IGNORECASE)
        except Exception:
            pat = None
        if pat:
            for c in df.columns:
                if re.fullmatch(pat, c):
                    return c
    return None

AVG_COL = _find_col(items_df, ["avg_vote", "mean_vote", "average_rating", "rating"])
VOTES_COL = _find_col(items_df, ["votes", "vote_count", "num_votes", "(?i)votes?"])
top_n_enabled = (AVG_COL is not None and VOTES_COL is not None)

show_search = st.session_state.get("show_search", False)
show_top_n = st.session_state.get("show_top_n", False)

top_results = pd.DataFrame()
top_error = None
top_selection = None

if show_top_n:
    with top_container:
        st.markdown("**Top N - Mejores películas**")
        if top_n_enabled:
            top_selection = st.slider(
                "Cantidad para el Top N (ordenado por avg_vote, mínimo 4000 votos)",
                min_value=10,
                max_value=250,
                value=24,
                step=1,
                key="topn_slider",
            )
        else:
            st.info("No se detectaron columnas de votos/promedio para calcular el Top N.")
else:
    with top_container:
        st.empty()

if show_top_n and top_n_enabled and top_selection is not None:
    try:
        df_top_candidates = items_df.dropna(subset=[AVG_COL, VOTES_COL]).copy()
        df_top_candidates[VOTES_COL] = pd.to_numeric(df_top_candidates[VOTES_COL], errors="coerce").fillna(0).astype(int)
        df_top_candidates[AVG_COL] = pd.to_numeric(df_top_candidates[AVG_COL], errors="coerce").fillna(0.0).astype(float)
        df_top_filtered = df_top_candidates[df_top_candidates[VOTES_COL] >= 4000]
        top_results = df_top_filtered.sort_values(by=[AVG_COL, VOTES_COL], ascending=[False, False]).head(top_selection)
    except Exception as e:
        top_error = str(e)

if not show_search and not show_top_n:
    st.subheader("Explora por portadas")
    home_cards = build_home_cards(limit=12, scan_rows=500, avg_col=AVG_COL, votes_col=VOTES_COL)
    if home_cards:
        chunk_size = 4
        for start in range(0, len(home_cards), chunk_size):
            cols = st.columns(min(chunk_size, len(home_cards) - start))
            for col, (card_title, card_poster) in zip(cols, home_cards[start:start + chunk_size]):
                safe_title = html.escape(card_title)
                safe_value = html.escape(card_title, quote=True)
                col.markdown(
                    f"""
                    <form method="get" style="text-align:center;">
                        <input type="hidden" name="home_select" value="{safe_value}" />
                        <button type="submit" style="background:none;border:none;padding:0;cursor:pointer;width:100%;">
                            <img src="{card_poster}" alt="{safe_title}" style="width:100%; border-radius:8px; height:240px; object-fit:cover; border:1px solid rgba(255,255,255,0.15);" />
                            <div style="margin-top:6px; text-align:center; font-weight:600; color:#f5f5f5;">{safe_title}</div>
                        </button>
                    </form>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No se encontraron portadas para mostrar en el inicio.")

# ---------------------
# Opciones de filtrado/selector (sidebar) - AUTOCOMPLETADO
# ---------------------
if show_search:
    with search_container:
        # ------------------------------------------------------------------
        # Construir lista completa de titulos (solo titulos con primer caracter alfabetico)
        # ------------------------------------------------------------------
        def _clean_title_option(value: str):
            if not isinstance(value, str):
                return None
            stripped = value.strip()
            if not stripped:
                return None
            return stripped if stripped[0].isalpha() else None

        raw_titles = items_df[title_col].dropna().astype(str).unique()
        all_titles = sorted(
            {
                cleaned
                for cleaned in (_clean_title_option(title) for title in raw_titles)
                if cleaned
            },
            key=lambda name: name.lower()
        )

        if not all_titles:
            st.warning("No hay titulos disponibles para mostrar en el selector.")
            st.stop()

        chosen = st.selectbox("Elige titulo", all_titles, index=0, key="movie_select_autocomplete")

        # slider k (key unico)
        k = st.slider("Numero de recomendaciones (k)", 1, 20, 12, key="k_recs")

    # ---------------------
    # Generar recomendaciones (usa el recommender ya pickled - sin transformar)
    # ---------------------
    try:
        recs = recommender.recommend_by_title(chosen, top_k=k)
    except Exception as e:
        st.error(f"No pude generar recomendaciones para '{chosen}': {e}")
        st.stop()

    # Detalles con imagen principal
    st.subheader("Detalles del titulo elegido")
    info_row = get_item_row_by_title(chosen)
    if info_row is None:
        st.write("No hay detalles para mostrar.")
    else:
        detail_container = st.container()
        poster_url = resolve_poster_url(chosen, info_row)
        poster_col, info_col = detail_container.columns([1, 2])
        if poster_url:
            poster_col.image(poster_url, use_container_width=True, caption=chosen)
        else:
            poster_col.caption("No encontré una imagen relacionada para este título.")

        detail_fields = [
            ("Titulo", ["original_title", title_col, "title"]),
            ("Calificacion", ["avg_vote", "mean_vote"]),
            ("Descripcion", ["description", "plot", "overview"]),
            ("Año", ["year", "start_year", "release_year"]),
            ("Genero", ["genre", "genres"]),
            ("Director", ["director", "directors"]),
            ("Actores", ["actors", "cast"]),
            ("Duracion", ["duration", "runtime", "minutes"]),
            ("Pais", ["country", "countries"]),
            ("Idioma", ["language", "languages", "original_language"]),
        ]

        sanitized_fields = []
        for label, candidates in detail_fields:
            flat_candidates = []
            for cand in candidates:
                if isinstance(cand, list):
                    flat_candidates.extend(cand)
                else:
                    flat_candidates.append(cand)
            value = _extract_row_value(info_row, flat_candidates)
            format_key = flat_candidates[0] if flat_candidates else label.lower()
            formatted = _format_detail_value(format_key, value)
            sanitized_fields.append((label, formatted))

        for label, formatted in sanitized_fields:
            info_col.markdown(f"**{label}:** {formatted}")

    # Recomendaciones con imagen
    if not recs.empty:
        st.subheader("Peliculas recomendadas")
        rec_titles = recs[recommender.title_col].astype(str).tolist()

        def _batched(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i:i + size]

        for batch in _batched(rec_titles, 4):
            cols = st.columns(len(batch))
            for col, rec_title in zip(cols, batch):
                rec_row = get_item_row_by_title(rec_title)
                rec_poster = resolve_poster_url(rec_title, rec_row)
                with col:
                    if rec_poster:
                        col.image(rec_poster, use_container_width=True)
                    else:
                        col.write("Sin imagen")
                    col.caption(rec_title)

    # Mostrar tabla de recomendaciones
    st.subheader(f"Recomendaciones para: {chosen}")
    st.dataframe(recs)

    # Altair: barra horizontal de similarity
    st.subheader("Cuanto se parecen las recomendaciones")
    st.write(
        "Estas barras muestran qué tan similares son las películas recomendadas al título elegido. "
        "La similitud se calcula comparando características de cada película —géneros, reparto, sinopsis y metadatos— "
        "convertidas a vectores numéricos. El sistema mide esa cercanía mediante técnicas como *similitud del coseno* "
        "y distintas distancias matemáticas, mostrando primero las películas más parecidas."
    )

    chart = alt.Chart(recs.reset_index()).mark_bar().encode(
        x=alt.X("similarity:Q", title="Similarity"),
        y=alt.Y(f"{recommender.title_col}:N", sort='-x', title="Titulo recomendado"),
        color=alt.Color("source:N", title="Metodo"),
        tooltip=[recommender.title_col, "similarity", "dist_euclidean", "dist_manhattan", "source"]
    ).properties(height=400, width=700)
    st.altair_chart(chart, use_container_width=True)

    # Boton para descargar resultado
    def to_csv_bytes(df_in):
        buff = BytesIO()
        df_in.to_csv(buff, index=False)
        buff.seek(0)
        return buff

    csv_bytes = to_csv_bytes(recs)
    st.download_button("Descargar recomendaciones (CSV)", data=csv_bytes, file_name=f"recs_for_{chosen}.csv", mime="text/csv")
else:
    with search_container:
        st.empty()

# Mostrar Top N en el panel principal
if show_top_n:
    if top_error:
        st.error(f"Error calculando el Top N: {top_error}")
    elif not top_results.empty:
        st.subheader(f"Top {len(top_results)} mejores películas")

        top_titles = top_results[title_col].astype(str).tolist()

        def _top_batches(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i:i + size]

        top_lower_series = top_results[title_col].astype(str).str.lower()

        for batch in _top_batches(top_titles, 4):
            cols = st.columns(len(batch))
            for col, top_title in zip(cols, batch):
                match_idx = top_lower_series[top_lower_series == top_title.lower()].index
                match_row = top_results.loc[match_idx[0]] if len(match_idx) > 0 else None
                row = get_item_row_by_title(top_title)
                poster_row = row if row is not None else match_row
                poster = resolve_poster_url(top_title, poster_row)
                with col:
                    if poster:
                        col.image(poster, use_container_width=True)
                    else:
                        col.write("Sin imagen")
                    col.markdown(f"**{top_title}**")
                    avg_val = match_row.get(AVG_COL) if match_row is not None and AVG_COL in match_row.index else None
                    col.caption(f"Calificación: {_format_detail_value('avg_vote', avg_val)}")

        cols_show = [title_col, AVG_COL, VOTES_COL]
        cols_show = [c for c in cols_show if c in top_results.columns]
        st.dataframe(top_results[cols_show].reset_index(drop=True))

        # 🔥 DETENEMOS TODO: Vista exclusiva para Top N
        st.stop()

    elif top_selection is not None and top_n_enabled:
        st.info("No hay películas que cumplan con los criterios del Top N.")
st.write("---")
st.caption("App adaptada: estilo IMDb — usando únicamente pickle del recommender (sin transformaciones runtime).")
