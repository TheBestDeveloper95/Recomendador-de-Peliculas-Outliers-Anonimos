# recommender_module.py
# Recomendador completo (transformadores, autoencoder, ContentRecommender)
# Uso: importar desde Streamlit y llamar build_pipeline_and_fit(df) o load_model(path)

import re
import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple, Dict, Optional

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import loguniform
from difflib import get_close_matches
from sklearn.feature_selection import VarianceThreshold

# --------------------------
# Utilidades
# --------------------------

def _coalesce_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        pattern = re.compile(cand, flags=re.IGNORECASE)
        for c in df.columns:
            if re.fullmatch(pattern, c):
                return c
    return None

_SPLIT_RX = re.compile(r'[,\|/;]+')

def _split_multi_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    return s.apply(lambda x: [p.strip() for p in _SPLIT_RX.split(x) if p.strip()])

def _token_norm(tok: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9]+', ' ', tok).strip()
    if not cleaned:
        return "UNKNOWN"
    return "".join(w.capitalize() for w in cleaned.split())

def _normalize_token_series(tokens: pd.Series) -> pd.Series:
    return tokens.map(_token_norm)

# --------------------------
# Transformador 1: Gate de nulos + flags
# --------------------------

class NullsGateAndFlags(BaseEstimator, TransformerMixin):
    def __init__(self, null_threshold: int = 12000, verbose: bool = True):
        self.null_threshold = null_threshold
        self.verbose = verbose
        self.cols_gate_ = None
        self.flagged_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        null_counts = X.isna().sum()
        self.cols_gate_ = null_counts[null_counts < self.null_threshold].index.tolist()

        gate_cols_present = [c for c in self.cols_gate_ if c in X.columns]
        if gate_cols_present:
            before = len(X)
            mask = X[gate_cols_present].notna().all(axis=1)
            after = int(mask.sum())
            lost = before - after
            if self.verbose:
                print(f"[NullsGateAndFlags] Filas antes: {before} | después: {after} | perdidas: {lost}")
            X_possible = X.loc[mask]
        else:
            X_possible = X
            if self.verbose:
                print(f"[NullsGateAndFlags] Aviso: ninguna columna del gate existe en X durante fit; no se filtran filas.")

        self.flagged_cols_ = [c for c in X_possible.columns if X_possible[c].isna().any()]
        if self.verbose and len(self.flagged_cols_) > 0:
            print(f"[NullsGateAndFlags] Columnas convertidas a flags: {self.flagged_cols_}")
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        gate_cols_present = [c for c in (self.cols_gate_ or []) if c in X.columns]
        if gate_cols_present:
            mask = X[gate_cols_present].notna().all(axis=1)
        else:
            mask = np.ones(len(X), dtype=bool)
        X = X.loc[mask].reset_index(drop=True)

        for col in (self.flagged_cols_ or []):
            if col in X.columns:
                X[f"had_{col}"] = X[col].isna().astype(int)
                X = X.drop(columns=[col])
            else:
                X[f"had_{col}"] = 0
        return X

# --------------------------
# Transformador 2: Filtros mínimos
# --------------------------

class MinCountsFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_votes=250, min_user_reviews=8, min_critic_reviews=7, verbose=True):
        self.min_votes = min_votes
        self.min_user_reviews = min_user_reviews
        self.min_critic_reviews = min_critic_reviews
        self.verbose = verbose
        self.col_votes_ = None
        self.col_users_ = None
        self.col_critics_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.col_votes_ = _coalesce_column(X, ["votes", "vote_count", "num_votes", "(?i)votes?"])
        self.col_users_ = _coalesce_column(X, ["reviews_from_users", "user_reviews", "(?i)reviews?_from_users?", "(?i)user_reviews?"])
        self.col_critics_ = _coalesce_column(X, ["reviews_from_critics", "critic_reviews", "(?i)reviews?_from_critics?", "(?i)critic_reviews?"])
        missing = [n for n in ["votes", "reviews_from_users", "reviews_from_critics"]
                   if getattr(self, f"col_{'votes' if n=='votes' else ('users' if 'users' in n else 'critics')}_") is None]
        if missing:
            raise ValueError(f"[MinCountsFilter] No pude resolver columnas requeridas: {missing}")
        return self

    def transform(self, X: pd.DataFrame):
        before = len(X)
        cond = (
                (X[self.col_votes_] >= self.min_votes) &
                (X[self.col_users_] >= self.min_user_reviews) &
                (X[self.col_critics_] >= self.min_critic_reviews)
        )
        X2 = X.loc[cond].reset_index(drop=True)
        lost = before - len(X2)
        if self.verbose:
            print(f"[MinCountsFilter] Filas antes: {before} | después: {len(X2)} | perdidas: {lost}")
        return X2

# --------------------------
# Transformador 3a: Señales de descripción (proporciones) — concatena con X
# --------------------------

class DescriptionSignalsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.desc_col_ = None
        self.LEX = {
            "accion": [
                "fight","battle","war","attack","assault","ambush","raid","gunfight",
                "shootout","chase","pursuit","escape","explosion","bomb","heist","robbery",
                "manhunt","showdown","duel","martial arts","karate","kung fu","sniper",
                "undercover","mission","infiltrate","sabotage","hostage"
            ],
            "emocional": [
                "love","romance","heartbreak","grief","mourning","fear","terror","hope",
                "friendship","betrayal","loyalty","revenge","jealousy","forgiveness",
                "despair","anger","guilt","compassion","redemption","courage"
            ],
            "temporal_months_days": [
                "january","february","march","april","may","june","july","august",
                "september","october","november","december",
                "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
                "century","era","medieval","renaissance","victorian","ancient","prehistoric"
            ],
            "geografico": [
                "new york","los angeles","london","paris","rome","tokyo","berlin","madrid",
                "chicago","san francisco","mumbai","hong kong","moscow","rio de janeiro",
                "usa","united states","uk","england","france","germany","italy","spain","japan","russia",
                "city","village","town","capital","suburb",
                "ocean","sea","island","desert","jungle","forest","mountain","valley","arctic",
                "space","planet","galaxy","mars","moon","colony","outpost"
            ],
            "interior": [
                "apartment","house","mansion","castle","palace","room","hall","basement",
                "attic","office","headquarters","head office","laboratory","lab",
                "hospital","clinic","prison","cell","courtroom","classroom","school",
                "warehouse","factory","bunker","shelter"
            ],
            "exterior": [
                "street","alley","highway","bridge","rooftop","market","square","harbor",
                "harbour","dock","port","farm","field","forest","mountain","desert","beach",
                "coast","island","jungle","canyon","wilderness","battlefield","camp"
            ],
            "fantasia": [
                "magic","spell","sorcery","wizard","witch","mage","magical","enchantment",
                "dragon","elf","dwarf","orc","troll","goblin","kingdom","realm","prophecy",
                "curse","fairy","knight","sword","medieval","myth","legend"
            ],
            "ficcion": [
                "alien","extraterrestrial","ufo","robot","android","cyborg","ai","a.i.","artificial intelligence",
                "spaceship","starship","warp","wormhole","galaxy","planet","colony","time travel","time-travel",
                "timeline","dystopia","utopia","apocalypse","post-apocalyptic","teleport","nanotechnology",
                "clone","virtual reality","simulated","cyberspace"
            ],
            "roles": [
                "detective","cop","police","officer","agent","spy","assassin","hitman","soldier","commander",
                "general","pilot","astronaut","scientist","researcher","engineer","doctor","nurse","lawyer",
                "attorney","judge","journalist","reporter","teacher","student","thief","hacker","priest","monk"
            ],
            "conflicto": [
                "attack","invade","murder","kill","assassinate","betray","revenge","retaliate","ambush",
                "kidnap","torture","blackmail","extort","threaten","bomb","explode","terrorist","uprising",
                "coup","riot","massacre","execute","enslave","conquer","dominate","hunt","chase","combat"
            ],
            "cooperacion": [
                "ally","alliance","cooperate","collaborate","team up","join forces","assist","help","rescue",
                "protect","defend","save","support","mentor","negotiate","truce","treaty","peace","reconcile",
                "unite","partnership"
            ],
            "infantil": [
                "family","kids","children","school","classroom","teen","friendship","adventure","magic","fairy",
                "cartoon","playful","wholesome","pet","puppy","kitten","holiday","christmas","birthday"
            ],
            "adulto": [
                "drug","narcotic","heroin","cocaine","meth","alcoholic","sex","erotic","nudity","explicit",
                "prostitute","brothel","violence","gore","graphic","abuse","rape","trafficking","addiction",
                "mafia","cartel","gang"
            ]
        }
        self.RE_YEARS   = re.compile(r"\b(18|19|20)\d{2}\b")
        self.RE_DECADES = re.compile(r"\b(18|19|20)\d0s\b")
        self.RE_PROPER  = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
        self.COMMON_TITLE_CASE = {"The","A","An","Of","And","In","On","At","To","From","By","With","For","As","La","El","Los","Las"}
        self._compiled: Dict[str, List[re.Pattern]] = {}

    def _find_desc_col(self, X: pd.DataFrame) -> Optional[str]:
        for cand in ["description","plot","storyline","synopsis","overview","resume","resumen"]:
            col = _coalesce_column(X, [cand])
            if col is not None:
                return col
        return None

    def fit(self, X: pd.DataFrame, y=None):
        self.desc_col_ = self._find_desc_col(X)
        self._compiled = {}
        for key, lst in self.LEX.items():
            pats = []
            for phrase in lst:
                pat = r"\b" + re.escape(phrase.lower()) + r"\b"
                pats.append(re.compile(pat))
            self._compiled[key] = pats
        if self.verbose:
            print(f"[DescriptionSignals] Columna de descripción usada: {self.desc_col_}")
        return self

    def transform(self, X: pd.DataFrame):
        n = len(X)
        if self.desc_col_ is None or self.desc_col_ not in X.columns:
            cols = [f"desc_prop_{k}" for k in [
                "accion","emocional","temporal","geografico","interior","exterior",
                "fantasia","ficcion","propios","roles","conflicto","cooperacion",
                "infantil","adulto"
            ]]
            zeros = pd.DataFrame(0.0, index=np.arange(n), columns=cols, dtype=np.float32)
            return pd.concat([X.reset_index(drop=True), zeros], axis=1)

        s_raw = X[self.desc_col_].fillna("").astype(str)
        s_low = s_raw.str.lower()

        token_counts = s_low.str.count(r"\w+").astype(np.int32)
        denom = token_counts.replace(0, 1)

        out = {}
        for grp in ["accion","emocional","geografico","interior","exterior","fantasia","ficcion",
                    "roles","conflicto","cooperacion","infantil","adulto"]:
            pats = self._compiled[grp]
            cnt = np.zeros(n, dtype=np.int32)
            for rx in pats:
                cnt += s_low.str.count(rx).astype(np.int32).values
            out[f"desc_prop_{grp}"] = (cnt / denom).astype(np.float32)

        pats = self._compiled["temporal_months_days"]
        cnt_temp = np.zeros(n, dtype=np.int32)
        for rx in pats:
            cnt_temp += s_low.str.count(rx).astype(np.int32).values
        cnt_years   = s_raw.str.count(self.RE_YEARS).astype(np.int32).values
        cnt_decades = s_raw.str.count(self.RE_DECADES).astype(np.int32).values
        out["desc_prop_temporal"] = ((cnt_temp + cnt_years + cnt_decades) / denom).astype(np.float32)

        proper_matches = s_raw.apply(lambda t: self.RE_PROPER.findall(t))
        cnt_propios = proper_matches.apply(
            lambda lst: sum(1 for m in lst if m.split()[0] not in self.COMMON_TITLE_CASE)
        ).astype(np.int32).values
        out["desc_prop_propios"] = (cnt_propios / denom).astype(np.float32)

        desc_df = pd.DataFrame(out, index=np.arange(n))
        return pd.concat([X.reset_index(drop=True), desc_df], axis=1)

# --------------------------
# Transformador 3b: Texto categórico + deciles relevancia (arrastra desc_prop_*)
# --------------------------

class CreativeTextFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, top_map: Dict[str, int] = None, verbose: bool = True):
        # Aseguramos que self.top_map siempre exista (evita errores en clone/get_params)
        self.top_map = top_map or {
            "country": 11,
            "language": 13,
            "director": None,
            "writer": None,
            "actors": None,
            "production_company": None
        }
        self.verbose = verbose
        self.col_votes_ = None
        self.col_avgvote_ = None
        self.genre_col_ = None
        self.text_cols_ = None
        self._genre_vocab_ = None
        self.selected_tokens_top_: Dict[str, List[str]] = {}
        self.decile_map_: Dict[str, Dict[str, int]] = {}
        self.decile_bins_count_: Dict[str, int] = {}
        self.feature_names_ = None

    def _find_col_like(self, X: pd.DataFrame, name: str) -> Optional[str]:
        pattern = re.compile(name, flags=re.IGNORECASE)
        candidates = [c for c in X.columns if re.search(pattern, c)]
        for c in candidates:
            if c.lower() == name.lower():
                return c
        return candidates[0] if candidates else None

    def _collect_relevancies(self, X: pd.DataFrame, col: str, col_votes: str, col_avg: str) -> pd.Series:
        toks_list = _split_multi_series(X[col])
        if len(X) == 0:
            return pd.Series(dtype=float)
        df_tok = pd.DataFrame({
            "_row": np.arange(len(X)).repeat(toks_list.str.len()),
            "token": np.concatenate(toks_list.values) if len(X) else np.array([], dtype=object)
        })
        if len(df_tok) == 0:
            return pd.Series(dtype=float)
        df_tok["token"] = _normalize_token_series(df_tok["token"])
        score = (X[col_votes].astype(float) * X[col_avg].astype(float)).values
        df_tok["score"] = score[df_tok["_row"].values]
        rel = df_tok.groupby("token", sort=False)["score"].sum().sort_values(ascending=False)
        return rel

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.col_votes_ = _coalesce_column(X, ["votes", "vote_count", "num_votes", "(?i)votes?"])
        self.col_avgvote_ = _coalesce_column(X, ["avg_vote", "mean_vote", "average_rating", "rating", "(?i)(avg|mean).*vote|rating"])
        if self.col_votes_ is None or self.col_avgvote_ is None:
            raise ValueError("[CreativeTextFeaturizer] No pude resolver 'votes' o 'avg_vote'.")

        self.genre_col_ = self._find_col_like(X, "genre")
        if self.genre_col_ is None:
            raise ValueError("[CreativeTextFeaturizer] No encontré columna de géneros (like 'genre').")

        self.text_cols_ = {}
        for key in self.top_map.keys():
            col_found = self._find_col_like(X, key)
            if col_found is not None:
                self.text_cols_[key] = col_found

        if self.verbose:
            print(f"[CreativeTextFeaturizer] Usando: votes='{self.col_votes_}', avg='{self.col_avgvote_}', genre='{self.genre_col_}', rankeables={self.text_cols_}")

        # géneros
        genres_list = _split_multi_series(X[self.genre_col_])
        genres_expl = genres_list.explode().dropna()
        genres_expl = genres_expl[genres_expl != ""]
        genres_norm = _normalize_token_series(genres_expl)
        self._genre_vocab_ = np.sort(genres_norm.unique())

        # top-K: country/language
        self.selected_tokens_top_.clear()
        for key in ["country", "language"]:
            if key in self.text_cols_:
                rel = self._collect_relevancies(X, self.text_cols_[key], self.col_votes_, self.col_avgvote_)
                topk = (self.top_map.get(key) or 0)
                sel = list(rel.index[:topk]) if topk > 0 else []
                self.selected_tokens_top_[key] = sel
                if self.verbose:
                    preview = ", ".join([_token_norm(t) for t in sel[:6]]) + ("..." if len(sel) > 6 else "")
                    print(f"[CreativeTextFeaturizer] Top-{topk} '{key}': {preview}")

        # deciles: director/writer/actors/company
        self.decile_map_.clear()
        self.decile_bins_count_.clear()
        for key in ["director", "writer", "actors", "production_company"]:
            if key not in self.text_cols_:
                continue
            rel = self._collect_relevancies(X, self.text_cols_[key], self.col_votes_, self.col_avgvote_)
            if len(rel) == 0:
                self.decile_map_[key] = {}
                self.decile_bins_count_[key] = 0
                continue
            uniq = len(rel)
            bins = min(10, uniq)
            ranks = np.arange(len(rel))  # 0..N-1
            borders = np.linspace(0, len(rel), bins + 1, dtype=int)
            dec_array = np.empty(len(rel), dtype=int)
            for d in range(bins):
                dec_array[borders[d]:borders[d+1]] = d + 1  # 1..bins
            tok2dec = {tok: int(dec) for tok, dec in zip(rel.index, dec_array)}
            self.decile_map_[key] = tok2dec
            self.decile_bins_count_[key] = int(bins)
            if self.verbose:
                print(f"[CreativeTextFeaturizer] '{key}' en {bins} deciles (tokens={len(rel)})")

        self.feature_names_ = None
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        n = len(X)
        parts = []

        # géneros
        genres_list = _split_multi_series(X[self.genre_col_])
        df_g = pd.DataFrame({"row": np.arange(n).repeat(genres_list.str.len()),
                             "genre": np.concatenate(genres_list.values) if n else np.array([], dtype=object)})
        if len(df_g) > 0:
            df_g["genre"] = _normalize_token_series(df_g["genre"])
            df_g = df_g[df_g["genre"].isin(self._genre_vocab_)]
            genre_ct = pd.crosstab(df_g["row"], df_g["genre"])
            genre_ct = (genre_ct > 0).astype(np.int8)
            for g in self._genre_vocab_:
                if g not in genre_ct.columns:
                    genre_ct[g] = 0
            genre_ct = genre_ct[self._genre_vocab_]
            genre_ct.columns = [f"genre_{g}" for g in genre_ct.columns]
        else:
            genre_ct = pd.DataFrame(0, index=np.arange(n), columns=[f"genre_{g}" for g in self._genre_vocab_], dtype=np.int8)
        genre_ct = genre_ct.reindex(index=np.arange(n), fill_value=0)
        parts.append(genre_ct)

        # country/language
        for key in ["country","language"]:
            if key not in self.text_cols_:
                continue
            col = self.text_cols_[key]
            sel = self.selected_tokens_top_.get(key, [])
            colname_other = f"is_other{key.capitalize()}"
            toks_list = _split_multi_series(X[col])
            if len(sel) == 0:
                has_any = toks_list.str.len().astype(bool).astype(np.int8)
                tmp = pd.DataFrame({colname_other: (1 - has_any).astype(np.int8)}, index=np.arange(n))
                parts.append(tmp)
            else:
                df_tok = pd.DataFrame({
                    "row": np.arange(n).repeat(toks_list.str.len()),
                    "token": np.concatenate(toks_list.values) if n else np.array([], dtype=object)
                })
                if len(df_tok) == 0:
                    tmp = pd.DataFrame(0, index=np.arange(n),
                                       columns=[f"is_{_token_norm(t)}" for t in sel] + [colname_other],
                                       dtype=np.int8)
                    tmp[colname_other] = 1
                    parts.append(tmp)
                else:
                    df_tok["token"] = _normalize_token_series(df_tok["token"])
                    df_tok = df_tok[df_tok["token"].isin(sel)]
                    if len(df_tok) == 0:
                        tmp = pd.DataFrame(0, index=np.arange(n),
                                           columns=[f"is_{_token_norm(t)}" for t in sel] + [colname_other],
                                           dtype=np.int8)
                        tmp[colname_other] = 1
                        parts.append(tmp)
                    else:
                        ct = pd.crosstab(df_tok["row"], df_tok["token"])
                        ct = (ct > 0).astype(np.int8)
                        for t in sel:
                            if t not in ct.columns:
                                ct[t] = 0
                        ct = ct[sel]
                        tmp = ct.reindex(index=np.arange(n), fill_value=0)
                        tmp.columns = [f"is_{_token_norm(c)}" for c in tmp.columns]
                        has_any_token = tmp.sum(axis=1)
                        had_list = toks_list.str.len() > 0
                        tmp[colname_other] = (~(has_any_token > 0) & had_list).astype(np.int8)
                        parts.append(tmp)

        # deciles
        for key in ["director","writer","actors","production_company"]:
            if key not in self.text_cols_:
                continue
            col = self.text_cols_[key]
            K = self.decile_bins_count_.get(key, 0)
            colname_other = f"is_other{key.capitalize()}"
            if K <= 0:
                has_any = _split_multi_series(X[col]).str.len().astype(bool).astype(np.int8)
                tmp = pd.DataFrame({colname_other: (1 - has_any).astype(np.int8)}, index=np.arange(n))
                parts.append(tmp)
                continue

            dec_cols = [f"is_{key}_decile_{d}" for d in range(1, K+1)]
            tmp = pd.DataFrame(0, index=np.arange(n), columns=dec_cols + [colname_other], dtype=np.int8)
            toks_list = _split_multi_series(X[col])
            df_tok = pd.DataFrame({
                "row": np.arange(n).repeat(toks_list.str.len()),
                "token": np.concatenate(toks_list.values) if n else np.array([], dtype=object)
            })
            if len(df_tok) == 0:
                tmp[colname_other] = 1
                parts.append(tmp)
            else:
                df_tok["token"] = _normalize_token_series(df_tok["token"])
                tok2dec = self.decile_map_.get(key, {})
                df_tok["dec"] = df_tok["token"].map(tok2dec).fillna(0).astype(int)
                df_tok = df_tok[df_tok["dec"] > 0]
                if len(df_tok) > 0:
                    ct = pd.crosstab(df_tok["row"], df_tok["dec"])
                    ct = (ct > 0).astype(np.int8)
                    for d in range(1, K+1):
                        if d not in ct.columns:
                            ct[d] = 0
                    ct = ct[list(range(1, K+1))]
                    tmp.loc[ct.index, [f"is_{key}_decile_{d}" for d in range(1, K+1)]] = ct.values
                has_any_token = toks_list.str.len() > 0
                any_dec = tmp[dec_cols].sum(axis=1) > 0
                tmp[colname_other] = (~any_dec & has_any_token).astype(np.int8)
                parts.append(tmp)

        # flags had_*
        had_cols = [c for c in X.columns if c.startswith("had_")]
        if had_cols:
            parts.append(X[had_cols].astype(np.int8))

        # num robustas
        num_cols_map = {
            "votes": ["votes", "vote_count", "num_votes", "(?i)votes?"],
            "avg_vote": ["avg_vote", "mean_vote", "average_rating", "rating", "(?i)(avg|mean).*vote|rating"],
            "reviews_from_users": ["reviews_from_users", "user_reviews", "(?i)reviews?_from_users?", "(?i)user_reviews?"],
            "reviews_from_critics": ["reviews_from_critics", "critic_reviews", "(?i)reviews?_from_critics?", "(?i)critic_reviews?"],
        }
        num_found = []
        for key, cands in num_cols_map.items():
            col = _coalesce_column(X, cands)
            if col is not None and col in X.columns:
                num_found.append((key, col))
        if num_found:
            nums = pd.DataFrame(index=np.arange(n))
            for key, col in num_found:
                series = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)
                if key == "votes":
                    series = np.log1p(series)
                nums[f"num_{key}"] = series.values.astype(np.float32)
            parts.append(nums.astype(np.float32))

        # arrastrar señales de descripción si existen
        desc_cols = [c for c in X.columns if c.startswith("desc_prop_")]
        if desc_cols:
            parts.append(X[desc_cols].astype(np.float32))

        out = pd.concat(parts, axis=1)

        if self.feature_names_ is None:
            self.feature_names_ = list(out.columns)
        else:
            keep = [c for c in self.feature_names_ if c in out.columns]
            extras = [c for c in out.columns if c not in keep]
            out = out.reindex(columns=keep + extras, fill_value=0)

        return out

    def get_feature_names_out(self):
        return np.array(self.feature_names_) if self.feature_names_ is not None else None

# --------------------------
# Estimador: AutoencoderRegressor
# --------------------------

class AutoencoderRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 hidden_layer_sizes=(128, 32, 128),
                 activation='relu',
                 alpha=1e-4,
                 learning_rate_init=1e-3,
                 max_iter=200,
                 batch_size='auto',
                 early_stopping=True,
                 random_state=42,
                 verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.verbose = verbose
        self._mlp = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self._mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self._mlp.fit(X, X)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self._mlp.predict(X)

    def score(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        recon = self.predict(X)
        return -mean_squared_error(X, recon)

# --------------------------
# Recomendador
# --------------------------

def _resolve_title_col(df: pd.DataFrame) -> str:
    col = _coalesce_column(df, ["title", "original_title", "movie", "movie_title", "name"])
    if col is None:
        raise ValueError("No encontré columna de título (ej: 'title').")
    return col

def _resolve_year_col(df: pd.DataFrame) -> Optional[str]:
    return _coalesce_column(df, ["year", "start_year", "release_year", "(?i).*year.*"])

def _resolve_votes_avg_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    votes = _coalesce_column(df, ["votes", "vote_count", "num_votes", "(?i)votes?"])
    avg = _coalesce_column(df, ["avg_vote", "mean_vote", "average_rating", "rating", "(?i)(avg|mean).*vote|rating"])
    return votes, avg

class ContentRecommender:
    def __init__(self, df: pd.DataFrame, fitted_pipeline, title_col: Optional[str] = None, year_col: Optional[str] = None):
        self.full_df = df.reset_index(drop=True).copy()
        self.pipe = fitted_pipeline
        # pipeline hasta antes del último paso (autoencoder)
        self.pipe_upto_scaler = fitted_pipeline[:-1]

        steps = fitted_pipeline.named_steps
        df_items = self.full_df.copy()
        if "nulls_gate" in steps:
            df_items = steps["nulls_gate"].transform(df_items)
        if "min_counts" in steps:
            df_items = steps["min_counts"].transform(df_items)

        self.items_df = df_items.reset_index(drop=True)
        if len(self.items_df) == 0:
            raise ValueError("Después de los filtros, no quedan items para recomendar.")

        self.title_col = title_col or _resolve_title_col(self.items_df)
        self.year_col = year_col or _resolve_year_col(self.items_df)

        # Representaciones
        self.X = self.pipe_upto_scaler.transform(self.items_df)
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.Xn = self.X / norms

        vcol, acol = _resolve_votes_avg_cols(self.items_df)
        if vcol is not None and acol is not None and len(self.items_df) > 0:
            pop = (self.items_df[vcol].astype(float) * self.items_df[acol].astype(float)).values
            pmin, pmax = float(np.nanmin(pop)), float(np.nanmax(pop))
            self.pop = (pop - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(pop)
        else:
            self.pop = np.zeros(len(self.items_df))

        self.title_index_map = {}
        for idx, t in enumerate(self.items_df[self.title_col].astype(str).values):
            self.title_index_map.setdefault(t.strip().lower(), []).append(idx)

    def _find_index(self, title_query: str, year_hint: Optional[int] = None) -> int:
        key = title_query.strip().lower()
        if key in self.title_index_map:
            candidates = self.title_index_map[key]
            if self.year_col is not None and year_hint is not None:
                years = self.items_df.loc[candidates, self.year_col]
                mask = (years.astype(str) == str(year_hint))
                if mask.any():
                    return int(np.asarray(candidates)[np.where(mask)[0][0]])
            if len(candidates) > 1:
                sub = np.asarray(candidates)
                return int(sub[np.argmax(self.pop[sub])])
            return int(candidates[0])

        all_titles = list(self.title_index_map.keys())
        match = get_close_matches(key, all_titles, n=1, cutoff=0.6)
        if match:
            return self.title_index_map[match[0]][0]
        raise ValueError(f"Título no encontrado (tras filtros): '{title_query}'")

    def recommend_by_title(self, title_query: str, top_k: int = 10, year_hint: Optional[int] = None) -> pd.DataFrame:
        """
        Top-k mixto en tres tercios:
          - ~1/3 por coseno (ángulo) en Xn (unit norm).
          - ~1/3 por distancia euclídea (L2) en X (estandarizado).
          - ~1/3 por distancia manhattan (L1) en X (estandarizado).
        Evita duplicados, anota 'source' y muestra similarity y ambas distancias.
        """
        if len(self.items_df) == 0:
            raise ValueError("No hay items disponibles para recomendar.")
        q_idx = self._find_index(title_query, year_hint=year_hint)

        N = len(self.items_df)
        if N <= 1:
            raise ValueError("No hay suficientes items para generar recomendaciones.")

        # Particiones k1, k2, k3
        k1 = top_k // 3
        k2 = (top_k - k1) // 2
        k3 = top_k - k1 - k2
        if k1 == 0 and top_k > 0:
            k1 = 1
            rem = top_k - k1
            k2 = rem // 2
            k3 = rem - k2

        # Coseno
        q_vec_n = self.Xn[q_idx:q_idx+1]
        sims = (self.Xn @ q_vec_n.T).ravel()
        sims[q_idx] = -np.inf
        sel_cos = []
        if k1 > 0:
            take = min(k1, max(0, N - 1))
            idx_cos = np.argpartition(-sims, range(take))[:take]
            idx_cos = idx_cos[np.argsort(-sims[idx_cos])]
            sel_cos = list(idx_cos)

        # Euclídea
        q_vec = self.X[q_idx:q_idx+1]
        used = np.zeros(N, dtype=bool)
        used[q_idx] = True
        if sel_cos:
            used[np.array(sel_cos, dtype=int)] = True
        rest_idx = np.where(~used)[0]
        dists_l2 = np.empty(N, dtype=np.float32)
        dists_l2.fill(np.inf)
        if len(rest_idx) > 0:
            diffs = self.X[rest_idx] - q_vec
            l2 = np.sqrt(np.sum(diffs * diffs, axis=1))
            dists_l2[rest_idx] = l2
        sel_l2 = []
        if k2 > 0 and len(rest_idx) > 0:
            take = min(k2, len(rest_idx))
            idx_part = np.argpartition(dists_l2, range(take))[:take]
            idx_part = idx_part[np.argsort(dists_l2[idx_part])]
            sel_l2 = list(idx_part)
            used[np.array(sel_l2, dtype=int)] = True

        # Manhattan
        rest_idx2 = np.where(~used)[0]
        dists_l1 = np.empty(N, dtype=np.float32)
        dists_l1.fill(np.inf)
        if len(rest_idx2) > 0:
            l1 = np.sum(np.abs(self.X[rest_idx2] - q_vec), axis=1)
            dists_l1[rest_idx2] = l1
        sel_l1 = []
        if k3 > 0 and len(rest_idx2) > 0:
            take = min(k3, len(rest_idx2))
            idx_part = np.argpartition(dists_l1, range(take))[:take]
            idx_part = idx_part[np.argsort(dists_l1[idx_part])]
            sel_l1 = list(idx_part)

        # Armar salida
        cols_to_show = [self.title_col]
        if self.year_col is not None:
            cols_to_show.append(self.year_col)
        vcol, acol = _resolve_votes_avg_cols(self.items_df)
        if acol is not None: cols_to_show.append(acol)
        if vcol is not None: cols_to_show.append(vcol)

        rows = []
        for i in sel_cos:
            row = self.items_df.loc[i, cols_to_show].copy().to_dict()
            row["similarity"] = float(np.round(sims[i], 4))
            dv = self.X[i] - q_vec
            row["dist_euclidean"] = float(np.round(np.sqrt(np.sum(dv * dv)), 4))
            row["dist_manhattan"] = float(np.round(np.sum(np.abs(dv)), 4))
            row["source"] = "angle(cosine)"
            rows.append(row)
        for i in sel_l2:
            row = self.items_df.loc[i, cols_to_show].copy().to_dict()
            row["similarity"] = float(np.round(sims[i], 4))
            dv = self.X[i] - q_vec
            row["dist_euclidean"] = float(np.round(np.sqrt(np.sum(dv * dv)), 4))
            row["dist_manhattan"] = float(np.round(np.sum(np.abs(dv)), 4))
            row["source"] = "distance(euclidean)"
            rows.append(row)
        for i in sel_l1:
            row = self.items_df.loc[i, cols_to_show].copy().to_dict()
            row["similarity"] = float(np.round(sims[i], 4))
            dv = self.X[i] - q_vec
            row["dist_euclidean"] = float(np.round(np.sqrt(np.sum(dv * dv)), 4))
            row["dist_manhattan"] = float(np.round(np.sum(np.abs(dv)), 4))
            row["source"] = "distance(manhattan)"
            rows.append(row)

        out = pd.DataFrame(rows)
        preferred = [self.title_col, "similarity", "dist_euclidean", "dist_manhattan", "source"]
        ordered = preferred + [c for c in cols_to_show if c not in preferred]
        out = out.reindex(columns=ordered)
        return out.reset_index(drop=True)

# --------------------------
# Funciones públicas para la app
# --------------------------

def build_pipeline_and_fit(df: pd.DataFrame, n_iter: int = 1, verbose: bool = True):
    """
    Construye y entrena el pipeline completo del recomendador a partir de df.
    Retorna el pipeline entrenado (best_estimator_ del RandomizedSearchCV).
    """
    pipe = Pipeline(steps=[
        ("nulls_gate", NullsGateAndFlags(null_threshold=12000, verbose=verbose)),
        ("min_counts", MinCountsFilter(min_votes=250, min_user_reviews=8, min_critic_reviews=7, verbose=verbose)),
        ("desc_sigs", DescriptionSignalsTransformer(verbose=verbose)),
        ("creative_text", CreativeTextFeaturizer(verbose=verbose)),
        ("varth", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("auto", AutoencoderRegressor(
            hidden_layer_sizes=(128, 64, 128),
            activation='relu',
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=True,
            random_state=42,
            verbose=False
        ))
    ])

    idx = np.arange(len(df))
    param_distributions = {
        "auto__hidden_layer_sizes": [(128, 64, 16, 64, 128)],
        "auto__activation": ["relu"],
        "auto__alpha": loguniform(1e-6, 1e-2),
        "auto__learning_rate_init": loguniform(1e-4, 5e-2),
        "auto__batch_size": [64, 128, 'auto'],
        "auto__max_iter": [150],
        "auto__early_stopping": [True],
    }

    rnd = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=max(1, int(n_iter)),
        cv=[(idx, idx)],
        n_jobs=-1,
        verbose=10 if verbose else 0,
        random_state=42,
        return_train_score=False
    )

    rnd.fit(df)
    if verbose:
        print("[build_pipeline_and_fit] Mejor score:", rnd.best_score_)
    return rnd.best_estimator_

def save_model(path: str, model):
    """Guarda pipeline / objeto en pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path: str):
    """Carga modelo desde pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)

def make_recommender_from_pipeline(pipeline, df: pd.DataFrame) -> ContentRecommender:
    """Crea instancia ContentRecommender ya preparada desde pipeline y df."""
    return ContentRecommender(df=df, fitted_pipeline=pipeline)

# No ejecutar procesos pesados al importar
if __name__ == "__main__":
    print("recommender_module: ejecución directa (pruebas). Para usar en Streamlit importa el módulo y llama a build_pipeline_and_fit(df) o load_model(path).")
