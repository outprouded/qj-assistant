import datetime as dt
import json
import os
import re
import difflib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

matplotlib.use("Agg")

try:
    from pymorphy3 import MorphAnalyzer as _MorphAnalyzer
except ImportError:
    try:
        from pymorphy2 import MorphAnalyzer as _MorphAnalyzer
    except ImportError:
        _MorphAnalyzer = None

MORPH = _MorphAnalyzer() if _MorphAnalyzer else None

st.set_page_config(page_title="Freedom QJ League Analytics", layout="wide")

load_dotenv(override=True)

DATA_PATH = Path("team_stats_with_season_ru_with_alt.csv")
ALIASES_PATH = Path("stat_aliases.json")
LOG_MATCH_PATH = Path("stat_match_logs.jsonl")
MISSED_QUERIES_PATH = Path("missed_queries.txt")

COL_TOURNAMENT = "Турнир"
COL_MATCH = "Матч"
COL_DATE = "Дата"
COL_GOALS_FOR = "Голы забитые"
COL_EXPECTED_GOALS = "Ожидаемые голы xG"
COL_GOALS_AGAINST = "Голы пропущенные"
COL_POSSESSION = "Процент владения мячом"
COL_SHOTS_ON_TARGET = "Удары в створ"
COL_PASSES = "Передачи"
COL_DUELS = "Единоборства"
COL_INTERCEPTIONS = "Перехваты"
COL_TEAM = "Команда"
COL_HOME = "Дома"
COL_AWAY = "Гости"
COL_OPPONENT = "Противник"
COL_SEASON = "Сезон"

META_COLUMNS = {
    COL_TEAM,
    COL_MATCH,
    COL_DATE,
    COL_SEASON,
    COL_TOURNAMENT,
    COL_HOME,
    COL_AWAY,
    COL_OPPONENT,
}

STOPWORDS = {"и", "в", "во", "на", "с", "по", "к", "а", "от", "за", "%"}
TOKEN_SPLIT_RE = re.compile(r"[^\w%]+")


@st.cache_data(show_spinner=False)
def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_aliases(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k.lower().strip(): v for k, v in data.items() if not k.startswith("_")}


df = load_dataframe(DATA_PATH)
STAT_ALIASES = load_aliases(ALIASES_PATH)
STAT_ALIAS_LEMMAS: Dict[str, str] = {}

if MORPH:
    for alias_key, target in STAT_ALIASES.items():
        lemma_key = None
        try:
            lemma_key = " ".join(
                t for t in TOKEN_SPLIT_RE.split(alias_key) if t and t not in STOPWORDS
            )
            if lemma_key:
                lemma_key = " ".join(
                    MORPH.parse(token)[0].normal_form for token in lemma_key.split()
                )
        except Exception:
            lemma_key = None
        if lemma_key:
            STAT_ALIAS_LEMMAS[lemma_key] = target


ALT_NAME_COL = "Альтернативное имя"
STATS_LIST = sorted(col for col in df.columns if col not in META_COLUMNS and col != ALT_NAME_COL)
TEAMS_LIST = sorted(df[COL_TEAM].dropna().unique().tolist())


def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def log_stat_match(query: str, resolved: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "timestamp": dt.datetime.now().isoformat(),
        "query": query,
        "resolved": resolved,
        "metadata": metadata or {},
    }
    try:
        with LOG_MATCH_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_stat_miss(query: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    try:
        with MISSED_QUERIES_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"query": query, "meta": metadata}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def lemmatize_word(word: str) -> str:
    if not MORPH or not word:
        return word
    try:
        return MORPH.parse(word)[0].normal_form
    except Exception:
        return word


def normalize_tokens(text: str, use_lemmas: bool = False) -> List[str]:
    tokens = [t for t in TOKEN_SPLIT_RE.split(text) if t]
    result: List[str] = []
    for token in tokens:
        low = token.lower()
        if not low or low in STOPWORDS:
            continue
        if use_lemmas:
            lemma = lemmatize_word(low)
            if lemma and lemma not in STOPWORDS:
                result.append(lemma)
        else:
            result.append(low)
    return result


def tokens_to_key(text: str) -> str:
    return " ".join(normalize_tokens(text, use_lemmas=True))


STAT_RAG_VECTORIZER: Optional["TfidfVectorizer"] = None
STAT_RAG_MATRIX = None
STAT_RAG_INDEX: List[str] = []


def build_stat_rag() -> None:
    """Pre-compute TF-IDF vectors for stat names and their aliases."""
    global STAT_RAG_VECTORIZER, STAT_RAG_MATRIX, STAT_RAG_INDEX
    if not (TfidfVectorizer and cosine_similarity):
        STAT_RAG_VECTORIZER = None
        STAT_RAG_MATRIX = None
        STAT_RAG_INDEX = []
        return

    docs: List[str] = []
    index: List[str] = []
    for stat in STATS_LIST:
        alias_terms = [alias for alias, target in STAT_ALIASES.items() if target == stat]
        parts: set[str] = {
            stat,
            stat.replace("_", " "),
            stat.lower(),
            " ".join(normalize_tokens(stat, use_lemmas=False)),
            " ".join(normalize_tokens(stat, use_lemmas=True)),
        }
        for alias in alias_terms:
            parts.add(alias)
            parts.add(alias.replace("_", " "))
            parts.add(alias.lower())
            parts.add(" ".join(normalize_tokens(alias, use_lemmas=False)))
            parts.add(" ".join(normalize_tokens(alias, use_lemmas=True)))
        doc = " ".join(sorted(p for p in parts if p))
        docs.append(doc)
        index.append(stat)

    if not docs:
        STAT_RAG_VECTORIZER = None
        STAT_RAG_MATRIX = None
        STAT_RAG_INDEX = []
        return

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(docs)
    STAT_RAG_VECTORIZER = vectorizer
    STAT_RAG_MATRIX = matrix
    STAT_RAG_INDEX = index


def rag_resolve_stat_col(query: str, min_score: float = 0.35) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    """Resolve stat name using TF-IDF similarity."""
    if not query or not query.strip():
        return None, []
    if not (STAT_RAG_VECTORIZER and STAT_RAG_MATRIX is not None and STAT_RAG_INDEX):
        return None, []

    vec = STAT_RAG_VECTORIZER.transform([query])
    scores = cosine_similarity(vec, STAT_RAG_MATRIX).flatten()
    ranked_indices = scores.argsort()[::-1]
    ranked = [(STAT_RAG_INDEX[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]
    if not ranked:
        return None, []

    best_name, best_score = ranked[0]
    if best_score >= min_score:
        return best_name, ranked[:5]
    return None, ranked[:5]


build_stat_rag()

def fuzzy_match_team(query: str, teams: Iterable[str], threshold: int = 80) -> Tuple[Optional[str], List[str]]:
    query_norm = (query or "").lower().strip()
    if not query_norm:
        return None, []
    best_match: Optional[str] = None
    best_score = threshold
    for team in teams:
        score = max(fuzz.ratio(query_norm, team.lower()), fuzz.token_sort_ratio(query_norm, team.lower()))
        if score >= best_score:
            best_match = team
            best_score = score
    if not best_match:
        close = difflib.get_close_matches(query_norm, [t.lower() for t in teams], n=3, cutoff=0.6)
        lower_map = {t.lower(): t for t in teams}
        suggestions = [lower_map.get(item, item) for item in close]
        return None, suggestions
    return best_match, []


def resolve_stat_col(
    query: str,
    columns: Iterable[str],
    aliases: Optional[Dict[str, str]] = None,
    threshold: int = 85,
) -> Tuple[Optional[str], List[Tuple[str, int]], Dict[str, Any]]:
    aliases = aliases or {}
    original_query = query or ""
    q = original_query.lower().strip()
    if not q:
        return None, [], {"method": "empty", "score": None}

    rag_candidate, rag_rankings = rag_resolve_stat_col(original_query)
    rag_suggestions = [(name, int(round(score * 100))) for name, score in rag_rankings]
    rag_meta: Dict[str, Any] = {}
    if rag_rankings:
        rag_meta = {
            "rag_top": rag_rankings[0][0],
            "rag_score": round(rag_rankings[0][1], 3),
            "rag_suggestions": rag_suggestions[:5],
        }

    alias_hit = aliases.get(q)
    if MORPH and not alias_hit:
        lemma_key = tokens_to_key(q)
        alias_hit = STAT_ALIAS_LEMMAS.get(lemma_key)
    if alias_hit and alias_hit in columns:
        meta = {"method": "alias", "score": 100, "matched": alias_hit, **rag_meta}
        alternatives = [item for item in rag_suggestions if item[0] != alias_hit][:5]
        return alias_hit, alternatives, meta

    if rag_candidate and rag_candidate in columns:
        meta = {
            "method": "rag",
            "score": int(round(rag_rankings[0][1] * 100)),
            "matched": rag_candidate,
            **rag_meta,
        }
        alternatives = [item for item in rag_suggestions if item[0] != rag_candidate][:5]
        return rag_candidate, alternatives, meta

    raw_tokens = normalize_tokens(q, use_lemmas=False)
    lemma_tokens = normalize_tokens(q, use_lemmas=True) if MORPH else []

    column_lemma_cache: Dict[str, str] = {}
    for tokens, use_lemmas in ((raw_tokens, False), (lemma_tokens, True)):
        if not tokens:
            continue
        for c in columns:
            haystack = c.lower()
            if use_lemmas:
                if c not in column_lemma_cache:
                    column_lemma_cache[c] = " ".join(normalize_tokens(c.lower(), use_lemmas=True))
                haystack = column_lemma_cache[c]
            if tokens and all(t in haystack for t in tokens):
                meta = {
                    "method": "token",
                    "score": 100,
                    "matched": c,
                    "tokens": tokens,
                    "lemmas": use_lemmas,
                    **rag_meta,
                }
                alternatives = [item for item in rag_suggestions if item[0] != c][:5]
                return c, alternatives, meta

    scored: List[Tuple[str, int]] = []
    for c in columns:
        score = max(fuzz.partial_ratio(q, c.lower()), fuzz.token_sort_ratio(q, c.lower()))
        if score >= threshold:
            scored.append((c, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored:
        best = scored[0]
        meta = {"method": "fuzzy", "score": best[1], "matched": best[0], **rag_meta}
        alternatives: List[Tuple[str, int]] = []
        for name, score in rag_suggestions:
            if name != best[0]:
                alternatives.append((name, score))
        for candidate in scored[1:]:
            if not any(item[0] == candidate[0] for item in alternatives):
                alternatives.append(candidate)
        return best[0], alternatives[:5], meta

    suggestions: List[Tuple[str, int]] = []
    for c in columns:
        score = max(fuzz.partial_ratio(q, c.lower()), fuzz.token_set_ratio(q, c.lower()))
        suggestions.append((c, score))
    suggestions.sort(key=lambda item: item[1], reverse=True)
    merged: List[Tuple[str, int]] = []
    seen: set[str] = set()
    for name, score in rag_suggestions:
        if name not in seen:
            merged.append((name, score))
            seen.add(name)
    for name, score in suggestions:
        if name not in seen:
            merged.append((name, score))
            seen.add(name)
    return None, merged[:5], {"method": "fuzzy_none", "score": None, **rag_meta}


def clean_to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = (
        s.str.replace("%", "", regex=False)
        .str.replace("\u202f", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")


def last_n(team: str, n: int, frame: pd.DataFrame) -> pd.DataFrame:
    team_df = frame[frame[COL_TEAM] == team].copy()
    if team_df.empty:
        return team_df
    if not pd.api.types.is_datetime64_any_dtype(team_df[COL_DATE]):
        team_df[COL_DATE] = pd.to_datetime(team_df[COL_DATE], errors="coerce")
    return team_df.sort_values(COL_DATE, ascending=False).head(n)


def get_stat_last_n_matches(team: str, stat_col: str, n: int = 5, frame: pd.DataFrame = df) -> Dict[str, Any]:
    orig_team = team
    team, close_matches = fuzzy_match_team(team, frame[COL_TEAM].unique())
    if not team:
        return {
            "error": f"Команда '{orig_team}' не найдена.",
            "suggestions": close_matches or [],
        }

    resolved_col, stat_suggestions, stat_debug = resolve_stat_col(stat_col, frame.columns, STAT_ALIASES)
    if not resolved_col or resolved_col not in frame.columns:
        available_stats = [
            col
            for col in frame.columns
            if col not in META_COLUMNS and not frame.loc[frame[COL_TEAM] == team, col].isna().all()
        ]
        log_stat_miss(stat_col, {"available_for_team": available_stats, **(stat_debug or {})})
        if stat_suggestions:
            return {
                "error": f"Показатель '{stat_col}' не найден.",
                "suggestions": stat_suggestions,
                "team": team,
            }
        return {
            "error": f"Показатель '{stat_col}' не найден.",
            "available_for_team": available_stats[:10],
            "team": team,
        }

    log_stat_match(stat_col, resolved_col, stat_debug)

    team_df = frame[frame[COL_TEAM] == team].copy()
    if team_df.empty:
        return {"error": f"Нет матчей для команды '{team}'."}
    if not pd.api.types.is_datetime64_any_dtype(team_df[COL_DATE]):
        team_df[COL_DATE] = pd.to_datetime(team_df[COL_DATE], errors="coerce")
    team_df = team_df.sort_values(COL_DATE, ascending=False).head(int(n))

    series = clean_to_numeric(team_df[resolved_col])
    values = series.dropna().tolist()
    if not values:
        return {"error": f"Нет числовых значений для '{resolved_col}' у команды {team}."}

    unit = "%"
    if "%" not in resolved_col and "процент" not in resolved_col.lower():
        unit = ""

    if unit == "%" and all(0 <= v <= 1 for v in values):
        values = [v * 100 for v in values]

    average = float(np.mean(values))
    return {
        "team": team,
        "stat": resolved_col,
        "n_requested": int(n),
        "matches_used": len(values),
        "average": round(average, 2),
        "unit": unit,
        "raw_values": values,
    }


def compare_stats(team1: str, team2: str, stat_col: str, n: int = 5, frame: pd.DataFrame = df) -> Dict[str, Any]:
    real_team1, close_matches1 = fuzzy_match_team(team1, frame[COL_TEAM].unique())
    real_team2, close_matches2 = fuzzy_match_team(team2, frame[COL_TEAM].unique())
    if not real_team1 or not real_team2:
        close = sorted(set((close_matches1 or []) + (close_matches2 or [])))
        return {
            "error": f"Команда '{team1}' или '{team2}' не найдена.",
            "suggestions": close,
        }

    real_col, stat_suggestions, stat_debug = resolve_stat_col(stat_col, frame.columns, STAT_ALIASES)
    if not real_col or real_col not in frame.columns:
        log_stat_miss(stat_col, {"team_hint": real_team1, "suggestions": stat_suggestions, **(stat_debug or {})})
        return {
            "error": f"Показатель '{stat_col}' не найден.",
            "suggestions": stat_suggestions,
            "team_hint": real_team1,
        }
    log_stat_match(stat_col, real_col, stat_debug)

    def collect(team_name: str) -> pd.Series:
        subset = last_n(team_name, int(n), frame)
        return clean_to_numeric(subset[real_col])

    s1 = collect(real_team1).dropna()
    s2 = collect(real_team2).dropna()
    if s1.empty or s2.empty:
        return {"error": "Недостаточно данных для сравнения."}

    unit = "%"
    if "%" not in real_col and "процент" not in real_col.lower():
        unit = ""

    mean1 = float(s1.mean())
    mean2 = float(s2.mean())
    if unit == "%" and (0 <= mean1 <= 1) and (0 <= mean2 <= 1):
        mean1 *= 100
        mean2 *= 100

    diff = mean1 - mean2
    ratio = mean1 / mean2 if mean2 not in (0, None) else None
    return {
        "team1": real_team1,
        "team2": real_team2,
        "stat": real_col,
        "value1": round(mean1, 2),
        "value2": round(mean2, 2),
        "diff": round(diff, 2),
        "ratio": round(ratio, 2) if ratio is not None else None,
        "unit": unit,
        "matches_used_team1": int(s1.count()),
        "matches_used_team2": int(s2.count()),
    }


def get_match_history(
    team1: str,
    team2: str,
    limit: int = 20,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    home_away: str = "any",
    frame: pd.DataFrame = df,
) -> Dict[str, Any]:
    real_team1, close_matches1 = fuzzy_match_team(team1, frame[COL_TEAM].unique())
    real_team2, close_matches2 = fuzzy_match_team(team2, frame[COL_TEAM].unique())
    if not real_team1 or not real_team2:
        close = sorted(set((close_matches1 or []) + (close_matches2 or [])))
        return {
            "error": f"Не удалось найти '{team1}' или '{team2}'.",
            "suggestions": close,
        }

    work_df = frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(work_df[COL_DATE]):
        work_df[COL_DATE] = pd.to_datetime(work_df[COL_DATE], errors="coerce")

    mask = (
        work_df[COL_MATCH].astype(str).str.contains(real_team1, case=False, na=False)
        & work_df[COL_MATCH].astype(str).str.contains(real_team2, case=False, na=False)
    )
    subset = work_df.loc[mask].copy()

    if from_date:
        subset = subset[subset[COL_DATE] >= pd.to_datetime(from_date, errors="coerce")]
    if to_date:
        subset = subset[subset[COL_DATE] <= pd.to_datetime(to_date, errors="coerce")]

    if home_away == "home":
        subset = subset[subset[COL_MATCH].str.lower().str.startswith(real_team1.lower())]
    elif home_away == "away":
        subset = subset[~subset[COL_MATCH].str.lower().str.startswith(real_team1.lower())]

    if subset.empty:
        return {"error": f"Совместных матчей {real_team1} и {real_team2} не найдено."}

    key_stats = [
        COL_EXPECTED_GOALS,
        COL_POSSESSION,
        COL_SHOTS_ON_TARGET,
        COL_PASSES,
        COL_DUELS,
        COL_INTERCEPTIONS,
    ]

    games: List[Dict[str, Any]] = []
    for (match_name, match_date), group in subset.groupby([COL_MATCH, COL_DATE]):
        row1 = group[group[COL_TEAM].str.lower() == real_team1.lower()]
        row2 = group[group[COL_TEAM].str.lower() == real_team2.lower()]
        if row1.empty or row2.empty:
            continue

        g1 = pd.to_numeric(row1[COL_GOALS_FOR], errors="coerce").iloc[0]
        g2 = pd.to_numeric(row2[COL_GOALS_FOR], errors="coerce").iloc[0]
        if pd.isna(g1) or pd.isna(g2):
            continue

        home_team = real_team1 if match_name.lower().startswith(real_team1.lower()) else real_team2
        away_team = real_team2 if home_team == real_team1 else real_team1
        home_goals = g1 if home_team == real_team1 else g2
        away_goals = g2 if home_team == real_team1 else g1

        stats_team1 = {stat: clean_to_numeric(row1[stat]).iloc[0] for stat in key_stats if stat in row1.columns}
        stats_team2 = {stat: clean_to_numeric(row2[stat]).iloc[0] for stat in key_stats if stat in row2.columns}

        games.append(
            {
                "match": match_name,
                "date": match_date.date().isoformat() if isinstance(match_date, pd.Timestamp) else None,
                "home_team": home_team,
                "away_team": away_team,
                "score": f"{int(home_goals)}:{int(away_goals)}",
                "team1_goals": float(g1),
                "team2_goals": float(g2),
                "team1_stats": stats_team1,
                "team2_stats": stats_team2,
            }
        )

    games_sorted = sorted(
        games,
        key=lambda x: pd.to_datetime(x["date"], errors="coerce") if x["date"] else pd.Timestamp.min,
        reverse=True,
    )[: int(limit)]

    if not games_sorted:
        return {"error": f"Нет пересечений матчей для {real_team1} и {real_team2}."}

    t1_wins = sum(1 for g in games_sorted if g["team1_goals"] > g["team2_goals"])
    t2_wins = sum(1 for g in games_sorted if g["team2_goals"] > g["team1_goals"])
    draws = sum(1 for g in games_sorted if g["team1_goals"] == g["team2_goals"])

    total_goals_t1 = sum(g["team1_goals"] for g in games_sorted)
    total_goals_t2 = sum(g["team2_goals"] for g in games_sorted)
    avg_goals_t1 = round(total_goals_t1 / len(games_sorted), 2)
    avg_goals_t2 = round(total_goals_t2 / len(games_sorted), 2)

    total_sot_t1 = sum(g["team1_stats"].get(COL_SHOTS_ON_TARGET, 0) for g in games_sorted)
    total_sot_t2 = sum(g["team2_stats"].get(COL_SHOTS_ON_TARGET, 0) for g in games_sorted)
    avg_possession_t1 = round(
        sum(g["team1_stats"].get(COL_POSSESSION, 0) for g in games_sorted) / len(games_sorted), 2
    )
    avg_possession_t2 = round(
        sum(g["team2_stats"].get(COL_POSSESSION, 0) for g in games_sorted) / len(games_sorted), 2
    )

    summary = {
        "team1": real_team1,
        "team2": real_team2,
        "matches": len(games_sorted),
        "team1_wins": int(t1_wins),
        "team2_wins": int(t2_wins),
        "draws": int(draws),
        "total_goals_team1": float(total_goals_t1),
        "total_goals_team2": float(total_goals_t2),
        "avg_goals_team1": avg_goals_t1,
        "avg_goals_team2": avg_goals_t2,
        "goal_difference_team1": float(total_goals_t1 - total_goals_t2),
        "total_shots_on_target_team1": float(total_sot_t1),
        "total_shots_on_target_team2": float(total_sot_t2),
        "avg_possession_team1": avg_possession_t1,
        "avg_possession_team2": avg_possession_t2,
    }

    return {
        "team1": real_team1,
        "team2": real_team2,
        "summary": summary,
        "matches": games_sorted,
    }


def home_vs_away_performance(
    team: str,
    n: Optional[int] = None,
    season: Optional[str] = None,
    frame: pd.DataFrame = df,
) -> Dict[str, Any]:
    real_team, suggestions = fuzzy_match_team(team, frame[COL_TEAM].unique())
    if not real_team:
        return {"error": "Команда не найдена.", "suggestions": suggestions}

    team_df = frame[frame[COL_TEAM] == real_team].copy()
    if season and COL_SEASON in team_df.columns:
        team_df = team_df[team_df[COL_SEASON] == season]
    if n:
        team_df = team_df.sort_values(COL_DATE, ascending=False).head(int(n))

    home_df = team_df[team_df[COL_MATCH].str.lower().str.startswith(real_team.lower())]
    away_df = team_df[~team_df[COL_MATCH].str.lower().str.startswith(real_team.lower())]

    key_stats = [COL_GOALS_FOR, COL_EXPECTED_GOALS, COL_POSSESSION, "Удары"]
    home_avgs = {stat: round(clean_to_numeric(home_df[stat]).mean(), 2) for stat in key_stats if stat in home_df.columns}
    away_avgs = {stat: round(clean_to_numeric(away_df[stat]).mean(), 2) for stat in key_stats if stat in away_df.columns}

    return {
        "team": real_team,
        "home_matches": int(len(home_df)),
        "away_matches": int(len(away_df)),
        "home_avgs": home_avgs,
        "away_avgs": away_avgs,
    }


def opponent_strength_analysis(team: str, n: int = 10, frame: pd.DataFrame = df) -> Dict[str, Any]:
    real_team, suggestions = fuzzy_match_team(team, frame[COL_TEAM].unique())
    if not real_team:
        return {"error": "Команда не найдена.", "suggestions": suggestions}

    goals_for = clean_to_numeric(frame[COL_GOALS_FOR])
    goals_against = clean_to_numeric(frame[COL_GOALS_AGAINST])
    points = np.where(goals_for > goals_against, 3, np.where(goals_for == goals_against, 1, 0))
    frame_points = frame.assign(_Points=points)
    team_points = frame_points.groupby(COL_TEAM)["_Points"].mean()

    team_df = last_n(real_team, int(n), frame_points)
    if team_df.empty:
        return {"error": f"Нет матчей для команды {real_team}."}
    team_df = team_df.assign(
        Opponent=team_df[COL_OPPONENT],
        Opp_Strength=team_df[COL_OPPONENT].map(team_points),
    )

    median_strength = team_points.median()
    vs_top = team_df[team_df["Opp_Strength"] > median_strength]
    vs_bottom = team_df[team_df["Opp_Strength"] <= median_strength]

    key_stats = [COL_GOALS_FOR, COL_EXPECTED_GOALS]
    vs_top_avgs = {stat: round(clean_to_numeric(vs_top[stat]).mean(), 2) for stat in key_stats if stat in vs_top.columns}
    vs_bottom_avgs = {
        stat: round(clean_to_numeric(vs_bottom[stat]).mean(), 2) for stat in key_stats if stat in vs_bottom.columns
    }

    return {
        "team": real_team,
        "matches_vs_top": int(len(vs_top)),
        "matches_vs_bottom": int(len(vs_bottom)),
        "vs_top_avgs": vs_top_avgs,
        "vs_bottom_avgs": vs_bottom_avgs,
    }


def plot_radar_matplotlib(
    team_avgs: Dict[str, float],
    comp_avgs: Dict[str, float],
    labels: List[str],
    team_label: str,
    comp_label: str,
) -> matplotlib.figure.Figure:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    team_data = list(team_avgs.values()) + [list(team_avgs.values())[0]]
    comp_data = list(comp_avgs.values()) + [list(comp_avgs.values())[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, team_data, color="tab:blue", alpha=0.25, label=team_label)
    ax.plot(angles, team_data, color="tab:blue", linewidth=2)
    ax.fill(angles, comp_data, color="tab:red", alpha=0.25, label=comp_label)
    ax.plot(angles, comp_data, color="tab:red", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, wrap=True)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    ax.set_title(f"{team_label} vs {comp_label} (нормированная шкала 0-10)")
    return fig


def multi_stat_radar(
    team: str,
    compare_to: Optional[str] = None,
    n: int = 5,
    stats: Optional[List[str]] = None,
    frame: pd.DataFrame = df,
) -> Dict[str, Any]:
    real_team, suggestions = fuzzy_match_team(team, frame[COL_TEAM].unique())
    if not real_team:
        return {"error": "Команда не найдена.", "suggestions": suggestions}

    stats = stats or [COL_GOALS_FOR, COL_EXPECTED_GOALS, COL_SHOTS_ON_TARGET, COL_POSSESSION, COL_PASSES]

    team_df = last_n(real_team, int(n), frame)
    if team_df.empty:
        return {"error": f"Нет матчей для команды {real_team}."}

    if COL_SEASON in team_df.columns and not team_df[COL_SEASON].isna().all():
        season = team_df[COL_SEASON].iloc[0]
    else:
        season = team_df[COL_DATE].dt.year.mode().iloc[0] if not team_df[COL_DATE].isna().all() else None
    if season is None:
        return {"error": f"Не удалось определить сезон для {real_team}."}

    if COL_SEASON in frame.columns:
        league_df = frame[frame[COL_SEASON] == season]
    else:
        league_df = frame[frame[COL_DATE].dt.year == season]
    if league_df.empty:
        return {"error": f"Нет данных лиги за сезон {season}."}

    team_avgs = {
        stat: clean_to_numeric(team_df[stat]).mean()
        for stat in stats
        if stat in team_df.columns and not clean_to_numeric(team_df[stat]).isna().all()
    }
    if not team_avgs:
        return {"error": f"Нет статистики по выбранным метрикам для {real_team}."}

    if compare_to:
        comp_team, comp_suggestions = fuzzy_match_team(compare_to, frame[COL_TEAM].unique())
        if not comp_team:
            return {"error": f"Команда для сравнения '{compare_to}' не найдена.", "suggestions": comp_suggestions}
        comp_df = last_n(comp_team, int(n), frame)
        if comp_df.empty:
            return {"error": f"Нет матчей для команды {comp_team}."}
        comp_avgs = {
            stat: clean_to_numeric(comp_df[stat]).mean()
            for stat in stats
            if stat in comp_df.columns and not clean_to_numeric(comp_df[stat]).isna().all()
        }
        if not comp_avgs:
            return {"error": f"Нет статистики по выбранным метрикам для {comp_team}."}
        compare_label = comp_team
    else:
        comp_avgs = {
            stat: clean_to_numeric(league_df[stat]).mean()
            for stat in stats
            if stat in league_df.columns and not clean_to_numeric(league_df[stat]).isna().all()
        }
        compare_label = "среднее по лиге"

    all_stats = sorted(set(team_avgs.keys()) | set(comp_avgs.keys()))
    normalized_team: Dict[str, float] = {}
    normalized_comp: Dict[str, float] = {}

    for stat in all_stats:
        if stat not in league_df.columns:
            continue
        league_values = clean_to_numeric(league_df[stat]).dropna()
        if league_values.empty:
            league_min, league_max = 0, 1
        else:
            league_min, league_max = league_values.min(), league_values.max()
        if league_max == league_min or pd.isna(league_max):
            norm_team = norm_comp = 5
        else:
            norm_team = 10 * (team_avgs.get(stat, league_min) - league_min) / (league_max - league_min)
            norm_comp = 10 * (comp_avgs.get(stat, league_min) - league_min) / (league_max - league_min)
        normalized_team[stat] = max(0, min(10, round(norm_team, 2)))
        normalized_comp[stat] = max(0, min(10, round(norm_comp, 2)))

    fig = plot_radar_matplotlib(normalized_team, normalized_comp, all_stats, real_team, compare_label)
    return {
        "message": f"Радар сравнения {real_team} против {compare_label}.",
        "figure": fig,
        "team": real_team,
        "compare_label": compare_label,
        "stats": all_stats,
    }


def plot_stat(team: str, stat_col: str, n: Optional[int] = None, frame: pd.DataFrame = df) -> Dict[str, Any]:
    real_team, suggestions = fuzzy_match_team(team, frame[COL_TEAM].unique())
    if not real_team:
        return {"error": f"Команда '{team}' не найдена.", "suggestions": suggestions}

    real_col, stat_suggestions, stat_debug = resolve_stat_col(stat_col, frame.columns, STAT_ALIASES)
    if not real_col or real_col not in frame.columns:
        log_stat_miss(stat_col, {"suggestions": stat_suggestions, **(stat_debug or {})})
        return {"error": f"Показатель '{stat_col}' не найден.", "suggestions": stat_suggestions}
    log_stat_match(stat_col, real_col, stat_debug)

    work_df = frame[frame[COL_TEAM] == real_team].copy()
    if not pd.api.types.is_datetime64_any_dtype(work_df[COL_DATE]):
        work_df[COL_DATE] = pd.to_datetime(work_df[COL_DATE], errors="coerce")
    work_df = work_df.sort_values(COL_DATE, ascending=True)
    if n is not None:
        work_df = work_df.tail(int(n))
    if work_df.empty:
        return {"error": f"Нет матчей для команды '{real_team}'."}

    series = clean_to_numeric(work_df[real_col])
    mask = series.notna()
    values = series[mask].tolist()
    dates = work_df.loc[mask, COL_DATE]
    if not values:
        return {"error": f"Нет числовых значений для '{real_col}' у команды {real_team}."}

    unit = "%"
    if "%" not in real_col and "процент" not in real_col.lower():
        unit = ""
    if unit == "%" and all(0 <= v <= 1 for v in values):
        values = [v * 100 for v in values]

    labels = [
        date.date().isoformat() if isinstance(date, pd.Timestamp) and not pd.isna(date) else ""
        for date in dates
    ]

    return {
        "team": real_team,
        "stat": real_col,
        "labels": labels,
        "values": values,
        "unit": unit,
        "n_points": len(values),
    }


def plot_stat_matplotlib(team: str, stat_col: str, n: Optional[int] = None, frame: pd.DataFrame = df) -> Dict[str, Any]:
    result = plot_stat(team, stat_col, n, frame)
    if "error" in result:
        return result

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["labels"], result["values"], marker="o", color="tab:blue")
    ax.set_title(f"{result['stat']} — {result['team']}")
    ax.set_xlabel("Дата")
    ax.set_ylabel(f"{result['stat']} ({result['unit']})" if result["unit"] else result["stat"])
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate(rotation=45)
    result["figure"] = fig
    return result


@st.cache_resource(show_spinner=False)
def get_openai_client():
    api_key: Optional[str] = None

    secrets_source = getattr(st, "secrets", None)
    if secrets_source:
        try:
            secrets_dict = dict(secrets_source)
        except Exception:
            secrets_dict = {}
        for key_option in ("OPENAI_API_KEY", "openai_api_key"):
            candidate = secrets_dict.get(key_option)
            if isinstance(candidate, str) and candidate.strip():
                api_key = candidate.strip()
                break
        if not api_key:
            nested = secrets_dict.get("openai")
            if isinstance(nested, dict):
                for key_option in ("api_key", "API_KEY"):
                    candidate = nested.get(key_option)
                    if isinstance(candidate, str) and candidate.strip():
                        api_key = candidate.strip()
                        break
                if not api_key:
                    candidate = nested.get("key")
                    if isinstance(candidate, str) and candidate.strip():
                        api_key = candidate.strip()

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None
    return OpenAI(api_key=api_key)


OPENAI_CLIENT = get_openai_client()

SYSTEM_PROMPT = (
    "Ты аналитик Freedom QJ League. Отвечай кратко (1-2 предложения), "
    "используй факты из статистики и предлагай метрики для дальнейшего анализа."
)

FUNCTION_SCHEMAS = [
    {
        "name": "get_stat_last_n_matches",
        "description": "Средний показатель команды за последние N матчей.",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": TEAMS_LIST},
                "stat_col": {"type": "string", "enum": STATS_LIST},
                "n": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            },
            "required": ["team", "stat_col"],
        },
    },
    {
        "name": "compare_stats",
        "description": "Сравнение показателя между двумя командами.",
        "parameters": {
            "type": "object",
            "properties": {
                "team1": {"type": "string", "enum": TEAMS_LIST},
                "team2": {"type": "string", "enum": TEAMS_LIST},
                "stat_col": {"type": "string", "enum": STATS_LIST},
                "n": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            },
            "required": ["team1", "team2", "stat_col"],
        },
    },
    {
        "name": "get_match_history",
        "description": "История очных встреч.",
        "parameters": {
            "type": "object",
            "properties": {
                "team1": {"type": "string", "enum": TEAMS_LIST},
                "team2": {"type": "string", "enum": TEAMS_LIST},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                "home_away": {"type": "string", "enum": ["home", "away", "any"], "default": "any"},
                "from_date": {"type": ["string", "null"]},
                "to_date": {"type": ["string", "null"]},
            },
            "required": ["team1", "team2"],
        },
    },
    {
        "name": "home_vs_away_performance",
        "description": "Домашние против гостевых показателей.",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": TEAMS_LIST},
                "n": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
                "season": {"type": ["string", "null"]},
            },
            "required": ["team"],
        },
    },
    {
        "name": "opponent_strength_analysis",
        "description": "Показатели против соперников разной силы.",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": TEAMS_LIST},
                "n": {"type": "integer", "minimum": 3, "maximum": 50, "default": 10},
            },
            "required": ["team"],
        },
    },
    {
        "name": "multi_stat_radar",
        "description": "Радарная диаграмма по нескольким показателям.",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": TEAMS_LIST},
                "compare_to": {"type": ["string", "null"], "enum": TEAMS_LIST + [None]},
                "n": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "stats": {"type": ["array", "null"], "items": {"type": "string", "enum": STATS_LIST}},
            },
            "required": ["team"],
        },
    },
    {
        "name": "plot_stat",
        "description": "Линейный график показателя.",
        "parameters": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": TEAMS_LIST},
                "stat_col": {"type": "string", "enum": STATS_LIST},
                "n": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
            },
            "required": ["team", "stat_col"],
        },
    },
]


def run_openai_query(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not OPENAI_CLIENT:
        return {"error": "OpenAI не настроен. Установите переменную окружения OPENAI_API_KEY."}

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=api_messages,
        functions=FUNCTION_SCHEMAS,
        function_call="auto",
    )
    message = response.choices[0].message
    figure = None
    chart_result: Optional[Dict[str, Any]] = None

    if message.function_call:
        func_name = message.function_call.name
        arguments = json.loads(message.function_call.arguments or "{}")
        if func_name == "get_stat_last_n_matches":
            result = get_stat_last_n_matches(**arguments)
        elif func_name == "compare_stats":
            result = compare_stats(**arguments)
        elif func_name == "get_match_history":
            result = get_match_history(**arguments)
        elif func_name == "home_vs_away_performance":
            result = home_vs_away_performance(**arguments)
        elif func_name == "opponent_strength_analysis":
            result = opponent_strength_analysis(**arguments)
        elif func_name == "multi_stat_radar":
            result = multi_stat_radar(**arguments)
            if isinstance(result, dict) and "figure" in result:
                figure = result["figure"]
                result = {k: v for k, v in result.items() if k != "figure"}
            chart_result = result if isinstance(result, dict) else None
        elif func_name == "plot_stat":
            result = plot_stat_matplotlib(**arguments)
            if isinstance(result, dict) and "figure" in result:
                figure = result["figure"]
                result = {k: v for k, v in result.items() if k != "figure"}
            chart_result = result if isinstance(result, dict) else None
        else:
            result = {"error": f"Функция {func_name} не поддерживается."}

        serializable = make_json_serializable(result)
        follow_up_messages = api_messages + [
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": func_name,
                    "arguments": message.function_call.arguments or "{}",
                },
            },
            {
                "role": "function",
                "name": func_name,
                "content": json.dumps(serializable, ensure_ascii=False),
            },
        ]

        follow_up = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=follow_up_messages,
        )
        final_text = follow_up.choices[0].message.content or ""
        return {"text": final_text, "result": result, "figure": figure, "chart": chart_result}

    return {"text": message.content or ""}


logo_col, title_col = st.columns([1, 12])
with logo_col:
    st.image("logo.png", width=64)
with title_col:
    st.title("Freedom QJ League — интерактивная аналитика")

sidebar_choice = st.sidebar.radio(
    "Разделы",
    [
        "Среднее по N матчам",
        "Сравнение команд",
        "История встреч",
        "Дом / Выезд",
        "Сила соперников",
        "Радарная диаграмма",
        "Линейный график показателя",
        "AI ассистент",
    ],
)

if sidebar_choice == "Среднее по N матчам":
    st.subheader("Средний показатель за последние N матчей")
    with st.form("avg_last_n"):
        team = st.selectbox("Команда", TEAMS_LIST, key="avg_team")
        stat = st.selectbox("Показатель", STATS_LIST, key="avg_stat")
        n = st.slider("Количество матчей (N)", 1, 50, 5, key="avg_n")
        submitted = st.form_submit_button("Рассчитать")
    if submitted:
        with st.spinner("Считаю..."):
            result = get_stat_last_n_matches(team, stat, n)
        if "error" in result:
            st.error(result["error"])
            if result.get("suggestions"):
                st.write("Варианты:", result["suggestions"])
        else:
            st.metric(
                f"{result['stat']} — {result['team']}",
                f"{result['average']}{result['unit']}",
                help=f"Использовано матчей: {result['matches_used']}",
            )
            st.write("Значения:", result["raw_values"])

elif sidebar_choice == "Сравнение команд":
    st.subheader("Сравнение показателя между командами")
    with st.form("compare_form"):
        team1 = st.selectbox("Команда 1", TEAMS_LIST, key="comp_team1")
        team2 = st.selectbox("Команда 2", TEAMS_LIST, key="comp_team2")
        stat = st.selectbox("Показатель", STATS_LIST, key="comp_stat")
        n = st.slider("Последние матчи (N)", 1, 50, 5, key="comp_n")
        submitted = st.form_submit_button("Сравнить")
    if submitted:
        with st.spinner("Сравниваю..."):
            result = compare_stats(team1, team2, stat, n)
        if "error" in result:
            st.error(result["error"])
            if result.get("suggestions"):
                st.write("Варианты:", result["suggestions"])
        else:
            col1, col2 = st.columns(2)
            col1.metric(result["team1"], f"{result['value1']}{result['unit']}")
            col2.metric(result["team2"], f"{result['value2']}{result['unit']}")
            st.write(f"Разница: {result['diff']}{result['unit']}")
            if result.get("ratio") is not None:
                st.write(f"Отношение: {result['ratio']}")

elif sidebar_choice == "История встреч":
    st.subheader("История очных противостояний")
    with st.form("history_form"):
        team1 = st.selectbox("Команда 1", TEAMS_LIST, key="hist_team1")
        team2 = st.selectbox("Команда 2", TEAMS_LIST, key="hist_team2")
        limit = st.slider("Предел матчей", 1, 50, 10, key="hist_limit")
        col_date1, col_date2 = st.columns(2)
        from_date = col_date1.text_input("c даты (YYYY-MM-DD)", key="hist_from")
        to_date = col_date2.text_input("по дату (YYYY-MM-DD)", key="hist_to")
        home_away = st.selectbox("Фильтр по месту", ["any", "home", "away"], key="hist_homeaway")
        submitted = st.form_submit_button("Показать историю")
    if submitted:
        with st.spinner("Готовлю историю..."):
            result = get_match_history(
                team1,
                team2,
                limit,
                from_date or None,
                to_date or None,
                home_away,
            )
        if "error" in result:
            st.error(result["error"])
            if result.get("suggestions"):
                st.write("Варианты:", result["suggestions"])
        else:
            summary = result["summary"]
            summary_text = (
                f"{summary['team1']} и {summary['team2']} сыграли {summary['matches']} матч(ей): "
                f"{summary['team1']} побед — {summary['team1_wins']}, {summary['team2']} — {summary['team2_wins']}, "
                f"ничьих — {summary['draws']}.\\n"
                f"Голы: {summary['team1']} {summary['total_goals_team1']} – {summary['total_goals_team2']} {summary['team2']} "
                f"(в среднем {summary['avg_goals_team1']} и {summary['avg_goals_team2']}).\\n"
                f"Удары в створ: {summary['total_shots_on_target_team1']} vs {summary['total_shots_on_target_team2']}; "
                f"владение: {summary['avg_possession_team1']}% vs {summary['avg_possession_team2']}%."
            )
            st.markdown(summary_text)
            st.dataframe(pd.DataFrame(result["matches"]))

elif sidebar_choice == "Дом / Выезд":
    st.subheader("Домашние и гостевые показатели")
    with st.form("home_away_form"):
        team = st.selectbox("Команда", TEAMS_LIST, key="ha_team")
        n = st.slider("Последние матчи", 0, 50, 0, key="ha_n")
        season_options = sorted(df[COL_SEASON].dropna().unique().tolist()) if COL_SEASON in df.columns else []
        season = st.selectbox("Сезон (опционально)", [""] + season_options, key="ha_season")
        submitted = st.form_submit_button("Показать")
    if submitted:
        season_value = season if season else None
        with st.spinner("Анализирую..."):
            result = home_vs_away_performance(team, n if n else None, season_value)
        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2 = st.columns(2)
            col1.metric("Домашние матчи", result["home_matches"])
            col2.metric("Гостевые матчи", result["away_matches"])
            home_avgs = ", ".join(f"{k}: {v}" for k, v in result["home_avgs"].items())
            away_avgs = ", ".join(f"{k}: {v}" for k, v in result["away_avgs"].items())
            summary_text = (
                f"Дома {result['team']} в среднем показывает {home_avgs or 'нет данных'}, "
                f"а на выезде — {away_avgs or 'нет данных'}."
            )
            st.markdown(summary_text)

elif sidebar_choice == "Сила соперников":
    st.subheader("Показатели против соперников разной силы")
    with st.form("strength_form"):
        team = st.selectbox("Команда", TEAMS_LIST, key="strength_team")
        n = st.slider("Последние матчи", 3, 50, 10, key="strength_n")
        submitted = st.form_submit_button("Показать")
    if submitted:
        with st.spinner("Сравниваю..."):
            result = opponent_strength_analysis(team, n)
        if "error" in result:
            st.error(result["error"])
        else:
            vs_top = ", ".join(f"{k}: {v}" for k, v in result["vs_top_avgs"].items())
            vs_bottom = ", ".join(f"{k}: {v}" for k, v in result["vs_bottom_avgs"].items())
            summary_text = (
                f"Против сильных соперников ({result['matches_vs_top']} матчей) {result['team']} в среднем имеет "
                f"{vs_top or 'нет данных'}, а против остальных ({result['matches_vs_bottom']} матчей) — "
                f"{vs_bottom or 'нет данных'}."
            )
            st.markdown(summary_text)

elif sidebar_choice == "Радарная диаграмма":
    st.subheader("Радарная диаграмма по метрикам")
    with st.form("radar_form"):
        team = st.selectbox("Команда", TEAMS_LIST, key="radar_team")
        compare_to = st.selectbox("Сравнить с", ["среднее по лиге"] + TEAMS_LIST, key="radar_compare")
        stats = st.multiselect("Метрики", STATS_LIST, default=[COL_GOALS_FOR, COL_EXPECTED_GOALS, COL_SHOTS_ON_TARGET])
        n = st.slider("Последние матчи", 1, 50, 5, key="radar_n")
        submitted = st.form_submit_button("Построить")
    if submitted:
        compare_value = None if compare_to == "среднее по лиге" else compare_to
        with st.spinner("Строю диаграмму..."):
            result = multi_stat_radar(team, compare_value, n, stats)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(result["message"])
            st.pyplot(result["figure"])
            plt.close(result["figure"])

elif sidebar_choice == "Линейный график показателя":
    st.subheader("Динамика показателя")
    with st.form("plot_form"):
        team = st.selectbox("Команда", TEAMS_LIST, key="plot_team")
        stat = st.selectbox("Показатель", STATS_LIST, key="plot_stat")
        n = st.slider("Последние матчи (0 — все)", 0, 50, 0, key="plot_n")
        submitted = st.form_submit_button("Построить")
    if submitted:
        n_value = n if n else None
        with st.spinner("Строю график..."):
            result = plot_stat_matplotlib(team, stat, n_value)
        if "error" in result:
            st.error(result["error"])
        else:
            st.pyplot(result["figure"])
            plt.close(result["figure"])

elif sidebar_choice == "AI ассистент":
    st.subheader("AI ассистент (требуется OPENAI_API_KEY)")
    if "ai_chat" not in st.session_state:
        st.session_state["ai_chat"] = []

    clear_col = st.columns([1, 5])[0]
    with clear_col:
        if st.button("Очистить диалог", use_container_width=True):
            st.session_state["ai_chat"] = []
            st.rerun()

    for msg in st.session_state["ai_chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("result"):
                with st.expander("Структурированный ответ", expanded=False):
                    st.json(make_json_serializable(msg["result"]))

    prompt = st.chat_input("Введите вопрос")
    if prompt:
        user_entry = {"role": "user", "content": prompt}
        st.session_state["ai_chat"].append(user_entry)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Получаю ответ..."):
            history_copy = [{"role": m["role"], "content": m["content"]} for m in st.session_state["ai_chat"]]
            response = run_openai_query(history_copy)

        if "error" in response:
            st.session_state["ai_chat"].pop()
            st.error(response["error"])
            st.stop()

        assistant_text = response.get("text") or ""
        assistant_entry: Dict[str, Any] = {"role": "assistant", "content": assistant_text}
        if response.get("result"):
            assistant_entry["result"] = response["result"]
        if response.get("chart"):
            assistant_entry["chart"] = response["chart"]

        st.session_state["ai_chat"].append(assistant_entry)

        with st.chat_message("assistant"):
            st.markdown(assistant_text)
            if assistant_entry.get("result"):
                with st.expander("Структурированный ответ", expanded=False):
                    st.json(make_json_serializable(assistant_entry["result"]))
            figure = response.get("figure")
            if figure:
                st.pyplot(figure)
                plt.close(figure)
