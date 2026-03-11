from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "database" / "player_features_base.db"
DEFAULT_TABLE = "player_features_base"

# Legacy heuristic components for constructing a proxy rating from base features.
# Prefer the data-driven rating in build_rating_pred (PCA + shrinkage).
POSITIVE_FEATURES = {
    "goals_assists_per_90": 1.0,
    "goals_per_90_overall": 0.7,
    "assists_per_90_overall": 0.7,
    "clean_sheets_per_90_derived": 0.6,
}

NEGATIVE_FEATURES = {
    "cards_per_90_derived": 0.5,
    "cards_per_90_overall": 0.3,
    "conceded_per_90_overall": 0.6,
    "red_cards_per_90_derived": 0.3,
}

COMPONENT_DEFS: dict[str, list[tuple[str, float]]] = {
    # Positive direction means higher is better; negative gets flipped before scoring.
    "attack_score": [
        ("goals_per_90_overall", 1.0),
        ("goals_involved_per_90_overall", 1.0),
        ("goals_assists_per_90", 1.0),
    ],
    "assist_score": [
        ("assists_per_90_overall", 1.0),
    ],
    "conceded_score": [
        ("conceded_per_90_overall", -1.0),
    ],
    "tackle_score": [
        ("tackles_per_90_overall", 1.0),
        ("tackles_successful_per_90_overall", 1.0),
    ],
    "foul_card_score": [
        ("cards_per_90_derived", -1.0),
        ("cards_per_90_overall", -1.0),
        ("red_cards_per_90_derived", -1.0),
        ("yellow_cards_per_90_derived", -1.0),
    ],
    "goalkeeper_score": [
        ("saves_per_90_overall", 1.0),
        ("save_percentage_overall", 1.0),
        ("clean_sheets_per_90_derived", 1.0),
    ],
    "appearance_score": [
        ("minutes_played_overall", 1.0),
        ("appearances_overall", 1.0),
        ("nineties", 1.0),
        ("minutes_per_appearance", 1.0),
        ("age", -0.2),
    ],
}

POSITION_TARGETS: dict[str, list[str]] = {
    "GK": ["goalkeeper_score", "conceded_score", "foul_card_score", "appearance_score"],
    "DEF": ["attack_score","assist_score","conceded_score", "foul_card_score", "appearance_score"],
    "MID": ["attack_score","assist_score","conceded_score", "foul_card_score", "appearance_score"],
    "FWD": ["attack_score","assist_score","conceded_score", "foul_card_score", "appearance_score"],
    "OTHER": ["attack_score","assist_score","conceded_score", "foul_card_score", "appearance_score"],
}


def load_base_dataframe(
    db_path: Path | str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    db_path = Path(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def _normalize_position(series: pd.Series) -> pd.Series:
    def normalize_value(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "OTHER"
        text = str(value).strip().lower()
        if "goalkeeper" in text or text == "gk":
            return "GK"
        if "defender" in text or text == "def":
            return "DEF"
        if "midfielder" in text or text == "mid":
            return "MID"
        if "forward" in text or text == "fw" or "striker" in text:
            return "FWD"
        return "OTHER"

    return series.apply(normalize_value)


def _choose_position_col(
    df: pd.DataFrame,
    desired_positions: set[str] | None,
    position_col_candidates: tuple[str, ...],
) -> str | None:
    best_col = None
    best_score = -1

    for col in position_col_candidates:
        if col not in df.columns:
            continue
        normalized = _normalize_position(df[col])
        unique = set(normalized.unique())
        unique.discard("OTHER")
        if desired_positions:
            if not unique.intersection(desired_positions):
                continue
            score = len(unique.intersection(desired_positions))
        else:
            score = len(unique)

        if score > best_score:
            best_col = col
            best_score = score

    return best_col


def _pca_first_component_score(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")

    df = df.dropna(axis=1, how="all")
    if df.empty:
        return pd.Series(np.nan, index=df.index)

    X = df.to_numpy(dtype="float64", copy=True)
    # Replace NaNs with column means to avoid dropping rows.
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    # Standardize per column.
    col_std = np.nanstd(X, axis=0, ddof=0)
    col_std[col_std == 0] = 1.0
    X = (X - col_means) / col_std

    if X.shape[1] == 1:
        return pd.Series(X[:, 0], index=df.index)

    # First principal component via SVD (data-driven weights).
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc = vt[0]
    scores = X @ pc

    # Align sign so higher roughly means better (positive correlation with mean).
    mean_signal = X.mean(axis=1)
    if np.std(scores) == 0 or np.std(mean_signal) == 0:
        corr = 1.0
    else:
        corr = np.corrcoef(scores, mean_signal)[0, 1]
        if np.isnan(corr):
            corr = 1.0
    if corr < 0:
        scores = -scores

    return pd.Series(scores, index=df.index)


def _resolve_group_cols(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] | None,
    include_league: bool,
) -> list[str]:
    resolved: list[str] = []
    for col in group_cols or ():
        if col in df.columns:
            resolved.append(col)
    if include_league and "league" in df.columns and "league" not in resolved:
        resolved.append("league")
    return resolved


def _group_zscore(series: pd.Series, keys: list[pd.Series]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not keys:
        return _zscore(values)

    grouped = values.groupby(keys)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))
    std = std.replace(0, np.nan)
    z = (values - mean) / std
    valid_mask = values.notna()
    z = z.where(valid_mask, np.nan)
    z = z.mask(valid_mask & z.isna(), 0.0)
    return z


def _group_percentile(series: pd.Series, keys: list[pd.Series]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not keys:
        return values.rank(pct=True)
    return values.groupby(keys).rank(pct=True)


def _shrink_scores_by_minutes(
    scores: pd.Series,
    minutes: pd.Series | None,
    keys: list[pd.Series],
    k_strategy: str = "median",
) -> pd.Series:
    if minutes is None:
        return scores

    score_values = pd.to_numeric(scores, errors="coerce")
    minute_values = pd.to_numeric(minutes, errors="coerce").clip(lower=0)

    valid_mask = score_values.notna()
    if not keys:
        group_mean = score_values.mean()
        if k_strategy == "median":
            k = minute_values.median()
        else:
            k = minute_values.mean()
        if pd.isna(k) or k == 0:
            k = 1.0
        weight = minute_values / (minute_values + k)
        shrunk = weight * score_values + (1 - weight) * group_mean
        return shrunk.where(valid_mask, np.nan)

    grouped_scores = score_values.groupby(keys)
    grouped_minutes = minute_values.groupby(keys)
    group_mean = grouped_scores.transform("mean")
    if k_strategy == "mean":
        group_k = grouped_minutes.transform("mean")
    else:
        group_k = grouped_minutes.transform("median")
    group_k = group_k.replace(0, np.nan)
    group_k = group_k.fillna(group_k.median())
    group_k = group_k.fillna(1.0)

    weight = minute_values / (minute_values + group_k)
    shrunk = weight * score_values + (1 - weight) * group_mean
    return shrunk.where(valid_mask, np.nan)


def build_component_scores(
    df: pd.DataFrame,
    component_defs: dict[str, list[tuple[str, float]]] | None = None,
) -> pd.DataFrame:
    component_defs = component_defs or COMPONENT_DEFS
    scores: dict[str, pd.Series] = {}

    for name, cols in component_defs.items():
        active_cols = []
        for col, direction in cols:
            if col not in df.columns:
                continue
            if df[col].isna().all():
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if direction < 0:
                series = -series
            active_cols.append(series.rename(col))

        if not active_cols:
            scores[name] = pd.Series(np.nan, index=df.index)
            continue

        component_df = pd.concat(active_cols, axis=1)
        scores[name] = _pca_first_component_score(component_df)

    return pd.DataFrame(scores)


def build_targets_by_position(
    df: pd.DataFrame,
    position_col_candidates: tuple[str, ...] = ("position_group", "position"),
    component_defs: dict[str, list[tuple[str, float]]] | None = None,
    position_targets: dict[str, list[str]] | None = None,
    group_cols: tuple[str, ...] = ("season", "position_bucket"),
    include_league: bool = False,
    minutes_col: str = "minutes_played_overall",
    shrink_k_strategy: str = "median",
    apply_shrinkage: bool = True,
    apply_zscore: bool = True,
    apply_percentile: bool = False,
    percentile_scale: float = 10.0,
) -> pd.DataFrame:
    component_defs = component_defs or COMPONENT_DEFS
    position_targets = position_targets or POSITION_TARGETS

    desired_positions = set(position_targets.keys())
    desired_positions.discard("OTHER")
    position_col = _choose_position_col(df, desired_positions, position_col_candidates)

    targets = build_component_scores(df, component_defs=component_defs)
    if position_col is None:
        targets["position_bucket"] = "OTHER"
        return targets

    buckets = _normalize_position(df[position_col])
    targets["position_bucket"] = buckets

    all_components = list(component_defs.keys())
    for bucket, allowed in position_targets.items():
        mask = buckets == bucket
        if not mask.any():
            continue
        disallowed = [col for col in all_components if col not in allowed]
        targets.loc[mask, disallowed] = np.nan

    resolved_cols = _resolve_group_cols(
        targets.join(df, how="left"),
        group_cols,
        include_league=include_league,
    )
    keys = [targets[col] if col in targets.columns else df[col] for col in resolved_cols]

    minutes = df[minutes_col] if minutes_col in df.columns else None
    if apply_shrinkage and minutes is not None:
        for col in targets.columns:
            if col in ("position_bucket", "appearance_score"):
                continue
            targets[col] = _shrink_scores_by_minutes(
                targets[col],
                minutes,
                keys,
                k_strategy=shrink_k_strategy,
            )

    if apply_zscore:
        for col in targets.columns:
            if col == "position_bucket":
                continue
            targets[col] = _group_zscore(targets[col], keys)

    if apply_percentile:
        for col in targets.columns:
            if col == "position_bucket":
                continue
            pct = _group_percentile(targets[col], keys)
            targets[col] = (pct * percentile_scale).clip(lower=0.0, upper=percentile_scale)

    return targets


def build_rating_pred(
    df: pd.DataFrame,
    position_col_candidates: tuple[str, ...] = ("position_group", "position"),
    component_defs: dict[str, list[tuple[str, float]]] | None = None,
    position_targets: dict[str, list[str]] | None = None,
    group_cols: tuple[str, ...] = ("season", "position_bucket"),
    include_league: bool = False,
    minutes_col: str = "minutes_played_overall",
    shrink_k_strategy: str = "median",
    percentile_scale: float = 10.0,
    use_legacy_weights: bool = False,
    apply_zscore: bool = True,
    positive: dict[str, float] | None = None,
    negative: dict[str, float] | None = None,
    scale_center: float = 5.0,
    scale_factor: float = 1.5,
) -> pd.Series:
    # If an explicit rating column exists, prefer it.
    for col in ("average_rating_overall", "rating", "rating_overall", "rating_pred"):
        if col in df.columns:
            rating = pd.to_numeric(df[col], errors="coerce")
            return rating.rename("rating_pred")

    if use_legacy_weights:
        positive = positive or POSITIVE_FEATURES
        negative = negative or NEGATIVE_FEATURES

        zscores: dict[str, pd.Series] = {}
        weights: dict[str, float] = {}

        for col, weight in positive.items():
            if col in df.columns:
                zscores[col] = _zscore(df[col])
                weights[col] = weight

        for col, weight in negative.items():
            if col in df.columns:
                zscores[col] = _zscore(df[col])
                weights[col] = -weight

        if not zscores:
            return pd.Series(np.nan, index=df.index, name="rating_pred")

        zdf = pd.DataFrame(zscores)
        weight_series = pd.Series(weights)
        weighted = zdf.mul(weight_series, axis=1)

        denom = weight_series.abs().sum()
        score = weighted.sum(axis=1) / (denom if denom else 1.0)

        rating = (scale_center + scale_factor * score).clip(lower=0.0, upper=10.0)
        rating.name = "rating_pred"
        return rating

    component_defs = component_defs or COMPONENT_DEFS
    position_targets = position_targets or POSITION_TARGETS

    desired_positions = set(position_targets.keys())
    desired_positions.discard("OTHER")
    position_col = _choose_position_col(df, desired_positions, position_col_candidates)
    if position_col is None:
        buckets = pd.Series(["OTHER"] * len(df), index=df.index)
    else:
        buckets = _normalize_position(df[position_col])

    component_scores = build_component_scores(df, component_defs=component_defs)
    component_scores["position_bucket"] = buckets

    rating = pd.Series(np.nan, index=df.index, name="rating_pred")
    for bucket, allowed in position_targets.items():
        mask = buckets == bucket
        if not mask.any():
            continue
        components = [c for c in allowed if c in component_scores.columns and c != "appearance_score"]
        if not components:
            continue
        comp_df = component_scores.loc[mask, components].dropna(axis=1, how="all")
        if comp_df.empty or comp_df.dropna(how="all").empty:
            continue
        rating.loc[mask] = _pca_first_component_score(comp_df)

    group_df = component_scores.copy()
    if "season" in df.columns:
        group_df["season"] = df["season"]
    if "league" in df.columns:
        group_df["league"] = df["league"]
    resolved_cols = _resolve_group_cols(group_df, group_cols, include_league=include_league)
    keys = [group_df[col] for col in resolved_cols]

    minutes = df[minutes_col] if minutes_col in df.columns else None
    rating = _shrink_scores_by_minutes(rating, minutes, keys, k_strategy=shrink_k_strategy)
    if apply_zscore:
        rating = _group_zscore(rating, keys)
    rating.name = "rating_pred"
    return rating


def load_rating_pred(
    db_path: Path | str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
) -> pd.Series:
    df = load_base_dataframe(db_path=db_path, table=table)
    return build_rating_pred(df)


def load_position_targets(
    db_path: Path | str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    position_col_candidates: tuple[str, ...] = ("position_group", "position"),
    group_cols: tuple[str, ...] = ("season", "position_bucket"),
    include_league: bool = False,
    minutes_col: str = "minutes_played_overall",
    shrink_k_strategy: str = "median",
    apply_shrinkage: bool = True,
    apply_zscore: bool = True,
    apply_percentile: bool = False,
    percentile_scale: float = 10.0,
) -> pd.DataFrame:
    df = load_base_dataframe(db_path=db_path, table=table)
    return build_targets_by_position(
        df,
        position_col_candidates=position_col_candidates,
        group_cols=group_cols,
        include_league=include_league,
        minutes_col=minutes_col,
        shrink_k_strategy=shrink_k_strategy,
        apply_shrinkage=apply_shrinkage,
        apply_zscore=apply_zscore,
        apply_percentile=apply_percentile,
        percentile_scale=percentile_scale,
    )
