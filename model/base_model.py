# base_model_train.py
# ------------------------------------------------------------
# Train a "Base student model" that stays consistent with your self-research scoring system.
#
# Core idea:
#   - X: your base feature matrix (limited/common fields)
#   - Y: your self-research teacher outputs (rating_pred + subscores) computed by target.py
#
# This script:
#   1) loads base features from DB (via features.py)
#   2) computes teacher targets (via target.py)
#   3) trains CatBoost regressors for:
#        - rating_pred (always)
#        - subscores that have enough non-null samples (assist/conceded/foul_card/appearance, etc.)
#   4) saves models + metadata
#
# NOTE:
#   This open-source version only keeps the base training pipeline.
#   All model targets are distilled from the base feature database.
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.features import DEFAULT_DROP_COLUMNS, build_feature_matrix, load_base_dataframe
from base.target import COMPONENT_DEFS, build_targets_by_position, build_rating_pred


# ----------------------------
# Config
# ----------------------------
OUT_DIR = PROJECT_ROOT / "artifacts_base_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum samples to train a subscore model
MIN_TRAIN_SAMPLES = 300
LEAGUE_AWARE_NORMALIZATION = False
USE_TIME_SPLIT = True
USE_GROUP_SPLIT = True
SPLIT_GROUP_COLS = ("player_id", "season")
RUN_SELF_CHECKS = False
RUN_SPLIT_CHECKS = True

# Hard split settings
HARD_SPLIT = True
# Modes:
#   - time:   train=older seasons, valid=most recent K seasons
#   - league: time split + holdout a fraction of leagues (valid restricted to those leagues)
#   - player: time split + valid only contains players not seen in train
#   - combo:  league + player on top of time split (very hard; use for stress test)
HARD_SPLIT_MODE = "time"  # "time" | "league" | "player" | "combo"
HARD_SPLIT_REQUIRE_NEW_PLAYERS = False
HARD_SPLIT_HOLDOUT_LEAGUE_FRAC = 0.1
HARD_TIME_VALID_YEARS = 2
HARD_SPLIT_SEED = 42
HARD_SPLIT_MIN_TRAIN = 1000
HARD_SPLIT_MIN_VALID = 200

# You can control which subscores you WANT to output in Base
# (Some may be too sparse in base data; we will skip if not enough samples.)
DESIRED_SUBSCORES = [
    "attack_score",
    "assist_score",
    "conceded_score",
    "tackle_score",        # likely missing in base; will be skipped unless you later distill from Pro
    "foul_card_score",
    "goalkeeper_score",
    "appearance_score",
]

# For Base student training, rating is the main always-on target
MAIN_TARGET = "rating_pred"


def _target_feature_cols(target_name: str) -> list[str]:
    if target_name == MAIN_TARGET:
        return sorted({c for cols in COMPONENT_DEFS.values() for c, _ in cols})
    if target_name in COMPONENT_DEFS:
        return [c for c, _ in COMPONENT_DEFS[target_name]]
    return []


# ----------------------------
# Helpers
# ----------------------------
def _parse_season_start_year(season: str) -> Optional[int]:
    """Parse a season string into the *start* year.

    Supports common formats found in football datasets:
      - '2015/2016' -> 2015
      - '2015/16'   -> 2015
      - '2015-2016' -> 2015
      - '2015–2016' -> 2015  (en dash)
      - '2015'      -> 2015

    Returns None if it cannot parse.
    """
    if season is None or (isinstance(season, float) and np.isnan(season)):
        return None

    s = str(season).strip()
    if not s or s.lower() in {"none", "nan"}:
        return None

    # Normalize dash variants
    s = s.replace("–", "-").replace("—", "-")

    # 1) YYYY/YYYY or YYYY-YYYY
    m = re.match(r"^(\d{4})\s*[/\-]\s*(\d{4})$", s)
    if m:
        return int(m.group(1))

    # 2) YYYY/YY or YYYY-YY (e.g., 2024/25)
    m = re.match(r"^(\d{4})\s*[/\-]\s*(\d{2})$", s)
    if m:
        return int(m.group(1))

    # 3) any 4-digit year token in the string (fallback)
    m = re.search(r"(19\d{2}|20\d{2})", s)
    if m:
        return int(m.group(1))

    return None


def fill_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure position_group exists and is consistent.
    Your base data has many None in position_group; we fill from 'position'.
    """
    df = df.copy()

    if "position_group" not in df.columns:
        df["position_group"] = None

    pos_col = "position"
    if pos_col not in df.columns and "postion" in df.columns:
        pos_col = "postion"

    def map_pos_to_group(pos: object) -> str:
        if pos is None or (isinstance(pos, float) and np.isnan(pos)):
            return "OTHER"
        t = str(pos).strip().lower()
        if "goalkeeper" in t or t == "gk":
            return "GK"
        if "defender" in t or t == "def":
            return "DEF"
        if "midfielder" in t or t == "mid":
            return "MID"
        if "forward" in t or t == "fw" or "striker" in t:
            return "FWD"
        return "OTHER"

    # Fill missing
    mask = df["position_group"].isna() | (df["position_group"].astype(str).str.lower().isin(["none", "nan", ""]))
    if pos_col in df.columns:
        df.loc[mask, "position_group"] = df.loc[mask, pos_col].apply(map_pos_to_group)
    else:
        df.loc[mask, "position_group"] = "OTHER"

    # Normalize values
    df["position_group"] = df["position_group"].apply(map_pos_to_group)

    return df


def build_sample_weight(
    df: pd.DataFrame,
    minutes_col: str = "minutes_played_overall",
    appearance_cols: tuple[str, ...] = (
        "appearances_overall",
        "games_played",
        "games_started",
    ),
    min_w: float = 0.2,
    max_w: float = 2.0,
) -> pd.Series:
    """
    Stability weighting using an empirical-Bayes style shrinkage:
    weight = m / (m + k), with k estimated from data (per position group if available).
    Optionally blends appearances to avoid over-weighting a few long matches.
    """
    if minutes_col not in df.columns:
        return pd.Series(1.0, index=df.index)

    m = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0).clip(lower=0)

    df_pos = fill_position_group(df)
    group_key = df_pos["position_group"] if "position_group" in df_pos.columns else None

    if group_key is not None:
        k_m = m.groupby(group_key).transform("median")
        k_m = k_m.fillna(k_m.median())
    else:
        k_m = pd.Series(m.median(), index=m.index)
    k_m = k_m.replace(0, np.nan).fillna(1.0)

    w_m = m / (m + k_m)

    app_series = None
    for col in appearance_cols:
        if col in df.columns:
            app_series = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
            break

    if app_series is not None:
        if group_key is not None:
            k_a = app_series.groupby(group_key).transform("median")
            k_a = k_a.fillna(k_a.median())
        else:
            k_a = pd.Series(app_series.median(), index=app_series.index)
        k_a = k_a.replace(0, np.nan).fillna(1.0)
        w_a = app_series / (app_series + k_a)
        w = w_m * np.sqrt(w_a)
    else:
        w = w_m

    w = min_w + (max_w - min_w) * w
    return w.clip(lower=min_w, upper=max_w)


def _random_group_split(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float = 0.85,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = df[group_col].dropna().unique()
    if len(groups) == 0:
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(train_frac * len(idx))
        return idx[:cut], idx[cut:]

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    cut = int(train_frac * len(groups))
    train_groups = set(groups[:cut])
    train_idx = df.index[df[group_col].isin(train_groups)].to_numpy()
    valid_idx = df.index[~df[group_col].isin(train_groups)].to_numpy()
    return train_idx, valid_idx


def make_time_split(
    df: pd.DataFrame,
    season_col: str = "season",
    valid_years: int = 2,          # ✅ 改成最近2个赛季
    min_valid: int = 50,
    min_train: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if season_col not in df.columns:
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(0.85 * len(idx))
        return idx[:cut], idx[cut:]

    years = df[season_col].apply(_parse_season_start_year)
    if years.notna().sum() < 0.5 * len(df):
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(0.85 * len(idx))
        return idx[:cut], idx[cut:]

    df2 = df.copy()
    df2["_season_year"] = years.fillna(years.median())
    uniq = sorted(df2["_season_year"].unique())
    if len(uniq) <= 1:
        # only one season bucket -> fallback random
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(0.85 * len(idx))
        return idx[:cut], idx[cut:]

    # ✅ 最近K个赛季作为valid
    k = min(valid_years, len(uniq) - 1)
    valid_year_set = set(uniq[-k:])
    valid_idx = df2.index[df2["_season_year"].isin(valid_year_set)].to_numpy()
    train_idx = df2.index[~df2["_season_year"].isin(valid_year_set)].to_numpy()

    if len(valid_idx) < min_valid or len(train_idx) < min_train:
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(0.85 * len(idx))
        return idx[:cut], idx[cut:]

    return train_idx, valid_idx


def make_group_split(
    df: pd.DataFrame,
    group_cols: tuple[str, ...],
    train_frac: float = 0.85,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        return make_time_split(df)

    if len(cols) == 1:
        return _random_group_split(df, cols[0], train_frac=train_frac, seed=seed)

    group_key = df[cols].astype(str).agg("|".join, axis=1)
    return _random_group_split(df.assign(_group_key=group_key), "_group_key", train_frac=train_frac, seed=seed)


def make_hard_split(
    df: pd.DataFrame,
    mode: str = "time",
    season_col: str = "season",
    league_col: str = "league",
    player_col: str = "player_id",
    holdout_league_frac: float = HARD_SPLIT_HOLDOUT_LEAGUE_FRAC,
    require_new_players: bool = HARD_SPLIT_REQUIRE_NEW_PLAYERS,
    valid_years: int = HARD_TIME_VALID_YEARS,
    min_train: int = HARD_SPLIT_MIN_TRAIN,
    min_valid: int = HARD_SPLIT_MIN_VALID,
    seed: int = HARD_SPLIT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hard split helper.

    IMPORTANT: We always start from a time-aware split, then optionally apply:
      - league OOD: restrict valid to a held-out subset of leagues
      - player OOD: restrict valid to players not present in train

    Modes:
      - time:   time split only
      - league: time + league holdout
      - player: time + new players
      - combo:  time + league holdout + new players
    """

    mode = (mode or "time").strip().lower()
    if mode not in {"time", "league", "player", "combo"}:
        mode = "time"

    # 1) baseline: time split (recent K seasons as validation)
    train_idx, valid_idx = make_time_split(
        df,
        season_col=season_col,
        valid_years=valid_years,
        seed=seed,
        min_valid=50,
        min_train=200,
    )

    # If we only want time split, return early (but still enforce minimums)
    if mode == "time":
        if len(train_idx) < min_train or len(valid_idx) < min_valid:
            return make_time_split(df, season_col=season_col, valid_years=valid_years, seed=seed)
        return train_idx, valid_idx

    # 2) league holdout (only if requested)
    if mode in {"league", "combo"} and holdout_league_frac > 0 and league_col in df.columns:
        # Sample holdout leagues from leagues present in TRAIN to avoid empty holdout.
        leagues = df.loc[train_idx, league_col].dropna().unique()
        leagues = sorted(leagues.tolist())
        if len(leagues) >= 2:
            n_holdout = max(1, int(round(len(leagues) * holdout_league_frac)))
            rng = np.random.default_rng(seed)
            holdout_leagues = set(rng.choice(leagues, size=n_holdout, replace=False))

            # Restrict valid to those leagues (intersection with time-valid)
            valid_mask = df.loc[valid_idx, league_col].isin(holdout_leagues)
            valid_idx2 = df.loc[valid_idx].index[valid_mask].to_numpy()

            # Optionally, you can also exclude holdout leagues from train to make a purer league-OOD:
            # train_mask = ~df.loc[train_idx, league_col].isin(holdout_leagues)
            # train_idx2 = df.loc[train_idx].index[train_mask].to_numpy()
            train_idx2 = train_idx

            if len(train_idx2) >= min_train and len(valid_idx2) >= min_valid:
                train_idx, valid_idx = train_idx2, valid_idx2

    # 3) new player restriction (only if requested)
    if mode in {"player", "combo"} and require_new_players and player_col in df.columns:
        train_players = set(df.loc[train_idx, player_col].dropna().unique())
        valid_mask = ~df.loc[valid_idx, player_col].isin(train_players)
        valid_idx2 = df.loc[valid_idx].index[valid_mask].to_numpy()
        if len(train_idx) >= min_train and len(valid_idx2) >= min_valid:
            valid_idx = valid_idx2

    # 4) final guardrail
    if len(train_idx) < min_train or len(valid_idx) < min_valid:
        return make_time_split(df, season_col=season_col, valid_years=valid_years, seed=seed)


    return train_idx, valid_idx


# ------------------------------------------------------------
# Split summary helper
# ------------------------------------------------------------
def print_split_summary(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    split_name: str,
    season_col: str = "season",
    league_col: str = "league",
    player_col: str = "player_id",
    hard_mode: str | None = None,
    holdout_league_frac: float = 0.0,
    seed: int = 42,
    top_n_leagues: int = 10,
) -> None:
    """Print a compact, actionable summary of the chosen split.

    Helps answer: 
      - Which season buckets are in valid?
      - Are leagues in valid skewed / held out?
      - How much player leakage exists?

    This is diagnostic only; it does not change training.
    """

    def _safe_value_counts(s: pd.Series, n: int) -> list[tuple[str, int]]:
        if s is None or s.empty:
            return []
        vc = s.astype(str).value_counts(dropna=True)
        pairs = [(str(k), int(v)) for k, v in vc.head(n).items()]
        return pairs

    df_train = df.loc[train_idx] if len(train_idx) else df.iloc[0:0]
    df_valid = df.loc[valid_idx] if len(valid_idx) else df.iloc[0:0]

    print("\n=== Split summary ===")
    print(f"split={split_name} train_rows={len(train_idx)} valid_rows={len(valid_idx)}")

    # Season buckets
    if season_col in df.columns:
        years = df[season_col].apply(_parse_season_start_year)
        y_train = years.loc[train_idx].dropna().astype(int).tolist() if len(train_idx) else []
        y_valid = years.loc[valid_idx].dropna().astype(int).tolist() if len(valid_idx) else []
        train_set = sorted(set(y_train))
        valid_set = sorted(set(y_valid))
        if train_set or valid_set:
            print(f"season_years train={train_set[:10]}{'...' if len(train_set) > 10 else ''} valid={valid_set}")
        else:
            print("season_years: n/a (parse failed)")

        # Show a few raw season values from valid for debugging parser/data issues
        raw_valid = df_valid[season_col].dropna().astype(str)
        if not raw_valid.empty:
            raw_examples = raw_valid.value_counts().head(5)
            examples = [f"{k}({int(v)})" for k, v in raw_examples.items()]
            print(f"valid season raw top: {examples}")

    # League distribution
    if league_col in df.columns:
        train_leagues = _safe_value_counts(df_train[league_col].dropna(), top_n_leagues)
        valid_leagues = _safe_value_counts(df_valid[league_col].dropna(), top_n_leagues)
        if train_leagues:
            print("train leagues top:")
            for k, v in train_leagues:
                print(f"  - {k}: {v}")
        else:
            print("train leagues top: n/a")

        if valid_leagues:
            print("valid leagues top:")
            for k, v in valid_leagues:
                print(f"  - {k}: {v}")
        else:
            print("valid leagues top: n/a")

        # If hard league split is enabled, show which leagues were sampled as holdout
        mode = (hard_mode or "").strip().lower()
        if mode in {"league", "combo"} and holdout_league_frac > 0:
            leagues = df_train[league_col].dropna().unique().tolist()
            leagues = sorted([str(x) for x in leagues])
            if len(leagues) >= 2:
                n_holdout = max(1, int(round(len(leagues) * holdout_league_frac)))
                rng = np.random.default_rng(seed)
                holdout = sorted(set(rng.choice(leagues, size=n_holdout, replace=False)))
                print(f"holdout_leagues (sampled from train, frac={holdout_league_frac}): {holdout}")

    # Player overlap / leakage
    if player_col in df.columns:
        train_players = set(df_train[player_col].dropna().astype(str).unique())
        valid_players = set(df_valid[player_col].dropna().astype(str).unique())
        overlap = train_players.intersection(valid_players)
        denom = max(1, len(valid_players))
        print(
            f"player overlap: {len(overlap)} / valid_unique_players={len(valid_players)} "
            f"(overlap_ratio={len(overlap)/denom:.3f})"
        )

        if "season" in df.columns:
            train_pairs = set(
                df_train[[player_col, "season"]].dropna().astype(str).agg("|".join, axis=1).tolist()
            )
            valid_pairs = set(
                df_valid[[player_col, "season"]].dropna().astype(str).agg("|".join, axis=1).tolist()
            )
            overlap_pairs = train_pairs.intersection(valid_pairs)
            print(f"player_id+season overlap: {len(overlap_pairs)}")
    else:
        print("player overlap: n/a (player_id missing)")

    # Optional match_id leakage
    if "match_id" in df.columns:
        train_matches = set(df_train["match_id"].dropna().astype(str).unique())
        valid_matches = set(df_valid["match_id"].dropna().astype(str).unique())
        overlap_m = train_matches.intersection(valid_matches)
        print(f"match_id overlap: {len(overlap_m)}")

    print("=== End split summary ===\n")


@dataclass
class TrainedModelInfo:
    target: str
    model_path: str
    feature_cols: List[str]
    cat_cols: List[str]
    metrics: Dict[str, float]
    n_train: int
    n_valid: int


def train_catboost_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    cat_cols: List[str],
    model_name: str,
    split_df: pd.DataFrame | None = None,
    use_time_split: bool = True,
    use_group_split: bool = True,
    group_cols: tuple[str, ...] = ("player_id", "season"),
    train_idx: np.ndarray | None = None,
    valid_idx: np.ndarray | None = None,
    save_model: bool = True,
) -> TrainedModelInfo:
    """
    Trains a CatBoostRegressor on (X,y) with a time-aware split.
    """
    X = X.copy()
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows where y is missing
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    w = sample_weight.loc[mask]

    split_source = split_df if split_df is not None else X
    if train_idx is None or valid_idx is None:
        if use_group_split:
            train_idx, valid_idx = make_group_split(split_source, group_cols=group_cols)
        elif use_time_split:
            train_idx, valid_idx = make_time_split(
                split_source,
                season_col="season" if "season" in split_source.columns else "season",
            )
        else:
            idx = np.arange(len(split_source))
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
            cut = int(0.85 * len(idx))
            train_idx, valid_idx = idx[:cut], idx[cut:]

    # Convert to positional indices for Pool
    train_rows = np.isin(X.index.to_numpy(), train_idx)
    valid_rows = np.isin(X.index.to_numpy(), valid_idx)

    X_train, y_train, w_train = X.loc[train_rows], y.loc[train_rows], w.loc[train_rows]
    X_valid, y_valid, w_valid = X.loc[valid_rows], y.loc[valid_rows], w.loc[valid_rows]

    # CatBoost categorical feature indices
    feature_cols = list(X.columns)
    cat_feature_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    train_pool = Pool(X_train, y_train, weight=w_train, cat_features=cat_feature_indices)
    valid_pool = Pool(X_valid, y_valid, weight=w_valid, cat_features=cat_feature_indices)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=6000,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=12.0,
        subsample=0.8,
        rsm=0.8,
        random_seed=42,
        verbose=200,
        early_stopping_rounds=300,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Metrics (RMSE)
    pred_valid = model.predict(X_valid)
    rmse = float(np.sqrt(np.mean((pred_valid - y_valid.to_numpy()) ** 2)))

    # Save model
    model_path = OUT_DIR / f"{model_name}.cbm"
    if save_model:
        model.save_model(str(model_path))

    std_y_valid = float(np.nanstd(y_valid.to_numpy(), ddof=0))
    rmse_over_std = float(rmse / std_y_valid) if std_y_valid > 0 else float("nan")
    rng = np.random.default_rng(42)
    y_shuffled = y_valid.to_numpy().copy()
    rng.shuffle(y_shuffled)
    rmse_shuffled = float(np.sqrt(np.mean((pred_valid - y_shuffled) ** 2)))

    return TrainedModelInfo(
        target=model_name,
        model_path=str(model_path),
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        metrics={
            "rmse_valid": rmse,
            "std_y_valid": std_y_valid,
            "rmse_over_std": rmse_over_std,
            "rmse_valid_shuffled_y": rmse_shuffled,
        },
        n_train=int(len(X_train)),
        n_valid=int(len(X_valid)),
    )




def _group_zscore_train_only(
    series: pd.Series,
    keys: list[pd.Series],
    train_mask: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not keys:
        train_vals = values[train_mask]
        mean = train_vals.mean()
        std = train_vals.std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        return (values - mean) / std

    key_df = pd.concat(keys, axis=1).astype(str)
    group_key = key_df.agg("|".join, axis=1)

    train_vals = values[train_mask]
    train_keys = group_key[train_mask]
    train_stats = (
        pd.DataFrame({"value": train_vals, "key": train_keys})
        .groupby("key")["value"]
        .agg(["mean", "std"])
    )
    global_mean = train_vals.mean()
    global_std = train_vals.std(ddof=0)
    if global_std == 0 or np.isnan(global_std):
        global_std = 1.0

    means = group_key.map(train_stats["mean"]).fillna(global_mean)
    stds = group_key.map(train_stats["std"]).replace(0, np.nan).fillna(global_std)
    return (values - means) / stds


# ----------------------------
# Main training
# ----------------------------
def main():
    # 1) Load raw base dataframe from DB
    db_path = os.getenv("BASE_DB_PATH")
    table = os.getenv("BASE_TABLE")
    load_kwargs = {}
    if db_path:
        load_kwargs["db_path"] = db_path
    if table:
        load_kwargs["table"] = table
    df_raw = load_base_dataframe(**load_kwargs)

    # 2) Build X feature matrix (base inputs)
    X = build_feature_matrix(df_raw, drop_cols=DEFAULT_DROP_COLUMNS)

    # Ensure consistent position_group
    X = fill_position_group(X)

    league_aware = LEAGUE_AWARE_NORMALIZATION and "league" in df_raw.columns
    if league_aware:
        group_cols = ("season", "league", "position_bucket")
    else:
        group_cols = ("season", "position_bucket")

    # 3) Build split indices (hard split preferred)
    if HARD_SPLIT:
        train_idx, valid_idx = make_hard_split(
            df_raw,
            mode=HARD_SPLIT_MODE,
            season_col="season" if "season" in df_raw.columns else "season",
            league_col="league",
            player_col="player_id",
            holdout_league_frac=HARD_SPLIT_HOLDOUT_LEAGUE_FRAC,
            require_new_players=HARD_SPLIT_REQUIRE_NEW_PLAYERS,
            valid_years=HARD_TIME_VALID_YEARS,
            min_train=HARD_SPLIT_MIN_TRAIN,
            min_valid=HARD_SPLIT_MIN_VALID,
            seed=HARD_SPLIT_SEED,
        )
        split_name = f"hard_split:{HARD_SPLIT_MODE}"
    elif USE_GROUP_SPLIT:
        train_idx, valid_idx = make_group_split(df_raw, group_cols=SPLIT_GROUP_COLS)
        split_name = f"group_split({','.join([c for c in SPLIT_GROUP_COLS if c in df_raw.columns])})"
    elif USE_TIME_SPLIT:
        train_idx, valid_idx = make_time_split(df_raw, season_col="season" if "season" in df_raw.columns else "season")
        split_name = "time_split(season)"
    else:
        idx = np.arange(len(df_raw))
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        cut = int(0.85 * len(idx))
        train_idx, valid_idx = idx[:cut], idx[cut:]
        split_name = "random_split"

    print(f"\nUsing split: {split_name} train={len(train_idx)} valid={len(valid_idx)}")
    print_split_summary(
        df=df_raw,
        train_idx=train_idx,
        valid_idx=valid_idx,
        split_name=split_name,
        season_col="season",
        league_col="league",
        player_col="player_id",
        hard_mode=HARD_SPLIT_MODE if HARD_SPLIT else None,
        holdout_league_frac=HARD_SPLIT_HOLDOUT_LEAGUE_FRAC if HARD_SPLIT else 0.0,
        seed=HARD_SPLIT_SEED if HARD_SPLIT else 42,
        top_n_leagues=10,
    )

    train_mask = df_raw.index.isin(train_idx)

    # 4) Build teacher targets (train-only zscore)
    position_cols = ("position", "postion", "position_group")
    targets_df = build_targets_by_position(
        df_raw,  # IMPORTANT: use raw df with all columns, not only X
        position_col_candidates=position_cols,
        group_cols=group_cols,
        include_league=league_aware,
        minutes_col="minutes_played_overall",
        apply_shrinkage=True,
        apply_zscore=False,
    )

    # Teacher rating (no zscore yet):
    rating_teacher = build_rating_pred(
        df_raw,
        position_col_candidates=position_cols,
        group_cols=group_cols,
        include_league=league_aware,
        minutes_col="minutes_played_overall",
        shrink_k_strategy="median",
        use_legacy_weights=False,  # keep consistent with your component/PCA system
        apply_zscore=False,
    )

    # Apply group zscore using train-only stats
    group_df = targets_df.copy()
    for col in group_cols:
        if col in df_raw.columns and col not in group_df.columns:
            group_df[col] = df_raw[col]

    keys = [group_df[col] for col in group_cols if col in group_df.columns]
    for col in targets_df.columns:
        if col == "position_bucket":
            continue
        targets_df[col] = _group_zscore_train_only(targets_df[col], keys, train_mask)

    rating_teacher = _group_zscore_train_only(rating_teacher, keys, train_mask)

    # 4) Assemble Y table aligned to X index
    Y = targets_df.copy()
    Y[MAIN_TARGET] = rating_teacher

    # 5) Decide which targets to train
    to_train = [MAIN_TARGET]
    for t in DESIRED_SUBSCORES:
        if t in Y.columns and t != MAIN_TARGET:
            non_null = int(Y[t].notna().sum())
            if non_null >= MIN_TRAIN_SAMPLES:
                to_train.append(t)
            else:
                print(f"[skip] target={t} non_null={non_null} (<{MIN_TRAIN_SAMPLES})")

    # Special note on tackle_score (your base data likely lacks tackle features)
    # TODO: To include tackle_score in Base later:
    #   - compute tackle_score teacher from a Pro/full dataset
    #   - align those rows with the same Base X columns
    #   - train a student model on that expanded dataset

    # 6) Sample weights from minutes
    weights = build_sample_weight(df_raw)

    # 7) Categorical columns for CatBoost
    # Keep these as strings, CatBoost handles them well.
    cat_cols = []
    for c in ["position", "postion", "position_group", "league", "season"]:
        if c in X.columns:
            cat_cols.append(c)

    # Drop any remaining non-numeric columns that aren't declared categorical.
    non_numeric = X.select_dtypes(exclude=["number"]).columns
    drop_extra = [c for c in non_numeric if c not in cat_cols]
    if drop_extra:
        X = X.drop(columns=drop_extra)

    # 9) Train models
    model_infos: List[TrainedModelInfo] = []
    for target_name in to_train:
        print(f"\n=== Training: {target_name} ===")

        # IMPORTANT: X columns used by the student model should be ONLY base features.
        # TODO: when you add Pro, you will create a separate X_pro with advanced columns.
        target_features = set(_target_feature_cols(target_name))
        allowed_cols = [c for c in X.columns if c in cat_cols or c in target_features]
        X_target = X.loc[:, allowed_cols].copy()

        info = train_catboost_regressor(
            X=X_target,
            y=Y[target_name],
            sample_weight=weights,
            cat_cols=[c for c in cat_cols if c in X_target.columns],
            model_name=f"base_{target_name}",
            split_df=df_raw,
            use_time_split=USE_TIME_SPLIT,
            use_group_split=USE_GROUP_SPLIT,
            group_cols=SPLIT_GROUP_COLS,
            train_idx=train_idx,
            valid_idx=valid_idx,
        )
        model_infos.append(info)
        print(f"[done] {target_name} rmse_valid={info.metrics['rmse_valid']:.4f} train={info.n_train} valid={info.n_valid}")

    # Self-checks disabled/removed

    if RUN_SPLIT_CHECKS:
        print("\n=== Split check: rating_pred ===")
        rating_name = f"base_{MAIN_TARGET}"
        if rating_name in [m.target for m in model_infos]:
            info_time = train_catboost_regressor(
                X=X,
                y=Y[MAIN_TARGET],
                sample_weight=weights,
                cat_cols=cat_cols,
                model_name="base_rating_pred_timecheck",
                split_df=df_raw,
                use_time_split=True,
                use_group_split=False,
                group_cols=SPLIT_GROUP_COLS,
                save_model=False,
            )
            info_group = train_catboost_regressor(
                X=X,
                y=Y[MAIN_TARGET],
                sample_weight=weights,
                cat_cols=cat_cols,
                model_name="base_rating_pred_groupcheck",
                split_df=df_raw,
                use_time_split=False,
                use_group_split=True,
                group_cols=SPLIT_GROUP_COLS,
                save_model=False,
            )
            print(
                f"time_split rmse={info_time.metrics['rmse_valid']:.4f} "
                f"std_y={info_time.metrics['std_y_valid']:.4f}"
            )
            print(
                f"group_split rmse={info_group.metrics['rmse_valid']:.4f} "
                f"std_y={info_group.metrics['std_y_valid']:.4f}"
            )
        else:
            print("rating_pred not trained; skip split check.")

    # 10) Save metadata for inference
    meta = {
        "version": "base_v1",
        "main_target": MAIN_TARGET,
        "trained_targets": [m.target for m in model_infos],
        "models": [m.__dict__ for m in model_infos],
        "notes": {
            "teacher_system": "target.py (components->PCA->shrink->group_zscore)",
            "todo_pro": "Add Pro teacher, train delta models, and stack base+delta for final outputs.",
            "todo_tackle": "Distill tackle_score from Pro/full data into Base X space.",
        },
    }

    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metadata to: {meta_path}")

    # 10) (Optional) quick inference demo on first 5 rows
    # NOTE: This is just a sanity check. You will implement a real predict API later.
    # TODO: Add confidence scoring (minutes/coverage/ood) in your API layer.
    from catboost import CatBoostRegressor as _CBR

    def _load_model(path: str) -> _CBR:
        m = _CBR()
        m.load_model(path)
        return m

    demo_X = X.head(5)
    for info in model_infos:
        m = _load_model(info.model_path)
        preds = m.predict(demo_X[info.feature_cols])
        print(f"[demo] {info.target} preds={np.round(preds, 3).tolist()}")


if __name__ == "__main__":
    main()
