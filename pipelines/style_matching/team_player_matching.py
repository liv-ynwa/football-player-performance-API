#!/usr/bin/env python3
"""Machine-learning player-team style matching based on roster-level style learning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline


# MANUAL_REQUIRED: This is the existing team-style database schema, not a player-feature mapping.
STYLE_DIMENSIONS = [
    "attacking_tempo",
    "possession_dominance",
    "high_pressing",
    "set_piece_reliance",
    "vertical_threat",
    "defensive_positioning",
    "wide_play",
    "physicality",
]

# MANUAL_REQUIRED: Identifier, label, and leakage columns must be excluded before ML training.
NON_FEATURE_COLUMNS = {
    "player_id",
    "age",
    "full_name",
    "league",
    "season",
    "position",
    "Current Club",
    "nationality",
    "team_name",
    "common_name",
    "country",
    "style_cluster_id",
    "style_cluster_label",
    "style_data_quality",
    *STYLE_DIMENSIONS,
}

# MANUAL_REQUIRED: These patterns remove value, rank, usage, and team-result leakage.
# This is not a hand-built style mapping; it prevents non-style shortcuts from entering the model.
NON_STYLE_FEATURE_PATTERNS = [
    "salary",
    "market_value",
    "rank_",
    "rating",
    "clean_sheets",
    "conceded",
    "appearance",
    "matches_played",
    "minutes_played",
    "min_per_match",
    "games_started",
    "games_subbed",
    "shirt_number",
]

ROLE_DIMENSION_MULTIPLIERS = {
    "FW": {
        "attacking_tempo": 1.20,
        "possession_dominance": 0.90,
        "high_pressing": 1.05,
        "set_piece_reliance": 0.85,
        "vertical_threat": 1.30,
        "defensive_positioning": 0.75,
        "wide_play": 1.05,
        "physicality": 1.00,
    },
    "MF": {
        "attacking_tempo": 1.00,
        "possession_dominance": 1.30,
        "high_pressing": 1.15,
        "set_piece_reliance": 0.90,
        "vertical_threat": 0.95,
        "defensive_positioning": 1.00,
        "wide_play": 1.00,
        "physicality": 0.90,
    },
    "DF": {
        "attacking_tempo": 0.80,
        "possession_dominance": 0.95,
        "high_pressing": 0.95,
        "set_piece_reliance": 0.85,
        "vertical_threat": 0.75,
        "defensive_positioning": 1.35,
        "wide_play": 0.95,
        "physicality": 1.20,
    },
    "GK": {
        "attacking_tempo": 0.65,
        "possession_dominance": 1.00,
        "high_pressing": 0.70,
        "set_piece_reliance": 0.75,
        "vertical_threat": 0.60,
        "defensive_positioning": 1.60,
        "wide_play": 0.65,
        "physicality": 1.00,
    },
    "UNK": {dimension: 1.0 for dimension in STYLE_DIMENSIONS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML-based player-team matching outputs.")
    parser.add_argument("--player-db", default="database/player_features.db", help="Path to player feature database.")
    parser.add_argument("--player-style-db", default="output/player/player_style_results.db", help="Path to player style output database.")
    parser.add_argument("--team-style-db", default="output/team_style/team_style_results.db", help="Path to team style output database.")
    parser.add_argument("--output-dir", default="output/matching", help="Directory for JSON/DB matching outputs.")
    parser.add_argument("--top-n-per-player", type=int, default=3, help="Number of best team matches stored per player.")
    parser.add_argument("--top-n-per-team", type=int, default=3, help="Number of best player matches stored per team row.")
    parser.add_argument(
        "--realistic-top-n-per-player",
        type=int,
        default=3,
        help="Number of season-filtered realistic team matches stored per player.",
    )
    parser.add_argument(
        "--realistic-season-window",
        type=int,
        default=0,
        help="Allowed absolute difference between player and team season start years for realistic matches.",
    )
    parser.add_argument(
        "--dimension-weight-floor",
        type=float,
        default=0.05,
        help="Minimum positive weight used when converting model R2 values into matching dimension weights.",
    )
    parser.add_argument(
        "--min-dimension-r2",
        type=float,
        default=0.10,
        help="Dimensions with cross-validated R2 below this threshold receive zero matching weight.",
    )
    parser.add_argument(
        "--realistic-min-minutes",
        type=float,
        default=300.0,
        help="Minimum player minutes required for realistic recommendation rows.",
    )
    parser.add_argument(
        "--include-current-club-in-realistic",
        action="store_true",
        help="Allow the player's current club to appear in realistic recommendation rows.",
    )
    parser.add_argument(
        "--allow-untrained-realistic-teams",
        action="store_true",
        help="Allow teams without matched roster training profiles in realistic recommendation rows.",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Player batch size for matching output.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds for model audit.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible ML output.")
    parser.add_argument("--n-estimators", type=int, default=120, help="Number of boosting estimators.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient boosting learning rate.")
    parser.add_argument("--max-depth", type=int, default=3, help="Gradient boosting tree depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=4, help="Gradient boosting leaf smoothing.")
    return parser.parse_args()


def clean_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def load_players(player_db: Path, player_style_db: Path) -> pd.DataFrame:
    with sqlite3.connect(player_db) as conn:
        players = pd.read_sql_query("SELECT * FROM player_features_pro", conn)
    if player_style_db.exists():
        with sqlite3.connect(player_style_db) as conn:
            style_columns = [
                "player_id",
                "full_name",
                "season",
                "style_confidence",
                "style_mix_index",
                "hdbscan_probability",
                *[f"latent_style_dim_{idx}" for idx in range(1, 9)],
            ]
            style = pd.read_sql_query(
                f"SELECT {', '.join(style_columns)} FROM player_style_cards",
                conn,
            )
        style = style.sort_values("style_confidence", ascending=False, na_position="last")
        style = style.drop_duplicates(["player_id", "full_name", "season"], keep="first")
        players = players.merge(style, on=["player_id", "full_name", "season"], how="left")
    players = players.reset_index(drop=True)
    players.insert(0, "player_row_id", np.arange(len(players), dtype=int))
    return players


def load_teams(team_style_db: Path) -> pd.DataFrame:
    with sqlite3.connect(team_style_db) as conn:
        teams = pd.read_sql_query("SELECT * FROM team_style_clusters", conn)
    teams = teams.dropna(subset=STYLE_DIMENSIONS).copy().reset_index(drop=True)
    teams.insert(0, "team_row_id", np.arange(len(teams), dtype=int))
    return teams


def is_non_style_feature(column: str) -> bool:
    normalized = column.lower()
    if any(pattern in normalized for pattern in NON_STYLE_FEATURE_PATTERNS):
        return True
    if "total" in normalized:
        return True
    raw_count_suffixes = ("_overall", "_home", "_away")
    rate_tokens = ("per_90", "per90", "percentage", "rate", "percentile")
    if normalized.endswith(raw_count_suffixes) and not any(token in normalized for token in rate_tokens):
        return True
    return False


def select_numeric_features(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = []
    feature_series = {}
    audit_rows = []

    for column in players.columns:
        if column in NON_FEATURE_COLUMNS or column == "player_row_id":
            continue
        numeric_for_audit = clean_numeric(players[column])
        if is_non_style_feature(column):
            audit_rows.append(
                {
                    "feature_name": column,
                    "selected": 0,
                    "usable_count": int(numeric_for_audit.notna().sum()),
                    "unique_count": int(numeric_for_audit.nunique(dropna=True)),
                    "reason": "manual_required_non_style_or_leakage_exclusion",
                }
            )
            continue
        usable_count = int(numeric_for_audit.notna().sum())
        unique_count = int(numeric_for_audit.nunique(dropna=True))
        if usable_count == 0 or unique_count <= 1:
            audit_rows.append(
                {
                    "feature_name": column,
                    "selected": 0,
                    "usable_count": usable_count,
                    "unique_count": unique_count,
                    "reason": "non_numeric_or_constant",
                }
            )
            continue
        feature_series[column] = numeric_for_audit
        feature_columns.append(column)
        audit_rows.append(
            {
                "feature_name": column,
                "selected": 1,
                "usable_count": usable_count,
                "unique_count": unique_count,
                "reason": "numeric_variable",
            }
        )

    feature_frame = pd.DataFrame(feature_series, index=players.index)
    return feature_frame[feature_columns], pd.DataFrame(audit_rows)


def weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    weighted = np.where(valid, values, 0.0) * weights[:, None]
    denom = (valid * weights[:, None]).sum(axis=0)
    return np.divide(weighted.sum(axis=0), denom, out=np.full(values.shape[1], np.nan), where=denom > 0)


def weighted_nanstd(values: np.ndarray, weights: np.ndarray, means: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    centered = np.where(valid, values - means.reshape(1, -1), 0.0)
    denom = (valid * weights[:, None]).sum(axis=0)
    variance = np.divide(
        (centered * centered * weights[:, None]).sum(axis=0),
        denom,
        out=np.full(values.shape[1], np.nan),
        where=denom > 0,
    )
    return np.sqrt(np.clip(variance, 0.0, None))


def build_profile_feature_names(base_columns: pd.Index) -> list[str]:
    names = []
    for stat in ["mean", "std", "p25", "p75"]:
        names.extend(f"{stat}__{column}" for column in base_columns)
    return names


def summarize_roster_features(values: np.ndarray, weights: np.ndarray, base_columns: pd.Index) -> dict[str, float]:
    means = weighted_nanmean(values, weights)
    stds = weighted_nanstd(values, weights, means)
    p25 = safe_nanpercentile(values, 25)
    p75 = safe_nanpercentile(values, 75)
    summary = {}
    for stat_name, stat_values in [("mean", means), ("std", stds), ("p25", p25), ("p75", p75)]:
        for idx, column in enumerate(base_columns):
            summary[f"{stat_name}__{column}"] = stat_values[idx]
    return summary


def safe_nanpercentile(values: np.ndarray, percentile: float) -> np.ndarray:
    output = np.full(values.shape[1], np.nan)
    for idx in range(values.shape[1]):
        column = values[:, idx]
        valid = column[~np.isnan(column)]
        if len(valid):
            output[idx] = np.percentile(valid, percentile)
    return output


def build_player_projection_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    projection = {}
    for column in feature_frame.columns:
        projection[f"mean__{column}"] = feature_frame[column]
        projection[f"std__{column}"] = 0.0
        projection[f"p25__{column}"] = feature_frame[column]
        projection[f"p75__{column}"] = feature_frame[column]
    return pd.DataFrame(projection, index=feature_frame.index)


def build_roster_training_data(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    join_columns = ["player_row_id", "player_id", "full_name", "season", "Current Club", "minutes_played_overall"]
    roster_rows = players[join_columns].merge(
        teams,
        left_on=["Current Club", "season"],
        right_on=["common_name", "season"],
        how="inner",
    )
    roster_rows = roster_rows.reset_index(drop=True)
    if roster_rows.empty:
        raise RuntimeError("No current-club player/team rows were found.")

    profile_rows = []
    audit_rows = []
    for team_row_id, group in roster_rows.groupby("team_row_id", sort=False):
        player_indices = group["player_row_id"].to_numpy(dtype=int)
        raw_minutes = clean_numeric(group["minutes_played_overall"]).fillna(0.0).to_numpy(dtype=float)
        # MANUAL_REQUIRED: Minutes are reliability weights for roster aggregation, not model features.
        weights = np.log1p(np.clip(raw_minutes, 0.0, None))
        if np.all(weights == 0):
            weights = np.ones_like(weights)
        values = feature_frame.iloc[player_indices].to_numpy(dtype=float)
        profile = summarize_roster_features(values, weights, feature_frame.columns)
        team_info = teams.loc[teams["team_row_id"] == team_row_id].iloc[0]
        profile_rows.append(
            {
                "team_row_id": int(team_row_id),
                **profile,
            }
        )
        audit_rows.append(
            {
                "team_row_id": int(team_row_id),
                "team_name": team_info["team_name"],
                "team_common_name": team_info["common_name"],
                "team_country": team_info["country"],
                "team_season": team_info["season"],
                "player_rows_used": int(len(group)),
                "minutes_weight_sum": round(float(weights.sum()), 4),
                "feature_coverage": round(float(np.isfinite(np.asarray(list(profile.values()), dtype=float)).mean()), 4),
            }
        )

    roster_profiles = pd.DataFrame(profile_rows).sort_values("team_row_id").reset_index(drop=True)
    team_targets = teams.merge(roster_profiles[["team_row_id"]], on="team_row_id", how="inner").sort_values("team_row_id").reset_index(drop=True)
    profile_feature_columns = build_profile_feature_names(feature_frame.columns)
    x_train = roster_profiles[profile_feature_columns].reset_index(drop=True)
    y_train = team_targets[STYLE_DIMENSIONS].reset_index(drop=True)
    profile_audit = pd.DataFrame(audit_rows).sort_values("team_row_id").reset_index(drop=True)
    return roster_rows, roster_profiles, x_train, y_train, profile_audit


def build_model(args: argparse.Namespace) -> Pipeline:
    # MANUAL_REQUIRED: These are ML hyperparameters, not football style rules.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=args.n_estimators,
                        learning_rate=args.learning_rate,
                        max_depth=args.max_depth,
                        min_samples_leaf=args.min_samples_leaf,
                        random_state=args.random_state,
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )


def fit_oof_predictions(model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, folds: int, random_state: int) -> np.ndarray:
    fold_count = max(2, min(folds, len(x_train)))
    cv = KFold(n_splits=fold_count, shuffle=True, random_state=random_state)
    oof = np.zeros((len(x_train), len(STYLE_DIMENSIONS)), dtype=float)

    for train_index, valid_index in cv.split(x_train):
        fold_model = clone(model)
        fold_model.fit(x_train.iloc[train_index], y_train.iloc[train_index])
        oof[valid_index] = fold_model.predict(x_train.iloc[valid_index])

    return np.clip(oof, 0.0, 100.0)


def build_model_quality(y_train: pd.DataFrame, oof_predictions: np.ndarray) -> pd.DataFrame:
    rows = []
    y_true = y_train.to_numpy(dtype=float)
    for idx, dimension in enumerate(STYLE_DIMENSIONS):
        mse = float(mean_squared_error(y_true[:, idx], oof_predictions[:, idx]))
        rows.append(
            {
                "dimension": dimension,
                "mae": round(float(mean_absolute_error(y_true[:, idx], oof_predictions[:, idx])), 4),
                "mse": round(mse, 4),
                "rmse": round(mse**0.5, 4),
                "r2": round(float(r2_score(y_true[:, idx], oof_predictions[:, idx])), 4),
            }
        )
    overall_mse = float(mean_squared_error(y_true, oof_predictions))
    rows.append(
        {
            "dimension": "overall",
            "mae": round(float(mean_absolute_error(y_true, oof_predictions)), 4),
            "mse": round(overall_mse, 4),
            "rmse": round(overall_mse**0.5, 4),
            "r2": round(float(r2_score(y_true, oof_predictions, multioutput="variance_weighted")), 4),
        }
    )
    return pd.DataFrame(rows)


def fit_final_model(model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    model.fit(x_train, y_train)
    return model


def build_dimension_weights(model_quality: pd.DataFrame, floor: float, min_r2: float) -> np.ndarray:
    quality = model_quality[model_quality["dimension"].isin(STYLE_DIMENSIONS)].set_index("dimension")
    raw = []
    for dimension in STYLE_DIMENSIONS:
        r2_value = float(quality.loc[dimension, "r2"]) if dimension in quality.index else floor
        raw.append(0.0 if r2_value < min_r2 else max(r2_value, floor))
    weights = np.asarray(raw, dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(STYLE_DIMENSIONS), dtype=float)
    return weights / weights.mean()


def build_dimension_filter_summary(model_quality: pd.DataFrame, min_r2: float) -> tuple[list[str], list[str]]:
    quality = model_quality[model_quality["dimension"].isin(STYLE_DIMENSIONS)].set_index("dimension")
    active_dimensions = []
    excluded_dimensions = []
    for dimension in STYLE_DIMENSIONS:
        r2_value = float(quality.loc[dimension, "r2"]) if dimension in quality.index else float("-inf")
        if r2_value < min_r2:
            excluded_dimensions.append(dimension)
        else:
            active_dimensions.append(dimension)
    return active_dimensions, excluded_dimensions


def infer_role_group(position: object) -> str:
    if pd.isna(position):
        return "UNK"
    text = str(position).lower()
    if any(token in text for token in ["goalkeeper", "keeper", "gk"]):
        return "GK"
    if any(token in text for token in ["forward", "attacker", "striker", "winger", "fw", "st", "cf", "lw", "rw"]):
        return "FW"
    if any(token in text for token in ["midfielder", "midfield", "half", "mf", "dm", "cm", "am"]):
        return "MF"
    if any(token in text for token in ["defender", "defence", "defense", "centre-back", "center-back", "full-back", "back", "df"]):
        return "DF"
    return "UNK"


def build_role_dimension_weight_matrix(player_predictions: pd.DataFrame, base_weights: np.ndarray) -> np.ndarray:
    role_weights = []
    for role in player_predictions["match_role_group"]:
        multipliers = ROLE_DIMENSION_MULTIPLIERS.get(str(role), ROLE_DIMENSION_MULTIPLIERS["UNK"])
        weights = np.asarray([float(multipliers[dimension]) for dimension in STYLE_DIMENSIONS], dtype=float)
        weights = weights * base_weights
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = base_weights.copy()
        role_weights.append(weights / weights.mean())
    return np.vstack(role_weights)


def weighted_rmse_distance(left: np.ndarray, right: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    squared = (left - right) ** 2
    if weights is None:
        return np.sqrt(np.mean(squared, axis=-1))
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim == 1:
        denominator = weight_array.sum()
        adjusted_weights = weight_array
    elif weight_array.ndim == 2:
        adjusted_weights = weight_array[:, None, :] if squared.ndim == 3 else weight_array
        denominator = adjusted_weights.sum(axis=-1)
    else:
        raise ValueError("weights must be a 1D dimension vector or 2D row-by-dimension matrix.")
    return np.sqrt(np.sum(squared * adjusted_weights, axis=-1) / denominator)


def percentage_from_distance(distance: np.ndarray, positive_reference_distances: np.ndarray) -> np.ndarray:
    half_life = float(np.percentile(positive_reference_distances, 75))
    if half_life <= 0:
        half_life = 1.0
    percentages = 100.0 * np.exp(np.log(0.5) * (distance / half_life) ** 2)
    return np.clip(percentages, 0.0, 100.0)


def build_player_predictions(players: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    out = players[
        [
            "player_row_id",
            "player_id",
            "full_name",
            "age",
            "league",
            "season",
            "position",
            "Current Club",
            "minutes_played_overall",
            "nationality",
        ]
    ].copy()
    out["match_role_group"] = out["position"].map(infer_role_group)
    for idx, dimension in enumerate(STYLE_DIMENSIONS):
        out[f"predicted_{dimension}"] = np.round(predictions[:, idx], 4)
    return out


def build_team_profile_predictions(teams: pd.DataFrame, roster_profiles: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    out = teams.merge(roster_profiles[["team_row_id"]], on="team_row_id", how="inner").sort_values("team_row_id").reset_index(drop=True)
    for idx, dimension in enumerate(STYLE_DIMENSIONS):
        out[f"predicted_{dimension}"] = np.round(predictions[:, idx], 4)
    return out


def empty_match_frame(include_explanation: bool = False) -> pd.DataFrame:
    columns = [
        "player_row_id",
        "player_id",
        "full_name",
        "player_season",
        "current_club",
        "position",
        "match_role_group",
        "team_row_id",
        "team_name",
        "team_common_name",
        "team_country",
        "team_season",
        "team_style_cluster_id",
        "team_style_cluster_label",
        "match_scope",
        "match_percentage",
        "style_distance",
        "is_current_club_match",
    ]
    if include_explanation:
        columns.extend(["best_fit_dimensions", "weak_fit_dimensions"])
        columns.extend(f"dimension_gap_{dimension}" for dimension in STYLE_DIMENSIONS)
    return pd.DataFrame(columns=columns)


def format_match_row(
    player: pd.Series,
    team: pd.Series,
    percentage: float,
    distance: float,
    match_scope: str,
    include_explanation: bool = False,
    explanation_dimensions: set[str] | None = None,
) -> dict[str, object]:
    row = {
        "player_row_id": int(player["player_row_id"]),
        "player_id": int(player["player_id"]),
        "full_name": player["full_name"],
        "player_season": player["season"],
        "current_club": player["Current Club"],
        "position": player["position"],
        "match_role_group": player.get("match_role_group", infer_role_group(player["position"])),
        "team_row_id": int(team["team_row_id"]),
        "team_name": team["team_name"],
        "team_common_name": team["common_name"],
        "team_country": team["country"],
        "team_season": team["season"],
        "team_style_cluster_id": int(team["style_cluster_id"]),
        "team_style_cluster_label": team["style_cluster_label"],
        "match_scope": match_scope,
        "match_percentage": round(float(percentage), 2),
        "style_distance": round(float(distance), 4),
        "is_current_club_match": int(player["Current Club"] == team["common_name"] and player["season"] == team["season"]),
    }
    if include_explanation:
        gaps = {}
        for dimension in STYLE_DIMENSIONS:
            predicted_value = float(player[f"predicted_{dimension}"])
            team_value = float(team[dimension])
            gaps[dimension] = abs(predicted_value - team_value)
        rankable_gaps = {
            dimension: gap
            for dimension, gap in gaps.items()
            if explanation_dimensions is None or dimension in explanation_dimensions
        }
        ordered_gaps = sorted(rankable_gaps.items(), key=lambda item: item[1])
        row["best_fit_dimensions"] = ", ".join(name for name, _ in ordered_gaps[:3])
        row["weak_fit_dimensions"] = ", ".join(name for name, _ in ordered_gaps[-3:][::-1])
        for dimension, gap in gaps.items():
            row[f"dimension_gap_{dimension}"] = round(float(gap), 4)
    return row


def build_player_top_team_matches(
    player_predictions: pd.DataFrame,
    prediction_matrix: np.ndarray,
    teams: pd.DataFrame,
    positive_reference_distances: np.ndarray,
    player_dimension_weights: np.ndarray,
    top_n: int,
    batch_size: int,
    match_scope: str = "global",
) -> pd.DataFrame:
    if top_n <= 0 or teams.empty:
        return empty_match_frame()
    team_matrix = teams[STYLE_DIMENSIONS].to_numpy(dtype=float)
    rows = []

    for start in range(0, len(player_predictions), batch_size):
        end = min(start + batch_size, len(player_predictions))
        distances = weighted_rmse_distance(
            prediction_matrix[start:end, None, :],
            team_matrix[None, :, :],
            player_dimension_weights[start:end],
        )
        percentages = percentage_from_distance(distances, positive_reference_distances)
        top_count = min(top_n, len(teams))
        top_indices = np.argpartition(distances, top_count - 1, axis=1)[:, :top_count]
        for local_idx, team_indices in enumerate(top_indices):
            player = player_predictions.iloc[start + local_idx]
            ordered = team_indices[np.argsort(distances[local_idx, team_indices])]
            for team_idx in ordered:
                rows.append(
                    format_match_row(
                        player,
                        teams.iloc[team_idx],
                        percentages[local_idx, team_idx],
                        distances[local_idx, team_idx],
                        match_scope,
                        include_explanation=False,
                    )
                )

    return pd.DataFrame(rows) if rows else empty_match_frame()


def season_start_year(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = pd.Series([str(value)]).str.extract(r"(\d{4})", expand=False).iloc[0]
    return float(match) if pd.notna(match) else np.nan


def build_realistic_player_team_matches(
    player_predictions: pd.DataFrame,
    prediction_matrix: np.ndarray,
    teams: pd.DataFrame,
    positive_reference_distances: np.ndarray,
    player_dimension_weights: np.ndarray,
    top_n: int,
    batch_size: int,
    season_window: int,
    min_minutes: float,
    exclude_current_club: bool,
    trained_team_ids: set[int] | None,
    explanation_dimensions: set[str] | None,
) -> pd.DataFrame:
    if top_n <= 0 or teams.empty:
        return empty_match_frame(include_explanation=True)
    team_matrix = teams[STYLE_DIMENSIONS].to_numpy(dtype=float)
    team_years = teams["season"].map(season_start_year).to_numpy(dtype=float)
    team_seasons = teams["season"].astype(str).to_numpy()
    team_names = teams["common_name"].fillna("").astype(str).to_numpy()
    eligible_team_mask = np.ones(len(teams), dtype=bool)
    if trained_team_ids is not None:
        eligible_team_mask &= teams["team_row_id"].isin(trained_team_ids).to_numpy()
    if not eligible_team_mask.any():
        return empty_match_frame(include_explanation=True)
    player_years = player_predictions["season"].map(season_start_year).to_numpy(dtype=float)
    player_seasons = player_predictions["season"].astype(str).to_numpy()
    player_clubs = player_predictions["Current Club"].fillna("").astype(str).to_numpy()
    player_minutes = clean_numeric(player_predictions["minutes_played_overall"]).fillna(0.0).to_numpy(dtype=float)
    eligible_player_mask = player_minutes >= min_minutes
    rows = []

    for start in range(0, len(player_predictions), batch_size):
        end = min(start + batch_size, len(player_predictions))
        distances = weighted_rmse_distance(
            prediction_matrix[start:end, None, :],
            team_matrix[None, :, :],
            player_dimension_weights[start:end],
        )
        year_allowed = (
            np.isfinite(player_years[start:end, None])
            & np.isfinite(team_years[None, :])
            & (np.abs(player_years[start:end, None] - team_years[None, :]) <= season_window)
        )
        exact_allowed = player_seasons[start:end, None] == team_seasons[None, :]
        allowed = (year_allowed | exact_allowed) & eligible_team_mask[None, :] & eligible_player_mask[start:end, None]
        if exclude_current_club:
            current_club_match = (player_clubs[start:end, None] == team_names[None, :]) & exact_allowed
            allowed &= ~current_club_match
        distances = np.where(allowed, distances, np.inf)
        percentages = percentage_from_distance(distances, positive_reference_distances)
        top_count = min(top_n, int(eligible_team_mask.sum()))
        if top_count <= 0:
            continue
        top_indices = np.argpartition(distances, top_count - 1, axis=1)[:, :top_count]
        for local_idx, team_indices in enumerate(top_indices):
            finite_indices = team_indices[np.isfinite(distances[local_idx, team_indices])]
            if len(finite_indices) == 0:
                continue
            player = player_predictions.iloc[start + local_idx]
            ordered = finite_indices[np.argsort(distances[local_idx, finite_indices])]
            for team_idx in ordered:
                rows.append(
                    format_match_row(
                        player,
                        teams.iloc[team_idx],
                        percentages[local_idx, team_idx],
                        distances[local_idx, team_idx],
                        f"season_window_{season_window}",
                        include_explanation=True,
                        explanation_dimensions=explanation_dimensions,
                    )
                )

    return pd.DataFrame(rows) if rows else empty_match_frame(include_explanation=True)


def build_team_top_player_matches(
    player_predictions: pd.DataFrame,
    prediction_matrix: np.ndarray,
    teams: pd.DataFrame,
    positive_reference_distances: np.ndarray,
    player_dimension_weights: np.ndarray,
    top_n: int,
    batch_size: int,
) -> pd.DataFrame:
    if top_n <= 0 or player_predictions.empty:
        return empty_match_frame()
    rows = []

    for _, team in teams.iterrows():
        team_vector = team[STYLE_DIMENSIONS].to_numpy(dtype=float)
        best_indices: list[int] = []
        best_distances: list[float] = []
        best_percentages: list[float] = []
        for start in range(0, len(player_predictions), batch_size):
            end = min(start + batch_size, len(player_predictions))
            distances = weighted_rmse_distance(
                prediction_matrix[start:end],
                team_vector.reshape(1, -1),
                player_dimension_weights[start:end],
            )
            percentages = percentage_from_distance(distances, positive_reference_distances)
            top_count = min(top_n, len(distances))
            local_indices = np.argpartition(distances, top_count - 1)[:top_count]
            best_indices.extend((start + local_indices).tolist())
            best_distances.extend(distances[local_indices].tolist())
            best_percentages.extend(percentages[local_indices].tolist())

        best_distances_array = np.asarray(best_distances)
        top_count = min(top_n, len(best_distances_array))
        keep = np.argpartition(best_distances_array, top_count - 1)[:top_count]
        keep = keep[np.argsort(best_distances_array[keep])]
        for idx in keep:
            player_idx = best_indices[int(idx)]
            rows.append(
                format_match_row(
                    player_predictions.iloc[player_idx],
                    team,
                    best_percentages[int(idx)],
                    best_distances[int(idx)],
                    "global",
                    include_explanation=False,
                )
            )

    return pd.DataFrame(rows) if rows else empty_match_frame()


def build_current_club_match_audit(
    roster_rows: pd.DataFrame,
    player_predictions: pd.DataFrame,
    player_prediction_matrix: np.ndarray,
    positive_reference_distances: np.ndarray,
    player_dimension_weights: np.ndarray,
) -> pd.DataFrame:
    merged = roster_rows.merge(
        player_predictions[["player_row_id", "position", "match_role_group"]],
        on="player_row_id",
        how="left",
    )
    target_matrix = merged[STYLE_DIMENSIONS].to_numpy(dtype=float)
    player_indices = merged["player_row_id"].to_numpy(dtype=int)
    predictions = player_prediction_matrix[player_indices]
    distances = weighted_rmse_distance(predictions, target_matrix, player_dimension_weights[player_indices])
    percentages = percentage_from_distance(distances, positive_reference_distances)
    out = merged[
        [
            "player_row_id",
            "player_id",
            "full_name",
            "season",
            "Current Club",
            "position",
            "match_role_group",
            "team_row_id",
            "team_name",
            "common_name",
            "country",
            "style_cluster_id",
            "style_cluster_label",
        ]
    ].copy()
    out = out.rename(
        columns={
            "season": "player_season",
            "Current Club": "current_club",
            "common_name": "team_common_name",
            "country": "team_country",
            "style_cluster_id": "team_style_cluster_id",
            "style_cluster_label": "team_style_cluster_label",
        }
    )
    out["team_season"] = out["player_season"]
    out["match_percentage"] = np.round(percentages, 2)
    out["style_distance"] = np.round(distances, 4)
    return out.sort_values("match_percentage", ascending=False).reset_index(drop=True)


def build_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    regressor = model.named_steps["regressor"]
    importances = np.mean([estimator.feature_importances_ for estimator in regressor.estimators_], axis=0)
    rows = [
        {
            "feature_name": feature_name,
            "importance": float(importance),
        }
        for feature_name, importance in zip(feature_names, importances)
    ]
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def build_experiment_summary(
    metadata: dict[str, object],
    model_quality: pd.DataFrame,
    dimension_weights: np.ndarray,
) -> pd.DataFrame:
    overall = model_quality[model_quality["dimension"] == "overall"]
    overall_row = overall.iloc[0].to_dict() if not overall.empty else {}
    return pd.DataFrame(
        [
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": metadata["scope"],
                "model": metadata["model"],
                "training_target": metadata["training_target"],
                "player_rows": metadata["player_rows"],
                "team_rows": metadata["team_rows"],
                "roster_player_rows": metadata["roster_player_rows"],
                "team_profile_rows": metadata["team_profile_rows"],
                "selected_numeric_features": metadata["selected_numeric_features"],
                "cv_folds": metadata["cv_folds"],
                "overall_mae": overall_row.get("mae"),
                "overall_mse": overall_row.get("mse"),
                "overall_rmse": overall_row.get("rmse"),
                "overall_r2": overall_row.get("r2"),
                "top_n_per_player": metadata["top_n_per_player"],
                "top_n_per_team": metadata["top_n_per_team"],
                "realistic_top_n_per_player": metadata["realistic_top_n_per_player"],
                "realistic_season_window": metadata["realistic_season_window"],
                "realistic_min_minutes": metadata["realistic_min_minutes"],
                "exclude_current_club_from_realistic": metadata["exclude_current_club_from_realistic"],
                "restrict_realistic_to_trained_teams": metadata["restrict_realistic_to_trained_teams"],
                "position_aware_dimension_weights": metadata["position_aware_dimension_weights"],
                "dimension_weight_floor": metadata["dimension_weight_floor"],
                "min_dimension_r2": metadata["min_dimension_r2"],
                "active_match_dimensions": json.dumps(metadata["active_match_dimensions"], ensure_ascii=False),
                "excluded_match_dimensions": json.dumps(metadata["excluded_match_dimensions"], ensure_ascii=False),
                "dimension_weights_json": json.dumps(
                    {
                        dimension: round(float(weight), 6)
                        for dimension, weight in zip(STYLE_DIMENSIONS, dimension_weights)
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    )


def write_outputs(
    player_predictions: pd.DataFrame,
    team_profile_predictions: pd.DataFrame,
    profile_audit: pd.DataFrame,
    feature_audit: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_quality: pd.DataFrame,
    player_top_team_matches: pd.DataFrame,
    player_realistic_team_matches: pd.DataFrame,
    team_top_player_matches: pd.DataFrame,
    current_club_match_audit: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, object],
    dimension_weights: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "team_player_match_results.db"
    if db_path.exists():
        db_path.unlink()
    cleanup_obsolete_json_outputs(
        output_dir,
        [
            "team_player_match_results.json",
            "team_player_match_metadata.json",
        ],
    )
    with sqlite3.connect(db_path) as conn:
        player_predictions.to_sql("ml_player_predicted_team_style", conn, if_exists="replace", index=False)
        team_profile_predictions.to_sql("ml_team_profile_predicted_style", conn, if_exists="replace", index=False)
        profile_audit.to_sql("ml_team_profile_audit", conn, if_exists="replace", index=False)
        feature_audit.to_sql("ml_feature_audit", conn, if_exists="replace", index=False)
        feature_importance.to_sql("ml_feature_importance", conn, if_exists="replace", index=False)
        model_quality.to_sql("ml_model_quality", conn, if_exists="replace", index=False)
        player_top_team_matches.to_sql("player_top_team_matches", conn, if_exists="replace", index=False)
        player_realistic_team_matches.to_sql("player_realistic_team_matches", conn, if_exists="replace", index=False)
        team_top_player_matches.to_sql("team_top_player_matches", conn, if_exists="replace", index=False)
        current_club_match_audit.to_sql("current_club_match_audit", conn, if_exists="replace", index=False)
        build_experiment_summary(metadata, model_quality, dimension_weights).to_sql(
            "ml_experiment_summary",
            conn,
            if_exists="replace",
            index=False,
        )
        create_output_indexes(conn)

    quality_payload = {
        "scope": metadata["scope"],
        "model": metadata["model"],
        "training_target": metadata["training_target"],
        "rows": {
            "players": metadata["player_rows"],
            "teams": metadata["team_rows"],
            "roster_player_rows": metadata["roster_player_rows"],
            "team_profile_rows": metadata["team_profile_rows"],
            "selected_numeric_features": metadata["selected_numeric_features"],
        },
        "cv_folds": metadata["cv_folds"],
        "style_dimensions": metadata["style_dimensions"],
        "dimension_weights": {
            dimension: round(float(weight), 6)
            for dimension, weight in zip(STYLE_DIMENSIONS, dimension_weights)
        },
        "min_dimension_r2": metadata["min_dimension_r2"],
        "active_match_dimensions": metadata["active_match_dimensions"],
        "excluded_match_dimensions": metadata["excluded_match_dimensions"],
        "realistic_match_filter": {
            "season_window": metadata["realistic_season_window"],
            "top_n_per_player": metadata["realistic_top_n_per_player"],
            "min_minutes": metadata["realistic_min_minutes"],
            "exclude_current_club": metadata["exclude_current_club_from_realistic"],
            "restrict_to_trained_teams": metadata["restrict_realistic_to_trained_teams"],
        },
        "position_aware_dimension_weights": metadata["position_aware_dimension_weights"],
        "metrics": model_quality.to_dict(orient="records"),
    }
    (output_dir / "model_quality.json").write_text(
        json.dumps(quality_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def create_output_indexes(conn: sqlite3.Connection) -> None:
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_player_top_matches_player ON player_top_team_matches(player_id, match_percentage DESC)",
        "CREATE INDEX IF NOT EXISTS idx_player_top_matches_team ON player_top_team_matches(team_row_id, match_percentage DESC)",
        "CREATE INDEX IF NOT EXISTS idx_realistic_matches_player ON player_realistic_team_matches(player_id, match_percentage DESC)",
        "CREATE INDEX IF NOT EXISTS idx_realistic_matches_team ON player_realistic_team_matches(team_row_id, match_percentage DESC)",
        "CREATE INDEX IF NOT EXISTS idx_team_top_matches_team ON team_top_player_matches(team_row_id, match_percentage DESC)",
        "CREATE INDEX IF NOT EXISTS idx_current_club_player ON current_club_match_audit(player_id, match_percentage DESC)",
        "CREATE INDEX IF NOT EXISTS idx_current_club_team ON current_club_match_audit(team_row_id, match_percentage DESC)",
    ]
    for sql in indexes:
        conn.execute(sql)


def cleanup_obsolete_json_outputs(output_dir: Path, filenames: list[str]) -> None:
    for filename in filenames:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    players = load_players(Path(args.player_db), Path(args.player_style_db))
    teams = load_teams(Path(args.team_style_db))
    feature_frame, feature_audit = select_numeric_features(players)
    if feature_frame.empty:
        raise RuntimeError("No usable numeric player features were found.")

    roster_rows, roster_profiles, x_train, y_train, profile_audit = build_roster_training_data(players, teams, feature_frame)
    model = build_model(args)
    oof_predictions = fit_oof_predictions(model, x_train, y_train, args.cv_folds, args.random_state)
    model_quality = build_model_quality(y_train, oof_predictions)
    dimension_weights = build_dimension_weights(model_quality, args.dimension_weight_floor, args.min_dimension_r2)
    active_match_dimensions, excluded_match_dimensions = build_dimension_filter_summary(model_quality, args.min_dimension_r2)
    final_model = fit_final_model(model, x_train, y_train)

    team_profile_predictions_matrix = np.clip(final_model.predict(x_train), 0.0, 100.0)
    player_projection_frame = build_player_projection_features(feature_frame)
    player_prediction_matrix = np.clip(final_model.predict(player_projection_frame[x_train.columns]), 0.0, 100.0)
    player_predictions = build_player_predictions(players, player_prediction_matrix)
    player_dimension_weights = build_role_dimension_weight_matrix(player_predictions, dimension_weights)
    team_profile_predictions = build_team_profile_predictions(teams, roster_profiles, team_profile_predictions_matrix)
    positive_reference_distances = weighted_rmse_distance(oof_predictions, y_train.to_numpy(dtype=float), dimension_weights)
    positive_reference_distances = positive_reference_distances[positive_reference_distances > 0]
    if len(positive_reference_distances) == 0:
        raise RuntimeError("Reference distance distribution is empty.")

    player_top_team_matches = build_player_top_team_matches(
        player_predictions,
        player_prediction_matrix,
        teams,
        positive_reference_distances,
        player_dimension_weights,
        args.top_n_per_player,
        args.batch_size,
    )
    player_realistic_team_matches = build_realistic_player_team_matches(
        player_predictions,
        player_prediction_matrix,
        teams,
        positive_reference_distances,
        player_dimension_weights,
        args.realistic_top_n_per_player,
        args.batch_size,
        args.realistic_season_window,
        args.realistic_min_minutes,
        not args.include_current_club_in_realistic,
        None if args.allow_untrained_realistic_teams else set(team_profile_predictions["team_row_id"].astype(int)),
        set(active_match_dimensions),
    )
    team_top_player_matches = build_team_top_player_matches(
        player_predictions,
        player_prediction_matrix,
        teams,
        positive_reference_distances,
        player_dimension_weights,
        args.top_n_per_team,
        args.batch_size,
    )
    current_club_match_audit = build_current_club_match_audit(
        roster_rows,
        player_predictions,
        player_prediction_matrix,
        positive_reference_distances,
        player_dimension_weights,
    )
    feature_importance = build_feature_importance(final_model, x_train.columns.tolist())

    metadata = {
        "scope": "ml_team_player_matching",
        "match_unit": "percentage",
        "model": "MultiOutputRegressor_GradientBoostingRegressor",
        "training_target": "team_roster_aggregate_player_features_to_team_style_8d",
        "player_projection": "final_roster_model_applied_to_single_player_features",
        "percentage_method": "exponential_distance_calibration_using_out_of_fold_team_profile_distances",
        "player_rows": int(len(players)),
        "team_rows": int(len(teams)),
        "roster_player_rows": int(len(roster_rows)),
        "team_profile_rows": int(len(x_train)),
        "selected_numeric_features": int(feature_audit["selected"].sum()),
        "top_n_per_player": int(args.top_n_per_player),
        "top_n_per_team": int(args.top_n_per_team),
        "realistic_top_n_per_player": int(args.realistic_top_n_per_player),
        "realistic_season_window": int(args.realistic_season_window),
        "realistic_min_minutes": float(args.realistic_min_minutes),
        "exclude_current_club_from_realistic": bool(not args.include_current_club_in_realistic),
        "restrict_realistic_to_trained_teams": bool(not args.allow_untrained_realistic_teams),
        "dimension_weight_floor": float(args.dimension_weight_floor),
        "min_dimension_r2": float(args.min_dimension_r2),
        "active_match_dimensions": active_match_dimensions,
        "excluded_match_dimensions": excluded_match_dimensions,
        "position_aware_dimension_weights": True,
        "cv_folds": int(max(2, min(args.cv_folds, len(x_train)))),
        "style_dimensions": STYLE_DIMENSIONS,
        "manual_required_items": [
            "STYLE_DIMENSIONS: existing team-style database schema",
            "NON_FEATURE_COLUMNS: identifier/leakage exclusion list",
            "NON_STYLE_FEATURE_PATTERNS: value/ranking/usage/team-result leakage exclusion list",
            "minutes_played_overall roster aggregation weights: reliability weighting only, not a matching feature",
            "GradientBoosting hyperparameters: operational ML settings",
            "top_n_per_player/top_n_per_team/batch_size: output-size controls only",
            "realistic_season_window: business filter for practical recommendation outputs",
            "dimension weights: derived from cross-validated R2 with a configurable floor and zeroed below min_dimension_r2",
            "role multipliers: position-aware interpretation layer applied after ML-derived R2 weights",
            "realistic_min_minutes/current-club/trained-team filters: business filters for practical outputs",
        ],
        "outputs": [
            "team_player_match_results.db",
            "model_quality.json",
        ],
        "database_tables": [
            "ml_player_predicted_team_style",
            "ml_team_profile_predicted_style",
            "ml_team_profile_audit",
            "ml_feature_audit",
            "ml_feature_importance",
            "ml_model_quality",
            "player_top_team_matches",
            "player_realistic_team_matches",
            "team_top_player_matches",
            "current_club_match_audit",
            "ml_experiment_summary",
        ],
    }

    write_outputs(
        player_predictions,
        team_profile_predictions,
        profile_audit,
        feature_audit,
        feature_importance,
        model_quality,
        player_top_team_matches,
        player_realistic_team_matches,
        team_top_player_matches,
        current_club_match_audit,
        Path(args.output_dir),
        metadata,
        dimension_weights,
    )

    print(f"Wrote matching outputs to {args.output_dir}")
    print(f"Players: {len(players)}")
    print(f"Teams: {len(teams)}")
    print(f"Roster player rows: {len(roster_rows)}")
    print(f"Team profile rows: {len(x_train)}")
    print(f"Selected numeric features: {int(feature_audit['selected'].sum())}")
    print(f"Player top team rows: {len(player_top_team_matches)}")
    print(f"Player realistic team rows: {len(player_realistic_team_matches)}")
    print(f"Team top player rows: {len(team_top_player_matches)}")
    print(f"Current club audit rows: {len(current_club_match_audit)}")


if __name__ == "__main__":
    main()
