#!/usr/bin/env python3
"""Build the first-pass team style vectors and clusters.

This is intentionally scoped to team style only. Player style and matching are
planned in memory.md but are not implemented here.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from typing import Iterable

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


TEAM_COLUMNS = [
    "team_name",
    "common_name",
    "season",
    "country",
    "minutes_per_goal_scored_overall",
    "minutes_per_goal_scored_home",
    "minutes_per_goal_scored_away",
    "minutes_per_goal_conceded_overall",
    "minutes_per_goal_conceded_home",
    "minutes_per_goal_conceded_away",
    "corners_total_per_match_overall",
    "corners_total_per_match_home",
    "corners_total_per_match_away",
    "corners_against_overall",
    "corners_against_per_match_overall",
    "corners_against_per_match_home",
    "corners_against_per_match_away",
    "cornerTimingRecorded_matches_overall",
    "corners_total_fh_overall",
    "corners_total_2h_overall",
    "corners_total_per_match_fh_overall",
    "corners_total_per_match_2h_overall",
    "shots_per_match_overall",
    "shots_per_match_home",
    "shots_per_match_away",
    "shots_on_target_per_match_overall",
    "shots_on_target_per_match_home",
    "shots_on_target_per_match_away",
    "shots_off_target_per_match_overall",
    "shots_off_target_per_match_home",
    "shots_off_target_per_match_away",
    "fouls_by_this_team_overall",
    "fouls_by_this_team_home",
    "fouls_by_this_team_away",
    "fouls_per_match_overall",
    "fouls_per_match_home",
    "fouls_per_match_away",
    "offsides_total_overall",
    "offsides_total_home",
    "offsides_total_away",
    "offsides_this_team_per_match_overall",
    "offsides_this_team_per_match_home",
    "offsides_this_team_per_match_away",
    "firstGoalScoredPercentage_overall",
    "firstGoalScoredPercentage_home",
    "firstGoalScoredPercentage_away",
    "scoredBothHalves_overall",
    "scoredBothHalvesPercentage_overall",
    "BTTS_both_halves_overall",
    "BTTS_both_halves_home",
    "BTTS_both_halves_away",
    "matches_goal_timings_recorded_overall",
    "average_total_goals_2h_per_match_overall",
    "average_total_goals_2h_per_match_home",
    "average_total_goals_2h_per_match_away",
    "total_goals_2h_overall",
    "goals_scored_2h_per_match_overall",
    "goals_scored_2h_per_match_home",
    "goals_scored_2h_per_match_away",
    "goals_scored_2h_overall",
    "goals_conceded_2h_per_match_overall",
    "goals_conceded_2h_per_match_home",
    "goals_conceded_2h_per_match_away",
    "goals_conceded_2h_overall",
    "points_2h_overall",
    "points_2h_home",
    "points_2h_away",
    "ppg_2h_overall",
    "ppg_2h_home",
    "ppg_2h_away",
    "wins_2h_overall",
    "wins_2h_percentage_overall",
    "draws_2h_overall",
    "losses_2h_overall",
    "btts_2h_overall",
    "btts_2h_percentage_overall",
    "clean_sheets_2h_overall",
    "failed_to_score_2h_overall",
    "over25_cards_total_percentage_overall",
]


SPORTMONKS_COLUMNS = [
    "season_id",
    "team_name",
    "played",
    "gf",
    "ga",
    "goals_scored_all_count",
    "goals_conceded_all_count",
    "avg_possession",
    "avg_possession_average",
    "total_passes",
    "passes_per_game",
    "pass_stats_passes_per_game",
    "passes_per_shot",
    "pass_stats_passes_per_shot",
    "shots",
    "shots_average",
    "shots_on_target_pct",
    "shot_on_target_percentage_pct_shots_on_target",
    "shots_shot_conversion_rate_pct",
    "conversion_conversion_rate_pct",
    "corners_average",
    "fouls_committed",
    "fouls_committed_average",
    "tackles",
    "tackles_average",
    "interception_stats_interceptions_per_game",
    "attacks_average",
    "dangerous_attacks_average",
]

SPORTMONKS_NUMERIC_COLUMNS = [f"sportmonks_{col}" for col in SPORTMONKS_COLUMNS if col != "team_name"]


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


PROXY_NOTES = {
    "attacking_tempo": "Proxy from goal scoring interval, offsides, shot frequency, and Sportmonks attack pressure when matched.",
    "possession_dominance": "Proxy from territorial pressure indicators, with Sportmonks possession/pass volume used where high-confidence team-season matches are available.",
    "high_pressing": "Proxy from own offsides, fouls, and Sportmonks defensive actions/attacks where matched; PPDA and event-level pressure data are absent.",
    "set_piece_reliance": "Proxy from corners per match and first-goal rate, with Sportmonks corner average used where matched.",
    "vertical_threat": "Proxy from offsides, shot accuracy, goal scoring speed, Sportmonks dangerous attacks, and pass-per-shot directness where matched.",
    "defensive_positioning": "Proxy from conceded-goal interval, second-half clean-sheet rate, second-half concession suppression, and limiting opponent corners.",
    "wide_play": "Proxy from corners and off-target shot volume because direct crossing/width team data are absent.",
    "physicality": "Proxy from fouls, card-threshold percentages, and Sportmonks tackles/fouls where matched.",
}


DIMENSION_FEATURE_CONFIG = {
    "attacking_tempo": [
        ("goal_scoring_speed", 0.40),
        ("offsides_this_team_per_match_overall", 0.15),
        ("shots_per_match_overall", 0.15),
        ("sportmonks_shots_per_match", 0.10),
        ("sportmonks_attacks_per_match", 0.10),
        ("sportmonks_dangerous_attacks_per_match", 0.10),
    ],
    "possession_dominance": [
        ("shots_per_match_overall", 0.18),
        ("shots_on_target_per_match_overall", 0.12),
        ("corners_total_per_match_overall", 0.15),
        ("low_corners_against", 0.15),
        ("sportmonks_possession", 0.20),
        ("sportmonks_pass_volume", 0.20),
    ],
    "high_pressing": [
        ("offsides_this_team_per_match_overall", 0.30),
        ("fouls_per_match_overall", 0.25),
        ("sportmonks_fouls_per_match", 0.15),
        ("sportmonks_tackles_per_match", 0.15),
        ("sportmonks_interceptions_per_match", 0.15),
    ],
    "set_piece_reliance": [
        ("corners_total_per_match_overall", 0.55),
        ("firstGoalScoredPercentage_overall", 0.25),
        ("sportmonks_corners_per_match", 0.20),
    ],
    "vertical_threat": [
        ("offsides_this_team_per_match_overall", 0.25),
        ("goal_scoring_speed", 0.20),
        ("shot_accuracy", 0.15),
        ("sportmonks_shot_accuracy", 0.10),
        ("sportmonks_dangerous_attacks_per_match", 0.15),
        ("sportmonks_directness", 0.15),
    ],
    "defensive_positioning": [
        ("goal_concession_spacing", 0.40),
        ("clean_sheets_2h_rate", 0.25),
        ("low_2h_concession_rate", 0.20),
        ("sportmonks_tackles_per_match", 0.08),
        ("sportmonks_interceptions_per_match", 0.07),
    ],
    "wide_play": [
        ("corners_total_per_match_overall", 0.55),
        ("shots_off_target_per_match_overall", 0.15),
        ("scoredBothHalvesPercentage_overall", 0.10),
        ("sportmonks_corners_per_match", 0.20),
    ],
    "physicality": [
        ("fouls_by_this_team_overall", 0.25),
        ("fouls_per_match_overall", 0.20),
        ("over25_cards_total_percentage_overall", 0.25),
        ("sportmonks_fouls_per_match", 0.15),
        ("sportmonks_tackles_per_match", 0.15),
    ],
}


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    best_k: int
    silhouette: float | None
    scanned_scores: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate team style vectors and clusters.")
    parser.add_argument("--team-db", default="database/teamdatabase.db", help="Path to team SQLite database.")
    parser.add_argument(
        "--sportmonks-team-db",
        default="database/sportmonks_team_season_stats.db",
        help="Optional Sportmonks team-season SQLite database to merge into the team style source.",
    )
    parser.add_argument(
        "--disable-sportmonks",
        action="store_true",
        help="Disable Sportmonks feature merging even if the database exists.",
    )
    parser.add_argument(
        "--sportmonks-match-threshold",
        type=float,
        default=0.25,
        help="Maximum normalized score for accepting a Sportmonks row as a high-confidence match.",
    )
    parser.add_argument("--output-dir", default="output/team_style", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--min-group-size", type=int, default=8, help="Minimum group size for country-season normalization.")
    parser.add_argument("--min-k", type=int, default=4, help="Minimum KMeans cluster count to scan.")
    parser.add_argument("--max-k", type=int, default=8, help="Maximum KMeans cluster count to scan.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for KMeans.")
    parser.add_argument(
        "--adaptive-style-alpha",
        type=float,
        default=0.0,
        help="Blend strength for adaptive reliability weights. 0=fixed prior, 1=fully adaptive experimental mode.",
    )
    return parser.parse_args()


def load_teams(db_path: Path, sportmonks_db_path: Path | None, sportmonks_match_threshold: float) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Team database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        columns_sql = ", ".join(f'"{col}"' for col in TEAM_COLUMNS)
        teams = pd.read_sql_query(f"SELECT rowid AS source_team_rowid, {columns_sql} FROM teams", conn)

    teams = teams.reset_index(drop=True)
    if sportmonks_db_path is None or not sportmonks_db_path.exists():
        return add_empty_sportmonks_columns(teams)
    return merge_sportmonks_features(teams, sportmonks_db_path, sportmonks_match_threshold)


def add_empty_sportmonks_columns(teams: pd.DataFrame) -> pd.DataFrame:
    result = teams.copy()
    for col in SPORTMONKS_NUMERIC_COLUMNS:
        result[col] = np.nan
    result["sportmonks_team_name"] = pd.NA
    result["sportmonks_match_status"] = "not_configured"
    result["sportmonks_match_score"] = np.nan
    result["sportmonks_candidate_count"] = 0
    result["sportmonks_matched_feature_count"] = 0
    return result


def load_sportmonks(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        available = {
            row[1]
            for row in conn.execute("PRAGMA table_info(team_season_stats)").fetchall()
        }
        select_cols = [col for col in SPORTMONKS_COLUMNS if col in available]
        if "team_name" not in select_cols:
            raise RuntimeError("Sportmonks team table is missing team_name.")
        columns_sql = ", ".join(f'"{col}"' for col in select_cols)
        sportmonks = pd.read_sql_query(f"SELECT {columns_sql} FROM team_season_stats", conn)

    for col in SPORTMONKS_COLUMNS:
        if col not in sportmonks.columns:
            sportmonks[col] = np.nan
    numeric_cols = [col for col in SPORTMONKS_COLUMNS if col != "team_name"]
    sportmonks = coerce_numeric(sportmonks, numeric_cols)
    sportmonks["name_key"] = sportmonks["team_name"].map(normalize_team_name)
    sportmonks["sportmonks_minutes_per_goal_scored"] = minutes_per_goal(
        sportmonks["played"], sportmonks["gf"].fillna(sportmonks["goals_scored_all_count"])
    )
    sportmonks["sportmonks_minutes_per_goal_conceded"] = minutes_per_goal(
        sportmonks["played"], sportmonks["ga"].fillna(sportmonks["goals_conceded_all_count"])
    )
    return sportmonks


def normalize_team_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stopwords = {"fc", "cf", "afc", "sc", "ac", "club", "de", "the"}
    tokens = [token for token in text.split() if token not in stopwords]
    return " ".join(tokens)


def minutes_per_goal(played: pd.Series, goals: pd.Series) -> pd.Series:
    played_numeric = pd.to_numeric(played, errors="coerce")
    goals_numeric = pd.to_numeric(goals, errors="coerce").where(lambda s: s > 0)
    return (played_numeric * 90.0) / goals_numeric


def merge_sportmonks_features(teams: pd.DataFrame, sportmonks_db_path: Path, threshold: float) -> pd.DataFrame:
    sportmonks = load_sportmonks(sportmonks_db_path)
    team_source = teams.copy()
    team_source["name_key"] = team_source["common_name"].map(normalize_team_name)
    fallback_key = team_source["team_name"].map(normalize_team_name)
    team_source["name_key"] = team_source["name_key"].where(team_source["name_key"] != "", fallback_key)

    groups = {
        name_key: group.reset_index(drop=True)
        for name_key, group in sportmonks.groupby("name_key", dropna=False)
        if name_key
    }

    matched_rows = []
    for _, team in team_source.iterrows():
        candidates = groups.get(team["name_key"], pd.DataFrame())
        match = choose_sportmonks_match(team, candidates, threshold)
        matched_rows.append(match)

    match_frame = pd.DataFrame(matched_rows, index=team_source.index)
    result = pd.concat([team_source.drop(columns=["name_key"]), match_frame], axis=1)
    result["sportmonks_matched_feature_count"] = result[SPORTMONKS_NUMERIC_COLUMNS].notna().sum(axis=1)
    return result


def choose_sportmonks_match(team: pd.Series, candidates: pd.DataFrame, threshold: float) -> dict[str, object]:
    empty = {col: np.nan for col in SPORTMONKS_NUMERIC_COLUMNS}
    empty.update(
        {
            "sportmonks_team_name": pd.NA,
            "sportmonks_match_status": "no_name_candidate" if candidates.empty else "low_confidence",
            "sportmonks_match_score": np.nan,
            "sportmonks_candidate_count": int(len(candidates)),
        }
    )
    if candidates.empty:
        return empty

    scored = candidates.copy()
    scored["match_score"] = score_sportmonks_candidates(team, scored)
    scored = scored.dropna(subset=["match_score"])
    if scored.empty:
        return empty

    best = scored.sort_values(["match_score", "season_id"]).iloc[0]
    if float(best["match_score"]) > threshold:
        empty["sportmonks_match_score"] = round(float(best["match_score"]), 6)
        return empty

    out = {}
    for col in SPORTMONKS_COLUMNS:
        if col == "team_name":
            out["sportmonks_team_name"] = best[col]
        else:
            out[f"sportmonks_{col}"] = best[col]
    out["sportmonks_match_status"] = "matched"
    out["sportmonks_match_score"] = round(float(best["match_score"]), 6)
    out["sportmonks_candidate_count"] = int(len(candidates))
    return out


def score_sportmonks_candidates(team: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    base_scored = pd.to_numeric(pd.Series([team["minutes_per_goal_scored_overall"]]), errors="coerce").iloc[0]
    base_conceded = pd.to_numeric(pd.Series([team["minutes_per_goal_conceded_overall"]]), errors="coerce").iloc[0]
    base_matches = pd.to_numeric(pd.Series([team["matches_goal_timings_recorded_overall"]]), errors="coerce").iloc[0]

    parts = []
    weights = []
    if pd.notna(base_scored) and base_scored > 0:
        parts.append(log_ratio_error(base_scored, candidates["sportmonks_minutes_per_goal_scored"]))
        weights.append(0.35)
    if pd.notna(base_conceded) and base_conceded > 0:
        parts.append(log_ratio_error(base_conceded, candidates["sportmonks_minutes_per_goal_conceded"]))
        weights.append(0.35)
    if pd.notna(base_matches) and base_matches > 0:
        played = pd.to_numeric(candidates["played"], errors="coerce")
        match_error = (played - base_matches).abs() / np.maximum(played, base_matches)
        parts.append(match_error)
        weights.append(0.15)
        if pd.notna(base_scored) and base_scored > 0:
            estimated_gf = (base_matches * 90.0) / base_scored
            parts.append(relative_error(estimated_gf, candidates["gf"].fillna(candidates["goals_scored_all_count"])))
            weights.append(0.075)
        if pd.notna(base_conceded) and base_conceded > 0:
            estimated_ga = (base_matches * 90.0) / base_conceded
            parts.append(relative_error(estimated_ga, candidates["ga"].fillna(candidates["goals_conceded_all_count"])))
            weights.append(0.075)

    if not parts:
        return pd.Series(np.nan, index=candidates.index, dtype=float)
    valid = pd.concat(parts, axis=1)
    weight_array = np.asarray(weights, dtype=float)
    present = valid.notna()
    weighted = valid.fillna(0.0).to_numpy(dtype=float) * weight_array
    denom = present.to_numpy(dtype=float) * weight_array
    return pd.Series(
        np.divide(weighted.sum(axis=1), denom.sum(axis=1), out=np.full(len(candidates), np.nan), where=denom.sum(axis=1) > 0),
        index=candidates.index,
    )


def log_ratio_error(base_value: float, candidate_values: pd.Series) -> pd.Series:
    candidates = pd.to_numeric(candidate_values, errors="coerce")
    candidates = candidates.where(candidates > 0)
    return np.log(candidates / base_value).abs()


def relative_error(base_value: float, candidate_values: pd.Series) -> pd.Series:
    candidates = pd.to_numeric(candidate_values, errors="coerce")
    denominator = pd.Series(np.maximum(np.abs(candidates), abs(base_value)), index=candidates.index)
    return (candidates - base_value).abs() / denominator.replace(0, np.nan)


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def safe_inverse(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric > 0)
    return 1.0 / numeric


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    den = den.where(den > 0)
    return num / den


def first_non_null(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[existing].bfill(axis=1).iloc[:, 0]


def percentile_rank(series: pd.Series) -> pd.Series:
    if series.notna().sum() <= 1:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return series.rank(method="average", pct=True) * 100.0


def normalized_feature(df: pd.DataFrame, column: str, min_group_size: int) -> pd.Series:
    country_season = df.groupby(["country", "season"], dropna=False)[column].transform(
        lambda s: percentile_rank(s) if s.notna().sum() >= min_group_size else np.nan
    )
    season = df.groupby("season", dropna=False)[column].transform(
        lambda s: percentile_rank(s) if s.notna().sum() >= min_group_size else np.nan
    )
    global_rank = percentile_rank(df[column])
    return country_season.fillna(season).fillna(global_rank).fillna(50.0)


def weighted_mean(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    weighted = None
    total_weight = 0.0
    for series, weight in parts:
        if weighted is None:
            weighted = series.astype(float) * weight
        else:
            weighted = weighted + series.astype(float) * weight
        total_weight += weight
    if weighted is None or total_weight == 0:
        raise ValueError("weighted_mean requires at least one weighted part")
    return (weighted / total_weight).clip(0, 100)


def feature_reliability(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    coverage_ratio = float(numeric.notna().mean())
    unique_count = int(valid.nunique(dropna=True))
    if len(valid) == 0 or unique_count <= 1:
        return {
            "coverage_ratio": round(coverage_ratio, 6),
            "non_zero_ratio": 0.0,
            "unique_count": unique_count,
            "distinctiveness_score": 0.0,
            "dispersion_score": 0.0,
            "reliability_score": 0.0,
        }

    non_zero_ratio = float((valid != 0).mean())
    distinctiveness_score = min(unique_count / 20.0, 1.0)
    std = float(valid.std(ddof=0))
    median_abs = float(valid.abs().median())
    dispersion_score = 0.0 if std <= 0 else std / (std + median_abs + 1e-9)
    reliability = (
        0.35 * np.sqrt(coverage_ratio)
        + 0.25 * non_zero_ratio
        + 0.20 * distinctiveness_score
        + 0.20 * dispersion_score
    )
    return {
        "coverage_ratio": round(coverage_ratio, 6),
        "non_zero_ratio": round(non_zero_ratio, 6),
        "unique_count": unique_count,
        "distinctiveness_score": round(float(distinctiveness_score), 6),
        "dispersion_score": round(float(dispersion_score), 6),
        "reliability_score": round(float(np.clip(reliability, 0.0, 1.0)), 6),
    }


def build_feature_reliability_table(feature_source: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for feature_name in feature_names:
        metrics = feature_reliability(feature_source[feature_name])
        rows.append({"feature_name": feature_name, **metrics})
    return pd.DataFrame(rows)


def adaptive_weighted_mean(
    normalized: dict[str, pd.Series],
    feature_source: pd.DataFrame,
    feature_reliability_lookup: dict[str, float],
    parts: list[tuple[str, float]],
) -> pd.Series:
    weighted = pd.Series(0.0, index=feature_source.index, dtype=float)
    denominator = pd.Series(0.0, index=feature_source.index, dtype=float)
    for feature_name, base_weight in parts:
        reliability = feature_reliability_lookup.get(feature_name, 0.0)
        effective_weight = base_weight * reliability
        if effective_weight <= 0:
            continue
        row_valid = feature_source[feature_name].notna()
        weighted = weighted + normalized[feature_name].astype(float).where(row_valid, 0.0) * effective_weight
        denominator = denominator + row_valid.astype(float) * effective_weight
    result = pd.Series(50.0, index=feature_source.index, dtype=float)
    valid = denominator > 0
    result.loc[valid] = weighted.loc[valid] / denominator.loc[valid]
    return result.clip(0, 100)


def fixed_prior_weighted_mean(
    normalized: dict[str, pd.Series],
    parts: list[tuple[str, float]],
) -> pd.Series:
    return weighted_mean([(normalized[feature_name], base_weight) for feature_name, base_weight in parts])


def blended_style_dimension(
    normalized: dict[str, pd.Series],
    feature_source: pd.DataFrame,
    feature_reliability_lookup: dict[str, float],
    parts: list[tuple[str, float]],
    adaptive_alpha: float,
) -> pd.Series:
    fixed_score = fixed_prior_weighted_mean(normalized, parts)
    adaptive_score = adaptive_weighted_mean(normalized, feature_source, feature_reliability_lookup, parts)
    return ((1.0 - adaptive_alpha) * fixed_score + adaptive_alpha * adaptive_score).clip(0, 100)


def build_dimension_weight_audit(
    feature_source: pd.DataFrame,
    reliability_table: pd.DataFrame,
) -> pd.DataFrame:
    reliability_lookup = reliability_table.set_index("feature_name")["reliability_score"].to_dict()
    rows = []
    for dimension, parts in DIMENSION_FEATURE_CONFIG.items():
        raw_weights = {feature: base * reliability_lookup.get(feature, 0.0) for feature, base in parts}
        total_global_weight = sum(raw_weights.values())
        row_denominator = pd.Series(0.0, index=feature_source.index, dtype=float)
        for feature, effective_weight in raw_weights.items():
            row_denominator = row_denominator + feature_source[feature].notna().astype(float) * effective_weight
        for feature, base_weight in parts:
            reliability = reliability_lookup.get(feature, 0.0)
            effective_weight = raw_weights[feature]
            rows.append(
                {
                    "dimension": dimension,
                    "feature_name": feature,
                    "base_weight": round(float(base_weight), 6),
                    "reliability_score": round(float(reliability), 6),
                    "global_effective_weight": round(float(effective_weight), 6),
                    "global_normalized_weight": round(float(effective_weight / total_global_weight), 6)
                    if total_global_weight > 0
                    else 0.0,
                    "row_usable_count": int(feature_source[feature].notna().sum()),
                    "row_avg_normalized_weight_when_usable": round(
                        float((effective_weight / row_denominator.where(row_denominator > 0)).where(feature_source[feature].notna()).mean()),
                        6,
                    )
                    if effective_weight > 0 and row_denominator.gt(0).any()
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_style_vectors(
    raw: pd.DataFrame,
    min_group_size: int,
    adaptive_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= adaptive_alpha <= 1.0:
        raise ValueError("--adaptive-style-alpha must be between 0 and 1.")
    numeric_cols = [
        c
        for c in [*TEAM_COLUMNS, *SPORTMONKS_NUMERIC_COLUMNS, "source_team_rowid", "sportmonks_match_score", "sportmonks_candidate_count", "sportmonks_matched_feature_count"]
        if c not in {"team_name", "common_name", "season", "country"}
        and c in raw.columns
    ]
    df = coerce_numeric(raw, numeric_cols)

    derived = pd.DataFrame(index=df.index)
    derived["goal_scoring_speed"] = safe_inverse(df["minutes_per_goal_scored_overall"])
    derived["goal_concession_spacing"] = df["minutes_per_goal_conceded_overall"].where(
        df["minutes_per_goal_conceded_overall"] > 0
    )
    derived["low_corners_against"] = -df["corners_against_per_match_overall"]
    derived["clean_sheets_2h_rate"] = safe_rate(
        df["clean_sheets_2h_overall"], df["matches_goal_timings_recorded_overall"]
    )
    derived["low_2h_concession_rate"] = -df["goals_conceded_2h_per_match_overall"]
    derived["shot_accuracy"] = safe_rate(
        df["shots_on_target_per_match_overall"], df["shots_per_match_overall"]
    )
    derived["sportmonks_possession"] = first_non_null(df, ["sportmonks_avg_possession_average", "sportmonks_avg_possession"])
    derived["sportmonks_pass_volume"] = first_non_null(
        df,
        ["sportmonks_pass_stats_passes_per_game", "sportmonks_passes_per_game"],
    ).fillna(safe_rate(df["sportmonks_total_passes"], df["sportmonks_played"]))
    derived["sportmonks_directness"] = -first_non_null(
        df,
        ["sportmonks_pass_stats_passes_per_shot", "sportmonks_passes_per_shot"],
    )
    derived["sportmonks_shots_per_match"] = first_non_null(df, ["sportmonks_shots_average"]).fillna(
        safe_rate(df["sportmonks_shots"], df["sportmonks_played"])
    )
    derived["sportmonks_shot_accuracy"] = first_non_null(
        df,
        ["sportmonks_shot_on_target_percentage_pct_shots_on_target", "sportmonks_shots_on_target_pct"],
    )
    derived["sportmonks_shot_conversion"] = first_non_null(
        df,
        ["sportmonks_conversion_conversion_rate_pct", "sportmonks_shots_shot_conversion_rate_pct"],
    )
    derived["sportmonks_corners_per_match"] = df["sportmonks_corners_average"]
    derived["sportmonks_fouls_per_match"] = df["sportmonks_fouls_committed_average"].fillna(
        safe_rate(df["sportmonks_fouls_committed"], df["sportmonks_played"])
    )
    derived["sportmonks_tackles_per_match"] = df["sportmonks_tackles_average"].fillna(
        safe_rate(df["sportmonks_tackles"], df["sportmonks_played"])
    )
    derived["sportmonks_interceptions_per_match"] = df["sportmonks_interception_stats_interceptions_per_game"]
    derived["sportmonks_attacks_per_match"] = df["sportmonks_attacks_average"]
    derived["sportmonks_dangerous_attacks_per_match"] = df["sportmonks_dangerous_attacks_average"]

    feature_source = pd.concat([df, derived], axis=1)
    configured_features = sorted({feature for parts in DIMENSION_FEATURE_CONFIG.values() for feature, _ in parts})
    normalized = {
        col: normalized_feature(feature_source, col, min_group_size)
        for col in configured_features
    }
    reliability_table = build_feature_reliability_table(feature_source, configured_features)
    reliability_lookup = reliability_table.set_index("feature_name")["reliability_score"].to_dict()
    dimension_weight_audit = build_dimension_weight_audit(feature_source, reliability_table)

    vector_columns = [
        "source_team_rowid",
        "team_name",
        "common_name",
        "country",
        "season",
        "sportmonks_season_id",
        "sportmonks_team_name",
        "sportmonks_match_status",
        "sportmonks_match_score",
        "sportmonks_candidate_count",
        "sportmonks_matched_feature_count",
    ]
    vectors = df[[col for col in vector_columns if col in df.columns]].copy()
    for dimension in STYLE_DIMENSIONS:
        vectors[dimension] = blended_style_dimension(
            normalized,
            feature_source,
            reliability_lookup,
            DIMENSION_FEATURE_CONFIG[dimension],
            adaptive_alpha,
        )

    vectors["style_data_quality"] = style_data_quality(df)
    return vectors, reliability_table, dimension_weight_audit


def style_data_quality(df: pd.DataFrame) -> pd.Series:
    quality_cols = [
        "minutes_per_goal_scored_overall",
        "minutes_per_goal_conceded_overall",
        "corners_total_per_match_overall",
        "shots_per_match_overall",
        "shots_on_target_per_match_overall",
        "fouls_by_this_team_overall",
        "fouls_per_match_overall",
        "offsides_this_team_per_match_overall",
        "matches_goal_timings_recorded_overall",
        "goals_scored_2h_per_match_overall",
        "goals_conceded_2h_per_match_overall",
        "over25_cards_total_percentage_overall",
    ]
    valid = pd.DataFrame(index=df.index)
    for col in quality_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        valid[col] = values.notna() & (values != 0)
    base_quality = valid.mean(axis=1) * 100.0

    sportmonks_quality_cols = [
        "sportmonks_avg_possession_average",
        "sportmonks_pass_stats_passes_per_game",
        "sportmonks_shots_average",
        "sportmonks_corners_average",
        "sportmonks_fouls_committed_average",
        "sportmonks_tackles_average",
        "sportmonks_attacks_average",
        "sportmonks_dangerous_attacks_average",
    ]
    sm_valid = pd.DataFrame(index=df.index)
    for col in sportmonks_quality_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            sm_valid[col] = values.notna() & (values != 0)
    if sm_valid.empty or "sportmonks_match_status" not in df.columns:
        return base_quality.round(1)

    sm_quality = sm_valid.mean(axis=1) * 100.0
    matched = df["sportmonks_match_status"].eq("matched")
    combined = base_quality.where(~matched, (base_quality * 0.75) + (sm_quality * 0.25))
    return combined.clip(0, 100).round(1)


def cluster_vectors(vectors: pd.DataFrame, min_k: int, max_k: int, random_state: int) -> ClusterResult:
    x = vectors[STYLE_DIMENSIONS].to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x)

    n = len(vectors)
    if n < 2:
        return ClusterResult(labels=np.zeros(n, dtype=int), best_k=1, silhouette=None, scanned_scores={})

    max_k = min(max_k, n - 1)
    min_k = min(min_k, max_k)
    scanned_scores: dict[int, float] = {}
    best_labels = None
    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit_predict(x_scaled)
        if len(set(labels)) <= 1:
            continue
        score = float(silhouette_score(x_scaled, labels))
        scanned_scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        best_labels = KMeans(n_clusters=min_k, n_init=25, random_state=random_state).fit_predict(x_scaled)
        best_score = None

    return ClusterResult(
        labels=np.asarray(best_labels, dtype=int),
        best_k=best_k,
        silhouette=best_score,
        scanned_scores=scanned_scores,
    )


def summarize_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in clustered.groupby("style_cluster_id"):
        center = group[STYLE_DIMENSIONS].mean().sort_values(ascending=False)
        high = center.head(3)
        low = center.tail(2)
        rows.append(
            {
                "style_cluster_id": int(cluster_id),
                "team_count": int(len(group)),
                "label": make_cluster_label(high, low),
                "top_dimensions": "; ".join(f"{name}={value:.1f}" for name, value in high.items()),
                "low_dimensions": "; ".join(f"{name}={value:.1f}" for name, value in low.items()),
                **{f"avg_{dimension}": round(float(center[dimension]), 2) for dimension in STYLE_DIMENSIONS},
            }
        )
    return pd.DataFrame(rows).sort_values("style_cluster_id")


def make_cluster_label(high: pd.Series, low: pd.Series) -> str:
    high_names = [dimension_to_label(name, "high") for name in high.index]
    low_names = [dimension_to_label(name, "low") for name in low.index]
    return " / ".join(high_names) + " | " + " / ".join(low_names)


def dimension_to_label(name: str, polarity: str) -> str:
    labels = {
        "attacking_tempo": ("fast attacking tempo", "slow attacking tempo"),
        "possession_dominance": ("possession dominance proxy", "low possession dominance"),
        "high_pressing": ("high pressing proxy", "low pressing proxy"),
        "set_piece_reliance": ("set-piece/corner reliant", "low set-piece reliance"),
        "vertical_threat": ("vertical threat", "low vertical threat"),
        "defensive_positioning": ("defensive positioning solidity", "defensive positioning fragility"),
        "wide_play": ("wide-play proxy", "low wide-play reliance"),
        "physicality": ("physicality", "low physicality"),
    }
    return labels[name][0 if polarity == "high" else 1]


def write_outputs(
    vectors: pd.DataFrame,
    result: ClusterResult,
    feature_reliability_table: pd.DataFrame,
    dimension_weight_audit: pd.DataFrame,
    adaptive_alpha: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_obsolete_json_outputs(
        output_dir,
        [
            "team_style_results.json",
            "team_style_run_metadata.json",
            "team_style_data_quality_issues.json",
        ],
    )

    clustered = vectors.copy()
    clustered["style_cluster_id"] = result.labels
    summary = summarize_clusters(clustered)
    clustered = clustered.merge(summary[["style_cluster_id", "label"]], on="style_cluster_id", how="left")
    clustered = clustered.rename(columns={"label": "style_cluster_label"})

    quality_issues = clustered[clustered["style_data_quality"] < 75].sort_values(
        ["style_data_quality", "country", "season", "common_name"]
    )
    audit_columns = [
        "source_team_rowid",
        "team_name",
        "common_name",
        "country",
        "season",
        "sportmonks_team_name",
        "sportmonks_season_id",
        "sportmonks_match_status",
        "sportmonks_match_score",
        "sportmonks_candidate_count",
        "sportmonks_matched_feature_count",
    ]
    sportmonks_audit = clustered[[col for col in audit_columns if col in clustered.columns]].copy()
    sportmonks_matched = (
        int(clustered["sportmonks_match_status"].eq("matched").sum())
        if "sportmonks_match_status" in clustered.columns
        else 0
    )
    sportmonks_low_confidence = (
        int(clustered["sportmonks_match_status"].eq("low_confidence").sum())
        if "sportmonks_match_status" in clustered.columns
        else 0
    )
    results_db_path = output_dir / "team_style_results.db"
    with sqlite3.connect(results_db_path) as conn:
        vectors.to_sql("team_style_vectors", conn, if_exists="replace", index=False)
        clustered.to_sql("team_style_clusters", conn, if_exists="replace", index=False)
        summary.to_sql("team_style_cluster_summary", conn, if_exists="replace", index=False)
        quality_issues.to_sql("team_style_data_quality_issues", conn, if_exists="replace", index=False)
        sportmonks_audit.to_sql("sportmonks_merge_audit", conn, if_exists="replace", index=False)
        feature_reliability_table.to_sql("team_style_feature_reliability", conn, if_exists="replace", index=False)
        dimension_weight_audit.to_sql("team_style_dimension_feature_weights", conn, if_exists="replace", index=False)
        pd.DataFrame(
            [
                {
                    "setting_name": "adaptive_style_alpha",
                    "setting_value": adaptive_alpha,
                    "description": "0=fixed semantic prior weights, 1=fully reliability-adaptive row-aware weights",
                },
                {
                    "setting_name": "style_dimension_method",
                    "setting_value": f"blend_fixed_prior_{1.0 - adaptive_alpha:.2f}_adaptive_{adaptive_alpha:.2f}",
                    "description": "Final team style dimensions blend fixed prior scores with adaptive reliability-weighted scores.",
                },
            ]
        ).to_sql("team_style_build_settings", conn, if_exists="replace", index=False)


def cleanup_obsolete_json_outputs(output_dir: Path, filenames: list[str]) -> None:
    for filename in filenames:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    team_db = Path(args.team_db)
    output_dir = Path(args.output_dir)
    sportmonks_db = None if args.disable_sportmonks else Path(args.sportmonks_team_db)

    raw = load_teams(team_db, sportmonks_db, args.sportmonks_match_threshold)
    vectors, feature_reliability_table, dimension_weight_audit = build_style_vectors(
        raw,
        min_group_size=args.min_group_size,
        adaptive_alpha=args.adaptive_style_alpha,
    )
    result = cluster_vectors(vectors, args.min_k, args.max_k, args.random_state)
    write_outputs(vectors, result, feature_reliability_table, dimension_weight_audit, args.adaptive_style_alpha, output_dir)

    silhouette = "n/a" if result.silhouette is None else f"{result.silhouette:.4f}"
    print(f"Wrote team style outputs to {output_dir}")
    print(f"Rows: {len(vectors)}")
    print(f"Best K: {result.best_k}")
    print(f"Silhouette: {silhouette}")


if __name__ == "__main__":
    main()
