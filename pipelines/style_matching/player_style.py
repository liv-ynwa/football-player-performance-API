#!/usr/bin/env python3
"""Data-driven player style clustering and style-card generation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*ensure_all_finite.*",
    category=FutureWarning,
)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ModuleNotFoundError:
    hdbscan = None


PLAYER_METADATA_COLUMNS = [
    "player_id",
    "full_name",
    "age",
    "league",
    "season",
    "position",
    "Current Club",
    "minutes_played_overall",
    "nationality",
    "appearances_overall",
    "average_rating_overall",
    "market_value",
    "annual_salary_eur",
]


ROLE_CONFIG = {
    "FW": {"role_name": "Forward", "positions": {"Forward"}, "k_range": (4, 6)},
    "MF": {"role_name": "Midfielder", "positions": {"Midfielder", "Defensive Midfield"}, "k_range": (5, 7)},
    "DF": {"role_name": "Defender", "positions": {"Defender", "Left Back", "Right Back"}, "k_range": (4, 6)},
    "GK": {"role_name": "Goalkeeper", "positions": {"Goalkeeper"}, "k_range": (2, 4)},
}


EXCLUDED_COLUMN_PARTS = (
    "salary",
    "market_value",
    "rank_in_",
    "rank_",
    "minutes_played",
    "appearances",
    "games_started",
    "games_subbed",
    "shirt_number",
    "booked_over05",
    "man_of_the_match",
    "hattricks",
    "two_goals_in_a_game",
    "three_goals_in_a_game",
    "matches_played",
    "min_per_",
    "shots_per_goal",
    "cards_per_90",
    "yellow_cards",
    "red_cards",
    "clean_sheets",
    "conceded",
    "pen_",
    "pens_",
    "hit_woodwork",
)

EXCLUDED_COLUMN_EXACT = {
    "player_id",
    "full_name",
    "age",
    "league",
    "season",
    "position",
    "Current Club",
    "nationality",
    "position_group",
    "average_rating_overall",
    "ratings_total_overall",
    "annual_salary_gbp",
    "annual_salary_usd",
    "annual_salary_eur",
}

SOURCE_SUFFIX_PATTERNS = (
    ("_auto_per90", "auto_per90"),
    ("_per_90_overall", "per90"),
    ("_percentage_percentile_overall", "percentile"),
    ("_per90_percentile_overall", "percentile"),
    ("_rate_percentile_overall", "percentile"),
    ("_percentile_overall", "percentile"),
    ("_percentage_overall", "percentage"),
    ("_rate_overall", "rate"),
    ("_derived", "derived"),
    ("_total_overall", "total"),
    ("_overall", "overall"),
)

ACTION_FAMILY_ALLOWLIST = {
    "aerial_duels",
    "aerial_duels_won",
    "assists",
    "blocks",
    "chances_created",
    "clearances",
    "cross_completion_rate",
    "crosses",
    "dribbled_past",
    "dribbles",
    "dribbles_successful",
    "dribbles_successful_percentage",
    "dispossesed",
    "duels",
    "duels_won",
    "fouls_committed",
    "fouls_drawn",
    "goals",
    "goals_assists",
    "goals_involved",
    "inside_box_saves",
    "interceptions",
    "key_passes",
    "long_passes",
    "npxg",
    "offsides",
    "pass_completion_rate",
    "passes",
    "passes_completed",
    "possession_regained",
    "progressive_passes",
    "punches",
    "save_percentage",
    "saves",
    "short_passes",
    "shot_accuracy",
    "shot_conversion",
    "shots",
    "shots_faced",
    "shots_off_target",
    "shots_on_target",
    "tackles",
    "tackles_success_rate",
    "tackles_successful",
    "through_passes",
    "xa",
    "xg",
    "accurate_crosses",
    "distance_travelled",
}

BROKEN_FAMILY_BLACKLIST = {
    "chances_created",
    "distance_travelled",
    "long_passes",
    "possession_regained",
    "progressive_passes",
    "short_passes",
    "tackles_successful",
    "through_passes",
}

ROLE_FAMILY_ALLOWLISTS = {
    "FW": {
        "aerial_duels",
        "aerial_duels_won",
        "accurate_crosses",
        "assists",
        "cross_completion_rate",
        "crosses",
        "dispossesed",
        "dribbled_past",
        "dribbles",
        "dribbles_successful",
        "dribbles_successful_percentage",
        "duels",
        "duels_won",
        "fouls_drawn",
        "goals",
        "goals_assists",
        "goals_involved",
        "key_passes",
        "npxg",
        "offsides",
        "pass_completion_rate",
        "passes",
        "passes_completed",
        "shot_accuracy",
        "shot_conversion",
        "shots",
        "shots_off_target",
        "shots_on_target",
        "xa",
        "xg",
    },
    "MF": {
        "accurate_crosses",
        "aerial_duels",
        "aerial_duels_won",
        "assists",
        "blocks",
        "chances_created",
        "cross_completion_rate",
        "crosses",
        "distance_travelled",
        "dribbled_past",
        "dribbles",
        "dribbles_successful",
        "dribbles_successful_percentage",
        "duels",
        "duels_won",
        "fouls_committed",
        "fouls_drawn",
        "goals",
        "goals_assists",
        "goals_involved",
        "interceptions",
        "key_passes",
        "long_passes",
        "npxg",
        "pass_completion_rate",
        "passes",
        "passes_completed",
        "possession_regained",
        "progressive_passes",
        "short_passes",
        "shot_accuracy",
        "shot_conversion",
        "shots",
        "shots_on_target",
        "tackles",
        "tackles_success_rate",
        "tackles_successful",
        "through_passes",
        "xa",
        "xg",
    },
    "DF": {
        "accurate_crosses",
        "aerial_duels",
        "aerial_duels_won",
        "blocks",
        "clearances",
        "crosses",
        "dribbled_past",
        "dribbles",
        "dribbles_successful",
        "duels",
        "duels_won",
        "fouls_committed",
        "interceptions",
        "key_passes",
        "long_passes",
        "pass_completion_rate",
        "passes",
        "passes_completed",
        "shot_accuracy",
        "tackles",
        "tackles_success_rate",
        "xa",
    },
    "GK": {
        "clearances",
        "inside_box_saves",
        "pass_completion_rate",
        "passes",
        "passes_completed",
        "punches",
        "save_percentage",
        "saves",
        "shots_faced",
    },
}


@dataclass(frozen=True)
class FeatureVariant:
    family: str
    column: str
    source_type: str
    quality_score: float
    coverage_ratio: float
    unique_count: int
    entropy_score: float
    variance_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Player style clustering and style-card generation.")
    parser.add_argument("--player-db", default="database/player_features.db", help="Path to player SQLite database.")
    parser.add_argument("--output-dir", default="output/player", help="Directory for JSON/DB outputs.")
    parser.add_argument("--minutes-threshold", type=float, default=400.0, help="Minimum minutes required.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for clustering.")
    return parser.parse_args()


def load_players(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Player database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM player_features_pro", conn)


def map_role(position: str) -> str | None:
    for role, config in ROLE_CONFIG.items():
        if position in config["positions"]:
            return role
    return None


def clean_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def percentile_rank(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return series.rank(method="average", pct=True) * 100.0


def feature_family(column: str) -> tuple[str | None, str | None]:
    for suffix, source_type in SOURCE_SUFFIX_PATTERNS:
        if column.endswith(suffix):
            family = column[: -len(suffix)]
            family = family.replace("shot_accuraccy", "shot_accuracy")
            return family, source_type
    return None, None


def is_candidate_column(column: str, role: str | None = None) -> bool:
    if column in EXCLUDED_COLUMN_EXACT:
        return False
    lower = column.lower()
    if any(part in lower for part in EXCLUDED_COLUMN_PARTS):
        return False
    if lower.endswith("_home") or lower.endswith("_away"):
        return False
    if lower.startswith("rank_"):
        return False
    if not any(lower.endswith(suffix) for suffix, _ in SOURCE_SUFFIX_PATTERNS):
        return False
    family, _ = feature_family(column)
    if column.endswith("_total_overall"):
        return False
    if family not in ACTION_FAMILY_ALLOWLIST:
        return False
    if family in BROKEN_FAMILY_BLACKLIST:
        return False
    if role is None:
        return True
    return family in ROLE_FAMILY_ALLOWLISTS[role]


def add_auto_per90_columns(df: pd.DataFrame, role: str) -> pd.DataFrame:
    enriched = df.copy()
    minutes = clean_numeric(enriched["minutes_played_overall"]).replace(0, np.nan)
    for column in list(df.columns):
        if not column.endswith("_total_overall"):
            continue
        family, _ = feature_family(column)
        if family is None:
            continue
        if family in BROKEN_FAMILY_BLACKLIST:
            continue
        if family not in ROLE_FAMILY_ALLOWLISTS[role]:
            continue
        preferred_raw = f"{family}_per_90_overall"
        preferred_rate = f"{family}_rate_overall"
        preferred_percentage = f"{family}_percentage_overall"
        if preferred_raw in enriched.columns or preferred_rate in enriched.columns or preferred_percentage in enriched.columns:
            continue
        derived_column = f"{family}_auto_per90"
        enriched[derived_column] = clean_numeric(enriched[column]).div(minutes).mul(90.0)
    return enriched


def normalized_entropy(series: pd.Series) -> float:
    valid = clean_numeric(series).replace(-1, np.nan).dropna()
    if len(valid) < 3:
        return 0.0
    bins = max(3, min(12, int(np.sqrt(valid.nunique()))))
    hist, _ = np.histogram(valid, bins=bins)
    hist = hist[hist > 0]
    if len(hist) <= 1:
        return 0.0
    probs = hist / hist.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / np.log(len(hist))


def feature_quality(series: pd.Series) -> dict[str, float]:
    usable = clean_numeric(series).replace(-1, np.nan)
    coverage = float(usable.notna().mean())
    unique_count = int(usable.nunique(dropna=True))
    if usable.notna().sum() == 0 or unique_count <= 1:
        return {
            "coverage_ratio": coverage,
            "unique_count": unique_count,
            "entropy_score": 0.0,
            "variance_score": 0.0,
            "quality_score": 0.0,
        }
    entropy_score = normalized_entropy(usable)
    variance_score = float(np.nanstd(usable.to_numpy(dtype=float), ddof=0))
    non_zero_ratio = float(usable.ne(0).mean())
    unique_ratio = float(unique_count / max(len(usable.dropna()), 1))
    quality_score = (
        coverage
        * non_zero_ratio
        * entropy_score
        * np.sqrt(max(unique_ratio, 0.0))
        * np.log1p(max(variance_score, 0.0))
    )
    return {
        "coverage_ratio": coverage,
        "unique_count": unique_count,
        "entropy_score": entropy_score,
        "variance_score": variance_score,
        "non_zero_ratio": non_zero_ratio,
        "unique_ratio": unique_ratio,
        "quality_score": float(quality_score),
    }


def build_feature_variants(role: str, role_df: pd.DataFrame) -> tuple[list[FeatureVariant], pd.DataFrame]:
    enriched_df = add_auto_per90_columns(role_df, role)
    variants: list[FeatureVariant] = []
    for column in enriched_df.columns:
        if not is_candidate_column(column, role=role):
            continue
        family, source_type = feature_family(column)
        if family is None or source_type is None:
            continue
        stats = feature_quality(enriched_df[column])
        variants.append(
            FeatureVariant(
                family=family,
                column=column,
                source_type=source_type,
                quality_score=stats["quality_score"],
                coverage_ratio=stats["coverage_ratio"],
                unique_count=stats["unique_count"],
                entropy_score=stats["entropy_score"],
                variance_score=stats["variance_score"],
            )
        )
    return variants, enriched_df


def evaluate_variant_map(
    role: str,
    enriched_df: pd.DataFrame,
    variant_map: dict[str, FeatureVariant],
    random_state: int,
) -> float:
    columns = [variant.column for variant in variant_map.values() if variant.quality_score > 0]
    if len(columns) < 4:
        return -1.0

    feature_df = enriched_df[columns].copy()
    if len(feature_df) > 1200:
        feature_df = feature_df.sample(n=1200, random_state=random_state)

    reduced_df, _ = remove_redundant_features(feature_df, list(variant_map.values()))
    if reduced_df.shape[1] < 4:
        return -1.0

    ranked_df, _, _ = rank_features_by_loading(reduced_df, random_state)
    _, scaled_df = standardize_frame(ranked_df)
    pca = PCA(n_components=0.85, random_state=random_state)
    embedding = pca.fit_transform(scaled_df.to_numpy())
    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 1)
    _, _, silhouette, _, _ = choose_kmeans_k(embedding, role, random_state)
    return float(silhouette) if silhouette is not None else -1.0


def choose_best_variants(
    role: str,
    variants: list[FeatureVariant],
    enriched_df: pd.DataFrame,
    random_state: int,
) -> tuple[list[FeatureVariant], pd.DataFrame]:
    audit_rows = []
    by_family: dict[str, list[FeatureVariant]] = {}
    for variant in variants:
        by_family.setdefault(variant.family, []).append(variant)

    quality_winners: dict[str, FeatureVariant] = {}
    family_ordered: dict[str, list[FeatureVariant]] = {}
    for family, family_variants in sorted(by_family.items()):
        ordered = sorted(
            family_variants,
            key=lambda item: (
                item.quality_score,
                item.coverage_ratio,
                item.unique_count,
                item.entropy_score,
                item.variance_score,
                item.column,
            ),
            reverse=True,
        )
        family_ordered[family] = ordered
        quality_winners[family] = ordered[0]

    optimized_winners = quality_winners.copy()
    family_variant_scores: dict[tuple[str, str], float] = {}

    if optimized_winners:
        best_total_score = evaluate_variant_map(role, enriched_df, optimized_winners, random_state)
        changed = False
        for family in sorted(family_ordered):
            current_winner = optimized_winners[family]
            family_best_variant = current_winner
            family_best_score = best_total_score
            for variant in family_ordered[family]:
                trial_map = optimized_winners.copy()
                trial_map[family] = variant
                trial_score = evaluate_variant_map(role, enriched_df, trial_map, random_state)
                family_variant_scores[(family, variant.column)] = trial_score
                if trial_score > family_best_score + 1e-9:
                    family_best_variant = variant
                    family_best_score = trial_score
                elif abs(trial_score - family_best_score) <= 1e-9 and variant.quality_score > family_best_variant.quality_score:
                    family_best_variant = variant
            if family_best_variant.column != current_winner.column:
                optimized_winners[family] = family_best_variant
                best_total_score = family_best_score
                changed = True

    winners = [variant for variant in optimized_winners.values() if variant.quality_score > 0]
    for family, ordered in sorted(family_ordered.items()):
        winner = optimized_winners[family]
        for variant in ordered:
            audit_rows.append(
                {
                    "feature_family": family,
                    "variant_column": variant.column,
                    "variant_source_type": variant.source_type,
                    "coverage_ratio": round(variant.coverage_ratio, 6),
                    "unique_count": int(variant.unique_count),
                    "entropy_score": round(variant.entropy_score, 6),
                    "variance_score": round(variant.variance_score, 6),
                    "quality_score": round(variant.quality_score, 6),
                    "cluster_effect_score": round(float(family_variant_scores.get((family, variant.column), best_total_score)), 6),
                    "is_quality_winner": variant.column == quality_winners[family].column,
                    "is_family_winner": variant.column == winner.column,
                    "passes_quality_gate": winner.quality_score > 0 and variant.column == winner.column,
                }
            )
    return winners, pd.DataFrame(audit_rows)


def standardize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filled = df.copy()
    for column in df.columns:
        series = clean_numeric(df[column]).replace(-1, np.nan)
        median = float(series.median()) if series.notna().any() else 0.0
        filled[column] = series.fillna(median)
    scaled = StandardScaler().fit_transform(filled)
    return filled, pd.DataFrame(scaled, columns=df.columns, index=df.index)


def remove_redundant_features(
    feature_df: pd.DataFrame,
    winners: list[FeatureVariant],
    correlation_cutoff: float = 0.92,
) -> tuple[pd.DataFrame, dict[str, str]]:
    _, scaled_df = standardize_frame(feature_df)
    quality_lookup = {variant.column: variant.quality_score for variant in winners}
    keep = list(scaled_df.columns)
    removal_reasons: dict[str, str] = {}

    variances = scaled_df.var(ddof=0)
    for column in list(keep):
        if float(variances[column]) <= 1e-8:
            keep.remove(column)
            removal_reasons[column] = "near_zero_variance"

    corr = scaled_df[keep].corr().abs() if keep else pd.DataFrame()
    for i, column in enumerate(list(keep)):
        if column not in keep:
            continue
        for other in keep[i + 1 :]:
            if other not in keep:
                continue
            if float(corr.loc[column, other]) < correlation_cutoff:
                continue
            drop = other if quality_lookup.get(column, 0.0) >= quality_lookup.get(other, 0.0) else column
            if drop in keep:
                keep.remove(drop)
                removal_reasons[drop] = f"high_correlation_with_{column if drop == other else other}"
            if column not in keep:
                break

    return feature_df[keep].copy(), removal_reasons


def rank_features_by_loading(feature_df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, dict[str, float], int]:
    _, scaled_df = standardize_frame(feature_df)
    max_components = min(len(feature_df.columns), len(feature_df) - 1)
    if max_components <= 0:
        raise ValueError("Insufficient data for PCA feature ranking.")
    pca = PCA(n_components=min(max_components, len(feature_df.columns)), random_state=random_state)
    pca.fit(scaled_df.to_numpy())
    loadings = np.abs(pca.components_.T)
    weighted = loadings * pca.explained_variance_ratio_
    scores = weighted.sum(axis=1)
    score_series = pd.Series(scores, index=scaled_df.columns).sort_values(ascending=False)
    score_share = score_series / score_series.sum()
    cumulative = score_share.cumsum()

    min_keep = min(len(score_series), max(6, int(np.sqrt(len(score_series)) * 2)))
    max_keep = min(len(score_series), 24)
    cutoff_count = max(min_keep, int((cumulative <= 0.90).sum()) + 1)
    cutoff_count = min(cutoff_count, max_keep)
    selected_columns = score_series.head(cutoff_count).index.tolist()
    return feature_df[selected_columns].copy(), score_series.to_dict(), cutoff_count


def choose_kmeans_k(embedding: np.ndarray, role: str, random_state: int) -> tuple[np.ndarray, int, float | None, dict[int, float], np.ndarray]:
    min_k, max_k = ROLE_CONFIG[role]["k_range"]
    n = len(embedding)
    max_k = min(max_k, n - 1)
    min_k = min(min_k, max_k)
    best_labels = None
    best_centers = None
    best_k = min_k
    best_score = -1.0
    scores: dict[int, float] = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, n_init=30, random_state=random_state)
        labels = model.fit_predict(embedding)
        if len(set(labels)) <= 1:
            continue
        score = float(silhouette_score(embedding, labels))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
            best_centers = model.cluster_centers_

    if best_labels is None or best_centers is None:
        model = KMeans(n_clusters=min_k, n_init=30, random_state=random_state)
        best_labels = model.fit_predict(embedding)
        best_centers = model.cluster_centers_
        return best_labels, min_k, None, scores, best_centers

    return best_labels, best_k, best_score, scores, best_centers


def kmeans_membership_probability(embedding: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = embedding[:, None, :] - centers[None, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))
    median_distance = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    scale = median_distance if median_distance > 0 else 1.0
    similarities = np.exp(-distances / scale)
    probabilities = similarities / similarities.sum(axis=1, keepdims=True)
    return probabilities[np.arange(len(labels)), labels]


def run_hdbscan(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hdbscan is None:
        return np.full(len(embedding), -1, dtype=int), np.zeros(len(embedding), dtype=float)
    min_cluster_size = max(8, int(len(embedding) * 0.015))
    min_samples = max(4, min_cluster_size // 2)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(embedding)
    probabilities = getattr(clusterer, "probabilities_", np.zeros(len(embedding)))
    return labels, probabilities


def clean_feature_name(name: str) -> str:
    value = name.replace("shot_accuraccy", "shot_accuracy")
    value = value.replace("__", "_")
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def feature_signature(cluster_center: pd.Series, top_n: int = 3) -> list[str]:
    ordered = cluster_center.reindex(cluster_center.abs().sort_values(ascending=False).index)
    signature = []
    for column, score in ordered.items():
        direction = "high" if score >= 0 else "low"
        signature.append(f"{direction}_{clean_feature_name(column)}")
        if len(signature) == top_n:
            break
    return signature


def cluster_label(role: str, cluster_center: pd.Series) -> str:
    role_name = ROLE_CONFIG[role]["role_name"]
    return " + ".join(feature_signature(cluster_center, top_n=3)) + f" {role_name}"


def secondary_tendencies(row_scaled: pd.Series, cluster_center: pd.Series) -> str:
    delta = row_scaled - cluster_center
    ordered = delta.reindex(delta.abs().sort_values(ascending=False).index)
    tags = []
    for column, value in ordered.items():
        direction = "extra_high" if value >= 0 else "extra_low"
        tags.append(f"{direction}_{clean_feature_name(column)}")
        if len(tags) == 2:
            break
    return " + ".join(tags)


def representative_players(role_df: pd.DataFrame, embedding: np.ndarray, labels: np.ndarray, cluster_id: int) -> str:
    mask = labels == cluster_id
    if mask.sum() == 0:
        return ""
    sub_embed = embedding[mask]
    center = sub_embed.mean(axis=0)
    distances = np.sqrt(((sub_embed - center) ** 2).sum(axis=1))
    sub_df = role_df.loc[mask, ["full_name", "season"]].copy()
    sub_df["distance"] = distances
    sample = sub_df.sort_values("distance").head(3)
    return "; ".join(f"{row.full_name} ({row.season})" for row in sample.itertuples())


def build_latent_dimensions(embedding: np.ndarray, index: pd.Index, count: int = 8) -> pd.DataFrame:
    latent = pd.DataFrame(index=index)
    usable_count = min(count, embedding.shape[1])
    for idx in range(usable_count):
        latent[f"latent_style_dim_{idx + 1}"] = percentile_rank(pd.Series(embedding[:, idx], index=index)).fillna(50.0).round(2)
    for idx in range(usable_count, count):
        latent[f"latent_style_dim_{idx + 1}"] = 50.0
    return latent


def process_role(role: str, role_df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    variants, enriched_df = build_feature_variants(role, role_df)
    winners, audit_df = choose_best_variants(role, variants, enriched_df, random_state)
    winner_lookup = {variant.column: variant for variant in winners}
    candidate_columns = [variant.column for variant in winners if variant.quality_score > 0]
    if len(candidate_columns) < 6:
        raise ValueError(f"{role} usable feature families too few: {len(candidate_columns)}")

    candidate_feature_df = enriched_df[candidate_columns].copy()
    reduced_feature_df, removal_reasons = remove_redundant_features(candidate_feature_df, winners)
    ranked_feature_df, loading_scores, ranked_count = rank_features_by_loading(reduced_feature_df, random_state)
    filled_df, scaled_df = standardize_frame(ranked_feature_df)

    pca = PCA(n_components=0.85, random_state=random_state)
    embedding = pca.fit_transform(scaled_df.to_numpy())
    if embedding.ndim == 1:
        embedding = embedding.reshape(-1, 1)

    labels, best_k, silhouette, scores, centers = choose_kmeans_k(embedding, role, random_state)
    hdb_labels, hdb_probs = run_hdbscan(embedding)
    membership_prob = kmeans_membership_probability(embedding, labels, centers)

    all_selected_columns = set(ranked_feature_df.columns)
    reduced_columns = set(reduced_feature_df.columns)
    winner_columns = {variant.column for variant in winners}
    for idx in audit_df.index:
        column = audit_df.at[idx, "variant_column"]
        audit_df.at[idx, "role"] = role
        audit_df.at[idx, "kept_after_correlation"] = column in reduced_columns
        audit_df.at[idx, "selected_for_model"] = column in all_selected_columns
        audit_df.at[idx, "loading_score"] = round(float(loading_scores.get(column, 0.0)), 6)
        if column not in winner_columns:
            audit_df.at[idx, "stage_reason"] = "not_best_variant_for_family"
        elif column in removal_reasons:
            audit_df.at[idx, "stage_reason"] = removal_reasons[column]
        elif column in all_selected_columns:
            audit_df.at[idx, "stage_reason"] = "selected_for_model"
        else:
            audit_df.at[idx, "stage_reason"] = "dropped_after_loading_ranking"

    cards = role_df[PLAYER_METADATA_COLUMNS].copy()
    cards["role"] = role
    cards["cluster_id"] = labels
    cards["style_confidence"] = np.round(membership_prob * 100.0, 2)
    cards["style_mix_index"] = np.round((1.0 - membership_prob) * 100.0, 2)
    cards["hdbscan_label"] = hdb_labels
    cards["hdbscan_probability"] = np.round(hdb_probs * 100.0, 2)
    cards["is_composite"] = hdb_labels == -1
    cards["main_style_label"] = ""
    cards["secondary_tendencies"] = ""

    latent_df = build_latent_dimensions(embedding, cards.index)
    for column in latent_df.columns:
        cards[column] = latent_df[column]

    cluster_summaries = []
    cluster_centers_scaled: dict[int, pd.Series] = {}
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_center = scaled_df.loc[mask].mean()
        cluster_centers_scaled[int(cluster_id)] = cluster_center
        style_label = cluster_label(role, cluster_center)
        cards.loc[mask, "main_style_label"] = style_label
        for idx in cards.index[mask]:
            cards.at[idx, "secondary_tendencies"] = secondary_tendencies(scaled_df.loc[idx], cluster_center)
        cluster_summaries.append(
            {
                "role": role,
                "role_name": ROLE_CONFIG[role]["role_name"],
                "cluster_id": int(cluster_id),
                "player_count": int(mask.sum()),
                "style_label": style_label,
                "feature_signature": " + ".join(feature_signature(cluster_center, top_n=4)),
                "representative_players": representative_players(role_df, embedding, labels, int(cluster_id)),
            }
        )

    role_summary = {
        "role": role,
        "role_name": ROLE_CONFIG[role]["role_name"],
        "player_rows": int(len(role_df)),
        "candidate_variant_count": int(len(variants)),
        "usable_family_count": int(len(candidate_columns)),
        "post_correlation_feature_count": int(reduced_feature_df.shape[1]),
        "selected_feature_count": int(ranked_feature_df.shape[1]),
        "pca_components": int(embedding.shape[1]),
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "best_k": int(best_k),
        "silhouette": silhouette,
        "hdbscan_noise_rate": float((hdb_labels == -1).mean()),
        "selected_features": [
            {
                "feature_family": winner_lookup[column].family,
                "column": column,
                "source_type": winner_lookup[column].source_type,
                "quality_score": round(float(winner_lookup[column].quality_score), 6),
                "loading_score": round(float(loading_scores.get(column, 0.0)), 6),
            }
            for column in ranked_feature_df.columns
        ],
        "scanned_k_scores": {str(k): score for k, score in scores.items()},
        "latent_dimensions": [column for column in latent_df.columns],
        "loading_rank_cutoff_count": int(ranked_count),
    }

    return cards, pd.DataFrame(cluster_summaries), audit_df, role_summary


def write_outputs(
    all_cards: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    feature_audit: pd.DataFrame,
    role_summaries: list[dict[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "player_style_results.db"
    cleanup_obsolete_json_outputs(
        output_dir,
        [
            "player_style_results.json",
            "player_style_run_metadata.json",
            "player_feature_audit.json",
        ],
    )
    role_summary_df = pd.DataFrame(role_summaries)
    role_summary_sql_df = role_summary_df.copy()
    for column in role_summary_sql_df.columns:
        if role_summary_sql_df[column].map(lambda value: isinstance(value, (list, dict))).any():
            role_summary_sql_df[column] = role_summary_sql_df[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            )

    with sqlite3.connect(db_path) as conn:
        all_cards.to_sql("player_style_cards", conn, if_exists="replace", index=False)
        cluster_summary.to_sql("player_cluster_summary", conn, if_exists="replace", index=False)
        feature_audit.to_sql("player_feature_audit", conn, if_exists="replace", index=False)
        role_summary_sql_df.to_sql("player_role_summary", conn, if_exists="replace", index=False)


def cleanup_obsolete_json_outputs(output_dir: Path, filenames: list[str]) -> None:
    for filename in filenames:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    players = load_players(Path(args.player_db))
    players["role"] = players["position"].map(map_role)
    players = players[players["role"].notna()].copy()
    players["minutes_played_overall"] = clean_numeric(players["minutes_played_overall"])
    players = players[players["minutes_played_overall"] >= args.minutes_threshold].copy()
    players = players.reset_index(drop=True)

    all_cards = []
    all_cluster_summaries = []
    all_feature_audits = []
    role_summaries = []

    for role in ROLE_CONFIG:
        role_df = players[players["role"] == role].copy().reset_index(drop=True)
        if role_df.empty:
            continue
        cards, cluster_summary, feature_audit, role_summary = process_role(role, role_df, args.random_state)
        all_cards.append(cards)
        all_cluster_summaries.append(cluster_summary)
        all_feature_audits.append(feature_audit)
        role_summaries.append(role_summary)

    all_cards_df = pd.concat(all_cards, ignore_index=True)
    cluster_summary_df = pd.concat(all_cluster_summaries, ignore_index=True)
    feature_audit_df = pd.concat(all_feature_audits, ignore_index=True)
    write_outputs(all_cards_df, cluster_summary_df, feature_audit_df, role_summaries, Path(args.output_dir))

    print(f"Wrote player style outputs to {args.output_dir}")
    print(f"Rows: {len(all_cards_df)}")
    for summary in role_summaries:
        silhouette = "n/a" if summary["silhouette"] is None else f"{summary['silhouette']:.4f}"
        print(
            f"{summary['role']}: players={summary['player_rows']} selected_features={summary['selected_feature_count']} "
            f"best_k={summary['best_k']} silhouette={silhouette}"
        )


if __name__ == "__main__":
    main()
