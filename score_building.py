from __future__ import annotations

from typing import Iterable, Sequence
from pathlib import Path
import sys
import argparse
import sqlite3
import math

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.features import (
    DEFAULT_DROP_COLUMNS,
    build_feature_matrix,
    load_base_dataframe,
)
from base.target import (
    build_targets_by_position,
)

import numpy as np
import pandas as pd

CLUB_COL_CANDIDATES = ("current club", "Current Club", "club", "club_name", "team", "team_name", "squad")


def _to_series(values: Iterable[object], name: str = "score") -> pd.Series:
    if isinstance(values, pd.Series):
        return values.rename(name)
    return pd.Series(values, name=name)


def _norm_ppf(p: pd.Series | np.ndarray) -> np.ndarray:
    """
    Approximate inverse CDF of the standard normal distribution.
    Acklam's approximation, accurate to ~1e-9 in the central region.
    """
    p = np.asarray(p, dtype="float64")

    # Coefficients in rational approximations.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    x = np.empty_like(p)

    # Lower region
    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2 * np.log(p[mask]))
        x[mask] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    # Central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q * q
        x[mask] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

    # Upper region
    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2 * np.log(1 - p[mask]))
        x[mask] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    return x


def _robust_z(x: pd.Series) -> pd.Series:
    values = pd.to_numeric(x, errors="coerce")
    if not values.notna().any():
        return pd.Series(np.nan, index=values.index)
    median = float(values.median(skipna=True))
    mad = float((values - median).abs().median(skipna=True))
    scale = 1.4826 * mad
    if scale == 0 or np.isnan(scale):
        scale = float(values.std(ddof=0, skipna=True))
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    return (values - median) / scale


def _group_robust_z(x: pd.Series, keys: Sequence[pd.Series] | None) -> pd.Series:
    values = pd.to_numeric(x, errors="coerce")
    if not keys:
        return _robust_z(values)

    df = pd.DataFrame({"x": values})
    for i, key in enumerate(keys):
        df[f"k{i}"] = key.values if isinstance(key, pd.Series) else key

    cols = [f"k{i}" for i in range(len(keys))]
    grouped = df.groupby(cols, dropna=False)

    # Per-group robust center
    median = grouped["x"].transform("median")

    # Per-group MAD (median absolute deviation)
    df["_abs_dev"] = (df["x"] - median).abs()
    mad = grouped["_abs_dev"].transform("median")

    # Robust scale with safe fallbacks
    scale = 1.4826 * mad
    scale = scale.replace(0, np.nan)

    std = grouped["x"].transform("std")
    scale = scale.fillna(std.replace(0, np.nan)).fillna(1.0)

    return (values - median) / scale


def _group_abs_percentile(z: pd.Series, keys: Sequence[pd.Series] | None, p: float) -> pd.Series:
    values = pd.to_numeric(z, errors="coerce").abs()
    if not keys:
        q = float(np.nanpercentile(values, p))
        if not np.isfinite(q) or q == 0:
            q = 1.0
        return pd.Series(q, index=values.index)

    df = pd.DataFrame({"z": values})
    for i, key in enumerate(keys):
        df[f"k{i}"] = key.values if isinstance(key, pd.Series) else key
    grouped = df.groupby([f"k{i}" for i in range(len(keys))], dropna=False)
    q = grouped["z"].transform(lambda s: np.nanpercentile(s, p))
    q = q.replace(0, np.nan).fillna(1.0)
    return q


def _group_counts(values: pd.Series, keys: Sequence[pd.Series] | None) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    if not keys:
        return pd.Series(vals.notna().sum(), index=vals.index)

    df = pd.DataFrame({"x": vals})
    for i, key in enumerate(keys):
        df[f"k{i}"] = key.values if isinstance(key, pd.Series) else key
    grouped = df.groupby([f"k{i}" for i in range(len(keys))], dropna=False)
    counts = grouped["x"].transform("count")
    return counts


def score_percentile(
    values: Iterable[object],
    *,
    group_keys: Sequence[pd.Series] | None = None,
    method: str = "average",
    out_scale: float = 10.0,
    center: float = 5.0,
    span: float = 4.0,
    clip_percentile: float = 90.0,
    group_shrink_k: float = 50.0,
) -> pd.Series:
    """
    Robust, distribution-calibrated score mapped to 0-10 by default.
    - group_keys: optional list of group series for per-group percentile.
    - center: midpoint score (default 5).
    - span: half-range around center (default 4 -> range ~1..9).
    - clip_percentile: percentile of |z| used to calibrate scale.
    """
    s = _to_series(values)
    z_group = _group_robust_z(s, group_keys)
    z_global = _robust_z(s)

    if group_keys:
        counts = _group_counts(s, group_keys)
        weight = counts / (counts + group_shrink_k)
        z = weight * z_group + (1.0 - weight) * z_global
        z_scale_group = _group_abs_percentile(z_group, group_keys, clip_percentile)
        z_scale_global = float(np.nanpercentile(np.abs(z_global), clip_percentile))
        if not np.isfinite(z_scale_global) or z_scale_global == 0:
            z_scale_global = 1.0
        z_scale = weight * z_scale_group + (1.0 - weight) * z_scale_global
    else:
        z = z_global
        z_scale = _group_abs_percentile(z, None, clip_percentile)

    score = center + span * np.tanh(z / z_scale)
    score = score.clip(lower=0.0, upper=out_scale)
    return pd.Series(score, index=s.index, name=s.name)


def score_sigmoid(
    values: Iterable[object],
    *,
    center: float | None = None,
    scale: float | None = None,
    soft_clip: float = 4.0,
    out_scale: float = 10.0,
    center_score: float = 5.0,
    span: float = 4.0,
    clip_percentile: float = 90.0,
    group_keys: Sequence[pd.Series] | None = None,
    group_shrink_k: float = 50.0,
) -> pd.Series:
    """
    Robust value-based score, mapped to 0-10 by default.
    - center: value that maps to center (default: median).
    - scale: spread (default: robust MAD*1.4826, fallback std dev, then 1.0).
    - soft_clip: tanh-based soft clipping to reduce extreme outliers.
    - center_score: midpoint score (default 5).
    - span: half-range around center (default 4 -> range ~1..9).
    - clip_percentile: percentile of |z| used to calibrate scale.
    """
    s = _to_series(values)
    x = pd.to_numeric(s, errors="coerce")

    if center is None or scale is None:
        z_group = _group_robust_z(x, group_keys)
        z_global = _robust_z(x)
    else:
        z_group = (x - center) / scale
        if scale == 0 or np.isnan(scale):
            z_group = _group_robust_z(x, group_keys)
        z_global = _robust_z(x)

    if soft_clip and np.isfinite(soft_clip) and soft_clip > 0:
        z_group = np.tanh(z_group / soft_clip) * soft_clip
        z_global = np.tanh(z_global / soft_clip) * soft_clip

    if group_keys:
        counts = _group_counts(x, group_keys)
        weight = counts / (counts + group_shrink_k)
        z = weight * z_group + (1.0 - weight) * z_global
        z_scale_group = _group_abs_percentile(z_group, group_keys, clip_percentile)
        z_scale_global = float(np.nanpercentile(np.abs(z_global), clip_percentile))
        if not np.isfinite(z_scale_global) or z_scale_global == 0:
            z_scale_global = 1.0
        z_scale = weight * z_scale_group + (1.0 - weight) * z_scale_global
    else:
        z = z_global
        z_scale = _group_abs_percentile(z, None, clip_percentile)

    score = center_score + span * np.tanh(z / z_scale)
    score = score.clip(lower=0.0, upper=out_scale)
    return pd.Series(score, index=s.index, name=s.name)


__all__ = [
    "score_percentile",
    "score_sigmoid",
    "build_scores",
    "build_scored_dataset",
    "build_rating_display",
]


def build_scores(
    targets: pd.DataFrame,
    method: str = "both",
    group_cols: Sequence[str] | None = ("position_bucket",),
    group_keys: Sequence[pd.Series] | None = None,
    exclude_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    score_df = targets.copy()
    if group_keys is None:
        group_keys = None
        if group_cols:
            group_keys = [score_df[col] for col in group_cols if col in score_df.columns]
    exclude_set = set(exclude_cols or [])

    for col in score_df.columns:
        if col == "position_bucket":
            continue
        if col in exclude_set:
            continue
        if not pd.api.types.is_numeric_dtype(score_df[col]):
            continue

        if method in ("percentile", "both"):
            score_df[f"{col}_score_pct"] = score_percentile(
                score_df[col], group_keys=group_keys
            )
        if method in ("sigmoid", "both"):
            score_df[f"{col}_score_sigmoid"] = score_sigmoid(
                score_df[col], group_keys=group_keys
            )

    return score_df


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def _weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [c for c in weights.keys() if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)

    w = pd.Series({c: weights[c] for c in cols}, dtype="float64")
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().astype("float64")

    denom = valid.mul(w, axis=1).sum(axis=1)
    numer = values.fillna(0.0).mul(w, axis=1).sum(axis=1)

    out = numer / denom.replace(0, np.nan)
    return out


BASE_WEIGHTS_BY_POS: dict[str, dict[str, float]] = {
    # appearance_score is intentionally excluded from rating_display
    "FWD": {
        "attack_score": 0.28,
        "assist_score": 0.22,
        "passing_score": 0.10,
        "defense_score": 0.08,
        "foul_card_score": 0.06,
        "aerial_score": 0.11,
        "conceded_score": 0.15,
    },
    "MID": {
        "passing_score": 0.24,
        "assist_score": 0.20,
        "attack_score": 0.16,
        "defense_score": 0.16,
        "foul_card_score": 0.06,
        "aerial_score": 0.08,
        "conceded_score": 0.10,
    },
    "DEF": {
        "defense_score": 0.26,
        "aerial_score": 0.14,
        "conceded_score": 0.12,
        "ball_security_score": 0.10,
        "passing_score": 0.12,
        "assist_score": 0.08,
        "attack_score": 0.05,
        "foul_card_score": 0.13,
    },
    "GK": {
        "goalkeeper_score": 0.60,
        "defense_score": 0.15,
        "conceded_score": 0.15,
        "foul_card_score": 0.10,
    },
    "OTHER": {
        "attack_score": 0.20,
        "assist_score": 0.20,
        "passing_score": 0.15,
        "defense_score": 0.15,
        "foul_card_score": 0.10,
        "aerial_score": 0.10,
        "conceded_score": 0.10,
    },
}


def build_rating_display(targets: pd.DataFrame) -> pd.Series:
    return build_rating_display_with_weights(targets)


def build_rating_display_with_weights(
    targets: pd.DataFrame,
    *,
    weights_by_pos: dict[str, dict[str, float]] | None = None,
    drop_components: Sequence[str] | None = None,
) -> pd.Series:
    if "position_bucket" not in targets.columns:
        return pd.Series(np.nan, index=targets.index, name="rating_display")

    weights_by_pos = weights_by_pos or BASE_WEIGHTS_BY_POS
    drop_set = {c.lower() for c in (drop_components or [])}

    rating = pd.Series(np.nan, index=targets.index, name="rating_display")
    for bucket, weights in weights_by_pos.items():
        mask = targets["position_bucket"] == bucket
        if not mask.any():
            continue
        if drop_set:
            weights = {k: v for k, v in weights.items() if k.lower() not in drop_set}
        weights = _normalize_weights(weights)
        rating.loc[mask] = _weighted_score(targets.loc[mask], weights)

    missing = rating.isna()
    if missing.any():
        fallback_weights = {
            "attack_score": 1.0,
            "assist_score": 1.0,
            "passing_score": 1.0,
            "defense_score": 1.0,
            "aerial_score": 1.0,
            "foul_card_score": 1.0,
            "conceded_score": 1.0,
            "goalkeeper_score": 1.0,
        }
        if drop_set:
            fallback_weights = {k: v for k, v in fallback_weights.items() if k.lower() not in drop_set}
        fallback_weights = _normalize_weights(fallback_weights)
        rating.loc[missing] = _weighted_score(targets.loc[missing], fallback_weights)

    rating.name = "rating_display"
    return rating


def build_scored_dataset(
    df_raw: pd.DataFrame,
    *,
    method: str = "both",
    group_cols: Sequence[str] | None = ("season", "position_bucket"),
    targets_group_cols: Sequence[str] | None = ("season", "position_bucket"),
    numeric_only: bool = False,
    include_rating_display: bool = True,
    rating_display_weights_by_pos: dict[str, dict[str, float]] | None = None,
    rating_display_drop_components: Sequence[str] | None = None,
    exclude_score_cols: Sequence[str] | None = None,
    identifier_cols: Sequence[str] = (
        "player_id",
        "full_name",
        "position",
        "position_group",
        "current club",
        "Current Club",
        "club",
        "club_name",
        "team",
        "team_name",
        "squad",
        "league",
        "season",
    ),
) -> pd.DataFrame:
    targets_group_cols = tuple(targets_group_cols or ())
    targets_include_league = "league" in targets_group_cols
    targets = build_targets_by_position(
        df_raw,
        group_cols=targets_group_cols,
        include_league=targets_include_league,
    )
    if include_rating_display:
        targets["rating_display"] = build_rating_display_with_weights(
            targets,
            weights_by_pos=rating_display_weights_by_pos,
            drop_components=rating_display_drop_components,
        )
    features = build_feature_matrix(
        df_raw,
        drop_cols=DEFAULT_DROP_COLUMNS,
        numeric_only=numeric_only,
    )
    group_keys: list[pd.Series] = []
    if group_cols:
        for col in group_cols:
            if col in df_raw.columns:
                group_keys.append(df_raw[col])
            elif col in targets.columns:
                group_keys.append(targets[col])
    scored = build_scores(
        targets,
        method=method,
        group_cols=group_cols,
        group_keys=group_keys or None,
        exclude_cols=exclude_score_cols,
    )

    frames = []
    if identifier_cols:
        id_cols = [c for c in identifier_cols if c in df_raw.columns]
        if id_cols:
            id_df = df_raw[id_cols].copy()
            club_col = next((c for c in CLUB_COL_CANDIDATES if c in id_df.columns), None)
            if club_col:
                id_df.rename(columns={club_col: "current club"}, inplace=True)
            frames.append(id_df)
    frames.extend([features, scored])

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    # Normalize club aliases to a single output column name.
    club_aliases = [c for c in CLUB_COL_CANDIDATES if c in out.columns]
    if club_aliases:
        club_series = out["current club"] if "current club" in out.columns else pd.Series(np.nan, index=out.index)
        for col in club_aliases:
            if col == "current club":
                continue
            club_series = club_series.combine_first(out[col])
        out["current club"] = club_series
        drop_cols = [c for c in club_aliases if c != "current club"]
        if drop_cols:
            out = out.drop(columns=drop_cols, errors="ignore")

    # SQLite treats column names case-insensitively; drop case-only duplicates.
    keep_cols: list[str] = []
    seen_lower: set[str] = set()
    for col in out.columns:
        lower = col.lower()
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        keep_cols.append(col)
    out = out.loc[:, keep_cols]

    return out


def _select_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep: list[str] = []
    drop_substrings = ("per_90", "per90")
    keep_keywords = ("game", "games", "season", "overall")

    for col in df.columns:
        col_l = col.lower()

        if col_l.endswith("_score_pct") or col_l.endswith("_score_sigmoid"):
            keep.append(col)
            continue
        if col_l.endswith("_score"):
            # Drop raw component scores; keep only *_score_pct/_score_sigmoid
            continue
        if any(s in col_l for s in drop_substrings):
            continue

        if col_l in {
            "player_id",
            "full_name",
            "position",
            "position_group",
            "current club",
            "league",
            "season",
        }:
            keep.append(col)
            continue

        if any(k in col_l for k in keep_keywords):
            keep.append(col)
            continue

    keep = [c for c in keep if c in df.columns]
    return df.loc[:, keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 0-10 scores from base targets.")
    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "database" / "player_features_base.db",
        help="Path to player_features_base.db",
    )
    parser.add_argument(
        "--table",
        "--table-base",
        dest="table",
        type=str,
        default="player_features_base",
        help="Table name for base features",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("percentile", "sigmoid", "both"),
        default="both",
        help="Scoring method",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        default="season,position_bucket",
        help="Comma-separated group columns for percentile scoring",
    )
    parser.add_argument(
        "--targets-group-cols",
        type=str,
        default="season,position_bucket",
        help="Comma-separated group columns for target normalization",
    )
    parser.add_argument(
        "--out-db",
        type=Path,
        default=PROJECT_ROOT / "database" / "base_rating.db",
        help="Output SQLite DB path",
    )
    parser.add_argument(
        "--out-table",
        type=str,
        default="base_scores",
        help="Output table name",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all columns (skip minimal display filtering).",
    )
    args = parser.parse_args()

    group_cols = tuple([c.strip() for c in args.group_cols.split(",") if c.strip()])
    targets_group_cols = tuple(
        [c.strip() for c in args.targets_group_cols.split(",") if c.strip()]
    )

    args.out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.out_db) as conn:
        df_base = load_base_dataframe(db_path=args.db, table=args.table)
        scored_base = build_scored_dataset(
            df_base,
            method=args.method,
            group_cols=group_cols,
            targets_group_cols=targets_group_cols,
            rating_display_weights_by_pos=BASE_WEIGHTS_BY_POS,
            rating_display_drop_components=("passing_score", "defense_score", "aerial_score"),
            exclude_score_cols=("passing_score", "defense_score", "aerial_score"),
        )
        if not args.keep_all:
            scored_base = _select_display_columns(scored_base)
        numeric_cols = scored_base.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            scored_base[numeric_cols] = scored_base[numeric_cols].round(2)
        scored_base.to_sql(args.out_table, conn, if_exists="replace", index=False)

    print(f"[done] saved base scores to {args.out_db} (table={args.out_table})")


if __name__ == "__main__":
    main()
