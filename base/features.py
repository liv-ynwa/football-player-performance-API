from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "database" / "player_features_base.db"
DEFAULT_TABLE = "player_features_base"

DEFAULT_DROP_COLUMNS = {"player_id", "full_name"}


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


def filter_by_position(
    df: pd.DataFrame,
    positions: str | list[str] | tuple[str, ...],
    position_col_candidates: tuple[str, ...] = ("position_group", "position"),
) -> pd.DataFrame:
    if isinstance(positions, str):
        wanted = {positions}
    else:
        wanted = set(positions)

    wanted = {p.upper() for p in wanted}
    position_col = _choose_position_col(df, wanted, position_col_candidates)

    if position_col is None:
        return df

    normalized = _normalize_position(df[position_col])
    mask = normalized.isin(wanted)
    return df.loc[mask].copy()


def load_base_dataframe(
    db_path: Path | str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    db_path = Path(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def build_feature_matrix(
    df: pd.DataFrame,
    drop_cols: set[str] | None = None,
    numeric_only: bool = False,
    positions: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if positions is not None:
        df = filter_by_position(df, positions)

    # Keep only columns that have at least one non-null value.
    non_null_cols = [col for col in df.columns if not df[col].isna().all()]
    X = df[non_null_cols].copy()

    # Derived availability features.
    if "minutes_played_overall" in X.columns:
        minutes = pd.to_numeric(X["minutes_played_overall"], errors="coerce")
        X["nineties"] = minutes / 90.0
    if "minutes_played_overall" in X.columns and "appearances_overall" in X.columns:
        minutes = pd.to_numeric(X["minutes_played_overall"], errors="coerce")
        apps = pd.to_numeric(X["appearances_overall"], errors="coerce")
        X["minutes_per_appearance"] = minutes / apps.replace(0, np.nan)

    drop_cols = drop_cols or set()
    if drop_cols:
        X = X.drop(columns=list(drop_cols), errors="ignore")

    if numeric_only:
        X = X.select_dtypes(include="number")

    return X


def load_feature_matrix(
    db_path: Path | str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    drop_cols: set[str] | None = None,
    numeric_only: bool = False,
    positions: str | list[str] | tuple[str, ...] | None = None,
    return_feature_names: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    df = load_base_dataframe(db_path=db_path, table=table)
    X = build_feature_matrix(
        df,
        drop_cols=drop_cols if drop_cols is not None else DEFAULT_DROP_COLUMNS,
        numeric_only=numeric_only,
        positions=positions,
    )
    if return_feature_names:
        return X, list(X.columns)
    return X
