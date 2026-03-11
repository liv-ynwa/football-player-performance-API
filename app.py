from __future__ import annotations

from pathlib import Path
import os
import json
import sqlite3
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from score_building import BASE_WEIGHTS_BY_POS, build_scored_dataset
from base.features import load_base_dataframe


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DB_PATH = Path(os.getenv("BASE_DB_PATH", PROJECT_ROOT / "database" / "player_features_base.db"))
BASE_TABLE = os.getenv("BASE_TABLE", "player_features_base")
SCORES_DB_PATH = Path(os.getenv("SCORES_DB_PATH", PROJECT_ROOT / "database" / "base_rating.db"))
SCORES_TABLE = os.getenv("SCORES_TABLE", "base_scores")
META_PATH = Path(os.getenv("META_PATH", PROJECT_ROOT / "meta.json"))

DEFAULT_ORDER_COL = "rating_display_score_pct"
PLAYER_COLUMNS = [
    "player_id",
    "full_name",
    "position",
    "position_group",
    "current club",
    "league",
    "season",
    "minutes_played_overall",
    "appearances_overall",
    "goals_overall",
    "assists_overall",
    "clean_sheets_overall",
    "conceded_overall",
    "yellow_cards_overall",
    "red_cards_overall",
    "attack_score_score_pct",
    "assist_score_score_pct",
    "conceded_score_score_pct",
    "foul_card_score_score_pct",
    "goalkeeper_score_score_pct",
    "appearance_score_score_pct",
    "rating_display_score_pct",
    "rating_display_score_sigmoid",
]

app = FastAPI(
    title="Football Player Performance API",
    version="1.0.0",
    description=(
        "Open-source base model API for football player performance scoring. "
        "The Pro model is available via the hosted API and Futrix Metrics Platform."
    ),
)


def _get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(db_path: Path, table_name: str) -> bool:
    if not db_path.exists():
        return False
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


def ensure_scores_db() -> None:
    if _table_exists(SCORES_DB_PATH, SCORES_TABLE):
        return

    if not BASE_DB_PATH.exists():
        raise RuntimeError(f"Base database not found: {BASE_DB_PATH}")

    df_base = load_base_dataframe(db_path=BASE_DB_PATH, table=BASE_TABLE)
    scored = build_scored_dataset(
        df_base,
        method="both",
        group_cols=("season", "position_bucket"),
        targets_group_cols=("season", "position_bucket"),
        rating_display_weights_by_pos=BASE_WEIGHTS_BY_POS,
        rating_display_drop_components=("passing_score", "defense_score", "aerial_score"),
        exclude_score_cols=("passing_score", "defense_score", "aerial_score"),
    )
    numeric_cols = scored.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        scored[numeric_cols] = scored[numeric_cols].round(2)

    SCORES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SCORES_DB_PATH) as conn:
        scored.to_sql(SCORES_TABLE, conn, if_exists="replace", index=False)


@app.on_event("startup")
def _startup() -> None:
    ensure_scores_db()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Football Player Performance API",
        "mode": "open-source base model",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/metadata",
        "players": "/players",
        "pro_api_docs": "https://footballperformanceapi.site/redoc",
        "pro_platform": "https://www.futrixmetrics.com/platform",
        "custom_solution": "https://www.futrixmetrics.com/customize.html",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "base_db_exists": BASE_DB_PATH.exists(),
        "scores_db_exists": SCORES_DB_PATH.exists(),
        "scores_table": SCORES_TABLE,
    }


@app.get("/metadata")
def metadata() -> Any:
    if not META_PATH.exists():
        raise HTTPException(status_code=404, detail="meta.json not found")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@app.get("/players")
def list_players(
    q: str | None = Query(default=None, description="Search by player name"),
    season: str | None = Query(default=None),
    league: str | None = Query(default=None),
    position_group: str | None = Query(default=None),
    club: str | None = Query(default=None, alias="current_club"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    ensure_scores_db()

    filters: list[str] = []
    params: list[Any] = []
    if q:
        filters.append("full_name LIKE ?")
        params.append(f"%{q}%")
    if season:
        filters.append("season = ?")
        params.append(season)
    if league:
        filters.append("league = ?")
        params.append(league)
    if position_group:
        filters.append("position_group = ?")
        params.append(position_group.upper())
    if club:
        filters.append("`current club` = ?")
        params.append(club)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    select_cols = ", ".join(f"`{col}`" if " " in col else col for col in PLAYER_COLUMNS)
    sql = (
        f"SELECT {select_cols} FROM {SCORES_TABLE} "
        f"{where_sql} "
        f"ORDER BY ({DEFAULT_ORDER_COL} IS NULL), {DEFAULT_ORDER_COL} DESC, minutes_played_overall DESC "
        "LIMIT ?"
    )
    params.append(limit)

    with _get_connection(SCORES_DB_PATH) as conn:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]

    return {"count": len(rows), "items": rows}


@app.get("/players/{player_id}")
def get_player(player_id: int) -> dict[str, Any]:
    ensure_scores_db()

    select_cols = ", ".join(f"`{col}`" if " " in col else col for col in PLAYER_COLUMNS)
    sql = (
        f"SELECT {select_cols} FROM {SCORES_TABLE} "
        "WHERE player_id = ? "
        f"ORDER BY ({DEFAULT_ORDER_COL} IS NULL), {DEFAULT_ORDER_COL} DESC, season DESC"
    )
    with _get_connection(SCORES_DB_PATH) as conn:
        rows = [dict(row) for row in conn.execute(sql, (player_id,)).fetchall()]

    if not rows:
        raise HTTPException(status_code=404, detail="player not found")
    return {"count": len(rows), "items": rows}
