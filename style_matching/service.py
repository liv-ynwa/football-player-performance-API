from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from fastapi import HTTPException

from .db import clamp_limit, connect, rows_to_dicts, table_exists


PLAYER_SEARCH_COLUMNS = [
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
    "match_role_group",
    "predicted_attacking_tempo",
    "predicted_possession_dominance",
    "predicted_high_pressing",
    "predicted_vertical_threat",
    "predicted_defensive_positioning",
    "predicted_physicality",
]

TEAM_SEARCH_COLUMNS = [
    "team_row_id",
    "team_name",
    "common_name",
    "country",
    "season",
    "style_cluster_id",
    "style_cluster_label",
    "attacking_tempo",
    "possession_dominance",
    "high_pressing",
    "vertical_threat",
    "defensive_positioning",
    "physicality",
    "style_data_quality",
]

MATCH_COLUMNS = [
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

REALISTIC_MATCH_EXTRA_COLUMNS = [
    "best_fit_dimensions",
    "weak_fit_dimensions",
    "dimension_gap_attacking_tempo",
    "dimension_gap_possession_dominance",
    "dimension_gap_high_pressing",
    "dimension_gap_vertical_threat",
    "dimension_gap_defensive_positioning",
    "dimension_gap_physicality",
]


def quoted_columns(columns: list[str]) -> str:
    return ", ".join(f"`{column}`" for column in columns)


def require_table(db_path: Path, table_name: str) -> None:
    if not table_exists(db_path, table_name):
        raise HTTPException(status_code=503, detail=f"Required table is unavailable: {table_name}")


def load_model_quality(model_quality_path: Path, matching_db_path: Path) -> dict[str, Any]:
    if model_quality_path.exists():
        return json.loads(model_quality_path.read_text(encoding="utf-8"))

    require_table(matching_db_path, "ml_model_quality")
    with connect(matching_db_path) as conn:
        metrics = rows_to_dicts(conn.execute("SELECT * FROM ml_model_quality").fetchall())
        summary = rows_to_dicts(conn.execute("SELECT * FROM ml_experiment_summary").fetchall())
    return {
        "scope": "ml_team_player_matching",
        "source": "sqlite",
        "summary": summary[0] if summary else {},
        "metrics": metrics,
    }


def search_players(
    db_path: Path,
    q: str | None,
    season: str | None,
    position: str | None,
    club: str | None,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    require_table(db_path, "ml_player_predicted_team_style")
    filters = []
    params: list[Any] = []
    if q:
        filters.append("full_name LIKE ?")
        params.append(f"%{q}%")
    if season:
        filters.append("season = ?")
        params.append(season)
    if position:
        filters.append("position LIKE ?")
        params.append(f"%{position}%")
    if club:
        filters.append("`Current Club` = ?")
        params.append(club)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    limited = clamp_limit(limit, max_limit)
    sql = (
        f"SELECT {quoted_columns(PLAYER_SEARCH_COLUMNS)} "
        "FROM ml_player_predicted_team_style "
        f"{where_sql} "
        "ORDER BY minutes_played_overall DESC, full_name ASC "
        "LIMIT ?"
    )
    params.append(limited)
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, params).fetchall())
    return {"count": len(items), "items": items}


def search_teams(
    db_path: Path,
    q: str | None,
    country: str | None,
    season: str | None,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    require_table(db_path, "ml_team_profile_predicted_style")
    filters = []
    params: list[Any] = []
    if q:
        filters.append("(team_name LIKE ? OR common_name LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    if country:
        filters.append("country = ?")
        params.append(country)
    if season:
        filters.append("season = ?")
        params.append(season)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    limited = clamp_limit(limit, max_limit)
    sql = (
        f"SELECT {quoted_columns(TEAM_SEARCH_COLUMNS)} "
        "FROM ml_team_profile_predicted_style "
        f"{where_sql} "
        "ORDER BY style_data_quality DESC, team_name ASC "
        "LIMIT ?"
    )
    params.append(limited)
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, params).fetchall())
    return {"count": len(items), "items": items}


def player_team_matches(
    db_path: Path,
    player_id: int,
    season: str | None,
    realistic: bool,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    table = "player_realistic_team_matches" if realistic else "player_top_team_matches"
    require_table(db_path, table)
    columns = MATCH_COLUMNS + (REALISTIC_MATCH_EXTRA_COLUMNS if realistic else [])
    filters = ["player_id = ?"]
    params: list[Any] = [player_id]
    if season:
        filters.append("player_season = ?")
        params.append(season)
    sql = (
        f"SELECT {quoted_columns(columns)} "
        f"FROM {table} "
        f"WHERE {' AND '.join(filters)} "
        "ORDER BY match_percentage DESC, style_distance ASC "
        "LIMIT ?"
    )
    params.append(clamp_limit(limit, max_limit))
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, params).fetchall())
    return {"count": len(items), "items": items}


def team_player_matches(
    db_path: Path,
    team_row_id: int,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    require_table(db_path, "team_top_player_matches")
    sql = (
        f"SELECT {quoted_columns(MATCH_COLUMNS)} "
        "FROM team_top_player_matches "
        "WHERE team_row_id = ? "
        "ORDER BY match_percentage DESC, style_distance ASC "
        "LIMIT ?"
    )
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, (team_row_id, clamp_limit(limit, max_limit))).fetchall())
    return {"count": len(items), "items": items}


def current_club_audit(
    db_path: Path,
    player_id: int | None,
    q: str | None,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    require_table(db_path, "current_club_match_audit")
    filters = []
    params: list[Any] = []
    if player_id is not None:
        filters.append("player_id = ?")
        params.append(player_id)
    if q:
        filters.append("full_name LIKE ?")
        params.append(f"%{q}%")
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = (
        "SELECT * FROM current_club_match_audit "
        f"{where_sql} "
        "ORDER BY match_percentage ASC, style_distance DESC "
        "LIMIT ?"
    )
    params.append(clamp_limit(limit, max_limit))
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, params).fetchall())
    return {"count": len(items), "items": items}


# ── Player style profile endpoints ──────────────────────────────────


def search_player_styles(
    db_path: Path,
    q: str | None,
    role: str | None,
    season: str | None,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    require_table(db_path, "player_style_cards")
    filters: list[str] = []
    params: list[Any] = []
    if q:
        filters.append("full_name LIKE ?")
        params.append(f"%{q}%")
    if role:
        filters.append("role = ?")
        params.append(role.upper())
    if season:
        filters.append("season = ?")
        params.append(season)
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = (
        "SELECT * FROM player_style_cards "
        f"{where_sql} "
        "ORDER BY minutes_played_overall DESC, full_name ASC "
        "LIMIT ?"
    )
    params.append(clamp_limit(limit, max_limit))
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, params).fetchall())
    return {"count": len(items), "items": items}


def player_style_profile(
    db_path: Path,
    player_id: int,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    require_table(db_path, "player_style_cards")
    sql = (
        "SELECT * FROM player_style_cards "
        "WHERE player_id = ? "
        "ORDER BY minutes_played_overall DESC "
        "LIMIT ?"
    )
    with connect(db_path) as conn:
        items = rows_to_dicts(
            conn.execute(sql, (player_id, clamp_limit(limit, max_limit))).fetchall()
        )
    return {"count": len(items), "items": items}


def player_clusters(
    db_path: Path,
    role: str | None,
) -> dict[str, Any]:
    require_table(db_path, "player_cluster_summary")
    filters: list[str] = []
    params: list[Any] = []
    if role:
        filters.append("role = ?")
        params.append(role.upper())
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"SELECT * FROM player_cluster_summary {where_sql} ORDER BY role, cluster_id"
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, params).fetchall())
    return {"count": len(items), "items": items}


def player_roles(db_path: Path) -> dict[str, Any]:
    require_table(db_path, "player_role_summary")
    sql = (
        "SELECT role, role_name, player_rows, best_k, silhouette, "
        "explained_variance_ratio, pca_components, selected_feature_count, "
        "hdbscan_noise_rate "
        "FROM player_role_summary ORDER BY role"
    )
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql).fetchall())
    return {"count": len(items), "items": items}


# ── Team style profile endpoints ────────────────────────────────────


def search_team_styles(
    db_path: Path,
    q: str | None,
    country: str | None,
    season: str | None,
    cluster_id: int | None,
    limit: int,
    max_limit: int,
) -> dict[str, Any]:
    require_table(db_path, "team_style_clusters")
    filters: list[str] = []
    params: list[Any] = []
    if q:
        filters.append("(team_name LIKE ? OR common_name LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    if country:
        filters.append("country = ?")
        params.append(country)
    if season:
        filters.append("season = ?")
        params.append(season)
    if cluster_id is not None:
        filters.append("style_cluster_id = ?")
        params.append(cluster_id)
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = (
        "SELECT * FROM team_style_clusters "
        f"{where_sql} "
        "ORDER BY style_data_quality DESC, team_name ASC "
        "LIMIT ?"
    )
    params.append(clamp_limit(limit, max_limit))
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, params).fetchall())
    return {"count": len(items), "items": items}


def team_style_profile(
    db_path: Path,
    source_team_rowid: int,
) -> dict[str, Any]:
    require_table(db_path, "team_style_clusters")
    sql = "SELECT * FROM team_style_clusters WHERE source_team_rowid = ?"
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql, (source_team_rowid,)).fetchall())
    if not items:
        raise HTTPException(status_code=404, detail="team not found in style data")
    return {"count": len(items), "items": items}


def team_clusters(db_path: Path) -> dict[str, Any]:
    require_table(db_path, "team_style_cluster_summary")
    sql = "SELECT * FROM team_style_cluster_summary ORDER BY style_cluster_id"
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql).fetchall())
    return {"count": len(items), "items": items}


def style_dimensions(db_path: Path) -> dict[str, Any]:
    require_table(db_path, "team_style_dimension_feature_weights")
    sql = (
        "SELECT * FROM team_style_dimension_feature_weights "
        "ORDER BY dimension, global_normalized_weight DESC"
    )
    with connect(db_path) as conn:
        items = rows_to_dicts(conn.execute(sql).fetchall())
    return {"count": len(items), "items": items}


def style_stats(
    matching_db_path: Path,
    player_style_db_path: Path,
    team_style_db_path: Path,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    def _count(db_path: Path, table: str) -> int | None:
        if not table_exists(db_path, table):
            return None
        with connect(db_path) as conn:
            row = conn.execute(f"SELECT COUNT(*) AS c FROM `{table}`").fetchone()
        return dict(row)["c"] if row else None

    stats["player_style_profiles"] = _count(player_style_db_path, "player_style_cards")
    stats["team_style_profiles"] = _count(team_style_db_path, "team_style_clusters")
    stats["predicted_players"] = _count(matching_db_path, "ml_player_predicted_team_style")
    stats["predicted_teams"] = _count(matching_db_path, "ml_team_profile_predicted_style")
    stats["realistic_matches"] = _count(matching_db_path, "player_realistic_team_matches")
    stats["top_player_matches"] = _count(matching_db_path, "player_top_team_matches")
    stats["team_player_matches"] = _count(matching_db_path, "team_top_player_matches")
    stats["club_audit_entries"] = _count(matching_db_path, "current_club_match_audit")
    return stats

