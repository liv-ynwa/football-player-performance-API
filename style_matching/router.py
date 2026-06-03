from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from .config import (
    STYLE_API_DEFAULT_LIMIT,
    STYLE_API_MAX_LIMIT,
    STYLE_MATCHING_DB_PATH,
    STYLE_MODEL_QUALITY_PATH,
    STYLE_PLAYER_STYLE_DB_PATH,
    STYLE_TEAM_STYLE_DB_PATH,
)
from .schemas import CountResponse, HealthResponse
from . import service


router = APIRouter(prefix="/style", tags=["style matching"])


@router.get("/health", response_model=HealthResponse)
def style_health() -> dict[str, Any]:
    return {
        "status": "ok" if STYLE_MATCHING_DB_PATH.exists() else "missing_data",
        "matching_db_exists": STYLE_MATCHING_DB_PATH.exists(),
        "model_quality_exists": STYLE_MODEL_QUALITY_PATH.exists(),
        "sample_mode": "sample" in str(STYLE_MATCHING_DB_PATH),
    }


@router.get("/metadata")
def style_metadata() -> dict[str, Any]:
    return service.load_model_quality(STYLE_MODEL_QUALITY_PATH, STYLE_MATCHING_DB_PATH)


@router.get("/players/search", response_model=CountResponse)
def search_players(
    q: str | None = Query(default=None, description="Search by player name"),
    season: str | None = Query(default=None),
    position: str | None = Query(default=None),
    current_club: str | None = Query(default=None),
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.search_players(STYLE_MATCHING_DB_PATH, q, season, position, current_club, limit, STYLE_API_MAX_LIMIT)


@router.get("/teams/search", response_model=CountResponse)
def search_teams(
    q: str | None = Query(default=None, description="Search by team name"),
    country: str | None = Query(default=None),
    season: str | None = Query(default=None),
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.search_teams(STYLE_MATCHING_DB_PATH, q, country, season, limit, STYLE_API_MAX_LIMIT)


@router.get("/players/{player_id}/team-matches", response_model=CountResponse)
def get_player_team_matches(
    player_id: int,
    season: str | None = Query(default=None),
    realistic: bool = Query(default=True),
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.player_team_matches(STYLE_MATCHING_DB_PATH, player_id, season, realistic, limit, STYLE_API_MAX_LIMIT)


@router.get("/teams/{team_row_id}/player-matches", response_model=CountResponse)
def get_team_player_matches(
    team_row_id: int,
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.team_player_matches(STYLE_MATCHING_DB_PATH, team_row_id, limit, STYLE_API_MAX_LIMIT)


@router.get("/current-club-audit", response_model=CountResponse)
def get_current_club_audit(
    player_id: int | None = Query(default=None),
    q: str | None = Query(default=None, description="Search by player name"),
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.current_club_audit(STYLE_MATCHING_DB_PATH, player_id, q, limit, STYLE_API_MAX_LIMIT)


@router.get("/player-styles/search", response_model=CountResponse)
def search_player_styles(
    q: str | None = Query(default=None, description="Search by player name"),
    role: str | None = Query(default=None, description="FW, MF, DF, GK"),
    season: str | None = Query(default=None),
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.search_player_styles(
        STYLE_PLAYER_STYLE_DB_PATH, q, role, season, limit, STYLE_API_MAX_LIMIT
    )


@router.get("/players/{player_id}/style-profile", response_model=CountResponse)
def get_player_style_profile(
    player_id: int,
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.player_style_profile(
        STYLE_PLAYER_STYLE_DB_PATH, player_id, limit, STYLE_API_MAX_LIMIT
    )


@router.get("/player-clusters", response_model=CountResponse)
def get_player_clusters(
    role: str | None = Query(default=None, description="FW, MF, DF, GK"),
) -> dict[str, Any]:
    return service.player_clusters(STYLE_PLAYER_STYLE_DB_PATH, role)


@router.get("/player-roles", response_model=CountResponse)
def get_player_roles() -> dict[str, Any]:
    return service.player_roles(STYLE_PLAYER_STYLE_DB_PATH)


@router.get("/team-styles/search", response_model=CountResponse)
def search_team_styles_endpoint(
    q: str | None = Query(default=None, description="Search by team name"),
    country: str | None = Query(default=None),
    season: str | None = Query(default=None),
    cluster_id: int | None = Query(default=None),
    limit: int = Query(default=STYLE_API_DEFAULT_LIMIT, ge=1, le=STYLE_API_MAX_LIMIT),
) -> dict[str, Any]:
    return service.search_team_styles(
        STYLE_TEAM_STYLE_DB_PATH, q, country, season, cluster_id, limit, STYLE_API_MAX_LIMIT
    )


@router.get("/teams/{source_team_rowid}/style-profile", response_model=CountResponse)
def get_team_style_profile(source_team_rowid: int) -> dict[str, Any]:
    return service.team_style_profile(STYLE_TEAM_STYLE_DB_PATH, source_team_rowid)


@router.get("/team-clusters", response_model=CountResponse)
def get_team_clusters() -> dict[str, Any]:
    return service.team_clusters(STYLE_TEAM_STYLE_DB_PATH)


@router.get("/dimensions", response_model=CountResponse)
def get_style_dimensions() -> dict[str, Any]:
    return service.style_dimensions(STYLE_TEAM_STYLE_DB_PATH)


@router.get("/stats")
def get_style_stats() -> dict[str, Any]:
    return service.style_stats(
        STYLE_MATCHING_DB_PATH, STYLE_PLAYER_STYLE_DB_PATH, STYLE_TEAM_STYLE_DB_PATH
    )

