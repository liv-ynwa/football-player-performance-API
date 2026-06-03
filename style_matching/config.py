from __future__ import annotations

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STYLE_MATCHING_DB_PATH = Path(
    os.getenv(
        "STYLE_MATCHING_DB_PATH",
        PROJECT_ROOT / "database" / "style_matching" / "sample" / "team_player_match_results_sample.db",
    )
)
STYLE_MODEL_QUALITY_PATH = Path(
    os.getenv(
        "STYLE_MODEL_QUALITY_PATH",
        PROJECT_ROOT / "database" / "style_matching" / "sample" / "model_quality_sample.json",
    )
)
STYLE_PLAYER_STYLE_DB_PATH = Path(
    os.getenv(
        "STYLE_PLAYER_STYLE_DB_PATH",
        PROJECT_ROOT / "database" / "style_matching" / "sample" / "player_style_results_sample.db",
    )
)
STYLE_TEAM_STYLE_DB_PATH = Path(
    os.getenv(
        "STYLE_TEAM_STYLE_DB_PATH",
        PROJECT_ROOT / "database" / "style_matching" / "sample" / "team_style_results_sample.db",
    )
)

STYLE_API_DEFAULT_LIMIT = int(os.getenv("STYLE_API_DEFAULT_LIMIT", "20"))
STYLE_API_MAX_LIMIT = int(os.getenv("STYLE_API_MAX_LIMIT", "50"))

