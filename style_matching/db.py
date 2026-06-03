from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(db_path: Path, table_name: str) -> bool:
    if not db_path.exists():
        return False
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def clamp_limit(limit: int, max_limit: int) -> int:
    return max(1, min(int(limit), int(max_limit)))

