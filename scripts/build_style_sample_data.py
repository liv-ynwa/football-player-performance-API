#!/usr/bin/env python3
"""Build small SQLite sample databases for the public style-matching demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd


def copy_table_subset(source: sqlite3.Connection, target: sqlite3.Connection, table: str, where_sql: str = "", params: tuple = ()) -> None:
    frame = pd.read_sql_query(f"SELECT * FROM {table} {where_sql}", source, params=params)
    frame.to_sql(table, target, if_exists="replace", index=False)


def build_matching_sample(source_db: Path, quality_json: Path, output_db: Path, output_quality: Path) -> None:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()

    with sqlite3.connect(source_db) as source, sqlite3.connect(output_db) as target:
        players = pd.read_sql_query(
            """
            SELECT DISTINCT player_id
            FROM player_realistic_team_matches
            ORDER BY match_percentage DESC, player_id
            LIMIT 40
            """,
            source,
        )
        player_ids = tuple(int(value) for value in players["player_id"].tolist())
        if not player_ids:
            raise RuntimeError("No player ids found for sample data.")

        placeholders = ",".join("?" for _ in player_ids)
        realistic = pd.read_sql_query(
            f"SELECT * FROM player_realistic_team_matches WHERE player_id IN ({placeholders})",
            source,
            params=player_ids,
        )
        team_ids = tuple(sorted({int(value) for value in realistic["team_row_id"].dropna().unique().tolist()}))
        team_placeholders = ",".join("?" for _ in team_ids)

        realistic.to_sql("player_realistic_team_matches", target, if_exists="replace", index=False)
        copy_table_subset(source, target, "player_top_team_matches", f"WHERE player_id IN ({placeholders})", player_ids)
        copy_table_subset(source, target, "team_top_player_matches", f"WHERE team_row_id IN ({team_placeholders})", team_ids)
        copy_table_subset(source, target, "current_club_match_audit", f"WHERE player_id IN ({placeholders})", player_ids)
        copy_table_subset(source, target, "ml_player_predicted_team_style", f"WHERE player_id IN ({placeholders})", player_ids)
        copy_table_subset(source, target, "ml_team_profile_predicted_style", f"WHERE team_row_id IN ({team_placeholders})", team_ids)
        copy_table_subset(source, target, "ml_model_quality")
        copy_table_subset(source, target, "ml_experiment_summary")

    quality = json.loads(quality_json.read_text(encoding="utf-8"))
    quality["sample_dataset"] = True
    quality["sample_note"] = "Small subset for local demo use. Full production data is not redistributed."
    output_quality.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_simple_sample(source_db: Path, output_db: Path, tables: list[str], limit: int) -> None:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    with sqlite3.connect(source_db) as source, sqlite3.connect(output_db) as target:
        for table in tables:
            frame = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit}", source)
            frame.to_sql(table, target, if_exists="replace", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build public sample databases for style matching.")
    parser.add_argument("--matching-db", required=True, type=Path)
    parser.add_argument("--model-quality", required=True, type=Path)
    parser.add_argument("--player-style-db", required=True, type=Path)
    parser.add_argument("--team-style-db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_matching_sample(
        args.matching_db,
        args.model_quality,
        args.output_dir / "team_player_match_results_sample.db",
        args.output_dir / "model_quality_sample.json",
    )
    build_simple_sample(
        args.player_style_db,
        args.output_dir / "player_style_results_sample.db",
        ["player_style_cards", "player_cluster_summary", "player_role_summary", "player_feature_audit"],
        200,
    )
    build_simple_sample(
        args.team_style_db,
        args.output_dir / "team_style_results_sample.db",
        [
            "team_style_vectors",
            "team_style_clusters",
            "team_style_cluster_summary",
            "team_style_data_quality_issues",
            "sportmonks_merge_audit",
            "team_style_feature_reliability",
            "team_style_dimension_feature_weights",
            "team_style_build_settings",
        ],
        200,
    )


if __name__ == "__main__":
    main()

