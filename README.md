# Football Player Performance API

Open-source football player performance scoring pipeline based on the Futrix Metrics base model.

This repository is the free self-hosted base version. You can deploy it yourself and call it for free on your own infrastructure. If you need the Pro model, production API access, or custom commercial integrations, use:

- API docs: [footballperformanceapi.site/redoc](https://footballperformanceapi.site/redoc)
- Platform: [futrixmetrics.com/platform](https://www.futrixmetrics.com/platform)
- Custom solutions: [futrixmetrics.com/customize.html](https://www.futrixmetrics.com/customize.html)

## What is included

- Base feature loading and target generation: `base/features.py`, `base/target.py`
- Base score building pipeline: `score_building.py`
- Base model training script: `model/base_model.py`
- Minimal self-hosted API: `app.py`
- Public model metadata: `meta.json`
- Demo database tracked via Git LFS: `database/player_features_base.db`

## What is not included

- `.cbm` model artifacts are intentionally excluded from git
- `base_rating.db` is generated locally
- Pro model weights and Pro data are not part of this open-source repository

## Demo database download

The demo database is distributed through Git LFS:

- [Download player_features_base.db](https://github.com/liv-ynwa/football-player-performance-API/raw/main/database/player_features_base.db)

## Quick start

```bash
git lfs install
git clone https://github.com/liv-ynwa/football-player-performance-API.git
cd football-player-performance-API
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate base scores locally

```bash
python score_building.py
```

This creates:

- `database/base_rating.db`
- table: `base_scores`

## Run the free self-hosted API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

- `GET /health`
- `GET /metadata`
- `GET /players`
- `GET /players/{player_id}`
- Interactive docs: `GET /docs`

Example:

```bash
curl "http://127.0.0.1:8000/players?q=Messi&limit=5"
```

## Train the base model yourself

The repository exposes the base training pipeline. Model artifacts are not committed, so if you want local `.cbm` files you can train them yourself:

```bash
python model/base_model.py
```

Generated model metadata is also published at `meta.json`.


## Style matching API

This repository also includes a style-matching module for player-to-team and team-to-player fit analysis.

Style-matching databases are not committed because they can contain third-party/vendor-derived data. Provide private database paths through environment variables on your server:

```bash
STYLE_MATCHING_DB_PATH=/srv/futrixmetrics/style_matching/team_player_match_results.db
STYLE_MODEL_QUALITY_PATH=/srv/futrixmetrics/style_matching/model_quality.json
STYLE_PLAYER_STYLE_DB_PATH=/srv/futrixmetrics/style_matching/player_style_results.db
STYLE_TEAM_STYLE_DB_PATH=/srv/futrixmetrics/style_matching/team_style_results.db
```

Useful endpoints:

- `GET /style/health`
- `GET /style/metadata`
- `GET /style/players/search?q=Messi&limit=5`
- `GET /style/teams/search?q=Basel&limit=5`
- `GET /style/players/{player_id}/team-matches`
- `GET /style/teams/{team_row_id}/player-matches`
- `GET /style/current-club-audit`

The frontend demo is served at `GET /demo` when the API is running.

See `docs/style-matching.md` for the full setup notes.

## Notes

- This repository is focused on the open-source base model only.
- For the Pro model, managed API, or enterprise customization, use the official links above.
- If you want a different open-source license, replace the default MIT license before wider redistribution.
