# FutrixMetrics Style Matching

This module adds player-to-team and team-to-player style-fit endpoints to the FutrixMetrics open model API.

The open-source repository does not commit style-matching databases because they can contain third-party/vendor-derived data. Mount private databases on your own server through environment variables.

## Data Files

Required private data files:

```bash
database/style_matching/sample/team_player_match_results_sample.db
database/style_matching/sample/player_style_results_sample.db
database/style_matching/sample/team_style_results_sample.db
database/style_matching/sample/model_quality_sample.json
```

Production environment variables:

```bash
STYLE_MATCHING_DB_PATH=/srv/futrixmetrics/style_matching/team_player_match_results.db
STYLE_MODEL_QUALITY_PATH=/srv/futrixmetrics/style_matching/model_quality.json
STYLE_PLAYER_STYLE_DB_PATH=/srv/futrixmetrics/style_matching/player_style_results.db
STYLE_TEAM_STYLE_DB_PATH=/srv/futrixmetrics/style_matching/team_style_results.db
STYLE_API_DEFAULT_LIMIT=20
STYLE_API_MAX_LIMIT=50
STYLE_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5500,http://127.0.0.1:5500
```

## API Endpoints

```text
GET /style/health
GET /style/metadata
GET /style/players/search?q=&season=&position=&current_club=&limit=
GET /style/teams/search?q=&country=&season=&limit=
GET /style/players/{player_id}/team-matches?season=&realistic=true&limit=
GET /style/teams/{team_row_id}/player-matches?limit=
GET /style/current-club-audit?player_id=&q=&limit=
```

## Model Quality Filter

The matching pipeline keeps all model metrics for audit, but dimensions with cross-validated `R2 < 0.10` receive zero matching weight.

Current active dimensions:

```text
attacking_tempo
possession_dominance
high_pressing
vertical_threat
defensive_positioning
physicality
```

Excluded from match scoring:

```text
set_piece_reliance
wide_play
```

## Frontend Demo

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/demo/
```

For local frontend development, run the Vite app with the backend URL:

```bash
cd frontend/demo
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

If the API is hosted on a different origin, set `VITE_API_BASE` before building or running the Vite frontend.

## Data Policy

Do not commit full vendor-derived databases unless the data license explicitly allows redistribution. The intended public setup is:

- Open-source code and sample data in GitHub.
- Full private SQLite/Postgres data on your own server.
- Hosted API returns limited model outputs, not bulk database exports.
