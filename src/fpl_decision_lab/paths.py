from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"

PLAYERS_CSV = PROCESSED_DIR / "player_features.csv"
PLAYER_HISTORY_CSV = PROCESSED_DIR / "player_gameweek_history.csv"
PLAYER_FORECASTS_CSV = PROCESSED_DIR / "player_fixture_forecasts.csv"
MODEL_JSON = PROCESSED_DIR / "model_summary.json"
STATIC_JSON = DOCS_DATA_DIR / "dashboard-data.json"
