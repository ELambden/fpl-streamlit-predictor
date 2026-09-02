import json
from pathlib import Path

from fpl_decision_lab.data import read_csv_records
from fpl_decision_lab.paths import MODEL_JSON, PLAYER_FORECASTS_CSV, PLAYER_HISTORY_CSV, PLAYERS_CSV


REQUIRED_COLUMNS = {
    "player_id",
    "web_name",
    "team",
    "position",
    "now_cost",
    "predicted_next_gw",
    "predicted_next3",
    "predicted_next5",
    "transfer_score",
    "risk_score",
    "season",
    "current_gameweek",
    "player_code",
    "blended_points_per_90",
    "blended_xgi_per_90",
}


def test_committed_player_snapshot_has_dashboard_contract() -> None:
    rows = read_csv_records(PLAYERS_CSV)
    assert rows
    assert REQUIRED_COLUMNS <= set(rows[0])


def test_history_and_forecast_outputs_exist() -> None:
    history = read_csv_records(PLAYER_HISTORY_CSV)
    forecasts = read_csv_records(PLAYER_FORECASTS_CSV)
    assert history
    assert forecasts
    assert {"player_id", "gw", "total_points", "minutes", "expected_goal_involvements"} <= set(history[0])
    assert {"player_id", "gw", "horizon_index", "fixture_difficulty", "forecast_points"} <= set(forecasts[0])
    assert max(int(row["horizon_index"]) for row in forecasts) <= 5


def test_model_summary_explains_random_forest_and_glossary() -> None:
    summary = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    assert "RandomForest" in summary["modelName"]
    assert summary["modelDiagnostics"]["status"] == "trained"
    assert "predicted_next5" in summary["metricGlossary"]


def test_docs_wrapper_references_streamlit_embed_and_assets() -> None:
    html = Path("docs/index.html").read_text(encoding="utf-8")
    assert "fpl-app-predictor-sklxegssnh6exvzw2av6vg.streamlit.app/?embed=true" in html
    assert "Portfolio Fit" not in html
    assert "What It Does" in html
    assert "predicted_next5" in html
    assert "styles.css" in html
    assert "app.js" in html
    assert "data/dashboard-data.json" in Path("docs/app.js").read_text(encoding="utf-8")
