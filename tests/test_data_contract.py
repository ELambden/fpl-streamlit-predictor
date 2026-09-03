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
    "minutes_per_game",
    "defcons_per_90",
    "defcon_success_pct",
    "bps_per_game",
    "bonus_per_game",
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
    assert {"player_id", "gw", "total_points", "minutes", "expected_goal_involvements", "bonus", "bps", "defensive_contribution"} <= set(history[0])
    assert {"player_id", "gw", "horizon_index", "fixture_difficulty", "forecast_points"} <= set(forecasts[0])
    assert max(int(row["horizon_index"]) for row in forecasts) <= 5


def test_model_summary_explains_random_forest_and_glossary() -> None:
    summary = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    assert "RandomForest" in summary["modelName"]
    assert summary["modelDiagnostics"]["status"] == "trained"
    assert "predicted_next5" in summary["metricGlossary"]


def test_docs_site_has_fast_app_home_and_explainer_pages() -> None:
    index = Path("docs/index.html").read_text(encoding="utf-8")
    what_it_does = Path("docs/what-it-does.html").read_text(encoding="utf-8")
    glossary = Path("docs/metric-glossary.html").read_text(encoding="utf-8")
    random_forest = Path("docs/random-forest.html").read_text(encoding="utf-8")

    assert "fpl-app-predictor-sklxegssnh6exvzw2av6vg.streamlit.app/?embed=true" in index
    assert "Import a public FPL team" not in index
    assert "predicted_next5" not in index
    assert "Portfolio Fit" not in index + what_it_does + glossary + random_forest
    assert "What It Does" in what_it_does
    assert "predicted_next5" in glossary
    assert "Why a Random Forest Helps" in random_forest
    assert "styles.css" in index
    assert "app.js" in index
    assert "data/dashboard-data.json" in Path("docs/app.js").read_text(encoding="utf-8")


def test_streamlit_app_exposes_decision_lab_controls() -> None:
    app = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "Only include players" in app
    assert "Remove players" in app
    assert "Remove clubs" in app
    assert "Minimum Defcons/90" in app
    assert "Defcon Success %" in app
    assert "BPS/game" in app
    assert "Rank table by" in app
    assert "Players shown" in app
    assert 'on_select="rerun"' in app
    assert "Remove selected players" in app
    assert "Primary y-axis" in app
    assert "Secondary y-axis" in app
    assert 'title_text="Gameweek"' in app
    assert "st.scatter_chart" not in app
