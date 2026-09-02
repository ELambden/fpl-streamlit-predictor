from pathlib import Path

from fpl_decision_lab.data import read_csv_records
from fpl_decision_lab.paths import PLAYERS_CSV


REQUIRED_COLUMNS = {
    "player_id",
    "web_name",
    "team",
    "position",
    "now_cost",
    "predicted_next_gw",
    "predicted_next3",
    "transfer_score",
    "season",
    "current_gameweek",
    "player_code",
    "prior_found",
    "prior_team_found",
    "blended_points_per_90",
    "blended_xgi_per_90",
}


def test_committed_player_snapshot_has_dashboard_contract() -> None:
    rows = read_csv_records(PLAYERS_CSV)
    assert rows
    assert REQUIRED_COLUMNS <= set(rows[0])


def test_docs_wrapper_references_streamlit_embed_and_assets() -> None:
    html = Path("docs/index.html").read_text(encoding="utf-8")
    assert "fpl-app-predictor-sklxegssnh6exvzw2av6vg.streamlit.app/?embed=true" in html
    assert "styles.css" in html
    assert "app.js" in html
    assert "data/dashboard-data.json" in Path("docs/app.js").read_text(encoding="utf-8")

