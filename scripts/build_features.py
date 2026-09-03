from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.data import build_player_history, load_latest_raw, normalize_bootstrap, write_csv_records
from fpl_decision_lab.features import add_history_summary_features, add_projection_features
from fpl_decision_lab.modeling import apply_forecasts_to_players, build_fixture_forecasts, prepare_training_frame, train_models, write_model_summary
from fpl_decision_lab.paths import MODEL_JSON, PLAYER_FORECASTS_CSV, PLAYER_HISTORY_CSV, PLAYERS_CSV, RAW_DIR


def main() -> None:
    raw_path = RAW_DIR / "latest.json"
    if not raw_path.exists():
        raise FileNotFoundError("No raw FPL snapshot found. Run scripts/fetch_fpl_data.py first.")
    raw_payload = load_latest_raw(raw_path)
    starter_rows = add_projection_features(normalize_bootstrap(raw_payload))
    history_rows = build_player_history(raw_payload)
    starter_rows = add_history_summary_features(starter_rows, history_rows)
    training = prepare_training_frame(
        raw_payload.get("prior_gameweeks_2025_26", []),
        history_rows,
        raw_payload.get("prior_fixtures_2025_26", []),
        raw_payload.get("fixtures", []),
    )
    model_info = train_models(training)
    forecasts = build_fixture_forecasts(starter_rows, raw_payload.get("fixtures", []), model_info, horizon=5)
    rows = apply_forecasts_to_players(starter_rows, forecasts)
    write_csv_records(PLAYERS_CSV, rows)
    if history_rows:
        write_csv_records(PLAYER_HISTORY_CSV, history_rows)
    if forecasts:
        write_csv_records(PLAYER_FORECASTS_CSV, forecasts)
    write_model_summary(MODEL_JSON, rows, model_info)
    print(f"Wrote {PLAYERS_CSV}")
    print(f"Wrote {PLAYER_HISTORY_CSV}")
    print(f"Wrote {PLAYER_FORECASTS_CSV}")
    print(f"Wrote {MODEL_JSON}")
    print(f"Training rows: {model_info.get('training_rows', 0)}")
    print(f"Model status: {model_info.get('status')}")


if __name__ == "__main__":
    main()
