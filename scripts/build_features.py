from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.data import load_latest_raw, normalize_bootstrap, write_csv_records
from fpl_decision_lab.features import add_projection_features
from fpl_decision_lab.modeling import write_model_summary
from fpl_decision_lab.paths import MODEL_JSON, PLAYERS_CSV, RAW_DIR


def main() -> None:
    raw_path = RAW_DIR / "latest.json"
    if not raw_path.exists():
        raise FileNotFoundError("No raw FPL snapshot found. Run scripts/fetch_fpl_data.py first.")
    rows = add_projection_features(normalize_bootstrap(load_latest_raw(raw_path)))
    write_csv_records(PLAYERS_CSV, rows)
    write_model_summary(MODEL_JSON, rows)
    print(f"Wrote {PLAYERS_CSV}")
    print(f"Wrote {MODEL_JSON}")


if __name__ == "__main__":
    main()

