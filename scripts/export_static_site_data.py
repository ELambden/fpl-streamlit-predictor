from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.modeling import METRIC_GLOSSARY
from fpl_decision_lab.paths import MODEL_JSON, PLAYER_FORECASTS_CSV, PLAYER_HISTORY_CSV, PLAYERS_CSV, STATIC_JSON

STREAMLIT_APP_URL = "https://fpl-app-predictor-sklxegssnh6exvzw2av6vg.streamlit.app/?embed=true"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    players = read_rows(PLAYERS_CSV)
    history = read_rows(PLAYER_HISTORY_CSV)
    forecasts = read_rows(PLAYER_FORECASTS_CSV)
    model = json.loads(MODEL_JSON.read_text(encoding="utf-8")) if MODEL_JSON.exists() else {}
    payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "source": "Official Fantasy Premier League public API plus 2025-26 prior-season features",
        "scope": {
            "players": len(players),
            "historyRows": len(history),
            "forecastRows": len(forecasts),
            "positions": sorted({player["position"] for player in players}),
            "teams": sorted({player["team"] for player in players}),
            "currentGameweek": players[0].get("current_gameweek") if players else None,
        },
        "streamlitAppUrl": STREAMLIT_APP_URL,
        "metricGlossary": METRIC_GLOSSARY,
        "modelDiagnostics": model.get("modelDiagnostics", {}),
        "priorCoverage": model.get("priorCoverage", {}),
        "players": players,
    }
    STATIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATIC_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {STATIC_JSON}")


if __name__ == "__main__":
    main()
