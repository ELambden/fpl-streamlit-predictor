from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.paths import PLAYERS_CSV, STATIC_JSON


def main() -> None:
    with PLAYERS_CSV.open(newline="", encoding="utf-8") as handle:
        players = list(csv.DictReader(handle))
    payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "source": "Official Fantasy Premier League public API plus committed sample snapshot",
        "scope": {
            "players": len(players),
            "positions": sorted({player["position"] for player in players}),
            "teams": sorted({player["team"] for player in players}),
        },
        "streamlitAppUrl": "https://fpl-decision-lab.streamlit.app/?embed=true",
        "players": players,
    }
    STATIC_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATIC_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {STATIC_JSON}")


if __name__ == "__main__":
    main()

