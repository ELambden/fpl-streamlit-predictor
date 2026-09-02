from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.modeling import build_model_summary
from fpl_decision_lab.paths import MODEL_JSON, PLAYERS_CSV


def main() -> None:
    with PLAYERS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = build_model_summary(rows, {"status": "summary-only"})
    MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODEL_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {MODEL_JSON}")
    print(f"Model rows: {summary['rows']}")


if __name__ == "__main__":
    main()
