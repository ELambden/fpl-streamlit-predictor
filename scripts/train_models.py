from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.modeling import write_model_summary
from fpl_decision_lab.paths import MODEL_JSON, PLAYERS_CSV


def main() -> None:
    with PLAYERS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = write_model_summary(MODEL_JSON, rows)
    print(f"Wrote {MODEL_JSON}")
    print(f"Model rows: {summary['rows']}")


if __name__ == "__main__":
    main()

