from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.data import fetch_current_fpl_data
from fpl_decision_lab.paths import RAW_DIR


def main() -> None:
    payload = fetch_current_fpl_data(RAW_DIR)
    print(f"Fetched {len(payload['bootstrap'].get('elements', []))} players")
    print(f"Fetched {len(payload.get('fixtures', []))} fixtures")


if __name__ == "__main__":
    main()

