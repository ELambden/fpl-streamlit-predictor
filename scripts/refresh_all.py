from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


def main() -> None:
    run("fetch_fpl_data.py")
    run("build_features.py")
    run("export_static_site_data.py")


if __name__ == "__main__":
    main()

