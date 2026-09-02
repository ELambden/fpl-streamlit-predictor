from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .features import as_float

FEATURES = [
    "points_per_game",
    "form",
    "points_per_90",
    "xgi_per_90",
    "fixture_ease",
    "start_rate",
    "value_points_per_million",
]


def explainable_projection_weights() -> dict[str, float]:
    return {
        "points_per_game": 0.34,
        "form": 0.26,
        "points_per_90": 0.16,
        "xgi_per_90": 0.58,
        "fixture_ease": 0.22,
        "start_rate": 0.18,
        "value_points_per_million": 0.04,
    }


def build_model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [as_float(row.get("predicted_next3")) for row in rows]
    baselines = [as_float(row.get("baseline_next3")) for row in rows]
    residuals = [p - b for p, b in zip(predictions, baselines, strict=False)]
    positions: dict[str, int] = {}
    for row in rows:
        positions[row.get("position", "UNK")] = positions.get(row.get("position", "UNK"), 0) + 1
    return {
        "generatedAt": datetime.now(UTC).isoformat(),
        "modelName": "Interpretable fixture-adjusted projection blend",
        "target": "next three gameweek expected points",
        "features": FEATURES,
        "weights": explainable_projection_weights(),
        "rows": len(rows),
        "positions": positions,
        "diagnostics": {
            "meanPrediction": round(mean(predictions), 3) if predictions else 0.0,
            "meanBaseline": round(mean(baselines), 3) if baselines else 0.0,
            "meanLift": round(mean(residuals), 3) if residuals else 0.0,
            "backtestStatus": "sample-snapshot-placeholder",
            "note": "Use scripts/refresh_all.py with live API data before treating diagnostics as current-season evidence.",
        },
    }


def write_model_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_model_summary(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

