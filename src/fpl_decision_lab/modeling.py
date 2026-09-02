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
    "ep_next",
    "blended_points_per_90",
    "blended_xgi_per_90",
    "fixture_ease",
    "blended_start_rate",
    "team_attack_prior",
    "team_defence_prior",
]


def explainable_projection_weights() -> dict[str, float]:
    return {
        "points_per_game": 0.27,
        "form": 0.20,
        "ep_next": 0.16,
        "blended_points_per_90": 0.17,
        "blended_xgi_per_90": 0.66,
        "fixture_ease": 0.25,
        "blended_start_rate": 0.18,
        "team_attack_prior": 0.42,
        "team_defence_prior": 0.30,
    }


def build_model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [as_float(row.get("predicted_next3")) for row in rows]
    baselines = [as_float(row.get("baseline_next3")) for row in rows]
    residuals = [p - b for p, b in zip(predictions, baselines, strict=False)]
    positions: dict[str, int] = {}
    prior_players = 0
    prior_teams = 0
    for row in rows:
        position = row.get("position", "UNK")
        positions[position] = positions.get(position, 0) + 1
        prior_players += int(as_float(row.get("prior_found")) > 0)
        prior_teams += int(as_float(row.get("prior_team_found")) > 0)
    return {
        "generatedAt": datetime.now(UTC).isoformat(),
        "modelName": "Current-season projection with 2025-26 player and team priors",
        "target": "next three gameweek expected points",
        "features": FEATURES,
        "weights": explainable_projection_weights(),
        "rows": len(rows),
        "positions": positions,
        "priorCoverage": {
            "playersWith2025_26Prior": prior_players,
            "teamsWith2025_26Prior": prior_teams,
            "playerPriorCoveragePct": round(100 * prior_players / len(rows), 1) if rows else 0.0,
            "teamPriorCoveragePct": round(100 * prior_teams / len(rows), 1) if rows else 0.0,
        },
        "diagnostics": {
            "meanPrediction": round(mean(predictions), 3) if predictions else 0.0,
            "meanBaseline": round(mean(baselines), 3) if baselines else 0.0,
            "meanLift": round(mean(residuals), 3) if residuals else 0.0,
            "backtestStatus": "live-current-season-with-prior-stabilisers",
            "note": "2026-27 live FPL data is authoritative; 2025-26 priors are used only where stable player/team codes match.",
        },
    }


def write_model_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_model_summary(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
