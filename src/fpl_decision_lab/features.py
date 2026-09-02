from __future__ import annotations

from typing import Any


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    return int(round(as_float(value, float(default))))


def add_projection_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add transparent, model-ready features and starter projections.

    The formulas use only aggregate information available before a future gameweek:
    cumulative production, rolling form-like FPL fields, price, ownership, and upcoming
    fixture difficulty. They are deliberately simple so the app can explain them.
    """

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        minutes = max(as_float(item.get("minutes")), 1.0)
        ninety = minutes / 90.0
        total_points = as_float(item.get("total_points"))
        cost = max(as_float(item.get("now_cost")), 1.0)
        form = as_float(item.get("form"))
        ppg = as_float(item.get("points_per_game"))
        xgi = as_float(item.get("expected_goal_involvements"))
        fixture = as_float(item.get("fixture_difficulty_next3"), 3.0)
        ownership = as_float(item.get("selected_by_percent"))
        starts = as_float(item.get("starts"))

        item["points_per_90"] = round(total_points / ninety, 3)
        item["xgi_per_90"] = round(xgi / ninety, 3)
        item["start_rate"] = round(starts / max(minutes / 90.0, starts, 1.0), 3)
        item["value_points_per_million"] = round(total_points / (cost / 10.0), 3)
        item["fixture_ease"] = round(6.0 - fixture, 3)
        item["differential_score"] = round(max(0.0, 25.0 - ownership), 3)

        predicted_next_gw = (
            0.34 * ppg
            + 0.26 * form
            + 0.16 * item["points_per_90"]
            + 0.58 * item["xgi_per_90"]
            + 0.22 * item["fixture_ease"]
            + 0.18 * item["start_rate"]
        )
        position = item.get("position")
        if position == "GKP":
            predicted_next_gw += 0.25 * item["fixture_ease"]
        elif position == "DEF":
            predicted_next_gw += 0.18 * item["fixture_ease"]
        elif position == "FWD":
            predicted_next_gw += 0.18 * item["xgi_per_90"]

        item["predicted_next_gw"] = round(max(0.0, predicted_next_gw), 3)
        item["predicted_next3"] = round(item["predicted_next_gw"] * max(as_float(item.get("opponent_count_next3"), 3.0), 1.0), 3)
        item["baseline_next3"] = round(max(ppg, form, 0.0) * 3.0, 3)
        item["model_residual"] = round(item["predicted_next3"] - item["baseline_next3"], 3)
        item["optimizer_value"] = round(item["predicted_next3"] / (cost / 10.0), 3)
        item["risk_score"] = round(max(0.0, 1.0 - min(1.0, minutes / 900.0)) + max(0.0, 0.65 - item["start_rate"]), 3)
        item["transfer_score"] = round(
            item["predicted_next3"]
            + 0.35 * item["fixture_ease"]
            + 0.035 * item["differential_score"]
            - 0.85 * item["risk_score"],
            3,
        )
        result.append(item)
    return result


def numeric_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for column in columns:
            if column in item:
                item[column] = as_float(item[column])
        converted.append(item)
    return converted

