from __future__ import annotations

from typing import Any


def as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None, "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    return int(round(as_float(value, float(default))))


def add_projection_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add transparent, model-ready features and starter projections.

    Current 2026-27 production is the authority. Prior 2025-26 player/team data is
    used as a stabilising prior where stable FPL player/team codes match. New
    players and promoted/new teams receive neutral priors and are driven by the
    current-season public API data.
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
        ep_next = as_float(item.get("ep_next"), ppg)
        xgi = as_float(item.get("expected_goal_involvements"))
        fixture = as_float(item.get("fixture_difficulty_next5"), as_float(item.get("fixture_difficulty_next3"), 3.0))
        ownership = as_float(item.get("selected_by_percent"))
        starts = as_float(item.get("starts"))
        prior_found = as_float(item.get("prior_found"))
        prior_team_found = as_float(item.get("prior_team_found"))
        prior_points_per_90 = as_float(item.get("prior_points_per_90"))
        prior_xgi_per_90 = as_float(item.get("prior_xgi_per_90"))
        prior_starts_per_90 = as_float(item.get("prior_starts_per_90"))
        prior_team_attack = as_float(item.get("prior_team_attack_strength"), 1000.0)
        prior_team_defence = as_float(item.get("prior_team_defence_strength"), 1000.0)
        chance = as_float(item.get("chance_of_playing_next_round"), 100.0) / 100.0

        current_points_per_90 = total_points / ninety
        current_xgi_per_90 = xgi / ninety
        start_rate = starts / max(minutes / 90.0, starts, 1.0)
        current_weight = min(0.72, max(0.28, minutes / 900.0))
        prior_weight = (1.0 - current_weight) * prior_found

        item["points_per_90"] = round(current_points_per_90, 3)
        item["xgi_per_90"] = round(current_xgi_per_90, 3)
        item["start_rate"] = round(start_rate, 3)
        item["blended_points_per_90"] = round(current_weight * current_points_per_90 + prior_weight * prior_points_per_90, 3)
        item["blended_xgi_per_90"] = round(current_weight * current_xgi_per_90 + prior_weight * prior_xgi_per_90, 3)
        item["blended_start_rate"] = round(max(start_rate, prior_weight * prior_starts_per_90), 3)
        item["team_attack_prior"] = round((prior_team_attack - 1000.0) / 300.0 if prior_team_found else 0.0, 3)
        item["team_defence_prior"] = round((prior_team_defence - 1000.0) / 300.0 if prior_team_found else 0.0, 3)
        item["value_points_per_million"] = round(total_points / (cost / 10.0), 3)
        item["fixture_ease"] = round(6.0 - fixture, 3)
        item["differential_score"] = round(max(0.0, 25.0 - ownership), 3)
        item["availability_factor"] = round(max(0.0, min(1.0, chance)), 3)

        predicted_next_gw = (
            0.27 * ppg
            + 0.20 * form
            + 0.16 * ep_next
            + 0.17 * item["blended_points_per_90"]
            + 0.66 * item["blended_xgi_per_90"]
            + 0.25 * item["fixture_ease"]
            + 0.18 * item["blended_start_rate"]
            + 0.42 * item["team_attack_prior"]
        )
        position = item.get("position")
        if position == "GKP":
            predicted_next_gw += 0.28 * item["fixture_ease"] + 0.35 * item["team_defence_prior"]
        elif position == "DEF":
            predicted_next_gw += 0.20 * item["fixture_ease"] + 0.30 * item["team_defence_prior"]
        elif position == "FWD":
            predicted_next_gw += 0.20 * item["blended_xgi_per_90"]

        predicted_next_gw *= item["availability_factor"]
        item["predicted_next_gw"] = round(max(0.0, predicted_next_gw), 3)
        item["predicted_next3"] = round(item["predicted_next_gw"] * max(as_float(item.get("opponent_count_next3"), 3.0), 1.0), 3)
        item["predicted_next5"] = round(item["predicted_next_gw"] * max(as_float(item.get("opponent_count_next5"), 5.0), 1.0), 3)
        item["baseline_next3"] = round(max(ppg, form, ep_next, 0.0) * 3.0, 3)
        item["baseline_next5"] = round(max(ppg, form, ep_next, 0.0) * 5.0, 3)
        item["model_residual"] = round(item["predicted_next5"] - item["baseline_next5"], 3)
        item["optimizer_value"] = round(item["predicted_next5"] / (cost / 10.0), 3)
        item["risk_score"] = round(
            max(0.0, 1.0 - min(1.0, minutes / 900.0))
            + max(0.0, 0.65 - item["blended_start_rate"])
            + (1.0 - item["availability_factor"]),
            3,
        )
        item["transfer_score"] = round(
            item["predicted_next5"]
            + 0.35 * item["fixture_ease"]
            + 0.035 * item["differential_score"]
            + 0.20 * prior_found
            + 0.10 * prior_team_found
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


def add_history_summary_features(rows: list[dict[str, Any]], history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in history_rows:
        grouped.setdefault(str(row.get("player_id")), []).append(row)

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        current_gameweek = max(as_float(item.get("current_gameweek"), 1.0), 1.0)
        minutes = as_float(item.get("minutes"))
        player_history = grouped.get(str(item.get("player_id")), [])
        played_rows = [history for history in player_history if as_float(history.get("minutes")) > 0]
        game_count = max(float(len(player_history)), current_gameweek, 1.0)
        played_count = max(float(len(played_rows)), 1.0)
        history_minutes = sum(as_float(history.get("minutes")) for history in player_history)
        history_defcons = sum(as_float(history.get("defensive_contribution")) for history in player_history)
        history_bonus = sum(as_float(history.get("bonus")) for history in player_history)
        history_bps = sum(as_float(history.get("bps")) for history in player_history)
        per_90_source = as_float(item.get("defensive_contribution_per_90"))

        item["minutes_per_game"] = round(minutes / current_gameweek, 2)
        item["bonus_per_game"] = round(history_bonus / game_count, 3)
        item["bps_per_game"] = round(history_bps / game_count, 3)
        success_threshold = 12.0 if item.get("position") == "MID" else 10.0
        item["defcons_per_90"] = round(per_90_source or (history_defcons * 90.0 / max(history_minutes, 1.0)), 3)
        item["defcon_success_pct"] = round(
            100.0 * sum(as_float(history.get("defensive_contribution")) >= success_threshold for history in played_rows) / played_count,
            1,
        )
        result.append(item)
    return result
