from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from .features import as_float

MODEL_FEATURES = [
    "lag_points_mean3",
    "lag_minutes_mean3",
    "lag_starts_mean3",
    "lag_xgi_mean3",
    "lag_ict_mean3",
    "fixture_difficulty",
    "was_home",
    "value",
    "selected_millions",
    "team_attack_prior",
    "team_defence_prior",
]

METRIC_GLOSSARY = {
    "predicted_next_gw": "Projected points for the next gameweek, including captain-free appearance risk and fixture difficulty.",
    "predicted_next5": "Projected points across the next five gameweeks, summing every scheduled fixture in that window.",
    "transfer_score": "A ranking score for transfer targets: projected points plus fixture/value upside, less availability and minutes risk.",
    "risk_score": "A caution score driven by minutes, start likelihood, injury status, and missing recent evidence. Lower is safer.",
    "fixture_ease": "A flipped fixture difficulty measure where larger numbers mean an easier upcoming run.",
    "optimizer_value": "Projected five-gameweek points per million of current FPL price.",
}


def _records_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    numeric = [
        "element", "player_id", "GW", "gw", "total_points", "minutes", "starts", "expected_goal_involvements",
        "ict_index", "value", "selected", "team_attack_prior", "team_defence_prior",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return frame


def fixture_difficulty_lookup(fixtures: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    lookup: dict[int, tuple[float, float]] = {}
    for fixture in fixtures:
        try:
            fixture_id = int(as_float(fixture.get("id")))
        except (TypeError, ValueError):
            continue
        lookup[fixture_id] = (as_float(fixture.get("team_h_difficulty"), 3.0), as_float(fixture.get("team_a_difficulty"), 3.0))
    return lookup


def prepare_training_frame(
    prior_gameweeks: list[dict[str, Any]],
    current_history: list[dict[str, Any]],
    prior_fixtures: list[dict[str, Any]] | None = None,
    current_fixtures: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    prior_fixture_lookup = fixture_difficulty_lookup(prior_fixtures or [])
    current_fixture_lookup = fixture_difficulty_lookup(current_fixtures or [])
    historical = _records_frame(prior_gameweeks)
    if not historical.empty:
        historical = historical.rename(columns={"GW": "gw", "element": "train_player_id", "xP": "fpl_expected_points"})
        historical["team_attack_prior"] = 0.0
        historical["team_defence_prior"] = 0.0
        historical["was_home"] = historical["was_home"].astype(str).str.lower().isin(["true", "1"]).astype(int)
        historical["fixture_difficulty"] = historical.apply(
            lambda row: prior_fixture_lookup.get(int(as_float(row.get("fixture"))), (3.0, 3.0))[0 if int(row.get("was_home")) else 1],
            axis=1,
        )
        historical["selected_millions"] = pd.to_numeric(historical.get("selected", 0), errors="coerce").fillna(0) / 1_000_000
        historical = historical[["train_player_id", "gw", "total_points", "minutes", "starts", "expected_goal_involvements", "ict_index", "fixture_difficulty", "was_home", "value", "selected_millions", "team_attack_prior", "team_defence_prior"]]

    current = _records_frame(current_history)
    if not current.empty:
        current = current.rename(columns={"player_id": "train_player_id"})
        current["selected_millions"] = pd.to_numeric(current.get("selected", 0), errors="coerce").fillna(0) / 1_000_000
        current["fixture_difficulty"] = current.apply(
            lambda row: current_fixture_lookup.get(int(as_float(row.get("fixture"))), (3.0, 3.0))[0 if int(as_float(row.get("was_home"))) else 1],
            axis=1,
        )
        current["team_attack_prior"] = 0.0
        current["team_defence_prior"] = 0.0
        current = current[["train_player_id", "gw", "total_points", "minutes", "starts", "expected_goal_involvements", "ict_index", "fixture_difficulty", "was_home", "value", "selected_millions", "team_attack_prior", "team_defence_prior"]]

    frame = pd.concat([part for part in [historical, current] if not part.empty], ignore_index=True)
    if frame.empty:
        return frame

    frame = frame.sort_values(["train_player_id", "gw"]).copy()
    grouped = frame.groupby("train_player_id", group_keys=False)
    frame["lag_points_mean3"] = grouped["total_points"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    frame["lag_minutes_mean3"] = grouped["minutes"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    frame["lag_starts_mean3"] = grouped["starts"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    frame["lag_xgi_mean3"] = grouped["expected_goal_involvements"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    frame["lag_ict_mean3"] = grouped["ict_index"].apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
    frame = frame.dropna(subset=MODEL_FEATURES + ["total_points"])
    return frame


def train_models(training_frame: pd.DataFrame) -> dict[str, Any]:
    if len(training_frame) < 50:
        return {"status": "insufficient-data", "model": None, "ridge": None, "mae": None, "ridge_mae": None, "importance": {}}

    if len(training_frame) > 18000:
        training_frame = training_frame.sample(n=18000, random_state=42).sort_values(["train_player_id", "gw"])
    x = training_frame[MODEL_FEATURES]
    y = training_frame["total_points"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    forest = RandomForestRegressor(n_estimators=120, min_samples_leaf=8, random_state=42, n_jobs=-1)
    forest.fit(x_train, y_train)
    ridge = Ridge(alpha=2.0)
    ridge.fit(x_train, y_train)
    pred = forest.predict(x_test)
    ridge_pred = ridge.predict(x_test)
    importance = dict(sorted(zip(MODEL_FEATURES, forest.feature_importances_, strict=False), key=lambda item: item[1], reverse=True))
    return {
        "status": "trained",
        "model": forest,
        "ridge": ridge,
        "mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "ridge_mae": round(float(mean_absolute_error(y_test, ridge_pred)), 3),
        "importance": {key: round(float(value), 4) for key, value in importance.items()},
        "training_rows": int(len(training_frame)),
    }


def build_forecast_features(player: dict[str, Any], fixture: dict[str, Any], was_home: int, difficulty: float) -> dict[str, float]:
    return {
        "lag_points_mean3": as_float(player.get("points_per_game")),
        "lag_minutes_mean3": min(90.0, as_float(player.get("minutes")) / max(as_float(player.get("current_gameweek"), 1), 1.0)),
        "lag_starts_mean3": as_float(player.get("blended_start_rate")),
        "lag_xgi_mean3": as_float(player.get("blended_xgi_per_90")),
        "lag_ict_mean3": as_float(player.get("ict_index")) / max(as_float(player.get("current_gameweek"), 1), 1.0),
        "fixture_difficulty": difficulty,
        "was_home": float(was_home),
        "value": as_float(player.get("now_cost")),
        "selected_millions": as_float(player.get("selected_by_percent")) / 10.0,
        "team_attack_prior": as_float(player.get("team_attack_prior")),
        "team_defence_prior": as_float(player.get("team_defence_prior")),
    }


def predict_fixture_points(model_info: dict[str, Any], features: dict[str, float], fallback: float, availability: float) -> float:
    if model_info.get("status") == "trained" and model_info.get("model") is not None:
        frame = pd.DataFrame([{key: features[key] for key in MODEL_FEATURES}])
        value = float(model_info["model"].predict(frame)[0])
    else:
        value = fallback
    return round(max(0.0, value * availability), 3)


def build_fixture_forecasts(players: list[dict[str, Any]], fixtures: list[dict[str, Any]], model_info: dict[str, Any], horizon: int = 5) -> list[dict[str, Any]]:
    upcoming = [fixture for fixture in fixtures if not fixture.get("finished")]
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, float]] = []
    fallbacks: list[float] = []
    availabilities: list[float] = []

    for player in players:
        team_id = int(as_float(player.get("team_id")))
        team_fixtures = [fixture for fixture in upcoming if int(as_float(fixture.get("team_h"))) == team_id or int(as_float(fixture.get("team_a"))) == team_id][:horizon]
        for index, fixture in enumerate(team_fixtures, start=1):
            is_home = int(as_float(fixture.get("team_h"))) == team_id
            difficulty = as_float(fixture.get("team_h_difficulty" if is_home else "team_a_difficulty"), 3.0)
            features = build_forecast_features(player, fixture, int(is_home), difficulty)
            rows.append({
                "player_id": str(player.get("player_id")),
                "web_name": player.get("web_name", ""),
                "team": player.get("team", ""),
                "position": player.get("position", ""),
                "gw": int(as_float(fixture.get("event"), 0)),
                "horizon_index": index,
                "was_home": int(is_home),
                "opponent_team_id": int(as_float(fixture.get("team_a" if is_home else "team_h"))),
                "fixture_difficulty": difficulty,
                "forecast_points": 0.0,
            })
            feature_rows.append({key: features[key] for key in MODEL_FEATURES})
            fallbacks.append(max(as_float(player.get("ep_next")), as_float(player.get("points_per_game"))) * max(0.45, (6.0 - difficulty) / 3.0))
            availabilities.append(as_float(player.get("availability_factor"), 1.0))

    if not rows:
        return rows

    if model_info.get("status") == "trained" and model_info.get("model") is not None:
        predictions = model_info["model"].predict(pd.DataFrame(feature_rows))
    else:
        predictions = fallbacks
    for row, prediction, availability in zip(rows, predictions, availabilities, strict=False):
        row["forecast_points"] = round(max(0.0, float(prediction) * availability), 3)
    return rows

def apply_forecasts_to_players(players: list[dict[str, Any]], forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for forecast in forecasts:
        grouped.setdefault(str(forecast["player_id"]), []).append(forecast)
    result = []
    for player in players:
        item = dict(player)
        rows = sorted(grouped.get(str(player.get("player_id")), []), key=lambda row: row["horizon_index"])
        next_gw = rows[0]["forecast_points"] if rows else as_float(item.get("predicted_next_gw"))
        next5 = sum(as_float(row.get("forecast_points")) for row in rows[:5])
        item["predicted_next_gw"] = round(next_gw, 3)
        item["predicted_next5"] = round(next5, 3)
        item["predicted_next3"] = round(sum(as_float(row.get("forecast_points")) for row in rows[:3]), 3)
        item["model_residual"] = round(item["predicted_next5"] - as_float(item.get("baseline_next5")), 3)
        item["optimizer_value"] = round(item["predicted_next5"] / max(as_float(item.get("now_cost")) / 10.0, 0.1), 3)
        item["transfer_score"] = round(item["predicted_next5"] + 0.35 * as_float(item.get("fixture_ease")) + 0.035 * as_float(item.get("differential_score")) - 0.85 * as_float(item.get("risk_score")), 3)
        result.append(item)
    return result


def build_model_summary(rows: list[dict[str, Any]], model_info: dict[str, Any] | None = None) -> dict[str, Any]:
    model_info = model_info or {}
    predictions = [as_float(row.get("predicted_next5")) for row in rows]
    baselines = [as_float(row.get("baseline_next5", row.get("baseline_next3"))) for row in rows]
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
        "modelName": "RandomForestRegressor with Ridge and form baselines",
        "target": "fixture-level FPL points, aggregated over the next five gameweeks",
        "features": MODEL_FEATURES,
        "metricGlossary": METRIC_GLOSSARY,
        "rows": len(rows),
        "positions": positions,
        "priorCoverage": {
            "playersWith2025_26Prior": prior_players,
            "teamsWith2025_26Prior": prior_teams,
            "playerPriorCoveragePct": round(100 * prior_players / len(rows), 1) if rows else 0.0,
            "teamPriorCoveragePct": round(100 * prior_teams / len(rows), 1) if rows else 0.0,
        },
        "modelDiagnostics": {
            "status": model_info.get("status", "not-run"),
            "trainingRows": model_info.get("training_rows", 0),
            "randomForestMae": model_info.get("mae"),
            "ridgeMae": model_info.get("ridge_mae"),
            "featureImportance": model_info.get("importance", {}),
        },
        "diagnostics": {
            "meanPredictionNext5": round(mean(predictions), 3) if predictions else 0.0,
            "meanBaseline": round(mean(baselines), 3) if baselines else 0.0,
            "meanLift": round(mean(residuals), 3) if residuals else 0.0,
            "note": "2026-27 live FPL data is authoritative; 2025-26 priors are used only where stable player/team codes match.",
        },
    }


def write_model_summary(path: Path, rows: list[dict[str, Any]], model_info: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = build_model_summary(rows, model_info)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
