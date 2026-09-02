from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
ENTRY_URL = "https://fantasy.premierleague.com/api/entry/{entry_id}/"
ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event_id}/picks/"
ENTRY_TRANSFERS_URL = "https://fantasy.premierleague.com/api/entry/{entry_id}/transfers/"
PRIOR_PLAYERS_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/players_raw.csv"
PRIOR_TEAMS_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/teams.csv"
PRIOR_GW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"
PRIOR_FIXTURES_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/fixtures.csv"

POSITION_BY_ID = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_NAME_TO_SHORT = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
NEUTRAL_TEAM_STRENGTH = 1000.0


def fetch_json(url: str, timeout: int = 30) -> Any:
    request = Request(url, headers={"User-Agent": "fpl-streamlit-predictor/0.3"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "fpl-streamlit-predictor/0.3"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8-sig")


def _csv_text_records(text: str) -> list[dict[str, Any]]:
    return [dict(row) for row in csv.DictReader(text.splitlines())]


def fetch_optional_csv(url: str) -> tuple[list[dict[str, Any]], str]:
    try:
        return _csv_text_records(fetch_text(url, timeout=60)), "ok"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return [], f"unavailable: {exc}"


def fetch_optional_json(url: str) -> tuple[Any | None, str]:
    try:
        return fetch_json(url, timeout=30), "ok"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return None, f"unavailable: {exc}"


def fetch_element_summaries(player_ids: list[int], max_workers: int = 16) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_optional_json, ELEMENT_SUMMARY_URL.format(player_id=player_id)): player_id
            for player_id in player_ids
        }
        for future in as_completed(futures):
            player_id = futures[future]
            payload, status = future.result()
            if payload is not None:
                summaries[str(player_id)] = payload
            else:
                summaries[str(player_id)] = {"history": [], "fixtures": [], "error": status}
    return summaries


def fetch_current_fpl_data(raw_dir: Path, include_element_summaries: bool = True) -> dict[str, Any]:
    """Fetch public FPL snapshots and optional prior-season priors."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = fetch_json(BOOTSTRAP_URL)
    prior_players, prior_players_status = fetch_optional_csv(PRIOR_PLAYERS_URL)
    prior_teams, prior_teams_status = fetch_optional_csv(PRIOR_TEAMS_URL)
    prior_gw, prior_gw_status = fetch_optional_csv(PRIOR_GW_URL)
    prior_fixtures, prior_fixtures_status = fetch_optional_csv(PRIOR_FIXTURES_URL)
    player_ids = [int(player["id"]) for player in bootstrap.get("elements", []) if player.get("id")]
    payload = {
        "bootstrap": bootstrap,
        "fixtures": fetch_json(FIXTURES_URL),
        "element_summaries": fetch_element_summaries(player_ids) if include_element_summaries else {},
        "prior_players_2025_26": prior_players,
        "prior_teams_2025_26": prior_teams,
        "prior_gameweeks_2025_26": prior_gw,
        "prior_fixtures_2025_26": prior_fixtures,
        "prior_source_status": {
            "players_raw": prior_players_status,
            "teams": prior_teams_status,
            "gameweeks": prior_gw_status,
            "fixtures": prior_fixtures_status,
        },
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (raw_dir / f"fpl-api-{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (raw_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_latest_raw(raw_path: Path) -> dict[str, Any]:
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None, "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value in ("", None, "None"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def current_gameweek(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("is_current"):
            return event
    for event in events:
        if event.get("is_next"):
            return event
    for event in reversed(events):
        if event.get("finished"):
            return event
    return events[0] if events else {}


def next_gameweek(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("is_next"):
            return event
    current = current_gameweek(events).get("id", 0)
    for event in events:
        if _as_int(event.get("id")) > _as_int(current):
            return event
    return {}


def latest_checked_gameweek(events: list[dict[str, Any]]) -> int:
    checked = [_as_int(event.get("id")) for event in events if event.get("finished") and event.get("data_checked")]
    return max(checked) if checked else _as_int(current_gameweek(events).get("id"))


def fixture_summary(fixtures: list[dict[str, Any]], next_n: int = 5) -> dict[int, dict[str, float]]:
    by_team: dict[int, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        if fixture.get("finished"):
            continue
        team_h = _as_int(fixture.get("team_h"))
        team_a = _as_int(fixture.get("team_a"))
        if not team_h or not team_a:
            continue
        by_team.setdefault(team_h, []).append({"difficulty": _as_float(fixture.get("team_h_difficulty"), 3.0), "home": 1.0})
        by_team.setdefault(team_a, []).append({"difficulty": _as_float(fixture.get("team_a_difficulty"), 3.0), "home": 0.0})

    summary: dict[int, dict[str, float]] = {}
    for team_id, team_fixtures in by_team.items():
        window = team_fixtures[:next_n]
        if not window:
            continue
        summary[team_id] = {
            "fixture_difficulty_next5": round(sum(item["difficulty"] for item in window) / len(window), 3),
            "fixture_difficulty_next3": round(sum(item["difficulty"] for item in window[:3]) / max(len(window[:3]), 1), 3),
            "next5_home_count": sum(item["home"] for item in window),
            "next3_home_count": sum(item["home"] for item in window[:3]),
            "opponent_count_next5": len(window),
            "opponent_count_next3": len(window[:3]),
        }
    return summary


def build_prior_player_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    priors: dict[int, dict[str, float]] = {}
    for row in rows:
        code = _as_int(row.get("code"))
        if not code:
            continue
        raw_minutes = _as_float(row.get("minutes"))
        minutes = max(raw_minutes, 1.0)
        ninety = minutes / 90.0
        prior_found = int(raw_minutes >= 90.0)
        starts_per_90 = _as_float(row.get("starts_per_90"), _as_float(row.get("starts")) / max(ninety, 1.0))
        priors[code] = {
            "prior_minutes": round(raw_minutes, 1),
            "prior_total_points": round(_as_float(row.get("total_points")), 1) if prior_found else 0.0,
            "prior_points_per_90": round(_as_float(row.get("total_points")) / ninety, 3) if prior_found else 0.0,
            "prior_xgi_per_90": round(_as_float(row.get("expected_goal_involvements")) / ninety, 3) if prior_found else 0.0,
            "prior_starts_per_90": round(max(0.0, min(1.05, starts_per_90)), 3) if prior_found else 0.0,
            "prior_found": prior_found,
        }
    return priors


def build_prior_team_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    priors: dict[int, dict[str, float]] = {}
    for row in rows:
        code = _as_int(row.get("code"))
        if not code:
            continue
        attack = (_as_float(row.get("strength_attack_home"), NEUTRAL_TEAM_STRENGTH) + _as_float(row.get("strength_attack_away"), NEUTRAL_TEAM_STRENGTH)) / 2
        defence = (_as_float(row.get("strength_defence_home"), NEUTRAL_TEAM_STRENGTH) + _as_float(row.get("strength_defence_away"), NEUTRAL_TEAM_STRENGTH)) / 2
        overall = (_as_float(row.get("strength_overall_home"), NEUTRAL_TEAM_STRENGTH) + _as_float(row.get("strength_overall_away"), NEUTRAL_TEAM_STRENGTH)) / 2
        priors[code] = {
            "prior_team_strength": round(overall, 1),
            "prior_team_attack_strength": round(attack, 1),
            "prior_team_defence_strength": round(defence, 1),
            "prior_team_found": 1,
        }
    return priors


def neutral_player_prior() -> dict[str, float]:
    return {"prior_minutes": 0.0, "prior_total_points": 0.0, "prior_points_per_90": 0.0, "prior_xgi_per_90": 0.0, "prior_starts_per_90": 0.0, "prior_found": 0}


def neutral_team_prior() -> dict[str, float]:
    return {"prior_team_strength": NEUTRAL_TEAM_STRENGTH, "prior_team_attack_strength": NEUTRAL_TEAM_STRENGTH, "prior_team_defence_strength": NEUTRAL_TEAM_STRENGTH, "prior_team_found": 0}


def normalize_bootstrap(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    bootstrap = raw_payload["bootstrap"]
    teams_by_id = {team["id"]: team for team in bootstrap.get("teams", [])}
    fixture_by_team = fixture_summary(raw_payload.get("fixtures", []))
    prior_players = build_prior_player_map(raw_payload.get("prior_players_2025_26", []))
    prior_teams = build_prior_team_map(raw_payload.get("prior_teams_2025_26", []))
    gameweek = current_gameweek(bootstrap.get("events", []))
    rows: list[dict[str, Any]] = []

    for player in bootstrap.get("elements", []):
        team_id = _as_int(player.get("team"))
        team = teams_by_id.get(team_id, {})
        team_code = _as_int(team.get("code"), _as_int(player.get("team_code")))
        player_code = _as_int(player.get("code"))
        fixture = fixture_by_team.get(team_id, {"fixture_difficulty_next5": 3.0, "fixture_difficulty_next3": 3.0, "next5_home_count": 2.5, "next3_home_count": 1.5, "opponent_count_next5": 5, "opponent_count_next3": 3})
        player_prior = prior_players.get(player_code, neutral_player_prior())
        team_prior = prior_teams.get(team_code, neutral_team_prior())
        rows.append(
            {
                "season": "2026-27",
                "current_gameweek": _as_int(gameweek.get("id")),
                "player_id": str(player.get("id")),
                "player_code": player_code,
                "web_name": player.get("web_name", ""),
                "full_name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
                "team": team.get("name", str(team_id)),
                "team_id": team_id,
                "team_code": team_code,
                "position": POSITION_BY_ID.get(_as_int(player.get("element_type")), "UNK"),
                "status": player.get("status", ""),
                "news": player.get("news", ""),
                "chance_of_playing_next_round": _as_int(player.get("chance_of_playing_next_round"), 100),
                "now_cost": _as_float(player.get("now_cost")),
                "selected_by_percent": _as_float(player.get("selected_by_percent")),
                "total_points": _as_int(player.get("total_points")),
                "minutes": _as_int(player.get("minutes")),
                "starts": _as_int(player.get("starts")),
                "goals_scored": _as_int(player.get("goals_scored")),
                "assists": _as_int(player.get("assists")),
                "clean_sheets": _as_int(player.get("clean_sheets")),
                "goals_conceded": _as_int(player.get("goals_conceded")),
                "expected_goals": _as_float(player.get("expected_goals")),
                "expected_assists": _as_float(player.get("expected_assists")),
                "expected_goal_involvements": _as_float(player.get("expected_goal_involvements")),
                "expected_goals_conceded": _as_float(player.get("expected_goals_conceded")),
                "ict_index": _as_float(player.get("ict_index")),
                "form": _as_float(player.get("form")),
                "points_per_game": _as_float(player.get("points_per_game")),
                "ep_next": _as_float(player.get("ep_next")),
                **fixture,
                **player_prior,
                **team_prior,
            }
        )
    return rows


def build_player_history(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    players = {str(player.get("id")): player for player in raw_payload.get("bootstrap", {}).get("elements", [])}
    teams = {team["id"]: team["name"] for team in raw_payload.get("bootstrap", {}).get("teams", [])}
    rows: list[dict[str, Any]] = []
    for player_id, summary in raw_payload.get("element_summaries", {}).items():
        player = players.get(str(player_id), {})
        for item in summary.get("history", []):
            rows.append(
                {
                    "player_id": str(player_id),
                    "web_name": player.get("web_name", ""),
                    "team": teams.get(_as_int(player.get("team")), ""),
                    "position": POSITION_BY_ID.get(_as_int(player.get("element_type")), "UNK"),
                    "gw": _as_int(item.get("round")),
                    "fixture": _as_int(item.get("fixture")),
                    "opponent_team_id": _as_int(item.get("opponent_team")),
                    "was_home": int(bool(item.get("was_home"))),
                    "minutes": _as_int(item.get("minutes")),
                    "starts": _as_int(item.get("starts")),
                    "total_points": _as_int(item.get("total_points")),
                    "goals_scored": _as_int(item.get("goals_scored")),
                    "assists": _as_int(item.get("assists")),
                    "clean_sheets": _as_int(item.get("clean_sheets")),
                    "expected_goals": _as_float(item.get("expected_goals")),
                    "expected_assists": _as_float(item.get("expected_assists")),
                    "expected_goal_involvements": _as_float(item.get("expected_goal_involvements")),
                    "ict_index": _as_float(item.get("ict_index")),
                    "bps": _as_int(item.get("bps")),
                    "value": _as_float(item.get("value")),
                    "selected": _as_int(item.get("selected")),
                    "transfers_in": _as_int(item.get("transfers_in")),
                    "transfers_out": _as_int(item.get("transfers_out")),
                }
            )
    return rows


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty dataset: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fetch_entry(entry_id: int) -> dict[str, Any]:
    return fetch_json(ENTRY_URL.format(entry_id=entry_id))


def fetch_entry_picks(entry_id: int, event_id: int) -> dict[str, Any]:
    return fetch_json(ENTRY_PICKS_URL.format(entry_id=entry_id, event_id=event_id))


def fetch_entry_transfers(entry_id: int) -> list[dict[str, Any]]:
    return fetch_json(ENTRY_TRANSFERS_URL.format(entry_id=entry_id))


def fetch_public_team(entry_id: int, event_id: int | None = None) -> dict[str, Any]:
    entry = fetch_entry(entry_id)
    event = event_id or _as_int(entry.get("current_event"))
    picks = fetch_entry_picks(entry_id, event)
    transfers = fetch_entry_transfers(entry_id)
    return {"entry": entry, "event": event, "picks": picks, "transfers": transfers}
