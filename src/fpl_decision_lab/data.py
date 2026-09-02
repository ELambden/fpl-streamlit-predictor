from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

POSITION_BY_ID = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_json(url: str, timeout: int = 30) -> Any:
    request = Request(url, headers={"User-Agent": "fpl-decision-lab/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_current_fpl_data(raw_dir: Path) -> dict[str, Any]:
    """Fetch public FPL snapshots and persist raw JSON for traceability."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bootstrap": fetch_json(BOOTSTRAP_URL),
        "fixtures": fetch_json(FIXTURES_URL),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (raw_dir / f"fpl-api-{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (raw_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_latest_raw(raw_path: Path) -> dict[str, Any]:
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fixture_summary(fixtures: list[dict[str, Any]], next_n: int = 3) -> dict[int, dict[str, float]]:
    """Return simple upcoming fixture features by FPL team ID."""

    by_team: dict[int, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        if fixture.get("finished"):
            continue
        team_h = _as_int(fixture.get("team_h"))
        team_a = _as_int(fixture.get("team_a"))
        if not team_h or not team_a:
            continue
        by_team.setdefault(team_h, []).append(
            {"difficulty": _as_float(fixture.get("team_h_difficulty"), 3.0), "home": 1.0}
        )
        by_team.setdefault(team_a, []).append(
            {"difficulty": _as_float(fixture.get("team_a_difficulty"), 3.0), "home": 0.0}
        )

    summary: dict[int, dict[str, float]] = {}
    for team_id, team_fixtures in by_team.items():
        window = team_fixtures[:next_n]
        if not window:
            continue
        summary[team_id] = {
            "fixture_difficulty_next3": round(sum(item["difficulty"] for item in window) / len(window), 3),
            "next3_home_count": sum(item["home"] for item in window),
            "opponent_count_next3": len(window),
        }
    return summary


def normalize_bootstrap(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    bootstrap = raw_payload["bootstrap"]
    teams = {team["id"]: team["name"] for team in bootstrap.get("teams", [])}
    fixture_by_team = fixture_summary(raw_payload.get("fixtures", []))
    rows: list[dict[str, Any]] = []

    for player in bootstrap.get("elements", []):
        team_id = _as_int(player.get("team"))
        fixture = fixture_by_team.get(
            team_id,
            {"fixture_difficulty_next3": 3.0, "next3_home_count": 1.5, "opponent_count_next3": 3},
        )
        expected_goal_involvements = _as_float(player.get("expected_goal_involvements"))
        total_points = _as_int(player.get("total_points"))
        minutes = _as_int(player.get("minutes"))
        starts = _as_int(player.get("starts"))
        cost = _as_float(player.get("now_cost"))
        rows.append(
            {
                "player_id": str(player.get("id")),
                "web_name": player.get("web_name", ""),
                "full_name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
                "team": teams.get(team_id, str(team_id)),
                "team_id": team_id,
                "position": POSITION_BY_ID.get(_as_int(player.get("element_type")), "UNK"),
                "now_cost": cost,
                "selected_by_percent": _as_float(player.get("selected_by_percent")),
                "total_points": total_points,
                "minutes": minutes,
                "starts": starts,
                "goals_scored": _as_int(player.get("goals_scored")),
                "assists": _as_int(player.get("assists")),
                "clean_sheets": _as_int(player.get("clean_sheets")),
                "goals_conceded": _as_int(player.get("goals_conceded")),
                "expected_goals": _as_float(player.get("expected_goals")),
                "expected_assists": _as_float(player.get("expected_assists")),
                "expected_goal_involvements": expected_goal_involvements,
                "ict_index": _as_float(player.get("ict_index")),
                "form": _as_float(player.get("form")),
                "points_per_game": _as_float(player.get("points_per_game")),
                **fixture,
            }
        )
    return rows


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty player dataset.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

