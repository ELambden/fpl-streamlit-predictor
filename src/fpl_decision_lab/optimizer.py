from __future__ import annotations

from itertools import combinations
from typing import Any

from .features import as_float

SQUAD_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
VALID_FORMATIONS = [
    {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3},
    {"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2},
    {"GKP": 1, "DEF": 4, "MID": 3, "FWD": 3},
    {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2},
    {"GKP": 1, "DEF": 4, "MID": 5, "FWD": 1},
    {"GKP": 1, "DEF": 5, "MID": 2, "FWD": 3},
    {"GKP": 1, "DEF": 5, "MID": 3, "FWD": 2},
    {"GKP": 1, "DEF": 5, "MID": 4, "FWD": 1},
]


def _player_id(player: dict[str, Any]) -> str:
    return str(player.get("player_id"))


def _score(player: dict[str, Any], key: str = "predicted_next3") -> float:
    return as_float(player.get(key))


def valid_squad(players: list[dict[str, Any]], budget: float = 1000.0) -> bool:
    if len(players) != 15:
        return False
    if sum(as_float(player.get("now_cost")) for player in players) > budget:
        return False
    by_position = {position: 0 for position in SQUAD_QUOTAS}
    by_team: dict[str, int] = {}
    for player in players:
        position = player.get("position")
        if position not in by_position:
            return False
        by_position[position] += 1
        team = str(player.get("team"))
        by_team[team] = by_team.get(team, 0) + 1
    return by_position == SQUAD_QUOTAS and all(count <= 3 for count in by_team.values())


def choose_starting_xi(squad: list[dict[str, Any]], key: str = "predicted_next_gw") -> dict[str, Any]:
    best: dict[str, Any] | None = None
    by_position = {position: [p for p in squad if p.get("position") == position] for position in SQUAD_QUOTAS}
    for position_players in by_position.values():
        position_players.sort(key=lambda player: _score(player, key), reverse=True)

    for formation in VALID_FORMATIONS:
        starters: list[dict[str, Any]] = []
        for position, count in formation.items():
            if len(by_position[position]) < count:
                starters = []
                break
            starters.extend(by_position[position][:count])
        if len(starters) != 11:
            continue
        total = sum(_score(player, key) for player in starters)
        if best is None or total > best["score"]:
            starter_ids = {_player_id(player) for player in starters}
            bench = [player for player in squad if _player_id(player) not in starter_ids]
            bench.sort(key=lambda player: _score(player, key), reverse=True)
            best = {"formation": formation, "starters": starters, "bench": bench, "score": round(total, 3)}
    if best is None:
        raise ValueError("Could not form a legal starting XI from the provided squad.")
    starters = best["starters"]
    best["captain"] = max(starters, key=lambda player: _score(player, key))
    best["vice_captain"] = sorted(starters, key=lambda player: _score(player, key), reverse=True)[1]
    best["score_with_captain"] = round(best["score"] + _score(best["captain"], key), 3)
    return best


def best_single_transfers(
    players: list[dict[str, Any]],
    squad_ids: set[str],
    bank: float,
    free_transfers: int = 1,
    limit: int = 20,
) -> list[dict[str, Any]]:
    by_id = {_player_id(player): player for player in players}
    squad = [by_id[player_id] for player_id in squad_ids if player_id in by_id]
    candidates = [player for player in players if _player_id(player) not in squad_ids]
    recommendations: list[dict[str, Any]] = []
    hit_cost = 0 if free_transfers >= 1 else 4

    for outgoing in squad:
        available_budget = as_float(outgoing.get("now_cost")) + bank
        for incoming in candidates:
            if incoming.get("position") != outgoing.get("position"):
                continue
            if as_float(incoming.get("now_cost")) > available_budget:
                continue
            team_count = sum(1 for player in squad if player.get("team") == incoming.get("team"))
            if outgoing.get("team") != incoming.get("team") and team_count >= 3:
                continue
            gain = _score(incoming) - _score(outgoing) - hit_cost
            recommendations.append(
                {
                    "out": outgoing,
                    "in": incoming,
                    "gain": round(gain, 3),
                    "net_budget": round(available_budget - as_float(incoming.get("now_cost")), 1),
                }
            )
    return sorted(recommendations, key=lambda item: item["gain"], reverse=True)[:limit]


def build_wildcard_squad(players: list[dict[str, Any]], budget: float = 1000.0) -> list[dict[str, Any]]:
    """Greedy-with-repair squad builder for a fast, explainable wildcard draft."""

    selected: list[dict[str, Any]] = []
    team_counts: dict[str, int] = {}
    remaining_budget = budget

    for position, quota in SQUAD_QUOTAS.items():
        pool = [player for player in players if player.get("position") == position]
        pool.sort(key=lambda player: (_score(player), as_float(player.get("optimizer_value"))), reverse=True)
        for player in pool:
            if len([item for item in selected if item.get("position") == position]) >= quota:
                break
            cost = as_float(player.get("now_cost"))
            team = str(player.get("team"))
            if cost > remaining_budget or team_counts.get(team, 0) >= 3:
                continue
            selected.append(player)
            remaining_budget -= cost
            team_counts[team] = team_counts.get(team, 0) + 1

    if len(selected) == 15 and valid_squad(selected, budget):
        return selected

    # Fallback exhaustive by position top slices, bounded for app responsiveness.
    top_by_position = {
        position: sorted(
            [player for player in players if player.get("position") == position],
            key=lambda player: _score(player),
            reverse=True,
        )[:12]
        for position in SQUAD_QUOTAS
    }
    best: list[dict[str, Any]] = []
    best_score = -1.0
    for gkp in combinations(top_by_position["GKP"], 2):
        for defs in combinations(top_by_position["DEF"], 5):
            for mids in combinations(top_by_position["MID"], 5):
                for fwds in combinations(top_by_position["FWD"], 3):
                    squad = list(gkp + defs + mids + fwds)
                    if not valid_squad(squad, budget):
                        continue
                    score = sum(_score(player) for player in squad)
                    if score > best_score:
                        best = squad
                        best_score = score
    if not best:
        raise ValueError("Could not build a legal wildcard squad within the budget.")
    return best

