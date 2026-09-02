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


def _score(player: dict[str, Any], key: str = "predicted_next5") -> float:
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
    score_key: str = "predicted_next5",
) -> list[dict[str, Any]]:
    by_id = {_player_id(player): player for player in players}
    squad = [by_id[player_id] for player_id in squad_ids if player_id in by_id]
    candidates = [player for player in players if _player_id(player) not in squad_ids and as_float(player.get("predicted_next5")) > 0]
    candidates = sorted(candidates, key=lambda player: as_float(player.get("transfer_score")), reverse=True)[:120]
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
            gain = _score(incoming, score_key) - _score(outgoing, score_key) - hit_cost
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
    """Build a legal wildcard squad quickly, then spend remaining budget on upgrades."""

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    team_counts: dict[str, int] = {}
    remaining_budget = budget

    for position, quota in SQUAD_QUOTAS.items():
        pool = sorted(
            [player for player in players if player.get("position") == position],
            key=lambda player: (as_float(player.get("now_cost")), -as_float(player.get("transfer_score"))),
        )
        for player in pool:
            if len([item for item in selected if item.get("position") == position]) >= quota:
                break
            player_id = _player_id(player)
            team = str(player.get("team"))
            cost = as_float(player.get("now_cost"))
            if player_id in selected_ids or team_counts.get(team, 0) >= 3 or cost > remaining_budget:
                continue
            selected.append(player)
            selected_ids.add(player_id)
            team_counts[team] = team_counts.get(team, 0) + 1
            remaining_budget -= cost
        if len([item for item in selected if item.get("position") == position]) < quota:
            raise ValueError(f"Could not find enough legal {position} players within budget.")

    improved = True
    while improved:
        improved = False
        best_swap: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        for outgoing in selected:
            for incoming in players:
                incoming_id = _player_id(incoming)
                if incoming_id in selected_ids or incoming.get("position") != outgoing.get("position"):
                    continue
                cost_delta = as_float(incoming.get("now_cost")) - as_float(outgoing.get("now_cost"))
                if cost_delta > remaining_budget:
                    continue
                outgoing_team = str(outgoing.get("team"))
                incoming_team = str(incoming.get("team"))
                incoming_team_count = team_counts.get(incoming_team, 0) - int(incoming_team == outgoing_team)
                if incoming_team_count >= 3:
                    continue
                gain = _score(incoming) - _score(outgoing)
                if gain > 0 and (best_swap is None or gain > best_swap[0]):
                    best_swap = (gain, outgoing, incoming)
        if best_swap is not None:
            _, outgoing, incoming = best_swap
            selected.remove(outgoing)
            selected.append(incoming)
            selected_ids.remove(_player_id(outgoing))
            selected_ids.add(_player_id(incoming))
            team_counts[str(outgoing.get("team"))] -= 1
            team_counts[str(incoming.get("team"))] = team_counts.get(str(incoming.get("team")), 0) + 1
            remaining_budget -= as_float(incoming.get("now_cost")) - as_float(outgoing.get("now_cost"))
            improved = True

    if not valid_squad(selected, budget):
        raise ValueError("Could not build a legal wildcard squad within the budget.")
    return selected


def forecast_lookup(forecasts: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    lookup: dict[tuple[str, int], float] = {}
    for row in forecasts:
        key = (str(row.get("player_id")), int(as_float(row.get("horizon_index"))))
        lookup[key] = lookup.get(key, 0.0) + as_float(row.get("forecast_points"))
    return lookup


def _score_for_week(player: dict[str, Any], lookup: dict[tuple[str, int], float], week: int) -> float:
    return lookup.get((_player_id(player), week), as_float(player.get("predicted_next_gw")))


def choose_starting_xi_for_week(squad: list[dict[str, Any]], lookup: dict[tuple[str, int], float], week: int) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    by_position = {position: [p for p in squad if p.get("position") == position] for position in SQUAD_QUOTAS}
    for position_players in by_position.values():
        position_players.sort(key=lambda player: _score_for_week(player, lookup, week), reverse=True)

    for formation in VALID_FORMATIONS:
        starters: list[dict[str, Any]] = []
        for position, count in formation.items():
            if len(by_position[position]) < count:
                starters = []
                break
            starters.extend(by_position[position][:count])
        if len(starters) != 11:
            continue
        total = sum(_score_for_week(player, lookup, week) for player in starters)
        if best is None or total > best["score"]:
            starter_ids = {_player_id(player) for player in starters}
            bench = [player for player in squad if _player_id(player) not in starter_ids]
            bench.sort(key=lambda player: _score_for_week(player, lookup, week), reverse=True)
            best = {"formation": formation, "starters": starters, "bench": bench, "score": round(total, 3)}
    if best is None:
        raise ValueError("Could not form a legal starting XI from the provided squad.")
    ranked = sorted(best["starters"], key=lambda player: _score_for_week(player, lookup, week), reverse=True)
    best["captain"] = ranked[0]
    best["vice_captain"] = ranked[1]
    best["score_with_captain"] = round(best["score"] + _score_for_week(best["captain"], lookup, week), 3)
    return best


def _candidate_transfers(players: list[dict[str, Any]], squad: list[dict[str, Any]], bank: float, lookup: dict[tuple[str, int], float], week: int, limit: int = 8) -> list[dict[str, Any]]:
    squad_ids = {_player_id(player) for player in squad}
    candidates = [player for player in players if _player_id(player) not in squad_ids and as_float(player.get("predicted_next5")) > 0]
    candidates = sorted(candidates, key=lambda player: as_float(player.get("transfer_score")), reverse=True)[:120]
    moves: list[dict[str, Any]] = []
    for outgoing in squad:
        budget = as_float(outgoing.get("now_cost")) + bank
        for incoming in candidates:
            if incoming.get("position") != outgoing.get("position"):
                continue
            if as_float(incoming.get("now_cost")) > budget:
                continue
            team_count = sum(1 for player in squad if player.get("team") == incoming.get("team"))
            if outgoing.get("team") != incoming.get("team") and team_count >= 3:
                continue
            gain = sum(_score_for_week(incoming, lookup, w) - _score_for_week(outgoing, lookup, w) for w in range(week, 6))
            moves.append({"out": outgoing, "in": incoming, "gain": round(gain, 3), "bank_delta": as_float(outgoing.get("now_cost")) - as_float(incoming.get("now_cost"))})
    return sorted(moves, key=lambda item: item["gain"], reverse=True)[:limit]


def _apply_moves(squad: list[dict[str, Any]], moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outgoing = {_player_id(move["out"]) for move in moves}
    return [player for player in squad if _player_id(player) not in outgoing] + [move["in"] for move in moves]


def plan_transfers(
    players: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    squad_ids: set[str],
    bank: float,
    free_transfers: int,
    horizon: int = 5,
    beam_width: int = 8,
    max_transfers_per_week: int = 2,
) -> list[dict[str, Any]]:
    by_id = {_player_id(player): player for player in players}
    initial_squad = [by_id[player_id] for player_id in squad_ids if player_id in by_id]
    if len(initial_squad) != 15:
        raise ValueError("A valid FPL squad needs 15 current players.")
    lookup = forecast_lookup(forecasts)
    states = [{"squad": initial_squad, "bank": bank, "free": max(1, min(5, int(free_transfers))), "score": 0.0, "steps": []}]

    for week in range(1, horizon + 1):
        next_states: list[dict[str, Any]] = []
        for state in states:
            possible_move_sets: list[list[dict[str, Any]]] = [[]]
            singles = _candidate_transfers(players, state["squad"], state["bank"], lookup, week)
            possible_move_sets.extend([[move] for move in singles[:6]])
            if max_transfers_per_week >= 2:
                for first in singles[:4]:
                    squad_after_first = _apply_moves(state["squad"], [first])
                    bank_after_first = state["bank"] + first["bank_delta"]
                    seconds = _candidate_transfers(players, squad_after_first, bank_after_first, lookup, week, limit=4)
                    for second in seconds[:2]:
                        if _player_id(second["in"]) == _player_id(first["out"]):
                            continue
                        possible_move_sets.append([first, second])
            for moves in possible_move_sets:
                new_squad = _apply_moves(state["squad"], moves)
                new_bank = state["bank"] + sum(move["bank_delta"] for move in moves)
                if not valid_squad(new_squad, budget=1000.0 + max(new_bank, 0)):
                    continue
                hits = max(0, len(moves) - state["free"]) * 4
                xi = choose_starting_xi_for_week(new_squad, lookup, week)
                weekly_score = xi["score_with_captain"] - hits
                next_free = min(5, state["free"] - len(moves) + 1) if len(moves) <= state["free"] else 1
                step = {
                    "week": week,
                    "moves": moves,
                    "hits": hits,
                    "formation": xi["formation"],
                    "captain": xi["captain"],
                    "score": round(weekly_score, 3),
                }
                next_states.append({"squad": new_squad, "bank": round(new_bank, 1), "free": next_free, "score": round(state["score"] + weekly_score, 3), "steps": state["steps"] + [step]})
        states = sorted(next_states, key=lambda item: item["score"], reverse=True)[:beam_width]
    return states
