from fpl_decision_lab.data import read_csv_records
from fpl_decision_lab.optimizer import build_wildcard_squad, choose_starting_xi, plan_transfers, valid_squad
from fpl_decision_lab.paths import PLAYER_FORECASTS_CSV, PLAYERS_CSV


def test_wildcard_squad_is_legal_for_current_pool() -> None:
    players = read_csv_records(PLAYERS_CSV)
    squad = build_wildcard_squad(players, budget=1000)

    assert valid_squad(squad, budget=1000)


def test_starting_xi_has_captain_and_legal_size() -> None:
    players = read_csv_records(PLAYERS_CSV)
    squad = build_wildcard_squad(players, budget=1000)
    xi = choose_starting_xi(squad)

    assert len(xi["starters"]) == 11
    assert xi["captain"] in xi["starters"]
    assert xi["vice_captain"] in xi["starters"]
    assert xi["score_with_captain"] >= xi["score"]


def test_five_gameweek_planner_returns_scored_path() -> None:
    players = read_csv_records(PLAYERS_CSV)
    forecasts = read_csv_records(PLAYER_FORECASTS_CSV)
    squad = build_wildcard_squad(players, budget=1000)
    plans = plan_transfers(players, forecasts, {player["player_id"] for player in squad}, bank=0, free_transfers=1, horizon=5, beam_width=3)

    assert plans
    assert len(plans[0]["steps"]) == 5
    assert plans[0]["score"] > 0
