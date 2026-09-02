from fpl_decision_lab.data import read_csv_records
from fpl_decision_lab.optimizer import build_wildcard_squad, choose_starting_xi, valid_squad
from fpl_decision_lab.paths import PLAYERS_CSV


def test_wildcard_squad_is_legal_for_sample_pool() -> None:
    players = read_csv_records(PLAYERS_CSV)
    squad = build_wildcard_squad(players, budget=1400)

    assert valid_squad(squad, budget=1400)


def test_starting_xi_has_captain_and_legal_size() -> None:
    players = read_csv_records(PLAYERS_CSV)
    squad = build_wildcard_squad(players, budget=1400)
    xi = choose_starting_xi(squad)

    assert len(xi["starters"]) == 11
    assert xi["captain"] in xi["starters"]
    assert xi["vice_captain"] in xi["starters"]
    assert xi["score_with_captain"] >= xi["score"]

