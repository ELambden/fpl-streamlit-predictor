from fpl_decision_lab.features import add_projection_features


def test_projection_features_are_positive_and_five_gameweek_ready() -> None:
    rows = [
        {
            "player_id": "1",
            "position": "MID",
            "now_cost": 75,
            "selected_by_percent": 10,
            "total_points": 30,
            "minutes": 450,
            "starts": 5,
            "expected_goal_involvements": 4.0,
            "form": 5,
            "points_per_game": 6,
            "ep_next": 5.5,
            "fixture_difficulty_next5": 2,
            "opponent_count_next3": 3,
            "opponent_count_next5": 5,
        }
    ]

    result = add_projection_features(rows)[0]

    assert result["points_per_90"] == 6
    assert result["fixture_ease"] == 4
    assert result["predicted_next_gw"] > 0
    assert result["predicted_next5"] > result["predicted_next3"] > result["predicted_next_gw"]
