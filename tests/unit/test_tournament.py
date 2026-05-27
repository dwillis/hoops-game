"""Tests for tournament auto-simulation."""

from __future__ import annotations

import numpy as np
import pytest

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.engine.bracket import Bracket
from hoops.engine.tournament import auto_sim_round
from hoops.league import League
from hoops.rules import Rules


def _priors(team_id: int, name: str) -> TeamPriors:
    return TeamPriors(
        league=League.WBB, season="2023-24",
        team_id=team_id, team_name=name, pace=70.0,
        shot_mix=ShotMix(rim=0.40, mid=0.25, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.40, three=0.35),
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.44, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )


def _mini_bracket() -> Bracket:
    data = {
        "season": "test",
        "regions": ["TestRegion"],
        "num_games": 3,
        "games": [
            {"game_id": 1, "round": 1, "region": 0,
             "home_team_id": 101, "away_team_id": 104,
             "home_seed": 1, "away_seed": 4,
             "home_score": None, "away_score": None},
            {"game_id": 2, "round": 1, "region": 0,
             "home_team_id": 102, "away_team_id": 103,
             "home_seed": 2, "away_seed": 3,
             "home_score": None, "away_score": None},
            {"game_id": 3, "round": 2, "region": 0,
             "home_team_id": None, "away_team_id": None,
             "home_seed": None, "away_seed": None,
             "home_score": None, "away_score": None},
        ],
    }
    return Bracket.from_json(data)


def test_auto_sim_round_plays_non_user_games():
    bracket = _mini_bracket()
    priors_map = {
        101: _priors(101, "A"), 102: _priors(102, "B"),
        103: _priors(103, "C"), 104: _priors(104, "D"),
    }
    rules = Rules(
        league=League.WBB, structure="quarters", quarter_minutes=10,
        shot_clock_seconds=30, three_point_distance_ft=22.146,
        bonus="per_quarter_5th_foul_two_shots", timeouts_per_team=4,
        ot_minutes=5, personal_foul_limit=5,
    )
    rng = np.random.default_rng(42)

    # User controls team 101 — game 0 skipped, game 1 auto-simmed
    results = auto_sim_round(bracket, 1, user_team_id=101, priors=priors_map,
                              rules=rules, rng=rng)

    assert bracket.games[0].winner_id is None  # user's game not played
    assert bracket.games[1].winner_id is not None  # non-user game played
    assert bracket.games[1].home_score is not None
    assert len(results) == 1


def test_auto_sim_round_skips_already_played():
    bracket = _mini_bracket()
    priors_map = {
        101: _priors(101, "A"), 102: _priors(102, "B"),
        103: _priors(103, "C"), 104: _priors(104, "D"),
    }
    rules = Rules(
        league=League.WBB, structure="quarters", quarter_minutes=10,
        shot_clock_seconds=30, three_point_distance_ft=22.146,
        bonus="per_quarter_5th_foul_two_shots", timeouts_per_team=4,
        ot_minutes=5, personal_foul_limit=5,
    )
    rng = np.random.default_rng(42)

    bracket.advance(1, winner_id=102, home_score=70, away_score=60)
    auto_sim_round(bracket, 1, user_team_id=101, priors=priors_map,
                   rules=rules, rng=rng)

    assert bracket.games[1].home_score == 70  # original result preserved


def test_auto_sim_round_returns_results_with_upset_info():
    bracket = _mini_bracket()
    priors_map = {
        101: _priors(101, "A"), 102: _priors(102, "B"),
        103: _priors(103, "C"), 104: _priors(104, "D"),
    }
    rules = Rules(
        league=League.WBB, structure="quarters", quarter_minutes=10,
        shot_clock_seconds=30, three_point_distance_ft=22.146,
        bonus="per_quarter_5th_foul_two_shots", timeouts_per_team=4,
        ot_minutes=5, personal_foul_limit=5,
    )
    rng = np.random.default_rng(42)

    results = auto_sim_round(bracket, 1, user_team_id=101, priors=priors_map,
                              rules=rules, rng=rng)

    assert len(results) == 1
    r = results[0]
    assert "home_score" in r
    assert "away_score" in r
    assert "is_upset" in r
    assert r["home_seed"] == 2
    assert r["away_seed"] == 3
