"""Tests for tournament bracket UI screens."""

from __future__ import annotations
import pytest
from hoops.engine.bracket import Bracket
from hoops.ui.bracket_screens import render_bracket_round, render_bracket_region


def _8team_bracket_json() -> dict:
    """8-team bracket with 2 regions."""
    return {
        "season": "test",
        "regions": ["East", "West"],
        "num_games": 7,
        "games": [
            {"game_id": 1, "round": 1, "region": 0,
             "home_team_id": 101, "away_team_id": 108,
             "home_seed": 1, "away_seed": 8, "home_score": None, "away_score": None},
            {"game_id": 2, "round": 1, "region": 0,
             "home_team_id": 104, "away_team_id": 105,
             "home_seed": 4, "away_seed": 5, "home_score": None, "away_score": None},
            {"game_id": 3, "round": 1, "region": 1,
             "home_team_id": 102, "away_team_id": 107,
             "home_seed": 2, "away_seed": 7, "home_score": None, "away_score": None},
            {"game_id": 4, "round": 1, "region": 1,
             "home_team_id": 103, "away_team_id": 106,
             "home_seed": 3, "away_seed": 6, "home_score": None, "away_score": None},
            {"game_id": 5, "round": 2, "region": 0,
             "home_team_id": None, "away_team_id": None,
             "home_seed": None, "away_seed": None, "home_score": None, "away_score": None},
            {"game_id": 6, "round": 2, "region": 1,
             "home_team_id": None, "away_team_id": None,
             "home_seed": None, "away_seed": None, "home_score": None, "away_score": None},
            {"game_id": 7, "round": 3, "region": None,
             "home_team_id": None, "away_team_id": None,
             "home_seed": None, "away_seed": None, "home_score": None, "away_score": None},
        ],
    }


def test_render_bracket_region_unplayed():
    b = Bracket.from_json(_8team_bracket_json())
    b.populate_names({101: "Tigers", 108: "Hawks", 104: "Bears", 105: "Eagles"})
    text = render_bracket_region(b, region_idx=0, round_num=1)
    assert "Tigers" in text
    assert "Hawks" in text
    assert "vs" in text


def test_render_bracket_region_with_scores():
    b = Bracket.from_json(_8team_bracket_json())
    b.populate_names({
        101: "Tigers", 108: "Hawks", 104: "Bears", 105: "Eagles",
        102: "Lions", 107: "Wolves", 103: "Cougars", 106: "Panthers",
    })
    b.advance(0, winner_id=101, home_score=75, away_score=60)
    b.advance(1, winner_id=105, home_score=62, away_score=68)  # upset: 5 beats 4
    text = render_bracket_region(b, region_idx=0, round_num=1)
    assert "75" in text
    assert "60" in text
    assert "!" in text  # upset marker


def test_render_bracket_round_all_regions():
    b = Bracket.from_json(_8team_bracket_json())
    b.populate_names({
        101: "Tigers", 108: "Hawks", 104: "Bears", 105: "Eagles",
        102: "Lions", 107: "Wolves", 103: "Cougars", 106: "Panthers",
    })
    text = render_bracket_round(b, round_num=1)
    assert "East" in text
    assert "West" in text
    assert "Round of 64" in text


# --- Conference tournament bracket tests ---

def _conf_bracket() -> Bracket:
    """Conference bracket for UI testing."""
    data = {
        "season": "2023-24",
        "conference_name": "SEC Tournament",
        "regions": [],
        "games": [
            {"game_id": 1, "round": 1, "region": None,
             "home_team_id": 101, "away_team_id": 102,
             "home_seed": 3, "away_seed": 6,
             "home_score": None, "away_score": None},
            {"game_id": 2, "round": 1, "region": None,
             "home_team_id": 103, "away_team_id": 104,
             "home_seed": 4, "away_seed": 5,
             "home_score": None, "away_score": None},
            {"game_id": 3, "round": 2, "region": None,
             "home_team_id": None, "away_team_id": None,
             "home_seed": None, "away_seed": None,
             "home_score": None, "away_score": None},
        ],
    }
    b = Bracket.from_json(data)
    b.populate_names({101: "Tigers", 102: "Eagles", 103: "Bears", 104: "Wolves"})
    return b


def test_render_conf_bracket_no_regions():
    b = _conf_bracket()
    b.advance(0, winner_id=101, home_score=70, away_score=60)
    text = render_bracket_round(b, round_num=1)
    assert "Tigers" in text
    # Should NOT have region headers
    assert "Region" not in text


def test_render_conf_bracket_round_name():
    b = _conf_bracket()
    text = render_bracket_round(b, round_num=1)
    # Should use conference round naming, not "Round of 64"
    assert "Round of 64" not in text
    assert "Semifinal" in text  # round 1 of a 2-max-round bracket is Semifinal
