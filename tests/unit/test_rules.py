import pytest

from hoops.league import League
from hoops.rules import Rules, UnsupportedSeasonError, available_seasons, rules_for


def test_loads_2023_24_wbb_rules():
    r = rules_for(League.WBB, "2023-24")
    assert isinstance(r, Rules)
    assert r.league == League.WBB
    assert r.structure == "quarters"
    assert r.quarter_minutes == 10
    assert r.shot_clock_seconds == 30
    assert r.bonus == "per_quarter_5th_foul_two_shots"
    assert r.timeouts_per_team == 4
    assert r.personal_foul_limit == 5


def test_three_point_line_moves_in_2021_22():
    pre_move = rules_for(League.WBB, "2020-21").three_point_distance_ft
    post_move = rules_for(League.WBB, "2021-22").three_point_distance_ft
    assert pre_move == pytest.approx(20.75)
    assert post_move == pytest.approx(22.146)


def test_three_point_line_was_moved_at_22_not_15():
    assert rules_for(League.WBB, "2015-16").three_point_distance_ft == pytest.approx(20.75)
    assert rules_for(League.WBB, "2024-25").three_point_distance_ft == pytest.approx(22.146)


def test_pre_2015_16_seasons_rejected_in_v1():
    with pytest.raises(UnsupportedSeasonError):
        rules_for(League.WBB, "2014-15")


def test_unknown_season_rejected():
    with pytest.raises(UnsupportedSeasonError):
        rules_for(League.WBB, "1999-00")


def test_mbb_not_yet_supported():
    with pytest.raises(UnsupportedSeasonError):
        rules_for(League.MBB, "2023-24")


def test_available_seasons_covers_2015_16_through_2025_26():
    seasons = available_seasons(League.WBB)
    assert "2015-16" in seasons
    assert "2023-24" in seasons
    assert "2025-26" in seasons
    assert "2014-15" not in seasons
    assert seasons == sorted(seasons)
