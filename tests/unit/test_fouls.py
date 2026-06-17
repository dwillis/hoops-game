"""Phase 3 rule tests for the per-quarter bonus.

The plan calls these out as the load-bearing rule tests for v0:

- 5th team foul in Q1 puts the other team in bonus.
- Team-foul counter resets at start of Q2.
- 6th team foul in Q2 (i.e., on a fresh quarter) also triggers bonus.
"""

from __future__ import annotations

from dataclasses import replace

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.engine.clock import end_period
from hoops.engine.fouls import is_in_bonus
from hoops.engine.machine import _shot_foul_prob
from hoops.engine.state import GameState, Side
from hoops.league import League
from hoops.rules import rules_for


def _team() -> TeamPriors:
    return TeamPriors(
        league=League.WBB, season="2023-24", team_id=1, team_name="Test",
        pace=70.0,
        off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )


def _state() -> GameState:
    return GameState.initial(rules_for(League.WBB, "2023-24"))


def test_no_bonus_with_zero_fouls():
    s = _state()
    assert not is_in_bonus(s, Side.HOME)
    assert not is_in_bonus(s, Side.AWAY)


def test_fourth_foul_does_not_trigger_bonus():
    s = _state()
    for _ in range(4):
        s = s.add_team_foul(Side.AWAY)
    assert not is_in_bonus(s, Side.HOME)


def test_fifth_team_foul_in_q1_puts_opponent_in_bonus():
    s = _state()
    for _ in range(5):
        s = s.add_team_foul(Side.AWAY)
    assert is_in_bonus(s, Side.HOME)
    assert not is_in_bonus(s, Side.AWAY)


def test_team_foul_counter_resets_at_quarter():
    s = _state()
    for _ in range(5):
        s = s.add_team_foul(Side.AWAY)
    s = replace(s, seconds_left=0)
    s, _ = end_period(s)
    assert s.quarter == 2
    assert s.away_team_fouls_q == 0
    assert s.home_team_fouls_q == 0
    assert not is_in_bonus(s, Side.HOME)


def test_q2_fouls_accumulate_independently():
    """A fresh quarter requires its own 5 fouls to trigger bonus."""
    s = _state()
    # Burn 5 fouls in Q1, end the quarter
    for _ in range(5):
        s = s.add_team_foul(Side.AWAY)
    s = replace(s, seconds_left=0)
    s, _ = end_period(s)
    # Now in Q2; 4 fouls should NOT be enough
    for _ in range(4):
        s = s.add_team_foul(Side.AWAY)
    assert not is_in_bonus(s, Side.HOME)
    # 5th foul of Q2 triggers
    s = s.add_team_foul(Side.AWAY)
    assert is_in_bonus(s, Side.HOME)


def test_only_offensive_team_in_bonus_via_defenders_fouls():
    """Bonus is asymmetric: the team with fouls is the one *not* in the bonus."""
    s = _state()
    for _ in range(5):
        s = s.add_team_foul(Side.HOME)
    assert is_in_bonus(s, Side.AWAY)
    assert not is_in_bonus(s, Side.HOME)


# --- zone-based shooting foul probability ------------------------------------


def test_three_point_foul_prob_lower_than_rim():
    team = _team()
    p_rim = _shot_foul_prob(team, "rim")
    p_mid = _shot_foul_prob(team, "mid")
    p_three = _shot_foul_prob(team, "three")
    assert p_rim > p_mid > p_three
    assert p_three < p_rim / 5


def test_zone_foul_probs_preserve_aggregate_fta_rate():
    """Mix-weighted expected FTA should equal the team's off_fta_rate."""
    team = _team()
    efg = {"rim": team.zone_efg.rim, "mid": team.zone_efg.mid, "three": team.zone_efg.three}
    mix = {"rim": team.shot_mix.rim, "mid": team.shot_mix.mid, "three": team.shot_mix.three}
    eft = {
        "rim": 2 - efg["rim"], "mid": 2 - efg["mid"],
        "three": 3 - 2 * efg["three"],
    }
    total_fta_rate = sum(
        mix[z] * _shot_foul_prob(team, z) * eft[z]
        for z in ("rim", "mid", "three")
    )
    assert abs(total_fta_rate - team.off_fta_rate) < 1e-10
