"""Phase 3 rule tests for the per-quarter bonus.

The plan calls these out as the load-bearing rule tests for v0:

- 5th team foul in Q1 puts the other team in bonus.
- Team-foul counter resets at start of Q2.
- 6th team foul in Q2 (i.e., on a fresh quarter) also triggers bonus.
"""

from __future__ import annotations

from dataclasses import replace

from hoops.engine.clock import end_period
from hoops.engine.fouls import is_in_bonus
from hoops.engine.state import GameState, Side
from hoops.league import League
from hoops.rules import rules_for


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
