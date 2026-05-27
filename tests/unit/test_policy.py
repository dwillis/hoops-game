"""Phase 6 policy tests: each control changes the relevant draw.

The plan §6 verification target is ``unit tests confirm each control
changes the relevant distribution draw; end-to-end test plays a
one-possession-left-down-3 scenario and verifies the foul-up-3 policy
actually fouls``.

We exercise the engine with a fixed seed and synthetic priors, toggling
one policy field at a time, and assert directional behavior:

- Defensive scheme (zone/press): scheme-adjusted priors shift in the
  expected direction.
- 2-for-1 / hold-for-last: possession-length sampler responds correctly
  to the current ``GameState``.
- Foul-up-3: in a Q4 down-3 endgame state, the trailing defense fouls
  before the leading offense can score.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.engine.machine import (
    _sample_possession_seconds,
    _should_intentionally_foul,
    simulate_possession,
)
from hoops.engine.matchup import apply_scheme
from hoops.engine.policy import CoachPolicies, CoachPolicy, DefensiveScheme
from hoops.engine.sampling import make_rng
from hoops.engine.state import GameState, Side
from hoops.league import League
from hoops.rules import rules_for

RULES = rules_for(League.WBB, "2023-24")


def _team(name="T") -> TeamPriors:
    return TeamPriors(
        league=League.WBB, season="2023-24", team_id=1, team_name=name,
        pace=70, off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )


# --- defensive scheme ---------------------------------------------------------


def test_man_scheme_is_identity():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.MAN)
    assert out.shot_mix.three == p.shot_mix.three
    assert out.off_tov_pct == p.off_tov_pct


def test_zone_reduces_3pt_share_and_make_rate():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.ZONE)
    assert out.shot_mix.three < p.shot_mix.three
    assert out.shot_mix.mid > p.shot_mix.mid
    assert out.zone_efg.three < p.zone_efg.three


def test_press_increases_tov_and_rim_efg():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.PRESS)
    assert out.off_tov_pct > p.off_tov_pct
    assert out.zone_efg.rim > p.zone_efg.rim


# --- end-of-quarter timing ---------------------------------------------------


def _state(seconds_left: int, quarter: int = 4) -> GameState:
    s = GameState.initial(RULES)
    return replace(s, seconds_left=seconds_left, quarter=quarter)


def test_two_for_one_compresses_in_window():
    """In the 35-50s window with two_for_one ON, possessions are short."""
    p = _team()
    rng = make_rng(0)
    state = _state(seconds_left=42)

    on = CoachPolicy(two_for_one=True, hold_for_last=False)
    off = CoachPolicy(two_for_one=False, hold_for_last=False)
    secs_on = [_sample_possession_seconds(p, p, state, on, make_rng(s)) for s in range(50)]
    secs_off = [_sample_possession_seconds(p, p, state, off, make_rng(s)) for s in range(50)]
    assert max(secs_on) <= 17 + 1, max(secs_on)
    assert np.mean(secs_on) < np.mean(secs_off)


def test_hold_for_last_extends_when_clock_low():
    p = _team()
    state = _state(seconds_left=22, quarter=2)

    on = CoachPolicy(hold_for_last=True, two_for_one=False)
    off = CoachPolicy(hold_for_last=False, two_for_one=False)
    secs_on = [_sample_possession_seconds(p, p, state, on, make_rng(s)) for s in range(20)]
    secs_off = [_sample_possession_seconds(p, p, state, off, make_rng(s)) for s in range(20)]
    # All hold-on draws should consume nearly the entire remaining clock.
    assert min(secs_on) >= 20
    assert np.mean(secs_on) > np.mean(secs_off)


def test_two_for_one_outside_window_is_ignored():
    p = _team()
    on = CoachPolicy(two_for_one=True, hold_for_last=False)
    state = _state(seconds_left=200, quarter=1)
    secs = [_sample_possession_seconds(p, p, state, on, make_rng(s)) for s in range(20)]
    # Should be the normal pace-derived draw (~17 mean ± noise) not forced.
    assert max(secs) > 12  # at least some long ones


# --- intentional foul detection ----------------------------------------------


def test_foul_up_3_fires_in_q4_down_3():
    state = GameState.initial(RULES)
    state = replace(state, quarter=4, seconds_left=8, home_score=80, away_score=83)
    # AWAY is on offense (leading). HOME is down 3 and decides to foul.
    state = state.with_possession(Side.AWAY)
    home_policy = CoachPolicy(foul_when_down_3=True)
    reason = _should_intentionally_foul(state, off_side=Side.AWAY, def_policy=home_policy)
    assert reason is not None and "foul-up-3" in reason


def test_foul_up_3_does_not_fire_in_q3():
    state = GameState.initial(RULES)
    state = replace(state, quarter=3, seconds_left=8, home_score=80, away_score=83)
    home_policy = CoachPolicy(foul_when_down_3=True)
    reason = _should_intentionally_foul(state, off_side=Side.AWAY, def_policy=home_policy)
    assert reason is None


def test_foul_up_3_does_not_fire_when_leading():
    """Up 3 instead of down 3 — no foul."""
    state = GameState.initial(RULES)
    state = replace(state, quarter=4, seconds_left=8, home_score=83, away_score=80)
    home_policy = CoachPolicy(foul_when_down_3=True)
    # AWAY on offense. HOME is *up* 3.
    reason = _should_intentionally_foul(state, off_side=Side.AWAY, def_policy=home_policy)
    assert reason is None


def test_foul_up_3_disabled_by_default():
    state = GameState.initial(RULES)
    state = replace(state, quarter=4, seconds_left=5, home_score=80, away_score=83)
    default = CoachPolicy()
    reason = _should_intentionally_foul(state, off_side=Side.AWAY, def_policy=default)
    assert reason is None


# --- end-to-end: down-3 with foul-up-3 ON actually fouls ---------------------


def test_simulate_possession_triggers_foul_up_3():
    """Plan §6 verification: with home down 3 in Q4, ≤12s left, and the
    foul-up-3 policy ON for home, simulate_possession must fire a
    defensive foul before the leading offense can score."""
    rules = RULES
    state = GameState.initial(rules)
    # Home down 3, AWAY (leading) has the ball.
    state = replace(
        state, quarter=4, seconds_left=10,
        home_score=80, away_score=83,
        possession=Side.AWAY,
        # Defense (home) is in bonus from earlier fouls — this is the
        # realistic late-game scenario where fouling actually sends the
        # offense to the line.
        away_team_fouls_q=5,
    )
    policies = CoachPolicies(home=CoachPolicy(foul_when_down_3=True))

    rng = make_rng(seed=123)
    new_state, events = simulate_possession(
        state, _team("Home"), _team("Away"), rng, policies=policies,
    )
    foul_events = [e for e in events if e.type == "foul_personal"]
    assert len(foul_events) == 1
    foul = foul_events[0]
    assert foul.team is Side.HOME, "the defense (home) should be the team committing the foul"
    assert "foul-up-3" in foul.detail


def test_foul_up_3_off_does_not_foul():
    """Sanity: same scenario with foul-up-3 OFF should not produce an
    intentional foul (the only events are normal possession outcomes)."""
    rules = RULES
    state = GameState.initial(rules)
    state = replace(
        state, quarter=4, seconds_left=10,
        home_score=80, away_score=83,
        possession=Side.AWAY,
        away_team_fouls_q=5,
    )
    policies = CoachPolicies()  # all defaults; foul_when_down_3=False
    rng = make_rng(seed=123)
    _, events = simulate_possession(
        state, _team("Home"), _team("Away"), rng, policies=policies,
    )
    # No 'intentional' foul detail should appear in any event.
    assert not any(
        e.type == "foul_personal" and "intentional" in e.detail
        for e in events
    )


# --- timeouts ----------------------------------------------------------------


def test_default_policy_starts_with_four_timeouts():
    """Doc rules table: timeouts_per_team = 4 in WBB 2015-16+."""
    p = CoachPolicy()
    assert p.timeouts_remaining == 4


def test_policy_for_side():
    pols = CoachPolicies(
        home=CoachPolicy(scheme=DefensiveScheme.ZONE),
        away=CoachPolicy(scheme=DefensiveScheme.PRESS),
    )
    assert pols.for_side(Side.HOME).scheme is DefensiveScheme.ZONE
    assert pols.for_side(Side.AWAY).scheme is DefensiveScheme.PRESS
