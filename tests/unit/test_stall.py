"""Tests for the STALL / SEMISTALL offensive schemes."""

from __future__ import annotations

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.engine.machine import _sample_possession_seconds
from hoops.engine.policy import CoachPolicy, OffensiveScheme
from hoops.engine.sampling import make_rng
from hoops.engine.state import GameState, Side
from hoops.league import League
from hoops.rules import rules_for

RULES = rules_for(League.WBB, "2023-24")


def _team() -> TeamPriors:
    return TeamPriors(
        league=League.WBB, season="2023-24", team_id=1, team_name="T",
        pace=70, off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )


def _durations(policy: CoachPolicy, seconds_left: int, quarter: int = 2, n=200):
    import dataclasses
    team = _team()
    out = []
    st = GameState.initial(RULES, opening_possession=Side.HOME)
    # GameState is a frozen dataclass; rebuild with the fields we need.
    st = dataclasses.replace(st, seconds_left=seconds_left, quarter=quarter)
    for seed in range(n):
        rng = make_rng(seed)
        out.append(_sample_possession_seconds(team, team, st, policy, rng))
    return out


def test_stall_burns_clock_to_shot_clock_window():
    shot_clock = RULES.shot_clock_seconds  # 30
    policy = CoachPolicy(off_scheme=OffensiveScheme.STALL)
    ds = _durations(policy, seconds_left=600, quarter=2)
    # burn = shot_clock - 1 - integers(0,4) -> 26..29 on a 30s clock
    assert min(ds) >= shot_clock - 4
    assert max(ds) <= shot_clock - 1
    assert len(set(ds)) > 1  # jitter present


def test_stall_capped_by_seconds_left():
    policy = CoachPolicy(off_scheme=OffensiveScheme.STALL)
    ds = _durations(policy, seconds_left=10, quarter=4)
    assert all(d <= 10 for d in ds)


def test_stall_overrides_two_for_one():
    # 42s left is squarely in the two-for-one window; STALL must win.
    policy = CoachPolicy(off_scheme=OffensiveScheme.STALL, two_for_one=True)
    ds = _durations(policy, seconds_left=42, quarter=2)
    # Two-for-one would compress to <=17; STALL burns >=26.
    assert min(ds) >= 26


def test_normal_two_for_one_still_compresses():
    # Sanity: without STALL, the two-for-one path still fires.
    policy = CoachPolicy(off_scheme=OffensiveScheme.NORMAL, two_for_one=True)
    ds = _durations(policy, seconds_left=42, quarter=2)
    assert max(ds) <= 18
