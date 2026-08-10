"""Tests for defensive intensity (safe / normal / tight)."""

from __future__ import annotations

import pytest

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.data.rosters import Player
from hoops.engine.lineup_rates import compute_lineup_rates
from hoops.engine.machine import simulate_possession
from hoops.engine.matchup import apply_scheme
from hoops.engine.policy import (
    CoachPolicies,
    CoachPolicy,
    DefensiveIntensity,
    DefensiveScheme,
)
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


# --- identity -------------------------------------------------------------

def test_man_normal_is_identity():
    p = _team()
    assert apply_scheme(p, DefensiveScheme.MAN, DefensiveIntensity.NORMAL) is p


def test_normal_intensity_matches_intensity_free_call():
    # For ZONE/PRESS, passing NORMAL intensity must equal the default arg.
    p = _team()
    for scheme in (DefensiveScheme.ZONE, DefensiveScheme.PRESS):
        a = apply_scheme(p, scheme, DefensiveIntensity.NORMAL)
        b = apply_scheme(p, scheme)
        assert a.zone_efg == b.zone_efg
        assert a.shot_mix == b.shot_mix
        assert a.off_tov_pct == b.off_tov_pct
        assert a.off_fta_rate == b.off_fta_rate


# --- base intensity block (under MAN) -------------------------------------

def test_tight_man_lowers_efg_raises_tov_and_fta():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.MAN, DefensiveIntensity.TIGHT)
    assert out.off_tov_pct == pytest.approx(p.off_tov_pct + 0.010)
    assert out.off_fta_rate == pytest.approx(p.off_fta_rate + 0.04)
    assert out.zone_efg.rim == pytest.approx(p.zone_efg.rim - 0.015)
    assert out.zone_efg.mid == pytest.approx(p.zone_efg.mid - 0.015)
    assert out.zone_efg.three == pytest.approx(p.zone_efg.three - 0.015)


def test_safe_man_raises_efg_lowers_tov_and_fta():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.MAN, DefensiveIntensity.SAFE)
    assert out.off_tov_pct == pytest.approx(p.off_tov_pct - 0.010)
    assert out.off_fta_rate == pytest.approx(p.off_fta_rate - 0.04)
    assert out.zone_efg.rim == pytest.approx(p.zone_efg.rim + 0.015)


# --- scheme x intensity net effects ---------------------------------------

def test_press_tight_tov_stacks_to_net_5_5pp():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.PRESS, DefensiveIntensity.TIGHT)
    assert out.off_tov_pct == pytest.approx(p.off_tov_pct + 0.055)
    assert out.zone_efg.rim == pytest.approx(p.zone_efg.rim + 0.030)
    assert out.off_fta_rate == pytest.approx(p.off_fta_rate + 0.05)


def test_press_safe_reduces_rim_concession():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.PRESS, DefensiveIntensity.SAFE)
    assert out.off_tov_pct == pytest.approx(p.off_tov_pct + 0.010)
    assert out.zone_efg.rim == pytest.approx(p.zone_efg.rim + 0.020)


def test_zone_tight_three_share_and_efg():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.ZONE, DefensiveIntensity.TIGHT)
    assert out.shot_mix.three == pytest.approx(p.shot_mix.three - 0.04)
    assert out.shot_mix.mid == pytest.approx(p.shot_mix.mid + 0.04)
    assert out.zone_efg.three == pytest.approx(p.zone_efg.three - 0.025)


def test_zone_safe_three_share_and_efg():
    p = _team()
    out = apply_scheme(p, DefensiveScheme.ZONE, DefensiveIntensity.SAFE)
    assert out.shot_mix.three == pytest.approx(p.shot_mix.three - 0.02)
    assert out.zone_efg.three == pytest.approx(p.zone_efg.three - 0.015)


def test_tov_clip_at_045_press_tight():
    p = _team(name="hi-tov")
    p = p.model_copy(update={"off_tov_pct": 0.44})
    out = apply_scheme(p, DefensiveScheme.PRESS, DefensiveIntensity.TIGHT)
    assert out.off_tov_pct == 0.45


def test_shot_mix_stays_balanced():
    p = _team()
    for scheme in DefensiveScheme:
        for intensity in DefensiveIntensity:
            out = apply_scheme(p, scheme, intensity)
            total = out.shot_mix.rim + out.shot_mix.mid + out.shot_mix.three
            assert total == pytest.approx(1.0, abs=0.01)


# --- possession-level: the P0 lineup-branch fix ---------------------------

def _five_players() -> list[Player]:
    def mk(pid, name, **kw):
        base = dict(
            player_id=pid, name=name, minutes=400.0,
            fga=100, fg3a=30, fta=40, orb=15, drb=50,
            fouls=20, tov=15, ast=30, blk=5, stl=10,
            usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
            ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
            drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
        )
        base.update(kw)
        return Player(**base)
    return [mk(i, f"P{i}") for i in range(1, 6)]


def _count_turnovers(intensity: DefensiveIntensity, scheme: DefensiveScheme, n=600) -> int:
    """Simulate n independent possessions, count turnovers.

    The offense uses lineup rates (so p_tov comes from the lineup blend);
    the *defense's* scheme+intensity must still reach that branch (P0 fix).
    """
    team = _team()
    players = _five_players()
    off_lr = compute_lineup_rates(players, team)
    tovs = 0
    for seed in range(n):
        rng = make_rng(seed)
        state = GameState.initial(RULES, opening_possession=Side.HOME)
        policies = CoachPolicies(
            home=CoachPolicy(),  # offense
            away=CoachPolicy(scheme=scheme, intensity=intensity),  # defense
        )
        _st, evs = simulate_possession(
            state, team, team, rng, policies=policies,
            off_lineup_rates=off_lr, def_lineup_rates=off_lr,
        )
        if any(e.type == "turnover" for e in evs):
            tovs += 1
    return tovs


def test_press_tight_raises_turnovers_in_lineup_branch():
    # This is the P0 regression: the defensive scheme's TOV bump must reach
    # the lineup-rates p_tov, which it did NOT before the fix.
    base = _count_turnovers(DefensiveIntensity.NORMAL, DefensiveScheme.MAN)
    press = _count_turnovers(DefensiveIntensity.TIGHT, DefensiveScheme.PRESS)
    assert press > base


def test_plain_press_raises_turnovers_in_lineup_branch():
    base = _count_turnovers(DefensiveIntensity.NORMAL, DefensiveScheme.MAN)
    press = _count_turnovers(DefensiveIntensity.NORMAL, DefensiveScheme.PRESS)
    assert press > base
