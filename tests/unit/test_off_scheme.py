"""Tests for offensive scheme adjustments."""

from __future__ import annotations

import pytest

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.engine.matchup import apply_off_scheme
from hoops.engine.policy import OffensiveScheme
from hoops.league import League


def _team_priors(**overrides) -> TeamPriors:
    base = dict(
        league=League.WBB, season="2023-24",
        team_id=1, team_name="Test",
        pace=70.0,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.38, three=0.33),
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.44, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )
    base.update(overrides)
    return TeamPriors(**base)


def test_normal_no_change():
    p = _team_priors()
    result = apply_off_scheme(p, OffensiveScheme.NORMAL)
    assert result.pace == p.pace
    assert result.off_tov_pct == p.off_tov_pct
    assert result.shot_mix == p.shot_mix
    assert result.zone_efg == p.zone_efg


def test_hurry_up_pace_and_tov():
    p = _team_priors()
    result = apply_off_scheme(p, OffensiveScheme.HURRY_UP)
    assert result.pace == p.pace + 3.0
    assert result.off_tov_pct == pytest.approx(p.off_tov_pct + 0.015)
    assert result.shot_mix == p.shot_mix
    assert result.zone_efg == p.zone_efg


def test_slow_down_pace_tov_and_efg():
    p = _team_priors()
    result = apply_off_scheme(p, OffensiveScheme.SLOW_DOWN)
    assert result.pace == p.pace - 3.0
    assert result.off_tov_pct == pytest.approx(p.off_tov_pct - 0.015)
    assert result.zone_efg.rim == pytest.approx(p.zone_efg.rim - 0.01)
    assert result.zone_efg.mid == pytest.approx(p.zone_efg.mid - 0.01)
    assert result.zone_efg.three == pytest.approx(p.zone_efg.three - 0.01)


def test_three_point_shot_mix_and_efg():
    p = _team_priors()
    result = apply_off_scheme(p, OffensiveScheme.THREE_POINT)
    assert result.shot_mix.three == pytest.approx(p.shot_mix.three + 0.05)
    assert result.shot_mix.mid == pytest.approx(p.shot_mix.mid - 0.05)
    assert result.shot_mix.rim == p.shot_mix.rim
    assert result.zone_efg.three == pytest.approx(p.zone_efg.three - 0.01)
    assert result.zone_efg.rim == p.zone_efg.rim
    assert result.pace == p.pace


def test_hurry_up_tov_clipped():
    p = _team_priors(off_tov_pct=0.395)
    result = apply_off_scheme(p, OffensiveScheme.HURRY_UP)
    assert result.off_tov_pct == 0.40


def test_slow_down_tov_clipped():
    p = _team_priors(off_tov_pct=0.06)
    result = apply_off_scheme(p, OffensiveScheme.SLOW_DOWN)
    assert result.off_tov_pct == 0.05
