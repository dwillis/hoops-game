"""Tests for home court advantage adjustments."""

from __future__ import annotations

import pytest

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.engine.matchup import apply_hca
from hoops.league import League


def _team_priors(**overrides) -> TeamPriors:
    base = dict(
        league=League.WBB, season="2023-24",
        team_id=1, team_name="Test",
        pace=70.0,
        shot_mix=ShotMix(rim=0.40, mid=0.25, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.40, three=0.35),
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.44, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )
    base.update(overrides)
    return TeamPriors(**base)


def test_apply_hca_reduces_ft_pct():
    priors = _team_priors(off_ft_pct=0.72)
    adjusted = apply_hca(priors)
    assert adjusted.off_ft_pct == pytest.approx(0.70, abs=0.001)


def test_apply_hca_increases_tov_pct():
    priors = _team_priors(off_tov_pct=0.18)
    adjusted = apply_hca(priors)
    assert adjusted.off_tov_pct == pytest.approx(0.19, abs=0.001)


def test_apply_hca_reduces_efg():
    priors = _team_priors(off_efg=0.48)
    adjusted = apply_hca(priors)
    assert adjusted.off_efg == pytest.approx(0.47, abs=0.001)


def test_apply_hca_scales_zone_efg_proportionally():
    priors = _team_priors(off_efg=0.48, zone_efg=ZoneEFG(rim=0.60, mid=0.40, three=0.35))
    adjusted = apply_hca(priors)
    ratio = 0.47 / 0.48
    assert adjusted.zone_efg.rim == pytest.approx(0.60 * ratio, abs=0.005)
    assert adjusted.zone_efg.mid == pytest.approx(0.40 * ratio, abs=0.005)
    assert adjusted.zone_efg.three == pytest.approx(0.35 * ratio, abs=0.005)


def test_apply_hca_clamps_low_ft_pct():
    priors = _team_priors(off_ft_pct=0.06)
    adjusted = apply_hca(priors)
    assert adjusted.off_ft_pct >= 0.05


def test_apply_hca_preserves_identity_fields():
    priors = _team_priors(team_id=42, team_name="Tigers")
    adjusted = apply_hca(priors)
    assert adjusted.team_id == 42
    assert adjusted.team_name == "Tigers"
    assert adjusted.pace == priors.pace
    assert adjusted.def_efg == priors.def_efg


def test_apply_hca_does_not_modify_original():
    priors = _team_priors()
    original_ft = priors.off_ft_pct
    _ = apply_hca(priors)
    assert priors.off_ft_pct == original_ft


# ---------------------------------------------------------------------------
# Integration: HCA applied in InteractiveGame
# ---------------------------------------------------------------------------

from hoops.data.rosters import Player, Roster
from hoops.engine.interactive import InteractiveGame
from hoops.engine.sampling import make_rng
from hoops.engine.state import Side
from hoops.rules import Rules


def _player(pid, name, minutes=200.0, **kw):
    base = dict(
        player_id=pid, name=name, minutes=minutes,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
        min_share=0.28,
    )
    base.update(kw)
    return Player(**base)


def _roster(team_id, name, n=10):
    players = tuple(
        _player(team_id * 100 + i, f"{name}_P{i}",
                usage_pct=0.25 - i * 0.015,
                min_share=0.30 - i * 0.02)
        for i in range(n)
    )
    return Roster(team_id=team_id, team_name=name, players=players)


def test_interactive_game_applies_hca_to_away():
    mix = ShotMix(rim=0.40, mid=0.25, three=0.35)
    efg = ZoneEFG(rim=0.55, mid=0.40, three=0.35)
    home_priors = TeamPriors(
        league=League.WBB, season="2023-24",
        team_id=1, team_name="Home", pace=70.0,
        shot_mix=mix, zone_efg=efg,
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.44, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )
    away_priors = TeamPriors(
        league=League.WBB, season="2023-24",
        team_id=2, team_name="Away", pace=68.0,
        shot_mix=mix, zone_efg=efg,
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.44, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )
    rules = Rules(
        league=League.WBB, structure="quarters", quarter_minutes=10,
        shot_clock_seconds=30, three_point_distance_ft=22.146,
        bonus="per_quarter_5th_foul_two_shots", timeouts_per_team=4,
        ot_minutes=5, personal_foul_limit=5,
    )
    game = InteractiveGame(
        home_priors, away_priors, rules, make_rng(seed=42),
        _roster(1, "Home"), _roster(2, "Away"),
        human_side=Side.HOME,
    )
    # Away team should have HCA penalty.
    assert game.away_priors.off_ft_pct < 0.72
    assert game.away_priors.off_tov_pct > 0.18
    assert game.away_priors.off_efg < 0.48
    # Home team unchanged (no league priors → no matchup adjust).
    assert game.home_priors.off_ft_pct == 0.72


def test_neutral_site_skips_hca():
    mix = ShotMix(rim=0.40, mid=0.25, three=0.35)
    efg = ZoneEFG(rim=0.55, mid=0.40, three=0.35)
    home_priors = TeamPriors(
        league=League.WBB, season="2023-24",
        team_id=1, team_name="Home", pace=70.0,
        shot_mix=mix, zone_efg=efg,
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.44, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )
    away_priors = TeamPriors(
        league=League.WBB, season="2023-24",
        team_id=2, team_name="Away", pace=68.0,
        shot_mix=mix, zone_efg=efg,
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.44, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )
    rules = Rules(
        league=League.WBB, structure="quarters", quarter_minutes=10,
        shot_clock_seconds=30, three_point_distance_ft=22.146,
        bonus="per_quarter_5th_foul_two_shots", timeouts_per_team=4,
        ot_minutes=5, personal_foul_limit=5,
    )
    game = InteractiveGame(
        home_priors, away_priors, rules, make_rng(seed=42),
        _roster(1, "Home"), _roster(2, "Away"),
        human_side=Side.HOME,
        neutral_site=True,
    )
    # Away team should NOT have HCA penalty on neutral site.
    assert game.away_priors.off_ft_pct == 0.72
    assert game.away_priors.off_tov_pct == 0.18
    assert game.away_priors.off_efg == 0.48
