"""Tests for per-player ball distribution (coach-set shot shares)."""

from __future__ import annotations

from collections import Counter

import pytest

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.data.rosters import Player
from hoops.engine.lineup_rates import compute_lineup_rates, sample_shooter
from hoops.engine.sampling import make_rng
from hoops.league import League

SEASON = "2023-24"


def _team() -> TeamPriors:
    return TeamPriors(
        league=League.WBB, season=SEASON, team_id=1, team_name="T",
        pace=70.0, off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )


def _player(pid, **kw):
    base = dict(
        player_id=pid, name=f"P{pid}", minutes=400.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
    )
    base.update(kw)
    return Player(**base)


def _five():
    return [_player(i) for i in range(1, 6)]


# --- identity -------------------------------------------------------------

def test_none_distribution_is_identity():
    team = _team()
    players = _five()
    a = compute_lineup_rates(players, team, shot_distribution=None)
    b = compute_lineup_rates(players, team)
    assert a.shot_weights is None
    assert b.shot_weights is None
    assert a.tov_pct == b.tov_pct
    assert [w for _, w in a.shooters] == [w for _, w in b.shooters]


def test_empty_distribution_is_identity():
    team = _team()
    lr = compute_lineup_rates(_five(), team, shot_distribution={})
    assert lr.shot_weights is None


def test_distribution_for_only_benched_players_is_identity():
    team = _team()
    # player_id 99 is not on court
    lr = compute_lineup_rates(_five(), team, shot_distribution={99: 0.5})
    assert lr.shot_weights is None


# --- normalization --------------------------------------------------------

def test_set_player_gets_requested_share():
    team = _team()
    lr = compute_lineup_rates(_five(), team, shot_distribution={1: 0.50})
    assert lr.shot_weights is not None
    assert sum(lr.shot_weights) == pytest.approx(1.0)
    assert lr.shot_weights[0] == pytest.approx(0.50, abs=0.02)


def test_unset_players_split_leftover_by_usage():
    team = _team()
    # give P1 0.40; the other four (equal usage) split 0.60 evenly
    lr = compute_lineup_rates(_five(), team, shot_distribution={1: 0.40})
    others = lr.shot_weights[1:]
    assert all(o == pytest.approx(0.15, abs=0.01) for o in others)


def test_share_floor_and_ceiling():
    team = _team()
    # Request an impossible 0.95 for P1 and 0.0 for P2.
    lr = compute_lineup_rates(_five(), team, shot_distribution={1: 0.95, 2: 0.0})
    assert lr.shot_weights is not None
    assert sum(lr.shot_weights) == pytest.approx(1.0)
    # P1 was clamped down from 0.95 (can't take nearly every shot).
    assert lr.shot_weights[0] < 0.95
    # P2 was floored up from 0.0 (can't be fully frozen out).
    assert lr.shot_weights[1] > 0.0


# --- sampling frequency ---------------------------------------------------

def test_sample_shooter_respects_distribution():
    team = _team()
    lr = compute_lineup_rates(_five(), team, shot_distribution={1: 0.50})
    rng = make_rng(0)
    counts = Counter()
    n = 6000
    for _ in range(n):
        counts[sample_shooter(lr, rng).player_id] += 1
    assert counts[1] / n == pytest.approx(0.50, abs=0.03)


# --- TS penalty -----------------------------------------------------------

def test_ts_penalty_applied_to_overused_shooter():
    team = _team()
    lr = compute_lineup_rates(_five(), team, shot_distribution={1: 0.50})
    # P1 is pushed from 0.20 natural to 0.50 -> +0.30 over, capped at +0.10
    # -> penalty 0.6 * 0.10 = 0.06 TS.
    shooter1 = next(p for p, _ in lr.shooters if p.player_id == 1)
    base = compute_lineup_rates(_five(), team)
    base1 = next(p for p, _ in base.shooters if p.player_id == 1)
    assert shooter1.ts_pct == pytest.approx(base1.ts_pct - 0.06, abs=1e-6)


def test_no_penalty_for_underused_shooter():
    team = _team()
    lr = compute_lineup_rates(_five(), team, shot_distribution={1: 0.50})
    # A player who drops below natural usage keeps full efficiency.
    base = compute_lineup_rates(_five(), team)
    for pid in (2, 3, 4, 5):
        cur = next(p for p, _ in lr.shooters if p.player_id == pid)
        b = next(p for p, _ in base.shooters if p.player_id == pid)
        assert cur.ts_pct == pytest.approx(b.ts_pct, abs=1e-6)


# --- team blends must be unaffected ---------------------------------------

def test_team_blends_unchanged_by_distribution():
    team = _team()
    a = compute_lineup_rates(_five(), team, shot_distribution={1: 0.50})
    b = compute_lineup_rates(_five(), team)
    assert a.tov_pct == pytest.approx(b.tov_pct)
    assert a.orb_pct == pytest.approx(b.orb_pct)
    assert a.drb_pct == pytest.approx(b.drb_pct)
    assert a.foul_rate == pytest.approx(b.foul_rate)
    assert a.ft_pct == pytest.approx(b.ft_pct)
    assert a.pace_adj == pytest.approx(b.pace_adj)
