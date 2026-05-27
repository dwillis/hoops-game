"""Tests for fatigue tracking and substitution logic."""

from __future__ import annotations

import numpy as np
import pytest

from hoops.data.rosters import Player, Roster
from hoops.engine.fatigue import (
    FatigueTracker, player_importance, apply_fatigue,
    check_substitutions, SubEvent,
)
from hoops.engine.state import Side
from hoops.ui.lineup import LineupState


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


def _roster(team_id, name, n=12):
    players = tuple(
        _player(team_id * 100 + i, f"{name}_P{i}",
                usage_pct=0.25 - i * 0.02,
                min_share=0.30 - i * 0.02)
        for i in range(n)
    )
    return Roster(team_id=team_id, team_name=name, players=players)


def test_fatigue_tracker_initializes_at_zero():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    for p in hr.players:
        assert ft.fatigue(p.player_id) == 0.0
        assert ft.fouls(p.player_id) == 0


def test_fatigue_accumulates_for_on_court():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    on_court_ids = [p.player_id for p in hr.players[:5]]
    ft.tick(on_court_ids, duration_seconds=20)
    for pid in on_court_ids:
        assert ft.fatigue(pid) > 0.0


def test_fatigue_decays_for_bench():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    bench_pid = hr.players[5].player_id
    ft._fatigue[bench_pid] = 0.5
    bench_ids = [p.player_id for p in hr.players[5:]]
    ft.rest(bench_ids, duration_seconds=20)
    assert ft.fatigue(bench_pid) < 0.5


def test_fatigue_never_goes_negative():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    bench_ids = [p.player_id for p in hr.players[5:]]
    ft.rest(bench_ids, duration_seconds=1000)
    for pid in bench_ids:
        assert ft.fatigue(pid) >= 0.0


def test_add_foul_increments():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    pid = hr.players[0].player_id
    ft.add_foul(pid)
    assert ft.fouls(pid) == 1
    ft.add_foul(pid)
    assert ft.fouls(pid) == 2


def test_player_importance_usage_weighted():
    star = _player(1, "Star", usage_pct=0.30, min_share=0.35)
    bench = _player(2, "Bench", usage_pct=0.10, min_share=0.10)
    assert player_importance(star) > player_importance(bench)


def test_apply_fatigue_degrades_ts_pct():
    p = _player(1, "Tired", ts_pct=0.55, tov_pct=0.15)
    adjusted = apply_fatigue(p, fatigue=0.8)
    assert adjusted.ts_pct < 0.55


def test_apply_fatigue_increases_tov_pct():
    p = _player(1, "Tired", tov_pct=0.15)
    adjusted = apply_fatigue(p, fatigue=0.8)
    assert adjusted.tov_pct > 0.15


def test_apply_fatigue_zero_is_identity():
    p = _player(1, "Fresh", ts_pct=0.55, tov_pct=0.15)
    adjusted = apply_fatigue(p, fatigue=0.0)
    assert adjusted.ts_pct == p.ts_pct
    assert adjusted.tov_pct == p.tov_pct


def test_apply_fatigue_preserves_non_affected_fields():
    p = _player(1, "Tired", orb_pct=3.0, drb_pct=8.0)
    adjusted = apply_fatigue(p, fatigue=0.8)
    assert adjusted.orb_pct == p.orb_pct
    assert adjusted.drb_pct == p.drb_pct
    assert adjusted.name == p.name
    assert adjusted.player_id == p.player_id


def test_apply_fatigue_handles_none_rates():
    raw = Player(
        player_id=1, name="Raw", minutes=200.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30,
    )
    adjusted = apply_fatigue(raw, fatigue=0.5)
    assert adjusted.ts_pct is None
    assert adjusted.tov_pct is None


def test_player_importance_handles_none():
    raw = Player(
        player_id=1, name="Raw", minutes=200.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30,
    )
    imp = player_importance(raw)
    assert imp > 0


# ---------------------------------------------------------------------------
# Substitution engine tests
# ---------------------------------------------------------------------------

def _lineup_state(hr, ar):
    rng = np.random.default_rng(42)
    return LineupState.with_default_starters(hr, ar, rng)


def test_no_subs_when_fresh():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    ls = _lineup_state(hr, ar)
    subs = check_substitutions(ls, ft, quarter=1, side=Side.HOME)
    assert subs == []


def test_sub_when_fatigued():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    ls = _lineup_state(hr, ar)
    low_imp_pid = hr.players[4].player_id
    ft._fatigue[low_imp_pid] = 0.80
    subs = check_substitutions(ls, ft, quarter=1, side=Side.HOME)
    assert len(subs) >= 1
    assert any(s.off_player_id == low_imp_pid for s in subs)


def test_star_stays_longer_when_fatigued():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    ls = _lineup_state(hr, ar)
    star_pid = hr.players[0].player_id
    ft._fatigue[star_pid] = 0.72
    subs = check_substitutions(ls, ft, quarter=1, side=Side.HOME)
    assert not any(s.off_player_id == star_pid for s in subs)


def test_foul_trouble_first_half():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    ls = _lineup_state(hr, ar)
    role_pid = hr.players[3].player_id
    ft.add_foul(role_pid)
    ft.add_foul(role_pid)
    subs = check_substitutions(ls, ft, quarter=2, side=Side.HOME)
    assert any(s.off_player_id == role_pid for s in subs)


def test_fouled_out_always_subbed():
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    ls = _lineup_state(hr, ar)
    pid = hr.players[0].player_id
    for _ in range(5):
        ft.add_foul(pid)
    subs = check_substitutions(ls, ft, quarter=4, side=Side.HOME)
    assert any(s.off_player_id == pid for s in subs)


def test_no_sub_when_no_bench_available():
    hr = Roster(team_id=1, team_name="Small", players=tuple(
        _player(100 + i, f"P{i}", usage_pct=0.20, min_share=0.20) for i in range(5)
    ))
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    rng = np.random.default_rng(42)
    ls = LineupState.with_default_starters(hr, ar, rng)
    ft._fatigue[hr.players[0].player_id] = 0.95
    subs = check_substitutions(ls, ft, quarter=1, side=Side.HOME)
    assert subs == []


def test_sub_cooldown_prevents_immediate_reentry():
    """A recently-subbed-out player can't re-enter until cooldown expires."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    tracker = FatigueTracker(hr, ar)
    tracker.start_cooldown(101)
    assert tracker.on_cooldown(101)
    tracker.tick_cooldowns()
    assert tracker.on_cooldown(101)
    tracker.tick_cooldowns()  # 2nd tick
    assert not tracker.on_cooldown(101)


def test_star_cooldown_is_shorter():
    """Stars get a 1-possession cooldown instead of 2."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    tracker = FatigueTracker(hr, ar)
    tracker.start_cooldown(101, is_star=True)
    assert tracker.on_cooldown(101)
    tracker.tick_cooldowns()  # 1st tick
    assert not tracker.on_cooldown(101)


def test_cooldown_skips_bench_in_check_substitutions():
    """Players on cooldown on the bench are not selected as replacements."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ls = _lineup_state(hr, ar)
    tracker = FatigueTracker(hr, ar)
    # Exhaust the starter so they need subbing
    tracker._fatigue[hr.players[4].player_id] = 0.95
    # Put the best bench player on cooldown
    tracker.start_cooldown(hr.players[5].player_id)
    subs = check_substitutions(ls, tracker, quarter=1, side=Side.HOME)
    if subs:
        assert subs[0].on_player_id != hr.players[5].player_id


def test_cooldown_does_not_block_fouled_out():
    """A fouled-out player is always subbed regardless of cooldown."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ls = _lineup_state(hr, ar)
    tracker = FatigueTracker(hr, ar)
    pid = hr.players[0].player_id
    # Put them on cooldown AND foul them out
    tracker.start_cooldown(pid)
    for _ in range(5):
        tracker.add_foul(pid)
    subs = check_substitutions(ls, tracker, quarter=1, side=Side.HOME)
    fouled_out_sub = [s for s in subs if s.off_player_id == pid]
    assert len(fouled_out_sub) == 1
