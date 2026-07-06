"""Tests for fatigue tracking and substitution logic."""

from __future__ import annotations

import numpy as np

from hoops.data.rosters import Player, Roster
from hoops.engine.fatigue import (
    FatigueTracker,
    apply_fatigue,
    check_substitutions,
    foul_trouble_hold,
    player_fatigue_threshold,
    player_importance,
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
                usage_pct=0.25 - i * 0.015,
                min_share=0.175 - i * 0.015)
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
    adjusted = apply_fatigue(p, fatigue=1.3)
    assert adjusted.ts_pct < 0.55


def test_apply_fatigue_increases_tov_pct():
    p = _player(1, "Tired", tov_pct=0.15)
    adjusted = apply_fatigue(p, fatigue=1.3)
    assert adjusted.tov_pct > 0.15


def test_apply_fatigue_zero_is_identity():
    p = _player(1, "Fresh", ts_pct=0.55, tov_pct=0.15)
    adjusted = apply_fatigue(p, fatigue=0.0)
    assert adjusted.ts_pct == p.ts_pct
    assert adjusted.tov_pct == p.tov_pct


def test_apply_fatigue_no_degradation_at_normal_workload():
    """Playing exactly (or under) your normal workload shouldn't degrade
    performance — only exceeding 100% should."""
    p = _player(1, "Normal", ts_pct=0.55, tov_pct=0.15)
    adjusted = apply_fatigue(p, fatigue=1.0)
    assert adjusted.ts_pct == p.ts_pct
    assert adjusted.tov_pct == p.tov_pct


def test_apply_fatigue_degrades_rebound_and_hustle_rates():
    p = _player(1, "Tired", orb_pct=3.0, drb_pct=8.0, stl_pct=2.5, blk_pct=0.8)
    adjusted = apply_fatigue(p, fatigue=1.3)
    assert adjusted.orb_pct < p.orb_pct
    assert adjusted.drb_pct < p.drb_pct
    assert adjusted.stl_pct < p.stl_pct
    assert adjusted.blk_pct < p.blk_pct


def test_apply_fatigue_increases_foul_rate():
    p = _player(1, "Tired", foul_rate=3.0)
    adjusted = apply_fatigue(p, fatigue=1.3)
    assert adjusted.foul_rate > p.foul_rate


def test_apply_fatigue_preserves_identity_fields():
    p = _player(1, "Tired")
    adjusted = apply_fatigue(p, fatigue=1.3)
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
    ft._fatigue[low_imp_pid] = 0.50  # above P4's threshold (~0.415)
    subs = check_substitutions(ls, ft, quarter=1, side=Side.HOME)
    assert len(subs) >= 1
    assert any(s.off_player_id == low_imp_pid for s in subs)


def test_high_minute_player_stays_longer():
    """A high-min_share player below their threshold is NOT subbed,
    even at fatigue that would bench a low-min_share player."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    ft = FatigueTracker(hr, ar)
    ls = _lineup_state(hr, ar)
    star_pid = hr.players[0].player_id  # threshold ~0.632
    ft._fatigue[star_pid] = 0.50  # below star's threshold but above P4's
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
    for _ in range(9):
        tracker.tick_cooldowns()
    assert tracker.on_cooldown(101)
    tracker.tick_cooldowns()  # 10th tick
    assert not tracker.on_cooldown(101)


def test_star_cooldown_is_shorter():
    """Stars get an 8-possession cooldown instead of 10."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    tracker = FatigueTracker(hr, ar)
    tracker.start_cooldown(101, is_star=True)
    assert tracker.on_cooldown(101)
    for _ in range(7):
        tracker.tick_cooldowns()
    assert tracker.on_cooldown(101)
    tracker.tick_cooldowns()  # 8th tick
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


# ---------------------------------------------------------------------------
# Per-player fatigue threshold tests
# ---------------------------------------------------------------------------

def test_player_fatigue_threshold_scales_with_min_share():
    high_min = _player(1, "Starter", min_share=0.175)
    low_min = _player(2, "Bench", min_share=0.05)
    assert player_fatigue_threshold(high_min) > player_fatigue_threshold(low_min)


def test_player_fatigue_threshold_none_uses_default():
    no_data = _player(1, "Unknown", min_share=None)
    default = _player(2, "Default", min_share=0.10)
    assert abs(player_fatigue_threshold(no_data) - player_fatigue_threshold(default)) < 1e-10


# ---------------------------------------------------------------------------
# Foul hold tests
# ---------------------------------------------------------------------------

def test_foul_hold_blocks_bench_selection():
    """A player on foul hold is not selected as a bench replacement."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    tracker = FatigueTracker(hr, ar)
    ls = _lineup_state(hr, ar)
    # Exhaust a starter
    tracker._fatigue[hr.players[4].player_id] = 0.95
    # Put best bench player on foul hold until Q3
    tracker.set_foul_hold(hr.players[5].player_id, until_quarter=3, until_seconds=600)
    subs = check_substitutions(ls, tracker, quarter=2, side=Side.HOME, seconds_left=300)
    if subs:
        assert subs[0].on_player_id != hr.players[5].player_id


def test_foul_hold_expires():
    """A foul hold expires when the game clock reaches the hold time."""
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    tracker = FatigueTracker(hr, ar)
    tracker.set_foul_hold(hr.players[5].player_id, until_quarter=3, until_seconds=600)
    assert tracker.on_foul_hold(hr.players[5].player_id, quarter=2, seconds_left=300)
    assert not tracker.on_foul_hold(hr.players[5].player_id, quarter=3, seconds_left=500)


def test_foul_trouble_hold_first_half_role_player():
    """Role players with 2 fouls in Q1 sit until Q3."""
    hold_q, hold_s = foul_trouble_hold(quarter=1, seconds_left=300, rank=3)
    assert hold_q == 3
    assert hold_s == 600


def test_foul_trouble_hold_first_half_star():
    """Stars with 2 fouls in the first half return with ~2 min left in Q2."""
    hold_q, hold_s = foul_trouble_hold(quarter=2, seconds_left=400, rank=0)
    assert hold_q == 3
    assert hold_s == 480


def test_foul_trouble_hold_q3_role_player():
    """Role player with 4 fouls in Q3 sits until Q4."""
    hold_q, hold_s = foul_trouble_hold(quarter=3, seconds_left=400, rank=3)
    assert hold_q == 4
    assert hold_s == 600


def test_foul_trouble_hold_q3_star():
    """Stars with 4 fouls in Q3 return with ~2 min left in Q3."""
    hold_q, hold_s = foul_trouble_hold(quarter=3, seconds_left=400, rank=0)
    assert hold_q == 3
    assert hold_s == 120


def test_foul_trouble_hold_q4_early():
    """Early Q4 foul trouble: stars sit ~2 min, role players ~3 min."""
    star_q, star_s = foul_trouble_hold(quarter=4, seconds_left=480, rank=0)
    role_q, role_s = foul_trouble_hold(quarter=4, seconds_left=480, rank=3)
    assert star_q == 4 and star_s == 360
    assert role_q == 4 and role_s == 300


def test_foul_trouble_hold_q4_late():
    """Late Q4 foul trouble: no hold, player returns immediately."""
    hold_q, hold_s = foul_trouble_hold(quarter=4, seconds_left=240, rank=3)
    assert hold_q == 4 and hold_s == 240


def test_is_fouled_out():
    tracker = FatigueTracker(_roster(1, "H"), _roster(2, "A"))
    pid = 101
    assert tracker.is_fouled_out(pid) is False
    for _ in range(4):
        tracker.add_foul(pid)
    assert tracker.is_fouled_out(pid) is False
    tracker.add_foul(pid)
    assert tracker.is_fouled_out(pid) is True
