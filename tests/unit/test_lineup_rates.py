"""Tests for LineupRates and compute_lineup_rates()."""

from __future__ import annotations

import math

import numpy as np

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.data.rosters import Player
from hoops.engine.lineup_rates import (
    LineupRates, compute_lineup_rates,
    sample_shooter, player_shot_zone, player_zone_make_prob,
    shrink_rate,
)
from hoops.engine.fatigue import FatigueTracker
from hoops.engine.policy import DefensiveScheme
from hoops.data.rosters import Roster
from hoops.league import League

SEASON = "2023-24"


def _team(name="Test", **kw):
    base = dict(
        league=League.WBB, season=SEASON, team_id=1, team_name=name,
        pace=70.0, off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )
    base.update(kw)
    return TeamPriors(**base)


def _player(pid, name, minutes=200.0, **kw):
    base = dict(
        player_id=pid, name=name, minutes=minutes,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
    )
    base.update(kw)
    return Player(**base)


def _five_players():
    return [
        _player(1, "PG", usage_pct=0.25, ts_pct=0.55, fg3a_share=0.40, tov_pct=0.16),
        _player(2, "SG", usage_pct=0.22, ts_pct=0.53, fg3a_share=0.45, tov_pct=0.14),
        _player(3, "SF", usage_pct=0.20, ts_pct=0.50, fg3a_share=0.35, tov_pct=0.13),
        _player(4, "PF", usage_pct=0.18, ts_pct=0.48, fg3a_share=0.15, tov_pct=0.12),
        _player(5, "C",  usage_pct=0.15, ts_pct=0.52, fg3a_share=0.05, tov_pct=0.18),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_lineup_rates_returns_lineup_rates():
    """compute_lineup_rates returns a LineupRates instance."""
    tp = _team()
    players = _five_players()
    result = compute_lineup_rates(players, tp)
    assert isinstance(result, LineupRates)


def test_lineup_rates_has_blended_team_rates():
    """Blended tov_pct, orb_pct, drb_pct are within plausible bounds."""
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)

    # tov_pct: players range from 0.12 to 0.18, so blended must be in that range
    assert 0.10 <= lr.tov_pct <= 0.20
    # orb_pct: all players have orb_pct=2.0 (unchanged from base)
    assert 1.0 <= lr.orb_pct <= 5.0
    # drb_pct: all players have drb_pct=8.0, shrunk toward team prior (0.70)
    assert 2.0 <= lr.drb_pct <= 12.0


def test_lineup_rates_has_shooter_data():
    """Shooters tuple has 5 entries and weights sum to 1.0."""
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)

    assert len(lr.shooters) == 5
    total_weight = sum(w for _, w in lr.shooters)
    assert math.isclose(total_weight, 1.0, rel_tol=1e-9)


def test_high_usage_player_gets_more_shooting_weight():
    """PG (usage 0.25) should have a higher weight than C (usage 0.15)."""
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)

    weight_map = {p.name: w for p, w in lr.shooters}
    assert weight_map["PG"] > weight_map["C"]


def test_high_tov_lineup_produces_higher_tov_rate():
    """A lineup of high-turnover players should have higher tov_pct."""
    tp = _team()

    normal = _five_players()
    lr_normal = compute_lineup_rates(normal, tp)

    high_tov = [
        _player(i, f"HT{i}", usage_pct=0.20, tov_pct=0.30) for i in range(1, 6)
    ]
    lr_high = compute_lineup_rates(high_tov, tp)

    assert lr_high.tov_pct > lr_normal.tov_pct


def test_fallback_to_team_priors_when_rates_are_none():
    """Players without advanced rates should fall back to team priors."""
    tp = _team(off_tov_pct=0.22, off_orb_pct=0.35, off_ft_pct=0.68,
               foul_rate_per_100=25.0)

    # All five players have None for every rate field.
    bare = [
        _player(i, f"Bare{i}",
                usage_pct=None, ts_pct=None, fg3a_share=None,
                ft_pct=None, tov_pct=None, orb_pct=None,
                drb_pct=None, stl_pct=None, blk_pct=None, foul_rate=None)
        for i in range(1, 6)
    ]
    lr = compute_lineup_rates(bare, tp)

    assert math.isclose(lr.tov_pct, 0.22, rel_tol=1e-9)
    assert math.isclose(lr.orb_pct, 0.35, rel_tol=1e-9)
    assert math.isclose(lr.drb_pct, 1.0 - 0.35, rel_tol=1e-9)
    assert math.isclose(lr.ft_pct, 0.68, rel_tol=1e-9)
    assert math.isclose(lr.foul_rate, 25.0, rel_tol=1e-9)
    assert math.isclose(lr.stl_rate, 2.0, rel_tol=1e-9)
    assert math.isclose(lr.blk_rate, 1.5, rel_tol=1e-9)


def test_lineup_rates_shooter_ts_pct_preserved():
    """Each shooter in the tuple retains a ts_pct (modified by shrinkage, but not None)."""
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)
    for p, _ in lr.shooters:
        assert p.ts_pct is not None


# ---------------------------------------------------------------------------
# Per-player shot resolution helpers
# ---------------------------------------------------------------------------


def test_sample_shooter_returns_player_from_lineup():
    lr = compute_lineup_rates(_five_players(), _team())
    rng = np.random.default_rng(42)
    shooter = sample_shooter(lr, rng)
    names = {p.name for p in _five_players()}
    assert shooter.name in names


def test_sample_shooter_respects_usage_weights():
    lr = compute_lineup_rates(_five_players(), _team())
    rng = np.random.default_rng(42)
    counts: dict[str, int] = {}
    for _ in range(2000):
        s = sample_shooter(lr, rng)
        counts[s.name] = counts.get(s.name, 0) + 1
    assert counts["PG"] > counts["C"]


def test_player_shot_zone_high_fg3a_share():
    team = _team()
    sniper = _player(1, "Sniper", fg3a_share=0.60)
    rng = np.random.default_rng(42)
    zones = [player_shot_zone(sniper, team, rng) for _ in range(1000)]
    three_pct = zones.count("three") / len(zones)
    assert three_pct > 0.45


def test_player_shot_zone_low_fg3a_share():
    team = _team()
    post = _player(2, "Post", fg3a_share=0.05)
    rng = np.random.default_rng(42)
    zones = [player_shot_zone(post, team, rng) for _ in range(1000)]
    three_pct = zones.count("three") / len(zones)
    assert three_pct < 0.15


def test_player_shot_zone_falls_back_to_team_mix():
    team = _team()
    raw = Player(
        player_id=1, name="Raw", minutes=200.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30,
    )
    rng = np.random.default_rng(42)
    zones = [player_shot_zone(raw, team, rng) for _ in range(2000)]
    three_pct = zones.count("three") / len(zones)
    assert 0.25 < three_pct < 0.45  # team shot_mix.three = 0.35


def test_player_zone_make_prob_scales_by_ts():
    team = _team()
    good = _player(1, "Good", ts_pct=0.60)
    bad = _player(2, "Bad", ts_pct=0.35)
    assert player_zone_make_prob(good, "mid", team) > player_zone_make_prob(bad, "mid", team)


def test_player_zone_make_prob_clamped():
    team = _team()
    extreme = _player(1, "Extreme", ts_pct=1.5)
    prob = player_zone_make_prob(extreme, "rim", team)
    assert 0.05 <= prob <= 0.95


def test_player_zone_make_prob_falls_back_to_team():
    team = _team()
    raw = Player(
        player_id=1, name="Raw", minutes=200.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30,
    )
    prob = player_zone_make_prob(raw, "rim", team)
    assert abs(prob - team.zone_efg.rim) < 1e-6


# ---------------------------------------------------------------------------
# Fatigue integration
# ---------------------------------------------------------------------------


def test_lineup_rates_with_fatigue_degrades_shooting():
    players = _five_players()
    team = _team()
    hr = Roster(team_id=1, team_name="Home", players=tuple(players))
    ar = Roster(team_id=2, team_name="Away", players=tuple(_five_players()))
    ft = FatigueTracker(hr, ar)

    lr_fresh = compute_lineup_rates(players, team)

    for p in players:
        ft._fatigue[p.player_id] = 0.8

    lr_tired = compute_lineup_rates(players, team, fatigue_tracker=ft)
    assert lr_tired.tov_pct > lr_fresh.tov_pct


def test_lineup_rates_without_fatigue_unchanged():
    players = _five_players()
    team = _team()
    lr1 = compute_lineup_rates(players, team)
    lr2 = compute_lineup_rates(players, team, fatigue_tracker=None)
    assert lr1.tov_pct == lr2.tov_pct
    assert lr1.ft_pct == lr2.ft_pct


# ---------------------------------------------------------------------------
# Scheme affinity integration
# ---------------------------------------------------------------------------


def _rim_protector():
    return _player(10, "BigBlock", blk_pct=4.0, drb_pct=15.0, stl_pct=1.0)


def _perimeter_stopper():
    return _player(11, "QuickHands", stl_pct=5.0, blk_pct=0.3, fg3a_share=0.40)


def test_lineup_rates_zone_with_rim_protector_boosts_defense():
    base = [_player(i, f"P{i}") for i in range(4)]
    zone_lineup = base + [_rim_protector()]
    team = _team()
    lr_man = compute_lineup_rates(zone_lineup, team, scheme=DefensiveScheme.MAN)
    lr_zone = compute_lineup_rates(zone_lineup, team, scheme=DefensiveScheme.ZONE)
    assert lr_zone.blk_rate > lr_man.blk_rate


def test_lineup_rates_press_with_perimeter_stopper_boosts_steals():
    base = [_player(i, f"P{i}") for i in range(4)]
    press_lineup = base + [_perimeter_stopper()]
    team = _team()
    lr_man = compute_lineup_rates(press_lineup, team, scheme=DefensiveScheme.MAN)
    lr_press = compute_lineup_rates(press_lineup, team, scheme=DefensiveScheme.PRESS)
    assert lr_press.stl_rate > lr_man.stl_rate


def test_lineup_rates_no_scheme_is_neutral():
    players = _five_players()
    team = _team()
    lr_none = compute_lineup_rates(players, team)
    lr_man = compute_lineup_rates(players, team, scheme=DefensiveScheme.MAN)
    assert abs(lr_none.blk_rate - lr_man.blk_rate) < 1e-6


# ---------------------------------------------------------------------------
# shrink_rate tests
# ---------------------------------------------------------------------------


def test_shrink_rate_zero_minutes_returns_team_prior():
    assert shrink_rate(player_rate=0.30, team_rate=0.18, minutes=0.0) == 0.18


def test_shrink_rate_at_k_returns_midpoint():
    result = shrink_rate(player_rate=0.30, team_rate=0.18, minutes=200.0)
    expected = 0.5 * 0.30 + 0.5 * 0.18
    assert abs(result - expected) < 1e-9


def test_shrink_rate_high_minutes_mostly_player():
    result = shrink_rate(player_rate=0.30, team_rate=0.18, minutes=600.0)
    w = 600.0 / (600.0 + 200.0)
    expected = w * 0.30 + (1 - w) * 0.18
    assert abs(result - expected) < 1e-9


def test_shrink_rate_low_minutes_mostly_team():
    result = shrink_rate(player_rate=0.30, team_rate=0.18, minutes=30.0)
    w = 30.0 / (30.0 + 200.0)
    expected = w * 0.30 + (1 - w) * 0.18
    assert abs(result - expected) < 1e-6


def test_shrink_rate_equal_rates_returns_same():
    assert abs(shrink_rate(0.20, 0.20, minutes=50.0) - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# Shrinkage integration tests
# ---------------------------------------------------------------------------


def test_compute_lineup_rates_shrinks_low_minute_players():
    """A player with 30 minutes should have their rate pulled toward team prior."""
    tp = _team(off_tov_pct=0.18)
    low_min = [
        _player(i, f"LowMin{i}", minutes=30.0, tov_pct=0.30, usage_pct=0.20)
        for i in range(1, 6)
    ]
    lr = compute_lineup_rates(low_min, tp)
    assert lr.tov_pct < 0.22  # well below raw 0.30
    assert lr.tov_pct > 0.18  # but above team prior


def test_compute_lineup_rates_high_minute_players_keep_rates():
    """A player with 600 minutes should mostly keep their raw rate."""
    tp = _team(off_tov_pct=0.18)
    high_min = [
        _player(i, f"HighMin{i}", minutes=600.0, tov_pct=0.30, usage_pct=0.20)
        for i in range(1, 6)
    ]
    lr = compute_lineup_rates(high_min, tp)
    assert lr.tov_pct > 0.25


def test_compute_lineup_rates_shrinks_shooter_ts_pct():
    """Shooters in the tuple should have shrunk ts_pct values."""
    tp = _team(off_efg=0.45, off_ft_pct=0.70)
    low_min_good_shooter = _player(
        1, "LowMinSniper", minutes=30.0, ts_pct=0.65, usage_pct=0.20,
    )
    lineup = [low_min_good_shooter] + [
        _player(i, f"P{i}", minutes=600.0) for i in range(2, 6)
    ]
    lr = compute_lineup_rates(lineup, tp)
    sniper_in_lineup = next(p for p, _ in lr.shooters if p.name == "LowMinSniper")
    assert sniper_in_lineup.ts_pct < 0.60
    assert sniper_in_lineup.ts_pct > 0.49


# ---------------------------------------------------------------------------
# pace_adj tests
# ---------------------------------------------------------------------------


def test_lineup_rates_has_pace_adj():
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)
    assert hasattr(lr, "pace_adj")
    assert isinstance(lr.pace_adj, float)


def test_pace_adj_guard_heavy_lineup_positive():
    tp = _team()
    guards = [
        _player(i, f"G{i}", position="G", usage_pct=0.22, minutes=400.0)
        for i in range(1, 6)
    ]
    lr = compute_lineup_rates(guards, tp)
    assert lr.pace_adj > 0


def test_pace_adj_center_heavy_lineup_negative():
    tp = _team()
    centers = [
        _player(i, f"C{i}", position="C", usage_pct=0.18, minutes=400.0)
        for i in range(1, 6)
    ]
    lr = compute_lineup_rates(centers, tp)
    assert lr.pace_adj < 0


def test_pace_adj_clamped():
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)
    assert -3.0 <= lr.pace_adj <= 3.0


# ---------------------------------------------------------------------------
# efg_adj tests
# ---------------------------------------------------------------------------


def test_lineup_rates_has_efg_adj():
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)
    assert hasattr(lr, "efg_adj")
    assert isinstance(lr.efg_adj, float)


def test_efg_adj_good_shooters_positive():
    tp = _team(off_efg=0.45, off_ft_pct=0.70)
    good = [
        _player(i, f"Good{i}", minutes=600.0, ts_pct=0.58, usage_pct=0.20)
        for i in range(1, 6)
    ]
    lr = compute_lineup_rates(good, tp)
    assert lr.efg_adj > 0


def test_efg_adj_bad_shooters_negative():
    tp = _team(off_efg=0.45, off_ft_pct=0.70)
    bad = [
        _player(i, f"Bad{i}", minutes=600.0, ts_pct=0.38, usage_pct=0.20)
        for i in range(1, 6)
    ]
    lr = compute_lineup_rates(bad, tp)
    assert lr.efg_adj < 0


def test_efg_adj_clamped():
    tp = _team()
    players = _five_players()
    lr = compute_lineup_rates(players, tp)
    assert -0.03 <= lr.efg_adj <= 0.03


# ---------------------------------------------------------------------------
# pace_adj wired into _sample_possession_seconds
# ---------------------------------------------------------------------------


def test_bench_lineup_scores_fewer_points():
    """A bench lineup (low minutes, mediocre rates) should score fewer
    points per possession than starters (high minutes, good rates)."""
    from hoops.engine.machine import simulate_possession
    from hoops.engine.state import GameState, Side
    from hoops.rules import rules_for

    tp = _team()
    rules = rules_for(League.WBB, SEASON)

    starters = [
        _player(i, f"Starter{i}", minutes=600.0, usage_pct=0.22,
                ts_pct=0.55, tov_pct=0.13, fg3a_share=0.30, ft_pct=0.78,
                position="G" if i <= 2 else ("F" if i <= 4 else "C"))
        for i in range(1, 6)
    ]
    bench = [
        _player(i + 10, f"Bench{i}", minutes=40.0, usage_pct=0.15,
                ts_pct=0.42, tov_pct=0.22, fg3a_share=0.25, ft_pct=0.62,
                position="G" if i <= 2 else ("F" if i <= 4 else "C"))
        for i in range(1, 6)
    ]

    def score_n_possessions(lineup, n=1000):
        lr = compute_lineup_rates(lineup, tp)
        total_pts = 0
        for i in range(n):
            rng = np.random.default_rng(42 + i)
            state = GameState.initial(rules, opening_possession=Side.HOME)
            new_state, events = simulate_possession(
                state, tp, tp, rng, off_lineup_rates=lr,
            )
            total_pts += new_state.home_score
        return total_pts / n

    starter_ppp = score_n_possessions(starters)
    bench_ppp = score_n_possessions(bench)

    assert starter_ppp > bench_ppp, (
        f"Starters ({starter_ppp:.3f}) should score more than bench ({bench_ppp:.3f})"
    )


def test_sample_possession_seconds_with_pace_adj():
    """Positive pace_adj should produce shorter possessions on average."""
    from hoops.engine.machine import _sample_possession_seconds
    from hoops.engine.state import GameState, Side
    from hoops.engine.policy import CoachPolicy
    from hoops.league import League
    from hoops.rules import rules_for

    tp_off = _team(pace=70.0)
    tp_def = _team(pace=70.0)
    rules = rules_for(League.WBB, SEASON)
    state = GameState.initial(rules)
    policy = CoachPolicy()

    durations_base = [
        _sample_possession_seconds(tp_off, tp_def, state, policy, np.random.default_rng(i))
        for i in range(500)
    ]

    durations_fast = [
        _sample_possession_seconds(tp_off, tp_def, state, policy, np.random.default_rng(i), pace_adj=3.0)
        for i in range(500)
    ]

    assert np.mean(durations_fast) < np.mean(durations_base)
