"""Tests for defensive man-to-man assignments."""

from __future__ import annotations

import pytest

from hoops.data.rosters import Player
from hoops.engine.assignments import (
    default_assignments,
    defensive_ratings,
    matchup_rating,
    resolve_actual_map,
    shooter_foul_delta,
    shooter_make_delta,
)


def _player(pid, position="G", **kw):
    base = dict(
        player_id=pid, name=f"P{pid}", position=position, minutes=400.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
    )
    base.update(kw)
    return Player(**base)


def _defenders():
    return [
        _player(1, "G", stl_pct=5.0, blk_pct=0.5, fg3a_share=0.45),   # perimeter stopper
        _player(2, "G", stl_pct=2.0, blk_pct=0.5, fg3a_share=0.45),
        _player(3, "F", stl_pct=2.0, blk_pct=1.5, fg3a_share=0.30),
        _player(4, "F", stl_pct=1.5, blk_pct=2.5, drb_pct=13.0, fg3a_share=0.10),
        _player(5, "C", stl_pct=0.5, blk_pct=3.5, drb_pct=15.0, fg3a_share=0.05),  # rim
    ]


def _opponents():
    return [
        _player(11, "G", usage_pct=0.30, fg3a_share=0.50),  # star guard
        _player(12, "G", usage_pct=0.22, fg3a_share=0.45),
        _player(13, "F", usage_pct=0.20, fg3a_share=0.30),
        _player(14, "F", usage_pct=0.16, fg3a_share=0.15),
        _player(15, "C", usage_pct=0.12, fg3a_share=0.05),
    ]


def test_default_map_is_deterministic():
    d, o = _defenders(), _opponents()
    m1 = default_assignments(d, o)
    m2 = default_assignments(d, o)
    assert m1 == m2
    # Every defender and opponent used exactly once.
    assert sorted(m1.keys()) == [1, 2, 3, 4, 5]
    assert sorted(m1.values()) == [11, 12, 13, 14, 15]


def test_default_map_gives_zero_delta():
    d, o = _defenders(), _opponents()
    ratings = defensive_ratings(d)
    default = resolve_actual_map(None, d, o)
    def_by_id = {p.player_id: p for p in d}
    for opp in o:
        assert shooter_make_delta(default, default, ratings, def_by_id, opp) == 0.0
        assert shooter_foul_delta(default, default, def_by_id, opp) == 0.0


def test_perimeter_stopper_rates_high_on_perimeter():
    d = _defenders()
    ratings = defensive_ratings(d)
    # Defender 1 (high steals) should out-rate defender 2 on perimeter.
    assert ratings[1][0] > ratings[2][0]
    # Defender 5 (high blocks/drb) should out-rate defender 1 on interior.
    assert ratings[5][1] > ratings[1][1]


def test_swapping_weak_defender_onto_star_raises_make_prob():
    d, o = _defenders(), _opponents()
    ratings = defensive_ratings(d)
    def_by_id = {p.player_id: p for p in d}
    default = resolve_actual_map(None, d, o)
    star = o[0]  # player 11
    default_def = next(did for did, oid in default.items() if oid == star.player_id)
    # Force the weakest perimeter defender (2) onto the star instead.
    weak = 2 if default_def != 2 else 1
    coach = {weak: star.player_id}
    actual = resolve_actual_map(coach, d, o)
    delta = shooter_make_delta(actual, default, ratings, def_by_id, star)
    # A weaker defender than the default should not lower the make prob.
    assert delta >= 0.0


def test_make_delta_is_bounded():
    d, o = _defenders(), _opponents()
    ratings = defensive_ratings(d)
    def_by_id = {p.player_id: p for p in d}
    default = resolve_actual_map(None, d, o)
    # Try every single-swap coach map and confirm the delta stays in +/-3pp.
    for did in [p.player_id for p in d]:
        for opp in o:
            actual = resolve_actual_map({did: opp.player_id}, d, o)
            for shooter in o:
                delta = shooter_make_delta(actual, default, ratings, def_by_id, shooter)
                assert -0.03 - 1e-9 <= delta <= 0.03 + 1e-9


def test_gc_mismatch_raises_foul_delta():
    d, o = _defenders(), _opponents()
    def_by_id = {p.player_id: p for p in d}
    default = resolve_actual_map(None, d, o)
    center_opp = o[4]  # player 15, a center
    guard_def = 1  # a guard
    # Put the guard on the center (a G<->C mismatch).
    actual = resolve_actual_map({guard_def: center_opp.player_id}, d, o)
    delta = shooter_foul_delta(actual, default, def_by_id, center_opp)
    assert delta > 0.0


def test_resolve_honors_coach_pair_and_fills_rest():
    d, o = _defenders(), _opponents()
    coach = {3: 11}  # defender 3 guards opponent 11
    actual = resolve_actual_map(coach, d, o)
    assert actual[3] == 11
    # Still a complete 1:1 map.
    assert sorted(actual.keys()) == [1, 2, 3, 4, 5]
    assert sorted(actual.values()) == [11, 12, 13, 14, 15]


def test_resolve_drops_offcourt_coach_pairs():
    d, o = _defenders(), _opponents()
    # Reference players not on court; should be ignored, fall back to default.
    coach = {99: 11, 3: 88}
    actual = resolve_actual_map(coach, d, o)
    default = resolve_actual_map(None, d, o)
    assert actual == default


def test_assignment_changes_make_rate_in_possession():
    """End-to-end: putting a weak defender on the star raises her make rate
    vs. putting the best defender on her."""
    from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
    from hoops.engine.lineup_rates import compute_lineup_rates
    from hoops.engine.machine import simulate_possession
    from hoops.engine.policy import CoachPolicies, CoachPolicy, DefensiveScheme
    from hoops.engine.sampling import make_rng
    from hoops.engine.state import GameState, Side
    from hoops.league import League
    from hoops.rules import rules_for

    rules = rules_for(League.WBB, "2023-24")
    team = TeamPriors(
        league=League.WBB, season="2023-24", team_id=1, team_name="T",
        pace=70, off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )
    # Heavy-usage star so she takes most shots; defenders span a wide range.
    offp = [_player(1, usage_pct=0.60)] + [_player(i, usage_pct=0.10) for i in range(2, 6)]
    defp = [
        _player(10, stl_pct=8.0), _player(11, stl_pct=6.0), _player(12, stl_pct=2.0),
        _player(13, stl_pct=1.0), _player(14, stl_pct=0.2),
    ]
    off_lr = compute_lineup_rates(offp, team)
    def_lr = compute_lineup_rates(defp, team)

    def make_rate(assign, n=3000):
        made = shots = 0
        for s in range(n):
            rng = make_rng(s)
            st = GameState.initial(rules, opening_possession=Side.HOME)
            pol = CoachPolicies(
                home=CoachPolicy(),
                away=CoachPolicy(scheme=DefensiveScheme.MAN, man_assignments=assign),
            )
            _st, evs = simulate_possession(
                st, team, team, rng, policies=pol,
                off_lineup_rates=off_lr, def_lineup_rates=def_lr,
            )
            for e in evs:
                if e.type in ("shot_made", "shot_missed") and e.player == "P1":
                    shots += 1
                if e.type == "shot_made" and e.player == "P1":
                    made += 1
        return made / shots if shots else 0.0

    best = make_rate({10: 1, 11: 2, 12: 3, 13: 4, 14: 5})
    worst = make_rate({14: 1, 10: 2, 11: 3, 12: 4, 13: 5})
    assert worst > best


def test_matchup_rating_uses_dimension_by_opponent_type():
    perim_opp = _player(20, "G", fg3a_share=0.50)
    post_opp = _player(21, "C", fg3a_share=0.05)
    rating = (0.8, -0.2)  # strong perimeter, weak interior
    assert matchup_rating(rating, perim_opp) == pytest.approx(0.8)
    assert matchup_rating(rating, post_opp) == pytest.approx(-0.2)
