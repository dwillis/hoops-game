"""Tests for the interactive (coaching) game engine."""

from __future__ import annotations

import pytest
import numpy as np

from hoops.data.rosters import Player, Roster
from hoops.engine.interactive import InteractiveGame, PossessionResult
from hoops.engine.policy import CoachPolicies, DefensiveScheme
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


def _make_game(seed=42, human_side=Side.HOME):
    from hoops.data.distributions import TeamPriors, ShotMix, ZoneEFG
    from hoops.engine.sampling import make_rng
    from hoops.league import League

    mix = ShotMix(rim=0.40, mid=0.25, three=0.35)
    efg = ZoneEFG(rim=0.55, mid=0.40, three=0.35)  # noqa: N806
    home_priors = TeamPriors(
        league=League.WBB, season="2023-24",
        team_id=1, team_name="Home", pace=70.0,
        shot_mix=mix, zone_efg=efg,
        off_efg=0.48, off_3pt_rate=0.35,
        off_tov_pct=0.18, off_orb_pct=0.30, off_fta_rate=0.30,
        off_ft_pct=0.72,
        def_efg=0.42, def_tov_pct=0.20, def_orb_pct=0.28, def_fta_rate=0.25,
        foul_rate_per_100=18.0,
    )
    away_priors = TeamPriors(
        league=League.WBB, season="2023-24",
        team_id=2, team_name="Away", pace=68.0,
        shot_mix=mix, zone_efg=efg,
        off_efg=0.46, off_3pt_rate=0.33,
        off_tov_pct=0.17, off_orb_pct=0.28, off_fta_rate=0.28,
        off_ft_pct=0.74,
        def_efg=0.44, def_tov_pct=0.19, def_orb_pct=0.30, def_fta_rate=0.27,
        foul_rate_per_100=17.0,
    )
    rules = Rules(
        league=League.WBB, structure="quarters", quarter_minutes=10,
        shot_clock_seconds=30, three_point_distance_ft=22.146,
        bonus="per_quarter_5th_foul_two_shots", timeouts_per_team=4,
        ot_minutes=5, personal_foul_limit=5,
    )
    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    rng = make_rng(seed=seed)
    return InteractiveGame(
        home_priors, away_priors, rules, rng, hr, ar,
        human_side=human_side,
    )


def test_game_completes():
    game = _make_game()
    poss = 0
    while not game.is_game_over:
        result = game.step_possession()
        poss += 1
        assert poss < 500
    assert game.state.home_score >= 0
    assert game.state.away_score >= 0
    assert len(game.all_events) > 10


def test_human_sub_changes_lineup():
    game = _make_game()
    for _ in range(5):
        game.step_possession()
    on_court = game.lineup.on_court(Side.HOME)
    bench = game.lineup.bench(Side.HOME)
    off = on_court[4]
    on = bench[0]
    game.human_substitute(off.player_id, on.player_id)
    new_ids = [p.player_id for p in game.lineup.on_court(Side.HOME)]
    assert on.player_id in new_ids
    assert off.player_id not in new_ids


def test_scheme_change_takes_effect():
    game = _make_game()
    assert game.human_policy().scheme is DefensiveScheme.MAN
    game.set_human_scheme(DefensiveScheme.ZONE)
    assert game.human_policy().scheme is DefensiveScheme.ZONE
    game.set_human_scheme(DefensiveScheme.PRESS)
    assert game.human_policy().scheme is DefensiveScheme.PRESS


def test_cpu_side_gets_auto_subs():
    game = _make_game(human_side=Side.HOME)
    sub_events = []
    while not game.is_game_over:
        result = game.step_possession()
        for e in result.events:
            if e.type == "substitution" and e.team is Side.AWAY:
                sub_events.append(e)
    assert len(sub_events) > 0, "CPU should make at least one sub"


def test_tip_off_is_first_event():
    game = _make_game()
    assert game.all_events[0].type == "tip_off"


def test_possession_result_reports_dead_ball():
    game = _make_game(seed=1)
    found_dead = False
    for _ in range(50):
        if game.is_game_over:
            break
        result = game.step_possession()
        if result.is_dead_ball:
            found_dead = True
            break
    assert found_dead, "Should encounter a dead ball within 50 possessions"


def test_call_timeout_decrements_and_emits_event():
    game = _make_game()
    for _ in range(3):
        game.step_possession()
    assert game.human_policy().timeouts_remaining == 4
    events = game.call_timeout(game.human_side)
    assert any(e.type == "timeout" for e in events)
    assert game.human_policy().timeouts_remaining == 3


def test_call_timeout_grants_fatigue_recovery():
    game = _make_game()
    for _ in range(20):
        if game.is_game_over:
            break
        game.step_possession()
    on_court = game.lineup.on_court(Side.HOME)
    fatigue_before = [game.fatigue.fatigue(p.player_id) for p in on_court]
    assert any(f > 0 for f in fatigue_before)
    game.call_timeout(Side.HOME)
    fatigue_after = [game.fatigue.fatigue(p.player_id) for p in on_court]
    for before, after in zip(fatigue_before, fatigue_after):
        assert after <= before


def test_call_timeout_with_zero_remaining_raises():
    game = _make_game()
    for _ in range(3):
        game.step_possession()
    for _ in range(4):
        game.call_timeout(game.human_side)
    assert game.human_policy().timeouts_remaining == 0
    with pytest.raises(ValueError, match="no timeouts remaining"):
        game.call_timeout(game.human_side)


def test_media_timeout_fires_under_five_minutes():
    game = _make_game(seed=7)
    media_events = []
    while not game.is_game_over:
        result = game.step_possession()
        for e in result.events:
            if e.type == "media_timeout":
                media_events.append(e)
    # Should have at least 1 media timeout across a full game.
    assert len(media_events) >= 1
    # Each should fire with clock < 300.
    for e in media_events:
        assert e.seconds_left < 300
    # No two media timeouts in the same quarter.
    quarters = [e.quarter for e in media_events]
    assert len(quarters) == len(set(quarters)), "duplicate media TO in same quarter"


def test_cpu_calls_timeout_on_scoring_run():
    """The CPU should call at least one timeout during a full game."""
    game = _make_game(seed=10, human_side=Side.HOME)
    cpu_timeouts = []
    for _ in range(500):
        if game.is_game_over:
            break
        result = game.step_possession()
        for e in result.events:
            if e.type == "timeout" and e.team is game.cpu_side:
                cpu_timeouts.append(e)
    # Over a full game the CPU should call at least one timeout.
    assert len(cpu_timeouts) >= 1
    # CPU timeouts should decrement CPU's count.
    assert game.cpu_policy().timeouts_remaining < 4


def test_to_save_dict_captures_game_state():
    game = _make_game()
    for _ in range(10):
        if game.is_game_over:
            break
        game.step_possession()
    d = game.to_save_dict()
    assert d["version"] == 1
    assert d["home_team_id"] == 1
    assert d["away_team_id"] == 2
    assert d["human_side"] == int(Side.HOME)
    assert d["game_state"]["quarter"] >= 1
    assert isinstance(d["events"], list)
    assert len(d["events"]) > 0
    assert "rng_state" in d
    assert isinstance(d["fatigue"]["fatigue"], dict)
    assert isinstance(d["lineup"]["home_on_court"], list)
    assert len(d["lineup"]["home_on_court"]) == 5
    # Must be JSON-serializable.
    import json
    json.dumps(d)


def test_save_and_load_roundtrip():
    game = _make_game(seed=99)
    for _ in range(30):
        if game.is_game_over:
            break
        game.step_possession()
    score_before = (game.state.home_score, game.state.away_score)
    quarter_before = game.state.quarter
    seconds_before = game.state.seconds_left
    events_before = len(game.all_events)
    human_tos_before = game.human_policy().timeouts_remaining
    on_court_ids_before = [p.player_id for p in game.lineup.on_court(Side.HOME)]

    d = game.to_save_dict()
    game2 = InteractiveGame.from_save_dict(
        d,
        _preloaded=(game.home_priors, game.away_priors, game.rules,
                     game.home_roster, game.away_roster),
    )

    assert game2.state.home_score == score_before[0]
    assert game2.state.away_score == score_before[1]
    assert game2.state.quarter == quarter_before
    assert game2.state.seconds_left == seconds_before
    assert len(game2.all_events) == events_before
    assert game2.human_policy().timeouts_remaining == human_tos_before
    assert [p.player_id for p in game2.lineup.on_court(Side.HOME)] == on_court_ids_before
    assert game2.human_side is game.human_side

    # The restored game should be playable.
    if not game2.is_game_over:
        result = game2.step_possession()
        assert isinstance(result, PossessionResult)


from hoops.engine.cpu_coach import CpuPersonality


def test_game_has_cpu_coach():
    game = _make_game()
    assert hasattr(game, 'cpu_coach')
    assert game.cpu_coach.personality in list(CpuPersonality)


def test_cpu_coach_trend_populated():
    game = _make_game()
    for _ in range(10):
        if game.is_game_over:
            break
        game.step_possession()
    assert len(game.cpu_coach.trend.recent) > 0


def test_save_load_preserves_cpu_coach():
    game = _make_game(seed=99)
    for _ in range(30):
        if game.is_game_over:
            break
        game.step_possession()

    personality_before = game.cpu_coach.personality
    scheme_before = game.cpu_coach.current_scheme
    trend_len_before = len(game.cpu_coach.trend.recent)

    d = game.to_save_dict()
    game2 = InteractiveGame.from_save_dict(
        d,
        _preloaded=(game.home_priors, game.away_priors, game.rules,
                     game.home_roster, game.away_roster),
    )

    assert game2.cpu_coach.personality is personality_before
    assert game2.cpu_coach.current_scheme is scheme_before
    assert len(game2.cpu_coach.trend.recent) == trend_len_before


# ---------------------------------------------------------------------------
# H2H (human vs human) mode tests
# ---------------------------------------------------------------------------


class TestH2HMode:
    def test_h2h_init_no_cpu_coach(self):
        game = _make_game(human_side=None)
        assert game.human_side is None
        assert game.cpu_side is None
        assert game.cpu_coach is None

    def test_h2h_step_possession(self):
        game = _make_game(human_side=None)
        result = game.step_possession()
        assert result.events
        assert not result.is_game_over

    def test_h2h_set_scheme_both_sides(self):
        game = _make_game(human_side=None)
        game.set_scheme(Side.HOME, DefensiveScheme.ZONE)
        game.set_scheme(Side.AWAY, DefensiveScheme.PRESS)
        assert game.policies.home.scheme == DefensiveScheme.ZONE
        assert game.policies.away.scheme == DefensiveScheme.PRESS

    def test_h2h_substitute_both_sides(self):
        game = _make_game(human_side=None)
        home_on = game.lineup.on_court(Side.HOME)
        home_bench = game.lineup.bench(Side.HOME)
        if home_on and home_bench:
            game.substitute(Side.HOME, home_on[0].player_id, home_bench[0].player_id)
        away_on = game.lineup.on_court(Side.AWAY)
        away_bench = game.lineup.bench(Side.AWAY)
        if away_on and away_bench:
            game.substitute(Side.AWAY, away_on[0].player_id, away_bench[0].player_id)

    def test_h2h_call_timeout_both_sides(self):
        game = _make_game(human_side=None)
        events = game.call_timeout(Side.HOME)
        assert any(e.type == "timeout" for e in events)
        events = game.call_timeout(Side.AWAY)
        assert any(e.type == "timeout" for e in events)

    def test_h2h_no_crash_many_possessions(self):
        game = _make_game(human_side=None)
        for _ in range(20):
            if game.is_game_over:
                break
            game.step_possession()

    def test_h2h_full_game(self):
        """Play an entire H2H game to completion with coaching actions."""
        game = _make_game(human_side=None)
        poss = 0
        for _ in range(400):
            if game.is_game_over:
                break
            result = game.step_possession()
            poss += 1
            # Simulate alternating coach decisions at dead balls
            if result.is_dead_ball and not game.is_game_over:
                # Home coach changes scheme partway through
                if poss == 10:
                    game.set_scheme(Side.HOME, DefensiveScheme.ZONE)
                # Away coach changes scheme later
                if poss == 20:
                    game.set_scheme(Side.AWAY, DefensiveScheme.PRESS)
        assert game.is_game_over
        assert game.state.home_score > 0
        assert game.state.away_score > 0
        assert len(game.all_events) > 10

    def test_h2h_human_substitute_asserts(self):
        game = _make_game(human_side=None)
        on = game.lineup.on_court(Side.HOME)
        bench = game.lineup.bench(Side.HOME)
        with pytest.raises(AssertionError, match="H2H"):
            game.human_substitute(on[0].player_id, bench[0].player_id)

    def test_h2h_set_human_scheme_asserts(self):
        game = _make_game(human_side=None)
        with pytest.raises(AssertionError, match="H2H"):
            game.set_human_scheme(DefensiveScheme.ZONE)

    def test_h2h_human_policy_asserts(self):
        game = _make_game(human_side=None)
        with pytest.raises(AssertionError):
            game.human_policy()

    def test_h2h_cpu_policy_asserts(self):
        game = _make_game(human_side=None)
        with pytest.raises(AssertionError):
            game.cpu_policy()

    def test_h2h_save_load_roundtrip(self):
        """Save and load an H2H game preserves state."""
        game = _make_game(human_side=None)
        for _ in range(5):
            if game.is_game_over:
                break
            game.step_possession()
        d = game.to_save_dict()
        assert d["human_side"] is None
        assert d.get("cpu_coach") is None

        restored = InteractiveGame.from_save_dict(d, _preloaded=(
            game.home_priors, game.away_priors, game.rules,
            game.home_roster, game.away_roster,
        ))
        assert restored.human_side is None
        assert restored.cpu_coach is None
        assert restored.state.quarter == game.state.quarter
