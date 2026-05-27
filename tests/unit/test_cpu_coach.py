"""Tests for CPU coaching intelligence."""

from __future__ import annotations

import pytest

from hoops.data.rosters import Player
from hoops.engine.cpu_coach import CpuCoach, CpuPersonality, PossessionSummary, TrendTracker, assign_personality
from hoops.engine.policy import CoachPolicy, DefensiveScheme, OffensiveScheme
from hoops.engine.state import Side


def test_trend_tracker_records_summaries():
    t = TrendTracker(window=5)
    s = PossessionSummary(side=Side.HOME, scored=True, points=3, zone="three", turnover=False, foul=False)
    t.record(s)
    assert len(t.recent) == 1
    assert t.recent[0].points == 3


def test_trend_tracker_respects_window():
    t = TrendTracker(window=3)
    for i in range(5):
        t.record(PossessionSummary(
            side=Side.HOME, scored=True, points=2,
            zone="rim", turnover=False, foul=False,
        ))
    assert len(t.recent) == 3


def test_trend_three_point_count():
    t = TrendTracker(window=10)
    for _ in range(3):
        t.record(PossessionSummary(side=Side.HOME, scored=True, points=3, zone="three", turnover=False, foul=False))
    for _ in range(2):
        t.record(PossessionSummary(side=Side.HOME, scored=False, points=0, zone="three", turnover=False, foul=False))
    assert t.made_threes_by(Side.HOME) == 3
    assert t.made_threes_by(Side.AWAY) == 0


def test_trend_rim_makes_count():
    t = TrendTracker(window=10)
    for _ in range(4):
        t.record(PossessionSummary(side=Side.AWAY, scored=True, points=2, zone="rim", turnover=False, foul=False))
    assert t.rim_makes_by(Side.AWAY) == 4


def test_trend_opponent_scored_count():
    t = TrendTracker(window=8)
    for _ in range(5):
        t.record(PossessionSummary(side=Side.HOME, scored=True, points=2, zone="rim", turnover=False, foul=False))
    for _ in range(3):
        t.record(PossessionSummary(side=Side.HOME, scored=False, points=0, zone="mid", turnover=True, foul=False))
    assert t.scored_possessions_by(Side.HOME) == 5


def test_trend_serialization_roundtrip():
    t = TrendTracker(window=10)
    t.record(PossessionSummary(side=Side.HOME, scored=True, points=3, zone="three", turnover=False, foul=False))
    t.record(PossessionSummary(side=Side.AWAY, scored=False, points=0, zone=None, turnover=True, foul=False))
    data = t.to_list()
    t2 = TrendTracker.from_list(data)
    assert len(t2.recent) == 2
    assert t2.recent[0].points == 3
    assert t2.recent[1].turnover is True


def test_aggressive_personality_high_pace():
    assert assign_personality(pace=74.0, off_tov_pct=0.18, def_efg=0.45) is CpuPersonality.AGGRESSIVE


def test_aggressive_personality_low_tov():
    assert assign_personality(pace=68.0, off_tov_pct=0.12, def_efg=0.45) is CpuPersonality.AGGRESSIVE


def test_conservative_personality_slow_pace():
    assert assign_personality(pace=62.0, off_tov_pct=0.18, def_efg=0.45) is CpuPersonality.CONSERVATIVE


def test_conservative_personality_strong_defense():
    assert assign_personality(pace=68.0, off_tov_pct=0.18, def_efg=0.38) is CpuPersonality.CONSERVATIVE


def test_balanced_personality():
    assert assign_personality(pace=68.0, off_tov_pct=0.18, def_efg=0.44) is CpuPersonality.BALANCED


def _make_coach(personality=CpuPersonality.BALANCED, current_scheme=DefensiveScheme.MAN):
    return CpuCoach(
        cpu_side=Side.AWAY,
        personality=personality,
        current_scheme=current_scheme,
    )


def test_switch_to_press_trailing_big_q4():
    coach = _make_coach()
    result = coach.should_switch_scheme(
        quarter=4, seconds_left=300, cpu_score=30, opp_score=42,
        opp_lineup_archetypes=[], total_possessions=80,
    )
    assert result is DefensiveScheme.PRESS


def test_no_press_when_leading():
    coach = _make_coach()
    result = coach.should_switch_scheme(
        quarter=4, seconds_left=300, cpu_score=45, opp_score=30,
        opp_lineup_archetypes=[], total_possessions=80,
    )
    assert result is not DefensiveScheme.PRESS


def test_switch_to_zone_on_three_point_barrage():
    coach = _make_coach()
    for _ in range(3):
        coach.trend.record(PossessionSummary(
            side=Side.HOME, scored=True, points=3,
            zone="three", turnover=False, foul=False,
        ))
    result = coach.should_switch_scheme(
        quarter=2, seconds_left=400, cpu_score=20, opp_score=25,
        opp_lineup_archetypes=[], total_possessions=40,
    )
    assert result is DefensiveScheme.ZONE


def test_switch_to_man_when_zone_failing():
    coach = _make_coach(current_scheme=DefensiveScheme.ZONE)
    # 5 scored possessions by opponent (HOME is opp of AWAY cpu)
    for _ in range(5):
        coach.trend.record(PossessionSummary(
            side=Side.HOME, scored=True, points=2,
            zone="rim", turnover=False, foul=False,
        ))
    # Add one more to make 6 opponent possessions
    coach.trend.record(PossessionSummary(
        side=Side.HOME, scored=True, points=2,
        zone="mid", turnover=False, foul=False,
    ))
    coach._last_scheme_poss = 0
    result = coach.should_switch_scheme(
        quarter=2, seconds_left=400, cpu_score=20, opp_score=25,
        opp_lineup_archetypes=[], total_possessions=40,
    )
    assert result is DefensiveScheme.MAN


def test_scheme_cooldown_prevents_thrash():
    coach = _make_coach()
    coach._last_scheme_poss = 35
    for _ in range(3):
        coach.trend.record(PossessionSummary(
            side=Side.HOME, scored=True, points=3,
            zone="three", turnover=False, foul=False,
        ))
    result = coach.should_switch_scheme(
        quarter=2, seconds_left=400, cpu_score=20, opp_score=25,
        opp_lineup_archetypes=[], total_possessions=38,
    )
    assert result is None


def test_aggressive_press_earlier():
    coach = _make_coach(personality=CpuPersonality.AGGRESSIVE)
    result = coach.should_switch_scheme(
        quarter=3, seconds_left=400, cpu_score=30, opp_score=39,
        opp_lineup_archetypes=[], total_possessions=60,
    )
    assert result is DefensiveScheme.PRESS


def test_zone_on_spacer_heavy_lineup():
    coach = _make_coach()
    result = coach.should_switch_scheme(
        quarter=2, seconds_left=400, cpu_score=20, opp_score=22,
        opp_lineup_archetypes=["floor_spacer", "versatile_wing", "floor_spacer", "default", "default"],
        total_possessions=40,
    )
    assert result is DefensiveScheme.ZONE


# ---------------------------------------------------------------------------
# Matchup substitution tests
# ---------------------------------------------------------------------------


def _player(pid, name="P", **kw):
    base = dict(
        player_id=pid, name=name, minutes=200.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
        min_share=0.28,
    )
    base.update(kw)
    return Player(**base)


def test_matchup_sub_perimeter_stopper_when_hot_from_three():
    coach = _make_coach()
    for _ in range(3):
        coach.trend.record(PossessionSummary(
            side=Side.HOME, scored=True, points=3,
            zone="three", turnover=False, foul=False,
        ))
    on_court = [_player(i, stl_pct=1.0, blk_pct=0.5) for i in range(5)]
    bench = [_player(10, stl_pct=5.0, blk_pct=0.5, name="Stopper")]
    subs = coach.should_matchup_sub(on_court, bench)
    assert len(subs) == 1
    assert subs[0][1] == 10


def test_matchup_sub_rim_protector_when_paint_dominated():
    coach = _make_coach()
    for _ in range(3):
        coach.trend.record(PossessionSummary(
            side=Side.HOME, scored=True, points=2,
            zone="rim", turnover=False, foul=False,
        ))
    on_court = [_player(i, blk_pct=1.0, drb_pct=5.0) for i in range(5)]
    bench = [_player(10, blk_pct=4.0, drb_pct=14.0, name="Rim protector")]
    subs = coach.should_matchup_sub(on_court, bench)
    assert len(subs) == 1
    assert subs[0][1] == 10


def test_matchup_sub_ball_handler_for_press():
    coach = _make_coach(current_scheme=DefensiveScheme.PRESS)
    on_court = [_player(i, usage_pct=0.15, ast_pct=3.0) for i in range(5)]
    bench = [_player(10, usage_pct=0.28, ast_pct=10.0, name="Ball handler")]
    subs = coach.should_matchup_sub(on_court, bench)
    assert len(subs) == 1
    assert subs[0][1] == 10


def test_matchup_sub_no_op_when_archetype_present():
    coach = _make_coach()
    for _ in range(3):
        coach.trend.record(PossessionSummary(
            side=Side.HOME, scored=True, points=3,
            zone="three", turnover=False, foul=False,
        ))
    on_court = [_player(i, stl_pct=1.0) for i in range(4)] + [_player(4, stl_pct=5.0)]
    bench = [_player(10, stl_pct=5.0)]
    subs = coach.should_matchup_sub(on_court, bench)
    assert len(subs) == 0


def test_matchup_sub_empty_bench():
    coach = _make_coach()
    for _ in range(3):
        coach.trend.record(PossessionSummary(
            side=Side.HOME, scored=True, points=3,
            zone="three", turnover=False, foul=False,
        ))
    on_court = [_player(i) for i in range(5)]
    subs = coach.should_matchup_sub(on_court, [])
    assert len(subs) == 0


# ---------------------------------------------------------------------------
# PossessionSummary player field tests
# ---------------------------------------------------------------------------


def test_possession_summary_player_field():
    s = PossessionSummary(
        side=Side.HOME, scored=True, points=3,
        zone="three", turnover=False, foul=False,
        player="Smith",
    )
    assert s.player == "Smith"


def test_possession_summary_player_default_none():
    s = PossessionSummary(
        side=Side.HOME, scored=False, points=0,
        zone=None, turnover=True, foul=False,
    )
    assert s.player is None


def test_trend_tracker_points_by_player():
    t = TrendTracker(window=10)
    t.record(PossessionSummary(side=Side.HOME, scored=True, points=3, zone="three", turnover=False, foul=False, player="Smith"))
    t.record(PossessionSummary(side=Side.HOME, scored=True, points=2, zone="rim", turnover=False, foul=False, player="Smith"))
    t.record(PossessionSummary(side=Side.HOME, scored=True, points=2, zone="mid", turnover=False, foul=False, player="Jones"))
    assert t.points_by_player("Smith") == 5
    assert t.points_by_player("Jones") == 2
    assert t.points_by_player("Nobody") == 0


def test_trend_serialization_with_player():
    t = TrendTracker(window=10)
    t.record(PossessionSummary(side=Side.HOME, scored=True, points=3, zone="three", turnover=False, foul=False, player="Smith"))
    data = t.to_list()
    t2 = TrendTracker.from_list(data)
    assert t2.recent[0].player == "Smith"


# ---------------------------------------------------------------------------
# Foul trouble substitution tests
# ---------------------------------------------------------------------------


def test_foul_trouble_first_half_3_fouls():
    """First half: pull any player with 3+ fouls."""
    coach = _make_coach()
    on_court = [_player(i) for i in range(5)]
    bench = [_player(10, name="Sub")]
    fouls = {i: 0 for i in range(11)}
    fouls[2] = 3  # player 2 has 3 fouls
    subs = coach.should_foul_trouble_sub(on_court, bench, fouls, quarter=2, seconds_left=300)
    assert len(subs) == 1
    assert subs[0][0] == 2  # off
    assert subs[0][1] == 10  # on


def test_foul_trouble_second_half_4_fouls():
    """Second half: pull any player with 4+ fouls."""
    coach = _make_coach()
    on_court = [_player(i) for i in range(5)]
    bench = [_player(10, name="Sub")]
    fouls = {i: 0 for i in range(11)}
    fouls[3] = 4
    subs = coach.should_foul_trouble_sub(on_court, bench, fouls, quarter=3, seconds_left=400)
    assert len(subs) == 1
    assert subs[0][0] == 3


def test_foul_trouble_second_half_3_fouls_no_sub():
    """Second half: 3 fouls is NOT enough to pull."""
    coach = _make_coach()
    on_court = [_player(i) for i in range(5)]
    bench = [_player(10)]
    fouls = {i: 0 for i in range(11)}
    fouls[3] = 3
    subs = coach.should_foul_trouble_sub(on_court, bench, fouls, quarter=3, seconds_left=400)
    assert len(subs) == 0


def test_foul_trouble_crunch_time_balanced():
    """BALANCED: crunch-time exception at <=2:00 -- don't pull."""
    coach = _make_coach(personality=CpuPersonality.BALANCED)
    on_court = [_player(i) for i in range(5)]
    bench = [_player(10)]
    fouls = {i: 0 for i in range(11)}
    fouls[0] = 4
    subs = coach.should_foul_trouble_sub(on_court, bench, fouls, quarter=4, seconds_left=100)
    assert len(subs) == 0


def test_foul_trouble_crunch_time_aggressive():
    """AGGRESSIVE: crunch-time exception at <=4:00 -- don't pull."""
    coach = _make_coach(personality=CpuPersonality.AGGRESSIVE)
    on_court = [_player(i) for i in range(5)]
    bench = [_player(10)]
    fouls = {i: 0 for i in range(11)}
    fouls[0] = 4
    subs = coach.should_foul_trouble_sub(on_court, bench, fouls, quarter=4, seconds_left=200)
    assert len(subs) == 0


def test_foul_trouble_conservative_always_protects():
    """CONSERVATIVE: no crunch-time exception -- always pull."""
    coach = _make_coach(personality=CpuPersonality.CONSERVATIVE)
    on_court = [_player(i) for i in range(5)]
    bench = [_player(10, name="Sub")]
    fouls = {i: 0 for i in range(11)}
    fouls[0] = 4
    subs = coach.should_foul_trouble_sub(on_court, bench, fouls, quarter=4, seconds_left=60)
    assert len(subs) == 1


def test_foul_trouble_empty_bench():
    """No bench players available -- can't sub."""
    coach = _make_coach()
    on_court = [_player(i) for i in range(5)]
    fouls = {i: 0 for i in range(5)}
    fouls[0] = 4
    subs = coach.should_foul_trouble_sub(on_court, [], fouls, quarter=3, seconds_left=400)
    assert len(subs) == 0


# ---------------------------------------------------------------------------
# Hot-hand veto tests
# ---------------------------------------------------------------------------


def test_hot_hand_veto_6_points():
    """BALANCED: veto fatigue sub if player scored 6+ in window."""
    coach = _make_coach()
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=3, zone="three", turnover=False, foul=False, player="Star"))
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=3, zone="three", turnover=False, foul=False, player="Star"))
    assert coach.should_veto_fatigue_sub("Star", fatigue=0.72) is True


def test_hot_hand_no_veto_below_threshold():
    """BALANCED: 4 points is below the 6-point threshold — no veto."""
    coach = _make_coach()
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=2, zone="rim", turnover=False, foul=False, player="Star"))
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=2, zone="mid", turnover=False, foul=False, player="Star"))
    assert coach.should_veto_fatigue_sub("Star", fatigue=0.72) is False


def test_hot_hand_no_veto_at_hard_ceiling():
    """Even a hot player gets pulled at fatigue >= 0.85."""
    coach = _make_coach()
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=3, zone="three", turnover=False, foul=False, player="Star"))
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=3, zone="three", turnover=False, foul=False, player="Star"))
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=3, zone="three", turnover=False, foul=False, player="Star"))
    assert coach.should_veto_fatigue_sub("Star", fatigue=0.86) is False


def test_hot_hand_aggressive_lower_threshold():
    """AGGRESSIVE: veto at 4+ points."""
    coach = _make_coach(personality=CpuPersonality.AGGRESSIVE)
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=2, zone="rim", turnover=False, foul=False, player="Star"))
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=2, zone="mid", turnover=False, foul=False, player="Star"))
    assert coach.should_veto_fatigue_sub("Star", fatigue=0.72) is True


def test_hot_hand_conservative_higher_threshold():
    """CONSERVATIVE: veto at 8+ points — 6 is not enough."""
    coach = _make_coach(personality=CpuPersonality.CONSERVATIVE)
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=3, zone="three", turnover=False, foul=False, player="Star"))
    coach.trend.record(PossessionSummary(side=Side.AWAY, scored=True, points=3, zone="three", turnover=False, foul=False, player="Star"))
    assert coach.should_veto_fatigue_sub("Star", fatigue=0.72) is False


# ---------------------------------------------------------------------------
# Late-game strategy tests
# ---------------------------------------------------------------------------


def test_intentional_foul_trailing_q4():
    """Trailing by 5 with 45s left: set intentional foul flags."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=45, cpu_score=60, opp_score=65)
    assert policy.intentional_foul_in_bonus_when_trailing is True
    assert policy.foul_when_down_3 is False


def test_intentional_foul_down_3_under_30():
    """Trailing by exactly 3 with 25s left: set foul_when_down_3."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=25, cpu_score=60, opp_score=63)
    assert policy.foul_when_down_3 is True
    assert policy.intentional_foul_in_bonus_when_trailing is True


def test_no_foul_when_leading():
    """Leading: never foul intentionally."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=45, cpu_score=65, opp_score=60)
    assert policy.intentional_foul_in_bonus_when_trailing is False
    assert policy.foul_when_down_3 is False


def test_no_foul_trailing_9_plus():
    """Trailing by 9+: don't bother fouling."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=45, cpu_score=55, opp_score=65)
    assert policy.intentional_foul_in_bonus_when_trailing is False


def test_no_foul_before_q4():
    """Not Q4: no late-game fouling."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=3, seconds_left=45, cpu_score=40, opp_score=45)
    assert policy.intentional_foul_in_bonus_when_trailing is False


def test_aggressive_fouls_earlier():
    """AGGRESSIVE: starts fouling at <=90 seconds."""
    coach = _make_coach(personality=CpuPersonality.AGGRESSIVE)
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=80, cpu_score=60, opp_score=65)
    assert policy.intentional_foul_in_bonus_when_trailing is True


def test_conservative_fouls_later():
    """CONSERVATIVE: only fouls at <=45 seconds -- 50s is too early."""
    coach = _make_coach(personality=CpuPersonality.CONSERVATIVE)
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=50, cpu_score=60, opp_score=65)
    assert policy.intentional_foul_in_bonus_when_trailing is False


def test_hold_for_last_shot():
    """Leading in Q4, <=35s left: hold for last shot."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=30, cpu_score=65, opp_score=60)
    assert policy.hold_for_last is True


def test_two_for_one():
    """<=40 seconds left in any quarter: set two_for_one."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=2, seconds_left=38, cpu_score=30, opp_score=32)
    assert policy.two_for_one is True


def test_run_clock_with_lead():
    """Q4, leading by 5+, <=2:00 remaining: hold continuously."""
    coach = _make_coach()
    policy = CoachPolicy()
    coach.update_late_game_policy(policy, quarter=4, seconds_left=100, cpu_score=70, opp_score=64)
    assert policy.hold_for_last is True


# ---------------------------------------------------------------------------
# Offensive scheme switching tests
# ---------------------------------------------------------------------------


def test_cpu_hurry_up_when_trailing_late():
    """Q4, 100s left, trailing by 10, balanced -> HURRY_UP."""
    coach = _make_coach()
    result = coach.should_switch_off_scheme(
        quarter=4, seconds_left=100, cpu_score=50, opp_score=60,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=80,
    )
    assert result is OffensiveScheme.HURRY_UP


def test_cpu_no_hurry_up_when_leading():
    """Q4, 100s left, leading by 10 -> not HURRY_UP."""
    coach = _make_coach()
    result = coach.should_switch_off_scheme(
        quarter=4, seconds_left=100, cpu_score=60, opp_score=50,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=80,
    )
    assert result is not OffensiveScheme.HURRY_UP


def test_cpu_slow_down_when_leading_late():
    """Q4, 100s left, leading by 10 -> SLOW_DOWN."""
    coach = _make_coach()
    result = coach.should_switch_off_scheme(
        quarter=4, seconds_left=100, cpu_score=60, opp_score=50,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=80,
    )
    assert result is OffensiveScheme.SLOW_DOWN


def test_cpu_three_point_vs_zone():
    """Q2, 300s left, tied, opp in ZONE -> THREE_POINT."""
    coach = _make_coach()
    result = coach.should_switch_off_scheme(
        quarter=2, seconds_left=300, cpu_score=30, opp_score=30,
        opp_def_scheme=DefensiveScheme.ZONE, total_possessions=40,
    )
    assert result is OffensiveScheme.THREE_POINT


def test_cpu_three_point_when_cold():
    """4 scoreless possessions, in NORMAL -> THREE_POINT."""
    coach = _make_coach()
    for _ in range(4):
        coach.trend.record(PossessionSummary(
            side=Side.AWAY, scored=False, points=0,
            zone="mid", turnover=False, foul=False,
        ))
    result = coach.should_switch_off_scheme(
        quarter=2, seconds_left=300, cpu_score=30, opp_score=30,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=40,
    )
    assert result is OffensiveScheme.THREE_POINT


def test_cpu_revert_hurry_up_when_deficit_shrinks():
    """Set current_off_scheme=HURRY_UP, deficit 2 -> NORMAL."""
    coach = _make_coach()
    coach.current_off_scheme = OffensiveScheme.HURRY_UP
    result = coach.should_switch_off_scheme(
        quarter=4, seconds_left=100, cpu_score=58, opp_score=60,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=80,
    )
    assert result is OffensiveScheme.NORMAL


def test_cpu_revert_slow_down_when_lead_shrinks():
    """Set current_off_scheme=SLOW_DOWN, lead 1 -> NORMAL."""
    coach = _make_coach()
    coach.current_off_scheme = OffensiveScheme.SLOW_DOWN
    result = coach.should_switch_off_scheme(
        quarter=4, seconds_left=100, cpu_score=61, opp_score=60,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=80,
    )
    assert result is OffensiveScheme.NORMAL


def test_cpu_off_scheme_cooldown():
    """_last_off_scheme_poss=18, total=20 -> None (cooldown not met)."""
    coach = _make_coach()
    coach._last_off_scheme_poss = 18
    result = coach.should_switch_off_scheme(
        quarter=4, seconds_left=100, cpu_score=50, opp_score=60,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=20,
    )
    assert result is None


def test_cpu_aggressive_hurry_up_earlier():
    """AGGRESSIVE, Q4, 170s left, trailing by 10 -> HURRY_UP."""
    coach = _make_coach(personality=CpuPersonality.AGGRESSIVE)
    result = coach.should_switch_off_scheme(
        quarter=4, seconds_left=170, cpu_score=50, opp_score=60,
        opp_def_scheme=DefensiveScheme.MAN, total_possessions=80,
    )
    assert result is OffensiveScheme.HURRY_UP
