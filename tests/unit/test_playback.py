"""Phase 5: PlaybackState should be the single source of truth for the UI.

The doc's UX requirements (§6) are tested here against the same fixtures
the Textual app will consume. Verifying:

- Box score derived correctly from events.
- Per-quarter team fouls reset at quarter rollover.
- Bonus indicator flips on once defenders have 5+ fouls in a quarter
  and resets at the next quarter (the load-bearing UX behavior the
  doc calls out — "the UI teaches the rule").
"""

from __future__ import annotations

import pytest

from hoops.engine.events import Event
from hoops.engine.state import Side
from hoops.ui.playback import PlaybackState


def _ev(quarter, seconds, type_, team=None, **kw):
    return Event(
        quarter=quarter, seconds_left=seconds, type=type_, team=team,
        home_score=kw.get("home_score", 0), away_score=kw.get("away_score", 0),
        detail=kw.get("detail", ""),
    )


def test_initial_state_from_tip_off():
    p = PlaybackState.from_events([_ev(1, 600, "tip_off", Side.HOME)])
    assert p.quarter == 1
    assert p.seconds_left == 600
    assert p.home_score == 0
    assert p.away_score == 0
    assert p.is_done is False


def test_step_one_advances_pointer():
    p = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_made", Side.HOME, home_score=2, detail="rim"),
    ])
    p.step_one()  # tip
    assert p.current.type == "tip_off"
    p.step_one()  # made shot
    assert p.home_score == 2
    assert p.home_box.fgm == 1
    assert p.home_box.fga == 1
    assert p.home_box.points == 2
    assert p.is_done is True


def test_three_pointer_box_attribution():
    p = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.AWAY),
        _ev(1, 580, "shot_made", Side.AWAY, away_score=3, detail="three"),
    ])
    p.step_to_end()
    assert p.away_box.fg3m == 1
    assert p.away_box.fg3a == 1
    assert p.away_box.points == 3


def test_team_fouls_reset_at_quarter_end():
    """Doc §6: bonus indicator must reset visually at start of each quarter."""
    p = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.HOME),
        # Five away-team fouls accumulate in Q1
        _ev(1, 540, "foul_personal", Side.AWAY),
        _ev(1, 500, "foul_personal", Side.AWAY),
        _ev(1, 460, "foul_personal", Side.AWAY),
        _ev(1, 400, "foul_personal", Side.AWAY),
        _ev(1, 300, "foul_personal", Side.AWAY),
        # End of Q1
        _ev(1, 0, "quarter_end"),
        # Q2 first event
        _ev(2, 600, "shot_made", Side.HOME, detail="rim", home_score=2),
    ])
    # Step through the 5 fouls + quarter_end
    for _ in range(7):
        p.step_one()
    # Foul counters cleared the instant Q1 ends, before any Q2 event runs.
    assert p.home_team_fouls_q == 0
    assert p.away_team_fouls_q == 0
    assert p.in_bonus(Side.HOME) is False  # bonus reset visually
    # Step the first Q2 event; quarter advances now.
    p.step_one()
    assert p.quarter == 2


def test_bonus_indicator_at_fifth_opponent_foul():
    """Doc §6: 'lights up when a team is in the bonus'."""
    p = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 540, "foul_personal", Side.AWAY),
        _ev(1, 500, "foul_personal", Side.AWAY),
        _ev(1, 460, "foul_personal", Side.AWAY),
        _ev(1, 400, "foul_personal", Side.AWAY),
    ])
    p.step_to_end()
    assert p.away_team_fouls_q == 4
    assert p.in_bonus(Side.HOME) is False

    # Add the 5th
    p2 = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 540, "foul_personal", Side.AWAY),
        _ev(1, 500, "foul_personal", Side.AWAY),
        _ev(1, 460, "foul_personal", Side.AWAY),
        _ev(1, 400, "foul_personal", Side.AWAY),
        _ev(1, 300, "foul_personal", Side.AWAY),
    ])
    p2.step_to_end()
    assert p2.in_bonus(Side.HOME) is True
    assert p2.in_bonus(Side.AWAY) is False


def test_quarter_scoreboard_records_running_total():
    p = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 500, "shot_made", Side.HOME, detail="rim", home_score=2),
        _ev(1, 0, "quarter_end", home_score=2),
        _ev(2, 600, "shot_made", Side.AWAY, detail="three", home_score=2, away_score=3),
        _ev(2, 0, "quarter_end", home_score=2, away_score=3),
    ])
    p.step_to_end()
    assert len(p.quarter_scores) >= 2
    assert p.quarter_scores[0].home == 2
    assert p.quarter_scores[0].away == 0
    assert p.quarter_scores[1].home == 2
    assert p.quarter_scores[1].away == 3


def test_step_to_end_of_quarter_stops_at_quarter_end():
    p = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 500, "shot_made", Side.HOME, detail="rim", home_score=2),
        _ev(1, 0, "quarter_end", home_score=2),
        _ev(2, 600, "shot_missed", Side.AWAY, detail="three", home_score=2),
    ])
    applied = p.step_to_end_of_quarter()
    assert applied[-1].type == "quarter_end"
    assert p.quarter == 1  # haven't entered Q2 yet
    assert p.away_box.fga == 0  # Q2 event not applied


def test_step_to_end_drains_log():
    p = PlaybackState.from_events([
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 0, "quarter_end"),
        _ev(2, 600, "shot_made", Side.HOME, detail="rim", home_score=2),
    ])
    p.step_to_end()
    assert p.is_done is True
    assert p.home_score == 2


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_real_engine_events_play_back_consistently(seed):
    """End-to-end: playing the engine's event log with PlaybackState should
    converge to the same final score the engine reports."""
    from hoops.data.distributions import LeaguePrior, ShotMix, TeamPriors, ZoneEFG
    from hoops.engine.machine import simulate_game
    from hoops.engine.sampling import make_rng
    from hoops.league import League
    from hoops.rules import rules_for

    rules = rules_for(League.WBB, "2023-24")

    def synth(name):
        return TeamPriors(
            league=League.WBB, season="2023-24", team_id=1, team_name=name,
            pace=70, off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
            off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
            def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
            shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
            zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
            foul_rate_per_100=20.0,
        )
    home, away = synth("Home"), synth("Away")
    final, events = simulate_game(home, away, rules, make_rng(seed=seed))

    p = PlaybackState.from_events(events)
    p.step_to_end()
    assert p.home_score == final.home_score
    assert p.away_score == final.away_score
