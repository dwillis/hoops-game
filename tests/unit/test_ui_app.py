"""Headless smoke tests for the Textual UI via Pilot.

The doc §6 requirements that need a live app to verify:
- Scoreboard widget renders with team names + Q1..Q4 + totals.
- Bonus indicator renders ON when defender fouls hit 5 in the quarter.
- Bonus indicator resets visually when the quarter rolls over.
- Pressing 'f' (run to end) drains the event log and the box panel
  reflects final totals.
"""

from __future__ import annotations

import pytest

from hoops.engine.events import Event
from hoops.engine.state import Side
from hoops.ui.app import HoopsApp


def _ev(quarter, seconds, type_, team=None, **kw):
    return Event(
        quarter=quarter, seconds_left=seconds, type=type_, team=team,
        home_score=kw.get("home_score", 0), away_score=kw.get("away_score", 0),
        detail=kw.get("detail", ""),
    )


def _scripted_events() -> list[Event]:
    """Five away fouls in Q1 (forces home into bonus), then Q1 ends, then Q2."""
    return [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 540, "foul_personal", Side.AWAY, detail="defensive"),
        _ev(1, 500, "foul_personal", Side.AWAY, detail="defensive"),
        _ev(1, 460, "foul_personal", Side.AWAY, detail="defensive"),
        _ev(1, 400, "foul_personal", Side.AWAY, detail="defensive"),
        _ev(1, 300, "foul_personal", Side.AWAY, detail="defensive"),
        _ev(1, 200, "shot_made", Side.HOME, detail="rim", home_score=2),
        _ev(1, 0, "quarter_end", home_score=2),
        _ev(2, 600, "shot_made", Side.AWAY, detail="three", home_score=2, away_score=3),
        _ev(2, 0, "quarter_end", home_score=2, away_score=3),
        _ev(3, 600, "tip_off"),
        _ev(3, 0, "quarter_end", home_score=2, away_score=3),
        _ev(4, 600, "tip_off"),
        _ev(4, 0, "quarter_end", home_score=2, away_score=3),
        _ev(4, 0, "game_end", home_score=2, away_score=3),
    ]


@pytest.mark.asyncio
async def test_app_mounts_and_shows_scoreboard():
    app = HoopsApp(events=_scripted_events(), home_name="South Carolina", away_name="Iowa")
    async with app.run_test() as pilot:
        await pilot.pause()
        # The scoreboard renders with both team names visible.
        rendered = app.screen.scoreboard.last_text
        assert "South Carolina" in rendered
        assert "Iowa" in rendered


@pytest.mark.asyncio
async def test_bonus_indicator_lights_up_after_five_fouls():
    """Doc §6 load-bearing UX behavior."""
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        # Mount auto-applies tip-off; press 's' five times to apply 5 fouls.
        for _ in range(5):
            await pilot.press("s")
        # Home should now be in the bonus.
        assert app.screen.playback.in_bonus(Side.HOME) is True
        rendered = app.screen.scoreboard.last_text
        assert "BONUS" in rendered


@pytest.mark.asyncio
async def test_bonus_resets_at_quarter_rollover():
    """Doc §6: bonus indicator must clear at the start of each quarter."""
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        # Step to end of Q1 (which includes 5 fouls + the made shot + quarter_end).
        await pilot.press("e")
        # After Q1 ends, fouls have been zeroed by PlaybackState.
        assert app.screen.playback.in_bonus(Side.HOME) is False
        assert app.screen.playback.home_team_fouls_q == 0
        assert app.screen.playback.away_team_fouls_q == 0


@pytest.mark.asyncio
async def test_run_to_end_drains_log():
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f")
        from hoops.ui.app import PostGameScreen
        assert isinstance(app.screen, PostGameScreen)
        assert app.screen._playback.is_done is True
        assert app.screen._playback.home_score == 2
        assert app.screen._playback.away_score == 3


@pytest.mark.asyncio
async def test_scoreboard_columns_align_with_uneven_team_names():
    """Q1..Q4 and TOT columns must align across header / home / away rows
    regardless of how long the team names are."""
    app = HoopsApp(events=_scripted_events(), home_name="South Carolina", away_name="Iowa")
    async with app.run_test() as pilot:
        await pilot.pause()
        lines = app.screen.scoreboard.last_text.splitlines()
        header, home_row, away_row = lines[0], lines[1], lines[2]

        # Q1 right-edge must align with each row's first score digit.
        q1_right = header.index("Q1") + 1
        assert home_row[q1_right] == "0"
        assert away_row[q1_right] == "0"

        # TOT right-edge must align with each row's total digit.
        tot_right = header.index("TOT") + 2
        assert home_row[tot_right] == "0"
        assert away_row[tot_right] == "0"

        # Different team-name lengths should produce same-length-prefix
        # padding such that the home and away rows have matching column
        # offsets.
        # The substring up to the first quarter column should differ only
        # by the team-name padding.
        assert home_row[: q1_right - 1].rstrip() == "South Carolina"
        assert away_row[: q1_right - 1].rstrip() == "Iowa"
        assert len(home_row[: q1_right - 1]) == len(away_row[: q1_right - 1])


@pytest.mark.asyncio
async def test_all_three_panels_render_with_nonzero_height():
    """Regression: scoreboard, possession log, and box score must each
    render with non-zero height. An earlier CSS bug had #top sized to
    auto + a 1fr child inside, which collapsed the log to height 0."""
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gs = app.screen
        assert gs.scoreboard.region.height > 0
        assert gs.event_log.region.height > 0
        assert gs.box.region.height > 0


@pytest.mark.asyncio
async def test_possession_log_shows_events_after_step():
    """Pressing 's' once should put the tip-off event in the log."""
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # tip_off was applied on mount; the log should already have a line.
        gs = app.screen
        # RichLog stores lines in `lines`.
        assert len(gs.event_log.lines) >= 1
        # Step once more.
        await pilot.press("s")
        await pilot.pause()
        assert len(gs.event_log.lines) >= 2


@pytest.mark.asyncio
async def test_auto_play_advances_log_over_time():
    """Pressing 'a' starts auto-play; events should accumulate without
    further keypresses."""
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gs = app.screen
        # Speed the timer up so the test isn't slow.
        from hoops.ui.app import _AUTO_SPEEDS
        gs._speed_idx = len(_AUTO_SPEEDS) - 1  # Turbo
        before = len(gs.event_log.lines)
        await pilot.press("a")
        # Advance enough virtual time for several ticks.
        await pilot.pause(0.5)
        after = len(gs.event_log.lines)
        # Even on a 9-event scripted game we should see at least one extra
        # line and the playback pointer should have moved.
        assert after > before or gs.playback.is_done


@pytest.mark.asyncio
async def test_auto_play_toggles_off():
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    from hoops.ui.app import GameScreen as _GS
    _GS.AUTO_INTERVAL_SECONDS = 0.05
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gs = app.screen
        await pilot.press("a")
        assert gs._auto_timer is not None
        await pilot.press("a")
        assert gs._auto_timer is None


@pytest.mark.asyncio
async def test_box_score_panel_updates():
    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f")
        # Post-game screen is now showing; press Esc to return to GameScreen.
        await pilot.press("escape")
        rendered = app.screen.box.last_text
        # Home: 1-1 FG (the rim shot), 2 PTS. Away: 1-1 3P, 3 PTS.
        assert "PTS" in rendered  # header
        assert "Home" in rendered
        assert "Away" in rendered


@pytest.mark.asyncio
async def test_post_game_screen_appears_after_run_to_end():
    from hoops.ui.app import PostGameScreen

    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f")
        assert isinstance(app.screen, PostGameScreen)
        # Shows final score in the title
        title = app.screen.query_one("Static.title").render().plain
        assert "2" in title and "3" in title


@pytest.mark.asyncio
async def test_post_game_screen_esc_returns_to_game():
    from hoops.ui.app import GameScreen as _GS, PostGameScreen

    app = HoopsApp(events=_scripted_events(), home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f")
        assert isinstance(app.screen, PostGameScreen)
        await pilot.press("escape")
        assert isinstance(app.screen, _GS)
