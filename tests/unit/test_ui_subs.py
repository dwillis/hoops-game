"""Pilot tests for the in-game substitution UI."""

from __future__ import annotations

import numpy as np
import pytest

from hoops.data.rosters import Player, Roster
from hoops.engine.events import Event
from hoops.engine.state import Side
from hoops.ui.app import GameScreen, HoopsApp, SubScreen
from hoops.ui.lineup import LineupState


def _player(pid: int, name: str) -> Player:
    return Player(
        player_id=pid, name=name, minutes=600.0, fga=200, fg3a=80, fta=60,
        orb=20, drb=80, fouls=40, tov=30, ast=40, blk=10, stl=15,
    )


def _roster(team_id: int, label: str, n: int = 8) -> Roster:
    return Roster(
        team_id=team_id, team_name=label,
        players=tuple(_player(i, f"{label}{i}") for i in range(1, n + 1)),
    )


def _events_with_a_few_possessions():
    return [
        Event(1, 600, "tip_off", Side.HOME),
        Event(1, 580, "foul_personal", Side.AWAY),  # dead ball
        Event(1, 560, "shot_made", Side.HOME, detail="rim", home_score=2),
        Event(1, 540, "shot_missed", Side.AWAY, detail="three"),
        Event(1, 540, "rebound_def", Side.HOME),
        Event(1, 520, "shot_made", Side.HOME, detail="three", home_score=5),
        Event(1, 0, "quarter_end", None, home_score=5),
        Event(2, 600, "tip_off", None),
        Event(2, 0, "quarter_end", None),
        Event(3, 600, "tip_off", None),
        Event(3, 0, "quarter_end", None),
        Event(4, 600, "tip_off", None),
        Event(4, 0, "quarter_end", None, home_score=5),
        Event(4, 0, "game_end", None, home_score=5, away_score=0),
    ]


@pytest.mark.asyncio
async def test_b_opens_sub_screen_at_dead_ball():
    home, away = _roster(1, "H"), _roster(2, "A")
    app = HoopsApp(
        events=_events_with_a_few_possessions(),
        home_name="H", away_name="A",
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gs = GameScreen(
            _events_with_a_few_possessions(),
            home_name="H", away_name="A",
            home_roster=home, away_roster=away,
        )
        await app.push_screen(gs)
        await pilot.pause()
        # First step (tip-off) is not a dead ball; b should be blocked.
        # Apply the foul event with `s`.
        await pilot.press("s")
        await pilot.pause()
        assert app.screen.playback.is_dead_ball is True
        await pilot.press("b")
        await pilot.pause()
        assert isinstance(app.screen, SubScreen)


@pytest.mark.asyncio
async def test_b_blocked_during_live_ball():
    """At tip-off (no dead ball yet), b should not open the SubScreen."""
    home, away = _roster(1, "H"), _roster(2, "A")
    app = HoopsApp(
        events=_events_with_a_few_possessions(),
        home_name="H", away_name="A",
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gs = GameScreen(
            _events_with_a_few_possessions(),
            home_name="H", away_name="A",
            home_roster=home, away_roster=away,
        )
        await app.push_screen(gs)
        await pilot.pause()
        # Tip-off has been applied on mount; not a dead ball.
        assert app.screen.playback.is_dead_ball is False
        await pilot.press("b")
        await pilot.pause()
        # SubScreen should not be on the stack.
        assert not isinstance(app.screen, SubScreen)


@pytest.mark.asyncio
async def test_b_no_op_without_lineup():
    """If GameScreen was constructed without rosters (legacy path), `b`
    is a no-op rather than crashing."""
    app = HoopsApp(
        events=_events_with_a_few_possessions(),
        home_name="H", away_name="A",
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gs = app.screen
        assert isinstance(gs, GameScreen)
        assert gs.lineup is None
        await pilot.press("b")
        await pilot.pause()
        # Still on the GameScreen.
        assert isinstance(app.screen, GameScreen)


@pytest.mark.asyncio
async def test_sub_screen_queues_substitution():
    """Subs from the screen go into the pending queue; the on-court lineup
    stays unchanged until a dead ball commits the queue."""
    home = _roster(1, "H", n=8)
    away = _roster(2, "A", n=8)
    rng = np.random.default_rng(0)
    lineup = LineupState.with_default_starters(home, away, rng)

    sub = SubScreen(lineup, "H", "A")

    from textual.app import App

    class _Host(App):
        def on_mount(self):
            self.push_screen(sub)

    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("1")  # pull starter #1
        await pilot.press("a")  # send in bench player a (player_id=6)
        await pilot.pause()
        # On-court is unchanged.
        actual_ids = {p.player_id for p in lineup.on_court(Side.HOME)}
        assert 1 in actual_ids and 6 not in actual_ids
        # But the pending shadow reflects the request.
        pending_ids = {p.player_id for p in lineup.pending_on_court(Side.HOME)}
        assert 1 not in pending_ids and 6 in pending_ids
        assert lineup.has_pending(Side.HOME)


@pytest.mark.asyncio
async def test_sub_screen_switches_sides_with_tab():
    home = _roster(1, "H", n=8)
    away = _roster(2, "A", n=8)
    rng = np.random.default_rng(0)
    lineup = LineupState.with_default_starters(home, away, rng)

    from textual.app import App

    class _Host(App):
        def on_mount(self):
            self.push_screen(SubScreen(lineup, "H", "A"))

    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        sub: SubScreen = app.screen  # type: ignore
        assert sub._active_side is Side.HOME
        await pilot.press("tab")
        await pilot.pause()
        assert sub._active_side is Side.AWAY


@pytest.mark.asyncio
async def test_subs_on_home_screen_take_effect_in_subsequent_attribution():
    """End-to-end: pull a starter, sim a possession, the new starter
    should be eligible for attribution and the pulled one should not."""
    home = _roster(1, "Home", n=8)
    away = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    # Construct a lineup where only player_id=6 has nonzero FGA after sub-in,
    # so attribution is deterministic.
    home_only_6 = Roster(
        team_id=10, team_name="Home",
        players=tuple(
            Player(
                player_id=i, name=f"H{i}",
                minutes=500, fga=(2000 if i == 6 else 0),
                fg3a=80, fta=60, orb=20, drb=80, fouls=40, tov=30,
                ast=40, blk=10, stl=15,
            ) for i in range(1, 9)
        ),
    )
    lineup = LineupState.with_default_starters(home_only_6, away, rng)
    # Sub player 1 OUT and player 6 IN.
    lineup.substitute(Side.HOME, off_player_id=1, on_player_id=6)

    # Now attribute a shot — must go to H6.
    e = Event(quarter=1, seconds_left=580, type="shot_made", team=Side.HOME, detail="rim")
    out = lineup.attribute(e)
    assert out.player == "H6"
