"""Pilot tests for the in-app team picker."""

from __future__ import annotations

import pytest

from hoops.data.paths import distributions_dir
from hoops.league import League
from hoops.ui.app import CoachGameScreen, GameScreen, HoopsApp, StartingLineupScreen, TeamSelectScreen


def _data_present() -> bool:
    return (distributions_dir(League.WBB, "2023-24") / "team_priors.parquet").exists()


pytestmark = pytest.mark.skipif(
    not _data_present(),
    reason="2023-24 priors missing; run scripts/fit_distributions.py",
)


@pytest.mark.asyncio
async def test_picker_screen_mounts_with_teams_loaded():
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TeamSelectScreen)
        assert len(screen._home_priors) > 100  # full D-I roster, not empty
        # Status starts in the unselected state.
        assert "(none)" in screen.last_status_text


@pytest.mark.asyncio
async def test_picker_play_action_no_op_without_selection():
    """Pressing P with nothing selected must not crash and must not advance."""
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        # Still on the picker.
        assert isinstance(app.screen, TeamSelectScreen)


@pytest.mark.asyncio
async def test_picker_routes_to_game_screen_after_play():
    """End-to-end: pick home, pick away, press P — should land in GameScreen."""
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TeamSelectScreen)

        # Pick the first two teams via direct mutation + the action; this
        # avoids needing to drive ListView focus precisely in a test.
        first_id = screen._home_priors[0].team_id
        second_id = screen._home_priors[1].team_id
        screen.home_id = first_id
        screen.away_id = second_id
        screen._refresh_status()

        await pilot.press("p")
        await pilot.pause()

        # In coaching mode we land on the starting lineup picker first.
        if isinstance(app.screen, StartingLineupScreen):
            await pilot.press("escape")  # use defaults
            await pilot.pause()
        assert isinstance(app.screen, (GameScreen, CoachGameScreen))
        first_name = screen._home_priors[0].team_name
        second_name = screen._home_priors[1].team_name
        assert app.screen.home_name == first_name
        assert app.screen.away_name == second_name


@pytest.mark.asyncio
async def test_picker_rejects_same_home_and_away():
    """Picking the same team for both sides should keep us on the picker."""
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TeamSelectScreen)

        same_id = screen._home_priors[0].team_id
        screen.home_id = same_id
        screen.away_id = same_id
        screen._refresh_status()

        await pilot.press("p")
        # Still on the picker.
        assert isinstance(app.screen, TeamSelectScreen)


@pytest.mark.asyncio
async def test_picker_status_updates_on_selection():
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen

        screen.home_id = screen._home_priors[0].team_id
        screen._refresh_status()
        assert screen._home_priors[0].team_name in screen.last_status_text
        assert "(none)" in screen.last_status_text  # away still unset

        screen.away_id = screen._home_priors[1].team_id
        screen._refresh_status()
        assert screen._home_priors[0].team_name in screen.last_status_text
        assert screen._home_priors[1].team_name in screen.last_status_text
        assert "Press P to play" in screen.last_status_text


@pytest.mark.asyncio
async def test_picker_filters_to_division_one_by_default():
    """The picker should hide non-D-I teams (those with very few games)
    so the user isn't scrolling through 600+ entries."""
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TeamSelectScreen)
        # 2023-24 D-I women's basketball has 360 teams.
        assert 300 <= len(screen._home_priors) <= 400, (
            f"expected ~360 D-I teams, got {len(screen._home_priors)}"
        )


@pytest.mark.asyncio
async def test_picker_can_show_all_teams_via_flag():
    app = HoopsApp(season="2023-24", division_one_only=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        # Should include the D-II opponents that play one-off games.
        assert len(screen._home_priors) > 500


@pytest.mark.asyncio
async def test_q_quits_app_from_picker_even_when_option_list_focused():
    """OptionList eats letter keys for jump-navigation; the app-level
    q binding must use priority=True to remain usable."""
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        # OptionList is focused at this point.
        await pilot.press("q")
        await pilot.pause()
        # The app should have started its quit sequence.
        assert app._exit is True or not app.is_running


@pytest.mark.asyncio
async def test_q_quits_app_from_game_screen():
    from hoops.engine.events import Event
    from hoops.engine.state import Side

    events = [
        Event(quarter=1, seconds_left=600, type="tip_off", team=Side.HOME),
        Event(quarter=1, seconds_left=0, type="quarter_end", team=None),
        Event(quarter=2, seconds_left=600, type="tip_off", team=Side.AWAY),
        Event(quarter=2, seconds_left=0, type="quarter_end", team=None),
        Event(quarter=3, seconds_left=600, type="tip_off", team=Side.HOME),
        Event(quarter=3, seconds_left=0, type="quarter_end", team=None),
        Event(quarter=4, seconds_left=600, type="tip_off", team=Side.AWAY),
        Event(quarter=4, seconds_left=0, type="quarter_end", team=None,
              home_score=2, away_score=0),
        Event(quarter=4, seconds_left=0, type="game_end", team=None,
              home_score=2, away_score=0),
    ]
    app = HoopsApp(events=events, home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert app._exit is True or not app.is_running


@pytest.mark.asyncio
async def test_picker_cycles_home_scheme():
    """Pressing 1 cycles the home defensive scheme MAN -> ZONE -> PRESS -> MAN."""
    from hoops.engine.policy import DefensiveScheme

    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen.home_policy.scheme is DefensiveScheme.MAN
        await pilot.press("1")
        assert screen.home_policy.scheme is DefensiveScheme.ZONE
        await pilot.press("1")
        assert screen.home_policy.scheme is DefensiveScheme.PRESS
        await pilot.press("1")
        assert screen.home_policy.scheme is DefensiveScheme.MAN


@pytest.mark.asyncio
async def test_picker_toggles_foul_up_3():
    """Pressing 3 toggles home foul-up-3, 4 toggles away."""
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen.home_policy.foul_when_down_3 is False
        assert screen.away_policy.foul_when_down_3 is False

        await pilot.press("3")
        assert screen.home_policy.foul_when_down_3 is True
        assert screen.away_policy.foul_when_down_3 is False

        await pilot.press("4")
        assert screen.away_policy.foul_when_down_3 is True

        await pilot.press("3")
        assert screen.home_policy.foul_when_down_3 is False


@pytest.mark.asyncio
async def test_picker_policy_text_reflects_state():
    """The rendered policy text on each side reflects the current policy."""
    app = HoopsApp(season="2023-24")
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert "MAN" in screen._policy_text(screen.home_policy)
        await pilot.press("1")
        assert "ZONE" in screen._policy_text(screen.home_policy)
        await pilot.press("3")
        assert "Foul-up-3: ON" in screen._policy_text(screen.home_policy)


@pytest.mark.asyncio
async def test_direct_construction_with_events_skips_picker():
    """The CLI's --home/--away path constructs HoopsApp with events=...
    and should land in GameScreen immediately, not the picker."""
    from hoops.engine.events import Event
    from hoops.engine.state import Side

    events = [
        Event(quarter=1, seconds_left=600, type="tip_off", team=Side.HOME),
        Event(quarter=1, seconds_left=0, type="quarter_end", team=None),
        Event(quarter=2, seconds_left=600, type="tip_off", team=Side.AWAY),
        Event(quarter=2, seconds_left=0, type="quarter_end", team=None),
        Event(quarter=3, seconds_left=600, type="tip_off", team=Side.HOME),
        Event(quarter=3, seconds_left=0, type="quarter_end", team=None),
        Event(quarter=4, seconds_left=600, type="tip_off", team=Side.AWAY),
        Event(quarter=4, seconds_left=0, type="quarter_end", team=None,
              home_score=2, away_score=0),
        Event(quarter=4, seconds_left=0, type="game_end", team=None,
              home_score=2, away_score=0),
    ]
    app = HoopsApp(events=events, home_name="Home", away_name="Away")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, GameScreen)
