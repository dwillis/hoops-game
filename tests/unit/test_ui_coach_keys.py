"""Pilot test: the new coaching keys work through a real app mount."""

from __future__ import annotations

import pytest

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.data.rosters import Player, Roster
from hoops.engine.interactive import InteractiveGame
from hoops.engine.policy import DefensiveIntensity
from hoops.engine.sampling import make_rng
from hoops.engine.state import Side
from hoops.league import League
from hoops.rules import rules_for
from hoops.ui.app import HoopsApp
from hoops.ui.game_screens import CoachGameScreen


def _player(pid, name):
    return Player(
        player_id=pid, name=name, minutes=400.0, games_played=30,
        fga=100, fg3a=30, fta=40, orb=15, drb=50, fouls=20, tov=15,
        ast=30, blk=5, stl=10, usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0, drb_pct=8.0, stl_pct=2.5,
        blk_pct=0.8, foul_rate=3.0, position="G",
    )


def _roster(team_id, label):
    return Roster(
        team_id=team_id, team_name=label,
        players=tuple(_player(team_id * 100 + i, f"{label}{i}") for i in range(1, 9)),
    )


def _coach_game():
    mix = ShotMix(rim=0.40, mid=0.25, three=0.35)
    efg = ZoneEFG(rim=0.55, mid=0.40, three=0.35)
    priors = dict(
        league=League.WBB, season="2023-24", pace=70.0, shot_mix=mix, zone_efg=efg,
        off_efg=0.48, off_3pt_rate=0.35, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_ft_pct=0.72, def_efg=0.42, def_tov_pct=0.20,
        def_orb_pct=0.28, def_fta_rate=0.25, foul_rate_per_100=18.0,
    )
    return InteractiveGame(
        TeamPriors(team_id=1, team_name="Home", **priors),
        TeamPriors(team_id=2, team_name="Away", **priors),
        rules_for(League.WBB, "2023-24"), make_rng(1),
        _roster(1, "H"), _roster(2, "A"), human_side=Side.HOME,
    )


@pytest.mark.asyncio
async def test_intensity_and_double_team_keys():
    game = _coach_game()
    app = HoopsApp(events=list(game.all_events), home_name="H", away_name="A")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(CoachGameScreen(game, "H", "A"))
        await pilot.pause()
        # 'i' cycles intensity off NORMAL; 'y' raises double-team off 0.
        await pilot.press("i")
        await pilot.pause()
        assert game.policies.for_side(Side.HOME).intensity is not DefensiveIntensity.NORMAL
        await pilot.press("y")
        await pilot.pause()
        assert game.policies.for_side(Side.HOME).double_team_pct > 0.0


@pytest.mark.asyncio
async def test_open_distribution_and_matchup_screens():
    game = _coach_game()
    app = HoopsApp(events=list(game.all_events), home_name="H", away_name="A")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(CoachGameScreen(game, "H", "A"))
        await pilot.pause()
        # 'u' opens the ball-distribution editor; escape closes it.
        await pilot.press("u")
        await pilot.pause()
        assert app.screen.__class__.__name__ == "BallDistributionScreen"
        await pilot.press("escape")
        await pilot.pause()
        # 'm' opens the matchup editor.
        await pilot.press("m")
        await pilot.pause()
        assert app.screen.__class__.__name__ == "DefenseAssignmentScreen"
        await pilot.press("escape")
        await pilot.pause()
