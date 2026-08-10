"""State-logic tests for the ball-distribution and matchup editor screens.

These exercise the screens' action handlers directly (with the widget
refresh stubbed out) so we cover the coach-input -> engine-policy wiring
without the full Textual app mount."""

from __future__ import annotations

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.data.rosters import Player, Roster
from hoops.engine.interactive import InteractiveGame
from hoops.engine.policy import DefensiveScheme
from hoops.engine.sampling import make_rng
from hoops.engine.state import Side
from hoops.league import League
from hoops.rules import rules_for
from hoops.ui.game_screens import BallDistributionScreen, DefenseAssignmentScreen


def _player(pid, name, **kw):
    base = dict(
        player_id=pid, name=name, minutes=400.0, games_played=30,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
        position="G",
    )
    base.update(kw)
    return Player(**base)


def _roster(team_id, label):
    return Roster(
        team_id=team_id, team_name=label,
        players=tuple(_player(team_id * 100 + i, f"{label}{i}") for i in range(1, 9)),
    )


def _game():
    mix = ShotMix(rim=0.40, mid=0.25, three=0.35)
    efg = ZoneEFG(rim=0.55, mid=0.40, three=0.35)
    priors = dict(
        league=League.WBB, season="2023-24", pace=70.0, shot_mix=mix, zone_efg=efg,
        off_efg=0.48, off_3pt_rate=0.35, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_ft_pct=0.72, def_efg=0.42, def_tov_pct=0.20,
        def_orb_pct=0.28, def_fta_rate=0.25, foul_rate_per_100=18.0,
    )
    home = TeamPriors(team_id=1, team_name="Home", **priors)
    away = TeamPriors(team_id=2, team_name="Away", **priors)
    rules = rules_for(League.WBB, "2023-24")
    return InteractiveGame(
        home, away, rules, make_rng(1),
        _roster(1, "H"), _roster(2, "A"), human_side=Side.HOME,
    )


class _NoRefreshDist(BallDistributionScreen):
    def _refresh(self):  # skip widget updates (no mounted app)
        pass

    def dismiss(self, result=None):  # no app to post the result to
        self._dismissed = result


class _NoRefreshAsg(DefenseAssignmentScreen):
    def _refresh(self):
        pass

    def dismiss(self, result=None):
        self._dismissed = result


# --- ball distribution screen ---------------------------------------------

def test_distribution_screen_sets_policy_on_close():
    game = _game()
    screen = _NoRefreshDist(game, Side.HOME, "H", "A")
    screen.action_select("0")
    for _ in range(5):
        screen.action_bump("1")  # push player 0's share up
    screen.action_close()
    dist = game.policies.for_side(Side.HOME).shot_distribution
    assert dist is not None
    first_pid = game.lineup.on_court(Side.HOME)[0].player_id
    # The bumped player should have the largest share.
    assert max(dist, key=dist.get) == first_pid


def test_distribution_screen_reset_clears_override():
    game = _game()
    game.set_shot_distribution(Side.HOME, {game.lineup.on_court(Side.HOME)[0].player_id: 0.5})
    screen = _NoRefreshDist(game, Side.HOME, "H", "A")
    screen.action_reset()
    screen.action_close()
    assert game.policies.for_side(Side.HOME).shot_distribution is None


def test_distribution_close_without_edit_leaves_none():
    game = _game()
    screen = _NoRefreshDist(game, Side.HOME, "H", "A")
    screen.action_close()
    assert game.policies.for_side(Side.HOME).shot_distribution is None


# --- matchup screen -------------------------------------------------------

def test_assignment_screen_swaps_and_sets_policy():
    game = _game()
    game.set_human_scheme(DefensiveScheme.MAN)
    screen = _NoRefreshAsg(game, Side.HOME, "H", "A")
    before = dict(screen._map)
    screen.action_pick("0")  # first defender
    screen.action_pick("1")  # swap with second
    screen.action_close()
    assigned = game.policies.for_side(Side.HOME).man_assignments
    assert assigned is not None
    d0 = game.lineup.on_court(Side.HOME)[0].player_id
    d1 = game.lineup.on_court(Side.HOME)[1].player_id
    # The two defenders' targets were exchanged.
    assert assigned[d0] == before[d1]
    assert assigned[d1] == before[d0]


def test_assignment_screen_reset_clears_policy():
    game = _game()
    game.set_human_scheme(DefensiveScheme.MAN)
    screen = _NoRefreshAsg(game, Side.HOME, "H", "A")
    screen.action_pick("0")
    screen.action_pick("1")
    screen.action_reset()
    screen.action_close()
    assert game.policies.for_side(Side.HOME).man_assignments is None
