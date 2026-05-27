"""Textual App for NCAA tournament bracket mode.

Orchestrates: load bracket -> for each round (auto-sim -> bracket view -> user game) -> champion/elimination.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from textual.app import App
from textual.binding import Binding

from hoops.data.distributions import (
    LeaguePrior,
    TeamPriors,
    load_league_prior,
    load_team_priors,
)
from hoops.data.paths import bracket_path
from hoops.data.rosters import Roster, load_roster
from hoops.engine.bracket import Bracket
from hoops.engine.sampling import make_rng
from hoops.engine.state import Side
from hoops.engine.tournament import auto_sim_round
from hoops.league import League
from hoops.rules import Rules, rules_for
from hoops.ui.bracket_screens import BracketViewScreen, ChampionScreen


class TournamentApp(App):
    """NCAA tournament bracket mode."""

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(
        self,
        season: str,
        user_team_id: int,
        user_team_name: str = "",
        seed: int = 42,
        bracket_path_override: Path | None = None,
        **kw,
    ):
        super().__init__(**kw)
        self._season = season
        self._user_team_id = user_team_id
        self._user_team_name = user_team_name
        self._seed = seed
        self._bracket_path_override = bracket_path_override
        self._bracket: Bracket | None = None
        self._priors: dict[int, TeamPriors] = {}
        self._league_prior: LeaguePrior | None = None
        self._rosters: dict[int, Roster] = {}
        self._rules: Rules | None = None
        self._rng: np.random.Generator | None = None
        self._current_round = 1
        self._eliminated_round: int | None = None

    def on_mount(self) -> None:
        """Load all data and start the tournament."""
        if self._bracket_path_override is not None:
            bp = self._bracket_path_override
        else:
            bp = bracket_path(League.WBB, self._season)
        if not bp.exists():
            self.exit(message=f"No bracket data found at {bp}")
            return

        self._bracket = Bracket.load(bp)
        if self._user_team_id not in self._bracket.team_ids():
            self.exit(message=f"Team {self._user_team_id} not in {self._season} tournament bracket.")
            return

        # Load priors, league prior, rules
        all_priors = load_team_priors(League.WBB, self._season)
        self._priors = {p.team_id: p for p in all_priors}
        self._league_prior = load_league_prior(League.WBB, self._season)
        self._rules = rules_for(League.WBB, self._season)
        self._rng = make_rng(seed=self._seed)

        # Populate team names
        names = {p.team_id: p.team_name for p in all_priors}
        self._bracket.populate_names(names)

        if not self._user_team_name:
            p = self._priors.get(self._user_team_id)
            if p:
                self._user_team_name = p.team_name

        # Pre-load rosters for all bracket teams
        bracket_ids = self._bracket.team_ids()
        for tid in bracket_ids:
            try:
                self._rosters[tid] = load_roster(tid, self._season)
            except Exception:
                pass

        # Start first round
        self._run_round()

    def _run_round(self) -> None:
        """Auto-sim non-user games, then show bracket view."""
        assert self._bracket is not None
        assert self._rules is not None
        assert self._rng is not None

        auto_sim_round(
            self._bracket, self._current_round,
            user_team_id=self._user_team_id,
            priors=self._priors,
            rules=self._rules,
            rng=self._rng,
            league=self._league_prior,
            rosters=self._rosters,
        )

        # Show bracket view
        self.push_screen(
            BracketViewScreen(
                self._bracket, self._current_round,
                self._user_team_id,
                user_eliminated=False,
            ),
            callback=self._on_bracket_view_dismissed,
        )

    def _on_bracket_view_dismissed(self, result) -> None:
        """After user sees bracket, launch their game."""
        assert self._bracket is not None

        game_idx = self._bracket.next_game_for(self._user_team_id)
        if game_idx is None:
            self._show_final()
            return

        game = self._bracket.games[game_idx]

        # Determine which side user is on
        if game.home.team_id == self._user_team_id:
            human_side = Side.HOME
            home_name = self._user_team_name
            away_name = game.away.team_name or "TBD"
        else:
            human_side = Side.AWAY
            home_name = game.home.team_name or "TBD"
            away_name = self._user_team_name

        home_priors = self._priors.get(game.home.team_id)
        away_priors = self._priors.get(game.away.team_id)

        if home_priors is None or away_priors is None:
            self._show_final()
            return

        home_roster = self._rosters.get(game.home.team_id)
        away_roster = self._rosters.get(game.away.team_id)

        if home_roster is None or away_roster is None:
            # Auto-sim if rosters missing
            from hoops.engine.machine import simulate_game
            final, _events = simulate_game(
                home_priors, away_priors, self._rules, self._rng,
                league=self._league_prior,
                neutral_site=True,
            )
            winner_id = (
                game.home.team_id if final.home_score > final.away_score
                else game.away.team_id
            )
            self._bracket.advance(game_idx, winner_id, final.home_score, final.away_score)
            self._after_user_game()
            return

        from hoops.engine.interactive import InteractiveGame
        interactive = InteractiveGame(
            home_priors, away_priors, self._rules, self._rng,
            home_roster, away_roster,
            human_side=human_side,
            league=self._league_prior,
            neutral_site=True,
        )

        # Add seed to display names
        home_seed = game.home.seed
        away_seed = game.away.seed
        home_label = f"({home_seed}) {home_name}" if home_seed else home_name
        away_label = f"({away_seed}) {away_name}" if away_seed else away_name

        from hoops.ui.app import CoachGameScreen
        self.push_screen(
            CoachGameScreen(interactive, home_label, away_label, tournament_mode=True),
            callback=lambda _result: self._on_game_finished(game_idx, interactive),
        )

    def _on_game_finished(self, game_idx: int, game) -> None:
        """Record result from user's interactive game."""
        assert self._bracket is not None

        state = game.state
        bg = self._bracket.games[game_idx]

        winner_id = (
            bg.home.team_id if state.home_score > state.away_score
            else bg.away.team_id
        )
        self._bracket.advance(game_idx, winner_id, state.home_score, state.away_score)
        self._after_user_game()

    def _after_user_game(self) -> None:
        """After user's game, check if eliminated or advance to next round."""
        assert self._bracket is not None

        champion = self._bracket.champion()
        if champion is not None:
            self._show_final()
            return

        user_game = self._bracket.next_game_for(self._user_team_id)
        if user_game is None:
            # User eliminated — sim remaining rounds
            self._eliminated_round = self._current_round
            for rnd in range(self._current_round + 1, self._bracket.max_round + 1):
                auto_sim_round(
                    self._bracket, rnd,
                    user_team_id=-1,  # sim everything
                    priors=self._priors,
                    rules=self._rules,
                    rng=self._rng,
                    league=self._league_prior,
                    rosters=self._rosters,
                )
            self._show_final()
        else:
            self._current_round += 1
            self._run_round()

    def _show_final(self) -> None:
        """Show champion/elimination screen."""
        assert self._bracket is not None
        user_won = self._bracket.champion() == self._user_team_id
        self.push_screen(ChampionScreen(
            self._bracket,
            self._user_team_id,
            user_won=user_won,
            user_team_name=self._user_team_name,
            eliminated_round=self._eliminated_round,
        ))
