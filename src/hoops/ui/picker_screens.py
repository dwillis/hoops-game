"""Team and season picker screens."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from hoops.data.distributions import (
    LeaguePrior,
    TeamPriors,
    division_one_team_ids,
    load_league_prior,
    load_team_priors,
)
from hoops.data.paths import fitted_seasons, teams_path
from hoops.data.rosters import load_roster
from hoops.engine.policy import (
    CoachPolicies,
    CoachPolicy,
    DefensiveIntensity,
    DefensiveScheme,
    OffensiveScheme,
)
from hoops.engine.state import Side
from hoops.league import League
from hoops.rules import rules_for
from hoops.ui.game_screens import CoachGameScreen, GameScreen, StartingLineupScreen


def _short_season(season: str) -> str:
    """'2025-26' -> '25-26'."""
    parts = season.split("-")
    if len(parts) == 2:
        return f"{parts[0][-2:]}-{parts[1]}"
    return season


def _load_team_records(league: League, season: str) -> dict[int, str]:
    """Return {team_id: 'W-L'} for display in the team picker."""
    path = teams_path(league, season)
    if not path.exists():
        return {}
    try:
        import polars as pl
        df = pl.read_parquet(path, columns=["team_id", "wins", "losses"])
        return {
            int(r["team_id"]): f"{r['wins']}-{r['losses']}"
            for r in df.iter_rows(named=True)
        }
    except Exception:
        return {}

class TeamSelectScreen(Screen):
    """Pre-game picker: choose home and away, optionally from different seasons.

    Two ``OptionList`` widgets side by side. Tab switches focus between
    columns; selecting an option (Enter) sets the side for that column;
    ``p`` plays the matchup once both are chosen.

    Each side has its own season, cycled with ``5``/``6``. When seasons
    differ, the more recent season's rules are used.
    """

    BINDINGS = [
        Binding("tab", "focus_next_column", "Switch column"),
        Binding("p", "play", "Play matchup", priority=True),
        Binding("c", "cycle_coach", "Coach side", priority=True),
        Binding("1", "cycle_home_scheme", "Home scheme", priority=True),
        Binding("2", "cycle_away_scheme", "Away scheme", priority=True),
        Binding("3", "toggle_home_foul_up_3", "Home foul-up-3", priority=True),
        Binding("4", "toggle_away_foul_up_3", "Away foul-up-3", priority=True),
        Binding("5", "cycle_home_season", "Home season", priority=True),
        Binding("6", "cycle_away_season", "Away season", priority=True),
        Binding("7", "cycle_home_off_scheme", "Home O-scheme", priority=True),
        Binding("8", "cycle_away_off_scheme", "Away O-scheme", priority=True),
        Binding("9", "cycle_home_intensity", "Home intensity", priority=True),
        Binding("0", "cycle_away_intensity", "Away intensity", priority=True),
        Binding("slash", "start_search", "/ search", priority=True),
        Binding("n", "toggle_neutral_site", "Neutral site", priority=True),
    ]

    DEFAULT_CSS = """
    TeamSelectScreen {
        layout: vertical;
    }
    TeamSelectScreen > Static.intro {
        height: auto;
        padding: 1 2;
    }
    TeamSelectScreen > Horizontal {
        height: 1fr;
    }
    TeamSelectScreen Vertical {
        width: 1fr;
        border: solid $accent;
    }
    TeamSelectScreen Static.column-header {
        height: auto;
        padding: 0 1;
        background: $accent;
        color: $background;
    }
    TeamSelectScreen #status {
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    TeamSelectScreen #search-bar {
        height: auto;
        padding: 0 2;
        display: none;
    }
    TeamSelectScreen #search-bar.visible {
        display: block;
    }
    """

    def __init__(
        self,
        seasons: list[str],
        default_season: str | None = None,
        seed: int | None = None,
        division_one_only: bool = True,
        neutral_site: bool = False,
    ):
        super().__init__()
        self.seasons = seasons
        self.seed = seed
        self.division_one_only = division_one_only
        self.neutral_site = neutral_site
        self.home_season = default_season or seasons[-1]
        self.away_season = default_season or seasons[-1]
        self._home_priors: list[TeamPriors] = []
        self._away_priors: list[TeamPriors] = []
        self._home_priors_by_id: dict[int, TeamPriors] = {}
        self._away_priors_by_id: dict[int, TeamPriors] = {}
        self._home_league_prior: LeaguePrior | None = None
        self._away_league_prior: LeaguePrior | None = None
        self.home_id: int | None = None
        self.away_id: int | None = None
        self.last_status_text: str = ""
        self.home_policy = CoachPolicy()
        self.away_policy = CoachPolicy()
        self.coach_side: Side | str | None = Side.HOME
        self._search_list_id: str = "home_list"

    def _load_priors(self, season: str) -> tuple[list[TeamPriors], LeaguePrior]:
        all_priors = load_team_priors(League.WBB, season)
        if self.division_one_only:
            d1 = division_one_team_ids(League.WBB, season)
            all_priors = [p for p in all_priors if p.team_id in d1]
        return (
            sorted(all_priors, key=lambda p: p.team_name.lower()),
            load_league_prior(League.WBB, season),
        )

    def _rebuild_list(
        self, list_id: str, priors: list[TeamPriors], records: dict[int, str]
    ) -> None:
        ol = self.query_one(f"#{list_id}", OptionList)
        ol.clear_options()
        ol.add_options([Option(self._team_label(p, records), id=str(p.team_id)) for p in priors])

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Pick teams (Tab/Enter)  ·  / search  ·  P play  ·  C coach side  ·  "
            "1/2 scheme  ·  9/0 intensity  ·  7/8 off-scheme  ·  3/4 foul-up-3  ·  "
            "5/6 season  ·  N neutral  ·  Q quit",
            classes="intro",
        )
        self._home_priors, self._home_league_prior = self._load_priors(self.home_season)
        self._away_priors, self._away_league_prior = self._load_priors(self.away_season)
        self._home_priors_by_id = {p.team_id: p for p in self._home_priors}
        self._away_priors_by_id = {p.team_id: p for p in self._away_priors}
        self._home_records = _load_team_records(League.WBB, self.home_season)
        self._away_records = _load_team_records(League.WBB, self.away_season)

        opts_home = [
            Option(self._team_label(p, self._home_records), id=str(p.team_id))
            for p in self._home_priors
        ]
        opts_away = [
            Option(self._team_label(p, self._away_records), id=str(p.team_id))
            for p in self._away_priors
        ]

        with Horizontal():
            with Vertical():
                yield Static(
                    self._header_text(Side.HOME),
                    classes="column-header",
                    id="home_header",
                )
                yield OptionList(*opts_home, id="home_list")
                yield Static(
                    self._policy_text(self.home_policy),
                    id="home_policy", classes="policy-line",
                )
            with Vertical():
                yield Static(
                    self._header_text(Side.AWAY),
                    classes="column-header",
                    id="away_header",
                )
                yield OptionList(*opts_away, id="away_list")
                yield Static(
                    self._policy_text(self.away_policy),
                    id="away_policy", classes="policy-line",
                )
        yield Input(placeholder="Type to jump to team…", id="search-bar")
        yield Static("(no teams selected)", id="status", markup=False)
        yield Footer()

    @staticmethod
    def _team_label(p: TeamPriors, records: dict[int, str]) -> str:
        """Format team name with W-L record if available."""
        rec = records.get(p.team_id)
        if rec:
            return f"{p.team_name} ({rec})"
        return p.team_name

    @staticmethod
    def _policy_text(p: CoachPolicy) -> str:
        scheme = p.scheme.value.upper()
        if p.intensity is not DefensiveIntensity.NORMAL:
            scheme = f"{scheme}-{p.intensity.value.upper()}"
        off = p.off_scheme.value.upper()
        foul = "ON" if p.foul_when_down_3 else "off"
        two = "ON" if p.two_for_one else "off"
        hold = "ON" if p.hold_for_last else "off"
        return (
            f"D: {scheme}  O: {off}   Foul-up-3: {foul}   "
            f"2-for-1: {two}   Hold-last: {hold}"
        )

    def _refresh_policy_panels(self) -> None:
        self.query_one("#home_policy", Static).update(self._policy_text(self.home_policy))
        self.query_one("#away_policy", Static).update(self._policy_text(self.away_policy))

    def on_mount(self) -> None:
        self._update_title()
        self.query_one("#home_list", OptionList).focus()
        self._refresh_status()

    def _update_title(self) -> None:
        if self.home_season == self.away_season:
            self.app.title = f"Hoops 2026 — pick matchup ({_short_season(self.home_season)})"
        else:
            self.app.title = (
                f"Hoops 2026 — pick matchup ({_short_season(self.home_season)} "
                f"vs {_short_season(self.away_season)})"
            )

    # --- selection events --------------------------------------------------

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        list_id = event.option_list.id
        team_id = int(event.option.id)
        if list_id == "home_list":
            self.home_id = team_id
            self.query_one("#away_list", OptionList).focus()
        elif list_id == "away_list":
            self.away_id = team_id
        self._refresh_status()

    def _side_word(self, side: Side) -> str:
        """Role label for *side* — HOME/AWAY normally, or a neutral-site
        placeholder (neither team is truly "home" on a neutral court)."""
        if self.neutral_site:
            return "TEAM A" if side is Side.HOME else "TEAM B"
        return "HOME" if side is Side.HOME else "AWAY"

    def _header_text(self, side: Side) -> str:
        season = self.home_season if side is Side.HOME else self.away_season
        return f"{self._side_word(side)} ({_short_season(season)})"

    def _coach_label(self) -> str:
        if self.coach_side == "h2h":
            return "H2H"
        if self.coach_side is None:
            return "WATCH"
        return f"Coach {self._side_word(self.coach_side)}"

    def _refresh_status(self) -> None:
        home = self._home_priors_by_id.get(self.home_id) if self.home_id else None
        away = self._away_priors_by_id.get(self.away_id) if self.away_id else None
        home_label = f"{home.team_name} ({_short_season(self.home_season)})" if home else "(none)"
        away_label = f"{away.team_name} ({_short_season(self.away_season)})" if away else "(none)"
        ready = bool(home and away)
        self_play = (
            ready
            and self.home_id == self.away_id
            and self.home_season == self.away_season
        )
        if self_play:
            suffix = "Cannot play a team against itself."
        elif ready:
            suffix = "Press P to play."
        else:
            suffix = ""
        coach = self._coach_label()
        neutral_tag = "  [NEUTRAL SITE]" if self.neutral_site else ""
        text = (
            f"{self._side_word(Side.HOME)}: {home_label}    "
            f"{self._side_word(Side.AWAY)}: {away_label}    [{coach}]"
            f"{neutral_tag}    {suffix}"
        ).rstrip()
        self.last_status_text = text
        self.query_one("#status", Static).update(text)

    # --- actions ----------------------------------------------------------

    def action_toggle_neutral_site(self) -> None:
        self.neutral_site = not self.neutral_site
        self.query_one("#home_header", Static).update(self._header_text(Side.HOME))
        self.query_one("#away_header", Static).update(self._header_text(Side.AWAY))
        self._refresh_status()

    def action_focus_next_column(self) -> None:
        focused = self.focused
        home = self.query_one("#home_list", OptionList)
        away = self.query_one("#away_list", OptionList)
        if focused is home:
            away.focus()
        else:
            home.focus()

    @staticmethod
    def _next_scheme(s: DefensiveScheme) -> DefensiveScheme:
        order = list(DefensiveScheme)
        return order[(order.index(s) + 1) % len(order)]

    def _cycle_season(self, current: str) -> str:
        idx = self.seasons.index(current)
        return self.seasons[(idx + 1) % len(self.seasons)]

    def action_cycle_home_season(self) -> None:
        self.home_season = self._cycle_season(self.home_season)
        self._home_priors, self._home_league_prior = self._load_priors(self.home_season)
        self._home_priors_by_id = {p.team_id: p for p in self._home_priors}
        self._home_records = _load_team_records(League.WBB, self.home_season)
        self.home_id = None
        self._rebuild_list("home_list", self._home_priors, self._home_records)
        self.query_one("#home_header", Static).update(self._header_text(Side.HOME))
        self._update_title()
        self._refresh_status()

    def action_cycle_away_season(self) -> None:
        self.away_season = self._cycle_season(self.away_season)
        self._away_priors, self._away_league_prior = self._load_priors(self.away_season)
        self._away_priors_by_id = {p.team_id: p for p in self._away_priors}
        self._away_records = _load_team_records(League.WBB, self.away_season)
        self.away_id = None
        self._rebuild_list("away_list", self._away_priors, self._away_records)
        self.query_one("#away_header", Static).update(self._header_text(Side.AWAY))
        self._update_title()
        self._refresh_status()

    def action_cycle_coach(self) -> None:
        if self.coach_side is None:
            self.coach_side = Side.HOME
        elif self.coach_side is Side.HOME:
            self.coach_side = Side.AWAY
        elif self.coach_side is Side.AWAY:
            self.coach_side = "h2h"
        else:
            self.coach_side = None
        self._refresh_status()
        # Flash the coach mode prominently so the change is obvious.
        label = self._coach_label()
        self.notify(f"Mode: {label}", timeout=1.5)

    def action_cycle_home_scheme(self) -> None:
        self.home_policy.scheme = self._next_scheme(self.home_policy.scheme)
        self._refresh_policy_panels()

    def action_cycle_away_scheme(self) -> None:
        self.away_policy.scheme = self._next_scheme(self.away_policy.scheme)
        self._refresh_policy_panels()

    def action_cycle_home_off_scheme(self) -> None:
        self.home_policy.off_scheme = self._next_off_scheme(self.home_policy.off_scheme)
        self._refresh_policy_panels()

    def action_cycle_away_off_scheme(self) -> None:
        self.away_policy.off_scheme = self._next_off_scheme(self.away_policy.off_scheme)
        self._refresh_policy_panels()

    @staticmethod
    def _next_off_scheme(s: OffensiveScheme) -> OffensiveScheme:
        order = list(OffensiveScheme)
        return order[(order.index(s) + 1) % len(order)]

    @staticmethod
    def _next_intensity(i: DefensiveIntensity) -> DefensiveIntensity:
        order = list(DefensiveIntensity)
        return order[(order.index(i) + 1) % len(order)]

    def action_cycle_home_intensity(self) -> None:
        self.home_policy.intensity = self._next_intensity(self.home_policy.intensity)
        self._refresh_policy_panels()

    def action_cycle_away_intensity(self) -> None:
        self.away_policy.intensity = self._next_intensity(self.away_policy.intensity)
        self._refresh_policy_panels()

    def action_toggle_home_foul_up_3(self) -> None:
        self.home_policy.foul_when_down_3 = not self.home_policy.foul_when_down_3
        self._refresh_policy_panels()

    def action_toggle_away_foul_up_3(self) -> None:
        self.away_policy.foul_when_down_3 = not self.away_policy.foul_when_down_3
        self._refresh_policy_panels()

    # --- team search -------------------------------------------------------

    def action_start_search(self) -> None:
        """Open the search bar and focus it for type-ahead team search."""
        focused = self.focused
        home = self.query_one("#home_list", OptionList)
        self._search_list_id = "home_list" if focused is home else "away_list"
        search = self.query_one("#search-bar", Input)
        search.value = ""
        search.add_class("visible")
        search.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Jump to the first matching team as the user types."""
        if event.input.id != "search-bar":
            return
        query = event.value.lower()
        if not query:
            return
        ol = self.query_one(f"#{self._search_list_id}", OptionList)
        for idx in range(ol.option_count):
            option = ol.get_option_at_index(idx)
            if option.prompt.lower().startswith(query):
                ol.highlighted = idx
                ol.scroll_to_highlight()
                break

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Close search and return focus to the list on Enter."""
        if event.input.id != "search-bar":
            return
        self._close_search()

    def on_key(self, event) -> None:
        """Close search on Escape."""
        if event.key == "escape":
            search = self.query_one("#search-bar", Input)
            if search.has_class("visible"):
                self._close_search()
                event.prevent_default()
                event.stop()

    def _close_search(self) -> None:
        search = self.query_one("#search-bar", Input)
        search.remove_class("visible")
        self.query_one(f"#{self._search_list_id}", OptionList).focus()

    def action_play(self) -> None:
        if not self.home_id or not self.away_id:
            self._refresh_status()
            return
        if self.home_id == self.away_id and self.home_season == self.away_season:
            self._refresh_status()
            return
        home = self._home_priors_by_id[self.home_id]
        away = self._away_priors_by_id[self.away_id]
        game_season = max(self.home_season, self.away_season)
        rules = rules_for(League.WBB, game_season)
        from hoops.engine.sampling import make_rng

        policies = CoachPolicies(home=self.home_policy, away=self.away_policy)
        home_roster = load_roster(home.team_id, self.home_season)
        away_roster = load_roster(away.team_id, self.away_season)
        rng = make_rng(seed=self.seed)
        home_label = (
            f"{home.team_name} ({_short_season(self.home_season)})"
            if self.home_season != self.away_season
            else home.team_name
        )
        away_label = (
            f"{away.team_name} ({_short_season(self.away_season)})"
            if self.home_season != self.away_season
            else away.team_name
        )

        # Store game params for the lineup callback chain.
        self._game_params = dict(
            home=home, away=away, rules=rules, rng=rng,
            home_roster=home_roster, away_roster=away_roster,
            policies=policies, home_label=home_label, away_label=away_label,
        )
        self._chosen_starters: dict[Side, list[int] | None] = {
            Side.HOME: None, Side.AWAY: None,
        }

        if self.coach_side == "h2h":
            # H2H: home picks first, then away, then launch game.
            self.app.push_screen(StartingLineupScreen(
                home_roster, home_label, Side.HOME, self._on_lineup_chosen,
            ))
        elif self.coach_side is not None:
            # Single-player coaching: pick lineup for human side.
            roster = home_roster if self.coach_side is Side.HOME else away_roster
            label = home_label if self.coach_side is Side.HOME else away_label
            self.app.push_screen(StartingLineupScreen(
                roster, label, self.coach_side, self._on_lineup_chosen,
            ))
        else:
            # Watch mode: no lineup picking, go straight to sim.
            self._launch_watch_game()

    def _on_lineup_chosen(self, side: Side, starter_ids: list[int] | None) -> None:
        """Callback from StartingLineupScreen after one side's lineup is set."""
        self._chosen_starters[side] = starter_ids
        gp = self._game_params

        if self.coach_side == "h2h" and side is Side.HOME:
            # Home is done — now pick away lineup.
            self.app.push_screen(StartingLineupScreen(
                gp["away_roster"], gp["away_label"], Side.AWAY,
                self._on_lineup_chosen,
            ))
            return

        # All lineups chosen — launch the game.
        self._launch_coaching_game()

    def _launch_coaching_game(self) -> None:
        gp = self._game_params
        from hoops.engine.interactive import InteractiveGame
        human_side = None if self.coach_side == "h2h" else self.coach_side
        game = InteractiveGame(
            gp["home"], gp["away"], gp["rules"], gp["rng"],
            gp["home_roster"], gp["away_roster"],
            human_side=human_side,
            policies=gp["policies"],
            league=self._home_league_prior,
            neutral_site=self.neutral_site,
            home_starters=self._chosen_starters.get(Side.HOME),
            away_starters=self._chosen_starters.get(Side.AWAY),
        )
        self.app.push_screen(CoachGameScreen(
            game, gp["home_label"], gp["away_label"],
        ))

    def _launch_watch_game(self) -> None:
        gp = self._game_params
        from hoops.engine.machine import simulate_game
        _final, events = simulate_game(
            gp["home"], gp["away"], gp["rules"], gp["rng"],
            opening_possession=Side.HOME,
            league=self._home_league_prior,
            policies=gp["policies"],
            home_roster=gp["home_roster"],
            away_roster=gp["away_roster"],
            neutral_site=self.neutral_site,
        )
        self.app.push_screen(GameScreen(
            events, gp["home_label"], gp["away_label"],
            policies=gp["policies"],
            home_roster=gp["home_roster"],
            away_roster=gp["away_roster"],
        ))


# ---------------------------------------------------------------------------
# Season picker
# ---------------------------------------------------------------------------


class SeasonSelectScreen(Screen):
    """Pick a season from the fitted data on disk."""

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    SeasonSelectScreen {
        layout: vertical;
    }
    SeasonSelectScreen > Static.intro {
        height: auto;
        padding: 1 2;
    }
    SeasonSelectScreen OptionList {
        height: 1fr;
        border: solid $accent;
    }
    """

    def __init__(self, seed: int, division_one_only: bool, neutral_site: bool = False):
        super().__init__()
        self.seed = seed
        self.division_one_only = division_one_only
        self.neutral_site = neutral_site

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("Select a season  ·  Enter to pick  ·  Esc to quit", classes="intro")
        seasons = fitted_seasons(League.WBB)
        opts = [Option(s, id=s) for s in reversed(seasons)]
        yield OptionList(*opts, id="season_list")
        yield Footer()

    def on_mount(self) -> None:
        self.app.title = "Hoops 2026 — pick season"
        self.query_one("#season_list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        season = event.option.id
        seasons = fitted_seasons(League.WBB)
        self.app.push_screen(TeamSelectScreen(
            seasons=seasons,
            default_season=season,
            seed=self.seed,
            division_one_only=self.division_one_only,
            neutral_site=self.neutral_site,
        ))

