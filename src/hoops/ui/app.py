"""Textual UI for a single simulated game.

Two screens compose the experience:

- :class:`TeamSelectScreen` — pick home and away from the fitted priors
  for the season. Two ``OptionList`` widgets side by side, tab to switch
  focus, ``p`` to play once both are chosen.
- :class:`GameScreen` — scoreboard, possession log, box score, controls.

The app routes to one or the other depending on how it's launched. Direct
construction with an ``events`` list (used in tests and by the CLI when
``--home`` / ``--away`` are passed) skips the picker and lands straight
in :class:`GameScreen`.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Footer,
    Header,
    Input,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from hoops.data.distributions import (
    LeaguePrior,
    TeamPriors,
    division_one_team_ids,
    load_league_prior,
    load_team_priors,
)
from hoops.data.paths import fitted_seasons, teams_path
from hoops.data.rosters import Roster, load_roster
from hoops.engine.events import Event, fmt_clock, fmt_event
from hoops.engine.policy import CoachPolicies, CoachPolicy, DefensiveScheme, OffensiveScheme
from hoops.engine.sampling import make_rng
from hoops.ui.lineup import LineupState


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
from hoops.engine.state import Side
from hoops.league import League
from hoops.rules import rules_for
from hoops.ui.playback import PlayerBox, PlaybackState


# ---------------------------------------------------------------------------
# Game-screen widgets
# ---------------------------------------------------------------------------


class Scoreboard(Static):
    """Top-left panel: Q1..Q4 + totals, fouls per quarter, bonus indicators.

    Re-rendered whenever the parent app's playback pointer advances. The
    fouls-per-quarter row resets visually at quarter rollover (doc §6).
    """

    DEFAULT_CSS = """
    Scoreboard {
        width: auto;
        min-width: 50;
        height: 100%;
        padding: 1 2;
        border: solid $accent;
    }
    """

    home_name: reactive[str] = reactive("Home")
    away_name: reactive[str] = reactive("Away")

    def __init__(self, home_name: str, away_name: str, **kw):
        super().__init__(**kw)
        self.home_name = home_name
        self.away_name = away_name
        self._playback: PlaybackState | None = None
        self.last_text = ""

    def bind_playback(self, p: PlaybackState) -> None:
        self._playback = p
        self.refresh_view()

    def _set(self, text: str) -> None:
        self.last_text = text
        self.update(text)

    def refresh_view(self) -> None:
        if self._playback is None:
            self._set("(no game loaded)")
            return
        p = self._playback
        from hoops.ui.playback import QuarterScore

        # Compute per-quarter scores directly from processed events.
        # Track the last cumulative score seen in each quarter, then
        # subtract quarter-by-quarter to get deltas.
        max_q = max(4, p.quarter)
        # cumulative[q] = (home, away) at end of quarter q (1-indexed)
        cumulative = {}
        for e in p.events[:p.pointer]:
            cumulative[e.quarter] = (e.home_score, e.away_score)

        cols: list[QuarterScore] = []
        prev_home, prev_away = 0, 0
        for q in range(1, max_q + 1):
            h, a = cumulative.get(q, (prev_home, prev_away))
            cols.append(QuarterScore(home=h - prev_home, away=a - prev_away))
            prev_home, prev_away = h, a

        # Pad the team-name column to the longer of the two names so the
        # quarter columns line up regardless of whether teams are short
        # ("Iowa") or long ("South Carolina").
        name_w = max(len(self.home_name), len(self.away_name))
        header = (
            " " * (name_w + 1)
            + "  ".join(f"Q{i+1}" for i in range(len(cols)))
            + "   TOT"
        )
        home_row = (
            f"{self.home_name:<{name_w}} "
            + "  ".join(f"{q.home:2d}" for q in cols)
            + f"   {p.home_score:3d}"
        )
        away_row = (
            f"{self.away_name:<{name_w}} "
            + "  ".join(f"{q.away:2d}" for q in cols)
            + f"   {p.away_score:3d}"
        )

        clock_line = f"\nQ{p.quarter}  {fmt_clock(p.seconds_left)}"
        fouls_line = (
            f"Team fouls (Q{p.quarter}):  "
            f"{self.home_name} {p.home_team_fouls_q}   "
            f"{self.away_name} {p.away_team_fouls_q}"
        )
        bonus_home = "[reverse] BONUS [/reverse]" if p.in_bonus(Side.HOME) else "       "
        bonus_away = "[reverse] BONUS [/reverse]" if p.in_bonus(Side.AWAY) else "       "
        bonus_line = f"Bonus:        {self.home_name} {bonus_home}   {self.away_name} {bonus_away}"

        self._set(
            "\n".join([header, home_row, away_row, clock_line, fouls_line, bonus_line])
        )


class PossessionLog(RichLog):
    """Right-hand panel: every event applied so far, newest at the bottom."""

    DEFAULT_CSS = """
    PossessionLog {
        width: 1fr;
        height: 100%;
        border: solid $accent;
    }
    """

    home_short = "Home"
    away_short = "Away"

    def configure_team_labels(self, home: str, away: str) -> None:
        # Team labels appear only as fallback when an event has no player
        # attributed; we keep them short to fit the line.
        self.home_short = home[:14]
        self.away_short = away[:14]

    _SCORING = {"shot_made", "free_throw_made"}

    def append_event(self, e: Event) -> None:
        line = fmt_event(e, self.home_short, self.away_short)
        if e.type in self._SCORING:
            line = f"[bold]{line}[/bold]"
        self.write(line)


class BoxScorePanel(Static):
    """Bottom panel: team-level and per-player box stats.

    Press ``x`` on the GameScreen to toggle between team summary and
    per-player detail view.
    """

    DEFAULT_CSS = """
    BoxScorePanel {
        height: auto;
        padding: 1 2;
        border: solid $accent;
    }
    """

    def __init__(self, home_name: str, away_name: str, **kw):
        super().__init__(**kw)
        self.home_name = home_name
        self.away_name = away_name
        self._playback: PlaybackState | None = None
        self._fatigue = None  # FatigueTracker, set by CoachGameScreen
        self._lineup = None   # LineupState, set by CoachGameScreen
        self.last_text = ""
        self.show_players = False

    def bind_playback(self, p: PlaybackState) -> None:
        self._playback = p
        self.refresh_view()

    def bind_fatigue(self, fatigue, lineup) -> None:
        self._fatigue = fatigue
        self._lineup = lineup

    def toggle_detail(self) -> None:
        self.show_players = not self.show_players
        self.refresh_view()

    def _set(self, text: str) -> None:
        self.last_text = text
        self.update(text)

    def refresh_view(self) -> None:
        if self._playback is None:
            self._set("(no game loaded)")
            return
        if self.show_players:
            self._render_player_view()
        else:
            self._render_team_view()

    def _render_team_view(self) -> None:
        p = self._playback
        header = (
            f"{'Team':<14} {'PTS':>3} {'FG':>7} {'3P':>7} "
            f"{'FT':>7} {'OREB':>4} {'DREB':>4} {'TOV':>3} {'PF':>3}"
        )

        def row(name: str, b) -> str:
            fg = f"{b.fgm}-{b.fga}"
            tp = f"{b.fg3m}-{b.fg3a}"
            ft = f"{b.ftm}-{b.fta}"
            return (
                f"{name:<14} {b.points:>3} {fg:>7} {tp:>7} "
                f"{ft:>7} {b.orb:>4} {b.drb:>4} {b.tov:>3} {b.pf:>3}"
            )

        self._set("\n".join([
            header,
            row(self.home_name, p.home_box),
            row(self.away_name, p.away_box),
            "",
            "[x] Toggle player box scores",
        ]))

    @staticmethod
    def _player_header() -> str:
        return (
            f"{'Player':<22} {'MIN':>5} {'PTS':>3} {'FG':>7} {'3P':>7} "
            f"{'FT':>7} {'REB':>3} {'AST':>3} {'STL':>3} "
            f"{'BLK':>3} {'TOV':>3} {'PF':>2}"
        )

    @staticmethod
    def _player_row(b: PlayerBox) -> str:
        name = b.name[:21]
        fg = f"{b.fgm}-{b.fga}"
        tp = f"{b.fg3m}-{b.fg3a}"
        ft = f"{b.ftm}-{b.fta}"
        return (
            f"{name:<22} {b.minutes_display:>5} {b.points:>3} {fg:>7} {tp:>7} "
            f"{ft:>7} {b.reb:>3} {b.ast:>3} {b.stl:>3} "
            f"{b.blk:>3} {b.tov:>3} {b.pf:>2}"
        )

    def _fatigue_for_side(self, side: Side) -> dict[str, float]:
        if self._fatigue is None or self._lineup is None:
            return {}
        roster = self._lineup.roster(side)
        return {
            p.name: self._fatigue.fatigue(p.player_id)
            for p in roster.players
            if p.player_id in self._fatigue._fatigue
        }

    @staticmethod
    def _fatigue_tag(level: float) -> str:
        if level >= 0.7:
            return " GASSED"
        if level >= 0.5:
            return " TIRED"
        return ""

    def _render_player_view(self) -> None:
        p = self._playback
        lines: list[str] = []

        for label, players, side in [
            (self.home_name, p.home_players, Side.HOME),
            (self.away_name, p.away_players, Side.AWAY),
        ]:
            fatigue_map = self._fatigue_for_side(side)
            on_court_names = (
                {pl.name for pl in self._lineup.on_court(side)}
                if self._lineup else set()
            )
            lines.append(f"── {label} ──")
            lines.append(self._player_header())
            sorted_players = sorted(
                players.values(), key=lambda b: b.points, reverse=True,
            )
            for pb in sorted_players:
                tag = self._fatigue_tag(fatigue_map.get(pb.name, 0.0))
                bench = "" if pb.name in on_court_names or not on_court_names else " [BCH]"
                lines.append(self._player_row(pb) + tag + bench)
            lines.append("")

        lines.append("[x] Toggle team summary")
        self._set("\n".join(lines))


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


_AUTO_SPEEDS: list[tuple[str, float]] = [
    ("Slow", 2.5),
    ("Normal", 1.5),
    ("Fast", 0.8),
    ("Turbo", 0.3),
]
_DEFAULT_SPEED_IDX = 1  # "Normal"


class GameScreen(Screen):
    """Plays back a single game's event log."""

    BINDINGS = [
        Binding("space", "next_possession", "Next poss"),
        Binding("s", "step_one", "Step"),
        Binding("e", "end_quarter", "End qtr"),
        Binding("a", "toggle_auto", "Auto-play"),
        Binding("f", "run_to_end", "Run to end"),
        Binding("b", "open_subs", "Subs"),
        Binding("x", "toggle_box_detail", "Box detail"),
        Binding("plus,equals", "speed_up", "+Speed", show=False),
        Binding("minus,underscore", "speed_down", "-Speed", show=False),
        Binding("escape", "back", "Back"),
    ]

    def __init__(
        self,
        events: list[Event],
        home_name: str,
        away_name: str,
        policies: CoachPolicies | None = None,
        home_roster: Roster | None = None,
        away_roster: Roster | None = None,
        lineup: LineupState | None = None,
    ):
        super().__init__()
        self.home_name = home_name
        self.away_name = away_name
        self.policies = policies
        self.home_roster = home_roster
        self.away_roster = away_roster
        if home_roster is not None and away_roster is not None:
            from hoops.engine.attribution import attribute_players
            events = attribute_players(
                events, home_roster, away_roster, make_rng(seed=1),
            )
        self.events = events
        if lineup is None and home_roster is not None and away_roster is not None:
            lineup = LineupState.with_default_starters(
                home_roster, away_roster, make_rng(seed=0),
            )
        self.lineup = lineup
        self.playback = PlaybackState.from_events(events, lineup=lineup)
        self._auto_timer = None
        self._speed_idx = _DEFAULT_SPEED_IDX

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="top"):
            self.scoreboard = Scoreboard(self.home_name, self.away_name)
            yield self.scoreboard
            self.event_log = PossessionLog(
                highlight=False, markup=True, wrap=False, id="log"
            )
            yield self.event_log
        self.box = BoxScorePanel(self.home_name, self.away_name)
        yield self.box
        yield Footer()

    def on_mount(self) -> None:
        self.app.title = f"Hoops 2026 — {self.home_name} @ {self.away_name}"
        self.scoreboard.bind_playback(self.playback)
        self.box.bind_playback(self.playback)
        self.event_log.configure_team_labels(self.home_name, self.away_name)
        e = self.playback.step_one()
        if e is not None:
            self.event_log.append_event(e)
        self._refresh_panels()

    # --- actions ----------------------------------------------------------

    def action_step_one(self) -> None:
        e = self.playback.step_one()
        if e is not None:
            self.event_log.append_event(e)
        self._refresh_panels()

    def action_next_possession(self) -> None:
        applied = self.playback.step_to_next_score_change()
        for e in applied:
            self.event_log.append_event(e)
        self._refresh_panels()

    def action_end_quarter(self) -> None:
        applied = self.playback.step_to_end_of_quarter()
        for e in applied:
            self.event_log.append_event(e)
        self._refresh_panels()

    def action_run_to_end(self) -> None:
        applied = self.playback.step_to_end()
        for e in applied:
            self.event_log.append_event(e)
        self._refresh_panels()

    def action_back(self) -> None:
        if self.playback.is_done:
            self._stop_auto()
            if len(self.app.screen_stack) > 2:
                self.app.pop_screen()
            return
        if len(self.app.screen_stack) > 2:
            self._stop_auto()
            self.app.push_screen(
                ConfirmQuitScreen(),
                callback=self._on_confirm_quit,
            )

    def _on_confirm_quit(self, confirmed: bool) -> None:
        if confirmed:
            self.app.pop_screen()

    def action_open_subs(self) -> None:
        """Open the substitution panel.

        Only fires when a lineup is bound and the game is in a dead-ball
        state (foul or non-steal turnover). Live-ball pressing is a
        no-op with a brief notification so the user understands the
        rule rather than thinking the binding is broken."""
        if self.lineup is None:
            return
        if not self.playback.is_dead_ball:
            self.app.notify(
                "Subs allowed only at a dead ball (foul or non-steal turnover).",
                severity="warning",
                timeout=3,
            )
            return
        self._stop_auto()
        self.app.push_screen(SubScreen(self.lineup, self.home_name, self.away_name))

    def _auto_interval(self) -> float:
        return _AUTO_SPEEDS[self._speed_idx][1]

    def _speed_label(self) -> str:
        return _AUTO_SPEEDS[self._speed_idx][0]

    def action_toggle_auto(self) -> None:
        """Auto-advance possessions until toggled off or game ends."""
        if self._auto_timer is not None:
            self._stop_auto()
            return
        self._auto_timer = self.set_interval(
            self._auto_interval(), self._auto_tick
        )

    def _auto_tick(self) -> None:
        if self.playback.is_done:
            self._stop_auto()
            return
        self.action_next_possession()

    def _stop_auto(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer = None

    def _restart_auto_if_running(self) -> None:
        """Restart the auto-play timer at the current speed if it's active."""
        if self._auto_timer is not None:
            self._stop_auto()
            self._auto_timer = self.set_interval(
                self._auto_interval(), self._auto_tick
            )

    def action_speed_up(self) -> None:
        if self._speed_idx < len(_AUTO_SPEEDS) - 1:
            self._speed_idx += 1
            self._restart_auto_if_running()
            self.notify(f"Speed: {self._speed_label()}", timeout=1.0)

    def action_speed_down(self) -> None:
        if self._speed_idx > 0:
            self._speed_idx -= 1
            self._restart_auto_if_running()
            self.notify(f"Speed: {self._speed_label()}", timeout=1.0)

    def action_toggle_box_detail(self) -> None:
        self.box.toggle_detail()

    # --- rendering --------------------------------------------------------

    def _refresh_panels(self) -> None:
        self.scoreboard.refresh_view()
        self.box.refresh_view()
        if self.playback.is_done and not self._showed_post_game:
            self._showed_post_game = True
            self._stop_auto()
            self.app.push_screen(PostGameScreen(
                self.playback, self.home_name, self.away_name,
            ))

    _showed_post_game: bool = False


class ConfirmQuitScreen(Screen):
    """Y/N confirmation before leaving a game in progress."""

    BINDINGS = [
        Binding("y", "confirm_yes", "Yes"),
        Binding("n", "confirm_no", "No"),
        Binding("escape", "confirm_no", "No"),
    ]

    DEFAULT_CSS = """
    ConfirmQuitScreen {
        align: center middle;
    }
    ConfirmQuitScreen > Static {
        width: 40;
        height: auto;
        padding: 2 4;
        background: $surface;
        border: thick $error;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Quit this game?\n\n[Y] Yes  [N] No")

    def action_confirm_yes(self) -> None:
        self.dismiss(True)

    def action_confirm_no(self) -> None:
        self.dismiss(False)


class PostGameScreen(Screen):
    """Final summary shown when the game ends."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back"),
        Binding("q", "quit_app", "Quit"),
    ]

    DEFAULT_CSS = """
    PostGameScreen {
        layout: vertical;
    }
    PostGameScreen > Static.title {
        height: auto;
        padding: 1 2;
        text-align: center;
        text-style: bold;
        background: $accent;
        color: $background;
    }
    PostGameScreen > Static.summary {
        height: auto;
        padding: 1 2;
    }
    PostGameScreen > Static.box {
        height: 1fr;
        padding: 0 2;
        overflow-y: auto;
    }
    PostGameScreen > Static.hint {
        height: auto;
        padding: 1 2;
        text-align: center;
    }
    """

    def __init__(self, playback: PlaybackState, home_name: str, away_name: str):
        super().__init__()
        self._playback = playback
        self.home_name = home_name
        self.away_name = away_name

    def compose(self) -> ComposeResult:
        p = self._playback
        winner = self.home_name if p.home_score > p.away_score else self.away_name
        margin = abs(p.home_score - p.away_score)
        if p.home_score == p.away_score:
            title = f"FINAL: {self.home_name} {p.home_score} — {self.away_name} {p.away_score} (TIE)"
        else:
            title = f"FINAL: {self.home_name} {p.home_score} — {self.away_name} {p.away_score}"

        yield Header(show_clock=False)
        yield Static(title, classes="title")

        summary_lines = self._build_summary(p, winner, margin)
        yield Static("\n".join(summary_lines), classes="summary")

        box_lines = self._build_box_scores(p)
        yield Static("\n".join(box_lines), classes="box")

        yield Static("Esc: back to game log  ·  Q: quit", classes="hint")
        yield Footer()

    def _build_summary(self, p: PlaybackState, winner: str, margin: int) -> list[str]:
        lines: list[str] = []
        hb, ab = p.home_box, p.away_box

        def pct(m: int, a: int) -> str:
            return f"{m / a * 100:.0f}%" if a > 0 else "—"

        lines.append(f"{'':>18} {'HOME':>8} {'AWAY':>8}")
        lines.append(f"{'FG':>18} {pct(hb.fgm, hb.fga):>8} {pct(ab.fgm, ab.fga):>8}")
        lines.append(f"{'3PT':>18} {pct(hb.fg3m, hb.fg3a):>8} {pct(ab.fg3m, ab.fg3a):>8}")
        lines.append(f"{'FT':>18} {pct(hb.ftm, hb.fta):>8} {pct(ab.ftm, ab.fta):>8}")
        lines.append(f"{'Rebounds':>18} {hb.orb + hb.drb:>8} {ab.orb + ab.drb:>8}")
        lines.append(f"{'Assists':>18} {hb.ast:>8} {ab.ast:>8}")
        lines.append(f"{'Turnovers':>18} {hb.tov:>8} {ab.tov:>8}")

        potg = self._player_of_game(p)
        if potg:
            lines.append("")
            lines.append(f"Player of the Game: {potg.name} — {potg.points} PTS, "
                         f"{potg.reb} REB, {potg.ast} AST")

        return lines

    @staticmethod
    def _player_of_game(p: PlaybackState) -> PlayerBox | None:
        all_players = list(p.home_players.values()) + list(p.away_players.values())
        if not all_players:
            return None
        def score(b: PlayerBox) -> float:
            return b.points + b.reb * 1.2 + b.ast * 1.5 + b.stl * 2.0 + b.blk * 2.0 - b.tov * 1.0
        return max(all_players, key=score)

    def _build_box_scores(self, p: PlaybackState) -> list[str]:
        lines: list[str] = []
        header = BoxScorePanel._player_header()
        for label, players in [
            (self.home_name, p.home_players),
            (self.away_name, p.away_players),
        ]:
            lines.append(f"── {label} ──")
            lines.append(header)
            sorted_players = sorted(
                players.values(), key=lambda b: b.seconds, reverse=True,
            )
            for pb in sorted_players:
                lines.append(BoxScorePanel._player_row(pb))
            lines.append("")
        return lines

    def action_dismiss_screen(self) -> None:
        self.app.pop_screen()

    def action_quit_app(self) -> None:
        self.app.exit()


class SubScreen(Screen):
    """In-game substitution panel.

    Two columns (HOME / AWAY). For each side, the on-court 5 sit above
    the bench. Press a number 1-5 to select the on-court player to pull,
    then choose a bench replacement with the up/down keys + Enter, or
    press the matching letter (a, b, c, ...) for the row. Press Tab to
    switch sides; Esc to close without further changes (changes already
    made via Enter take effect immediately).
    """

    BINDINGS = [
        Binding("tab", "switch_side", "Switch side"),
        Binding("escape", "close", "Done"),
        # Pull starters 1..5
        Binding("1", "pull('0')", "Pull #1"),
        Binding("2", "pull('1')", "Pull #2"),
        Binding("3", "pull('2')", "Pull #3"),
        Binding("4", "pull('3')", "Pull #4"),
        Binding("5", "pull('4')", "Pull #5"),
        # Send in bench a..h
        Binding("a", "send_in('0')", "In #a"),
        Binding("b", "send_in('1')", "In #b"),
        Binding("c", "send_in('2')", "In #c"),
        Binding("d", "send_in('3')", "In #d"),
        Binding("e", "send_in('4')", "In #e"),
        Binding("f", "send_in('5')", "In #f"),
        Binding("g", "send_in('6')", "In #g"),
        Binding("h", "send_in('7')", "In #h"),
    ]

    DEFAULT_CSS = """
    SubScreen {
        align: center middle;
    }
    SubScreen > Vertical {
        width: 90;
        height: 30;
        border: thick $primary;
        padding: 1 2;
    }
    SubScreen .col {
        width: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    SubScreen .col-header {
        background: $accent;
        color: $background;
        height: 1;
        padding: 0 1;
    }
    SubScreen .section-label {
        color: $accent;
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, lineup: LineupState, home_name: str, away_name: str):
        super().__init__()
        self.lineup = lineup
        self.home_name = home_name
        self.away_name = away_name
        self._active_side = Side.HOME
        self._pull_idx: int | None = None
        # Bindings for 1..5 (pull a starter) and a..h (sub a benchwarmer)
        # are registered dynamically in on_mount via bind().

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical():
            yield Static(
                "Substitutions  ·  Tab switch side  ·  1-5 pull starter  ·  "
                "a-h send in bench  ·  Esc done",
                classes="intro",
            )
            with Horizontal():
                with Vertical(classes="col"):
                    yield Static(self.home_name, classes="col-header", id="home-h")
                    yield Static(self._lineup_block(Side.HOME), id="home-body")
                with Vertical(classes="col"):
                    yield Static(self.away_name, classes="col-header", id="away-h")
                    yield Static(self._lineup_block(Side.AWAY), id="away-body")
            yield Static(self._status_text(), id="sub-status")
        yield Footer()

    def on_mount(self) -> None:
        self.app.title = "Substitutions"

    # --- helpers ----------------------------------------------------------

    def _lineup_block(self, side: Side) -> str:
        # Display is "post-pending" so the user sees what the lineup will
        # look like after the dead ball commits. A players who's been
        # queued to come in is marked with "(in →)"; the player they're
        # replacing is marked "(→ out)" beneath, in the bench section.
        post_on_court = self.lineup.pending_on_court(side)
        actual_on_court = self.lineup.on_court(side)
        actual_ids = [p.player_id for p in actual_on_court]
        post_ids = [p.player_id for p in post_on_court]
        active = " <-- selected" if side is self._active_side else ""
        pending_tag = "  (PENDING)" if self.lineup.has_pending(side) else ""
        rows = [f"On court{active}{pending_tag}:", ""]
        for idx, p in enumerate(post_on_court):
            marker = " *" if (
                self._pull_idx == idx and side is self._active_side
            ) else "  "
            tag = "  (in →)" if p.player_id not in actual_ids else ""
            rows.append(f"{marker}{idx + 1}. {p.name}  ({int(p.minutes)} min){tag}")
        rows += ["", "Bench:", ""]
        bench = self.lineup.pending_bench(side)[:8]
        for idx, p in enumerate(bench):
            letter = "abcdefgh"[idx]
            tag = "  (→ out)" if p.player_id in actual_ids else ""
            rows.append(f"  {letter}. {p.name}  ({int(p.minutes)} min){tag}")
        return "\n".join(rows)

    def _status_text(self) -> str:
        side_label = "HOME" if self._active_side is Side.HOME else "AWAY"
        any_pending = (
            self.lineup.has_pending(Side.HOME)
            or self.lineup.has_pending(Side.AWAY)
        )
        wait_note = (
            "  Subs take effect on the next dead ball (foul or non-steal turnover)."
            if any_pending else ""
        )
        if self._pull_idx is None:
            return f"[{side_label}] Pick a starter to pull (1-5).{wait_note}"
        on_court = self.lineup.pending_on_court(self._active_side)
        starter = on_court[self._pull_idx]
        return (
            f"[{side_label}] Pulling {starter.name}. "
            f"Press a-h to bring in a bench player, or 1-5 to pick a different starter."
            f"{wait_note}"
        )

    def _refresh(self) -> None:
        self.query_one("#home-body", Static).update(self._lineup_block(Side.HOME))
        self.query_one("#away-body", Static).update(self._lineup_block(Side.AWAY))
        self.query_one("#sub-status", Static).update(self._status_text())

    # --- actions ----------------------------------------------------------

    def action_switch_side(self) -> None:
        self._active_side = self._active_side.other
        self._pull_idx = None
        self._refresh()

    def action_pull(self, idx: str) -> None:
        i = int(idx)
        post_on_court = self.lineup.pending_on_court(self._active_side)
        if 0 <= i < len(post_on_court):
            self._pull_idx = i
        self._refresh()

    def action_send_in(self, idx: str) -> None:
        if self._pull_idx is None:
            return
        i = int(idx)
        bench = self.lineup.pending_bench(self._active_side)[:8]
        if not (0 <= i < len(bench)):
            return
        post_on_court = self.lineup.pending_on_court(self._active_side)
        try:
            self.lineup.request_substitution(
                self._active_side,
                off_player_id=post_on_court[self._pull_idx].player_id,
                on_player_id=bench[i].player_id,
            )
        except Exception:
            pass
        self._pull_idx = None
        self._refresh()

    def action_close(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Starting lineup picker
# ---------------------------------------------------------------------------


class StartingLineupScreen(Screen):
    """Pre-game screen for choosing starting five from the roster.

    Shows all players numbered 1-N. Selected players are marked with ``*``.
    Toggle a player with their number key. Must have exactly 5 selected
    to confirm with Enter. In H2H, shown once per side.
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm lineup"),
        Binding("escape", "skip", "Use defaults"),
    ]

    DEFAULT_CSS = """
    StartingLineupScreen {
        layout: vertical;
    }
    StartingLineupScreen .intro {
        height: auto;
        padding: 1 2;
    }
    StartingLineupScreen .roster-body {
        height: 1fr;
        padding: 0 2;
        border: solid $accent;
        overflow-y: auto;
    }
    StartingLineupScreen #lineup-status {
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    """

    def __init__(
        self,
        roster: "Roster",
        team_name: str,
        side: Side,
        callback: "Callable[[Side, list[int]], None]",
    ):
        super().__init__()
        from hoops.data.rosters import Roster
        self._roster: Roster = roster
        self._team_name = team_name
        self._side = side
        self._callback = callback
        # Start with default top-5 selected.
        self._selected: set[int] = set(range(min(5, len(roster.players))))

    def compose(self) -> ComposeResult:
        side_label = "HOME" if self._side is Side.HOME else "AWAY"
        yield Header(show_clock=False)
        yield Static(
            f"Pick starting 5 for {side_label}: {self._team_name}  ·  "
            "1-9,0/a-e toggle  ·  ★ = starter  ·  Enter confirm  ·  Esc use defaults",
            classes="intro",
        )
        yield Static(self._roster_text(), classes="roster-body", id="roster-body")
        yield Static(self._status_text(), id="lineup-status")
        yield Footer()

    def on_mount(self) -> None:
        self.app.title = f"Starting Lineup — {self._team_name}"
        # Bind number keys for up to 15 players (0-9 plus a-e).
        for i in range(min(len(self._roster.players), 10)):
            self._bind_key(str(i), i)
        for i, letter in enumerate("abcde"):
            idx = 10 + i
            if idx < len(self._roster.players):
                self._bind_key(letter, idx)

    def _bind_key(self, key: str, idx: int) -> None:
        """Dynamically bind a key to toggle player at idx."""
        # Use on_key instead of dynamic bindings for Textual 8 compat.
        pass  # handled in on_key below

    def on_key(self, event) -> None:
        key = event.key
        idx = None
        if key in "123456789":
            idx = int(key) - 1  # 1-indexed keys → 0-indexed roster
        elif key == "0":
            idx = 9  # key "0" maps to player 10
        elif key in "abcde":
            idx = 10 + "abcde".index(key)
        if idx is not None and 0 <= idx < len(self._roster.players):
            if idx in self._selected:
                self._selected.discard(idx)
            else:
                self._selected.add(idx)
            self._refresh()
            event.prevent_default()
            event.stop()

    def _roster_text(self) -> str:
        # Per-game stats table with selection markers.
        header = (
            "     KEY  PLAYER               POS   GP   MPG   PPG   RPG   APG  "
            "FG%   3P%   FT%"
        )
        sep = "     " + "─" * (len(header) - 5)
        rows = [header, sep]
        for i, p in enumerate(self._roster.players):
            if i < 9:
                key_label = str(i + 1)  # 1-9
            elif i == 9:
                key_label = "0"         # 0 = player 10
            elif i < 15:
                key_label = "abcde"[i - 10]
            else:
                key_label = " "
            marker = " ★ " if i in self._selected else "   "
            gp = max(p.games_played, 1)
            ppg = p.points / gp
            rpg = (p.orb + p.drb) / gp
            apg = p.ast / gp
            mpg = p.minutes / gp
            fg_pct = (p.fgm / p.fga * 100) if p.fga > 0 else 0.0
            fg3_pct = (p.fg3m / p.fg3a * 100) if p.fg3a > 0 else 0.0
            ft_pct = (p.ftm / p.fta * 100) if p.fta > 0 else 0.0
            pos = p.position or "—"
            name = p.name[:20]
            rows.append(
                f"{marker}  {key_label}   {name:<20s} {pos:<4s} {gp:3d}  "
                f"{mpg:4.1f}  {ppg:5.1f} {rpg:5.1f} {apg:5.1f}  "
                f"{fg_pct:4.1f}  {fg3_pct:4.1f}  {ft_pct:4.1f}"
            )
        return "\n".join(rows)

    def _status_text(self) -> str:
        n = len(self._selected)
        if n == 5:
            return "5 players selected. Press Enter to confirm, or keep adjusting."
        elif n < 5:
            return f"{n}/5 selected — need {5 - n} more."
        else:
            return f"{n}/5 selected — remove {n - 5} to get to 5."

    def _refresh(self) -> None:
        self.query_one("#roster-body", Static).update(self._roster_text())
        self.query_one("#lineup-status", Static).update(self._status_text())

    def action_confirm(self) -> None:
        if len(self._selected) != 5:
            self.notify(f"Need exactly 5 starters, have {len(self._selected)}", timeout=2.0)
            return
        starter_ids = [self._roster.players[i].player_id for i in sorted(self._selected)]
        self.app.pop_screen()
        self._callback(self._side, starter_ids)

    def action_skip(self) -> None:
        """Use default starters (top 5 by minutes)."""
        self.app.pop_screen()
        self._callback(self._side, None)


class CoachGameScreen(Screen):
    """Interactive coaching mode: possession-by-possession with human vs CPU."""

    BINDINGS = [
        Binding("space", "next_possession", "Next poss"),
        Binding("a", "toggle_auto", "Auto-play"),
        Binding("b", "open_subs", "Subs"),
        Binding("d", "cycle_scheme", "D-Scheme"),
        Binding("o", "cycle_off_scheme", "O-Scheme"),
        Binding("t", "call_timeout", "Timeout"),
        Binding("s", "save_game", "Save"),
        Binding("l", "load_game", "Load"),
        Binding("x", "toggle_box_detail", "Box detail"),
        Binding("f", "run_to_end", "Sim to end"),
        Binding("plus,equals", "speed_up", "+Speed", show=False),
        Binding("minus,underscore", "speed_down", "-Speed", show=False),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    CoachGameScreen {
        layout: vertical;
    }
    CoachGameScreen #top {
        height: 1fr;
    }
    CoachGameScreen #coach-bar {
        height: auto;
        padding: 0 2;
        background: $primary;
        color: $background;
    }
    """

    def __init__(self, game, home_name: str, away_name: str, tournament_mode: bool = False):
        super().__init__()
        from hoops.engine.interactive import InteractiveGame
        self.game: InteractiveGame = game
        self.home_name = home_name
        self.away_name = away_name
        self.tournament_mode = tournament_mode
        self.h2h_mode = game.human_side is None
        self._active_coach: Side = Side.HOME
        self._awaiting_away: bool = False
        # Sub-request flags: when True the sub screen opens at the next dead ball.
        self._sub_requested: dict[Side, bool] = {
            Side.HOME: False, Side.AWAY: False,
        }
        self._sub_queue: list[Side] = []
        self._at_dead_ball: bool = False
        self.playback = PlaybackState.from_events(
            list(self.game.all_events), lineup=self.game.lineup,
        )
        self._auto_timer = None
        self._speed_idx = _DEFAULT_SPEED_IDX
        self._showed_post_game = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(self._coach_bar_text(), id="coach-bar")
        with Horizontal(id="top"):
            self.scoreboard = Scoreboard(self.home_name, self.away_name)
            yield self.scoreboard
            self.event_log = PossessionLog(
                highlight=False, markup=True, wrap=False, id="log",
            )
            yield self.event_log
        self.box = BoxScorePanel(self.home_name, self.away_name)
        yield self.box
        yield Footer()

    def on_mount(self) -> None:
        if self.h2h_mode:
            self.app.title = "Hoops 2026 — H2H"
        else:
            side_label = "HOME" if self.game.human_side is Side.HOME else "AWAY"
            self.app.title = f"Hoops 2026 — Coaching {side_label}"
        self.scoreboard.bind_playback(self.playback)
        self.box.bind_playback(self.playback)
        self.box.bind_fatigue(self.game.fatigue, self.game.lineup)
        self.event_log.configure_team_labels(self.home_name, self.away_name)
        # Play the tip-off event.
        e = self.playback.step_one()
        if e is not None:
            self.event_log.append_event(e)
        self._refresh_panels()

    def _sub_requested_label(self) -> str:
        """Return a short label if subs are requested, or empty string."""
        parts = []
        for side, label in ((Side.HOME, "H"), (Side.AWAY, "A")):
            if self._sub_requested[side]:
                parts.append(label)
        if not parts:
            return ""
        return f"  Subs pending: {','.join(parts)}"

    def _open_subs_if_requested(self) -> None:
        """At a dead ball, open the sub screen for any side that requested it."""
        if self.game.is_game_over:
            return
        sides_to_sub = [s for s in (Side.HOME, Side.AWAY) if self._sub_requested[s]]
        if not sides_to_sub:
            return
        self._stop_auto()
        # Open subs for the first requesting side; when that screen closes
        # we chain to the next side if needed.
        self._sub_queue = list(sides_to_sub)
        self._open_next_sub_screen()

    def _open_next_sub_screen(self) -> None:
        """Pop the next side off the sub queue and push its CoachSubScreen."""
        if not self._sub_queue:
            return
        side = self._sub_queue.pop(0)
        self._sub_requested[side] = False

        def on_sub_screen_closed(_result) -> None:
            self._update_coach_bar()
            if self._sub_queue:
                self._open_next_sub_screen()

        self.app.push_screen(
            CoachSubScreen(
                self.game, self.home_name, self.away_name,
                sub_side=side,
            ),
            callback=on_sub_screen_closed,
        )

    def _coach_bar_text(self) -> str:
        h_to = self.game.policies.home.timeouts_remaining
        a_to = self.game.policies.away.timeouts_remaining
        h_scheme = self.game.policies.home.scheme.value.upper()
        a_scheme = self.game.policies.away.scheme.value.upper()
        pending = self._sub_requested_label()

        if self.h2h_mode:
            home_marker = ">" if self._active_coach is Side.HOME else " "
            away_marker = ">" if self._active_coach is Side.AWAY else " "
            active_name = self.home_name if self._active_coach is Side.HOME else self.away_name
            h_off = self.game.policies.home.off_scheme.value.upper()
            a_off = self.game.policies.away.off_scheme.value.upper()
            return (
                f"{home_marker}HOME: {self.home_name} [{h_scheme}/{h_off}] {h_to}TO  |  "
                f"{away_marker}AWAY: {self.away_name} [{a_scheme}/{a_off}] {a_to}TO  ·  "
                f"{active_name}'s turn{pending}  ·  "
                "D: scheme  O: off-scheme  B: subs  T: timeout  Space: done"
            )
        else:
            side = "HOME" if self.game.human_side is Side.HOME else "AWAY"
            scheme = self.game.human_policy().scheme.value.upper()
            off_scheme = self.game.human_policy().off_scheme.value.upper()
            cpu_scheme = self.game.cpu_policy().scheme.value.upper()
            cpu_off_scheme = self.game.cpu_policy().off_scheme.value.upper()
            cpu_personality = self.game.cpu_coach.personality.value.capitalize()
            return (
                f"Coaching: {side}  ·  D: {scheme}  O: {off_scheme}  ·  "
                f"CPU: {cpu_scheme}/{cpu_off_scheme} ({cpu_personality})  ·  "
                f"TOs: {self.home_name} {h_to} | {self.away_name} {a_to}{pending}  ·  "
                "T: timeout  ·  B: subs  ·  D: scheme  ·  O: off-scheme  ·  S: save  ·  L: load"
            )

    # --- actions ----------------------------------------------------------

    @staticmethod
    def _is_stoppage(result) -> bool:
        """True when the result represents a dead ball or between-quarter break."""
        if result.is_dead_ball:
            return True
        return any(e.type in ("quarter_end", "overtime_start") for e in result.events)

    def action_next_possession(self) -> None:
        if self.game.is_game_over:
            return
        self._at_dead_ball = False

        if self.h2h_mode and self._awaiting_away:
            # Away coach pressed Space -> advance possession
            self._awaiting_away = False
            self._active_coach = Side.HOME
            result = self.game.step_possession()
            self._sync_events(result.events)
            if self._is_stoppage(result) and not self.game.is_game_over:
                self._at_dead_ball = True
                self._open_subs_if_requested()
                self._active_coach = Side.HOME
                self._update_coach_bar()
            self._check_game_over()
        elif self.h2h_mode and self._active_coach is Side.HOME:
            # Home coach pressed Space -> switch to away
            self._active_coach = Side.AWAY
            self._awaiting_away = True
            self._update_coach_bar()
        else:
            # Single-player mode: advance directly
            result = self.game.step_possession()
            self._sync_events(result.events)
            if self._is_stoppage(result) and not self.game.is_game_over:
                self._at_dead_ball = True
                self._open_subs_if_requested()
                self._update_coach_bar()
            self._check_game_over()

    def _auto_interval(self) -> float:
        return _AUTO_SPEEDS[self._speed_idx][1]

    def _speed_label(self) -> str:
        return _AUTO_SPEEDS[self._speed_idx][0]

    def action_toggle_auto(self) -> None:
        if self.h2h_mode:
            return
        if self._auto_timer is not None:
            self._stop_auto()
            return
        self._auto_timer = self.set_interval(
            self._auto_interval(), self._auto_tick,
        )

    def _auto_tick(self) -> None:
        if self.game.is_game_over:
            self._stop_auto()
            return
        self.action_next_possession()

    def _stop_auto(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer = None

    def _restart_auto_if_running(self) -> None:
        if self._auto_timer is not None:
            self._stop_auto()
            self._auto_timer = self.set_interval(
                self._auto_interval(), self._auto_tick,
            )

    def action_speed_up(self) -> None:
        if self._speed_idx < len(_AUTO_SPEEDS) - 1:
            self._speed_idx += 1
            self._restart_auto_if_running()
            self.notify(f"Speed: {self._speed_label()}", timeout=1.0)

    def action_speed_down(self) -> None:
        if self._speed_idx > 0:
            self._speed_idx -= 1
            self._restart_auto_if_running()
            self.notify(f"Speed: {self._speed_label()}", timeout=1.0)

    def action_run_to_end(self) -> None:
        if self.h2h_mode:
            return
        self._stop_auto()
        while not self.game.is_game_over:
            result = self.game.step_possession()
            self._sync_events(result.events)
        self._check_game_over()

    def action_open_subs(self) -> None:
        if self.game.is_game_over:
            return
        sub_side = self._active_coach if self.h2h_mode else self.game.human_side
        if self._at_dead_ball:
            # Already at a dead ball — open the sub screen immediately.
            self._stop_auto()
            self._sub_requested[sub_side] = False

            def on_sub_screen_closed(_result) -> None:
                self._update_coach_bar()

            self.app.push_screen(
                CoachSubScreen(
                    self.game, self.home_name, self.away_name,
                    sub_side=sub_side,
                ),
                callback=on_sub_screen_closed,
            )
        elif self._sub_requested[sub_side]:
            # Toggle off — cancel the request.
            self._sub_requested[sub_side] = False
            self.notify("Sub request cancelled.", timeout=1.5)
            self._update_coach_bar()
        else:
            self._sub_requested[sub_side] = True
            self.notify("Subs queued — will open at next dead ball.", timeout=1.5)
            self._update_coach_bar()

    def action_cycle_scheme(self) -> None:
        order = list(DefensiveScheme)
        if self.h2h_mode:
            current = self.game.policies.for_side(self._active_coach).scheme
            next_scheme = order[(order.index(current) + 1) % len(order)]
            self.game.set_scheme(self._active_coach, next_scheme)
        else:
            current = self.game.human_policy().scheme
            next_scheme = order[(order.index(current) + 1) % len(order)]
            self.game.set_human_scheme(next_scheme)
        self._update_coach_bar()

    def action_cycle_off_scheme(self) -> None:
        order = list(OffensiveScheme)
        if self.h2h_mode:
            current = self.game.policies.for_side(self._active_coach).off_scheme
            next_scheme = order[(order.index(current) + 1) % len(order)]
            self.game.set_off_scheme(self._active_coach, next_scheme)
        else:
            current = self.game.human_policy().off_scheme
            next_scheme = order[(order.index(current) + 1) % len(order)]
            self.game.set_human_off_scheme(next_scheme)
        self._update_coach_bar()

    def action_call_timeout(self) -> None:
        if self.game.is_game_over:
            return
        side = self._active_coach if self.h2h_mode else self.game.human_side
        try:
            events = self.game.call_timeout(side)
        except ValueError:
            return
        self._at_dead_ball = True
        self._sync_events(events)
        self._open_subs_if_requested()
        self._update_coach_bar()

    def action_save_game(self) -> None:
        from hoops.engine.save import save_game
        season = max(self.game.home_priors.season, self.game.away_priors.season)
        d = self.game.to_save_dict()
        if self.h2h_mode:
            d["h2h_active_coach"] = int(self._active_coach)
            d["h2h_awaiting_away"] = self._awaiting_away
        save_game(d, self.home_name, self.away_name, season)
        self.query_one("#coach-bar", Static).update("Game saved!")
        self.set_timer(2.0, lambda: self._update_coach_bar())

    def action_load_game(self) -> None:
        from hoops.engine.save import has_save, load_save, save_path_for
        from hoops.engine.interactive import InteractiveGame
        season = max(self.game.home_priors.season, self.game.away_priors.season)
        if not has_save(self.home_name, self.away_name, season):
            self.query_one("#coach-bar", Static).update("No save found")
            self.set_timer(2.0, lambda: self._update_coach_bar())
            return
        path = save_path_for(self.home_name, self.away_name, season)
        d = load_save(path)
        self.game = InteractiveGame.from_save_dict(d)
        self.h2h_mode = self.game.human_side is None
        if self.h2h_mode:
            self._active_coach = Side(d.get("h2h_active_coach", 0))
            self._awaiting_away = d.get("h2h_awaiting_away", False)
        self._stop_auto()
        self._showed_post_game = False
        # Rebuild UI state from restored game.
        self.playback = PlaybackState.from_events(
            list(self.game.all_events), lineup=self.game.lineup,
        )
        self.playback.pointer = 0
        while not self.playback.is_done:
            self.playback.step_one()
        self.scoreboard.bind_playback(self.playback)
        self.box.bind_playback(self.playback)
        self.event_log.clear()
        for e in self.playback.events:
            self.event_log.append_event(e)
        self._refresh_panels()
        self._update_coach_bar()
        self.query_one("#coach-bar", Static).update("Game loaded!")
        self.set_timer(2.0, lambda: self._update_coach_bar())

    def action_toggle_box_detail(self) -> None:
        self.box.toggle_detail()

    def action_back(self) -> None:
        if self.game.is_game_over:
            self._stop_auto()
            if len(self.app.screen_stack) > 2:
                self.app.pop_screen()
            return
        if len(self.app.screen_stack) > 2:
            self._stop_auto()
            self.app.push_screen(
                ConfirmQuitScreen(),
                callback=self._on_confirm_quit,
            )

    def _on_confirm_quit(self, confirmed: bool) -> None:
        if confirmed:
            self.app.pop_screen()

    # --- helpers ----------------------------------------------------------

    def _sync_events(self, new_events: list[Event]) -> None:
        # Extend incrementally so _credit_minutes sees the lineup that was
        # on court when each event occurred, not the current lineup.
        start = len(self.playback.events)
        self.playback.events = tuple(list(self.playback.events) + new_events)
        while not self.playback.is_done:
            self.playback.step_one()
        # Use the already-attributed events from playback (not a second
        # attribution pass) so the event log and box score agree on player
        # names.  A second .attribute() call would consume different RNG
        # draws and potentially pick a different random fouler/shooter.
        for i in range(start, len(self.playback.events)):
            self.event_log.append_event(self.playback.events[i])
        self._refresh_panels()

    def _refresh_panels(self) -> None:
        self.scoreboard.refresh_view()
        self.box.refresh_view()

    def _update_coach_bar(self) -> None:
        self.query_one("#coach-bar", Static).update(self._coach_bar_text())

    def _check_game_over(self) -> None:
        if self.game.is_game_over and not self._showed_post_game:
            self._showed_post_game = True
            self._stop_auto()
            if self.tournament_mode:
                self.app.push_screen(
                    PostGameScreen(self.playback, self.home_name, self.away_name),
                    callback=lambda _: self.dismiss(True),
                )
            else:
                self.app.push_screen(
                    PostGameScreen(self.playback, self.home_name, self.away_name),
                )


class CoachSubScreen(Screen):
    """Substitution screen for coaching mode — only shows the human side."""

    BINDINGS = [
        Binding("escape", "close", "Done"),
        Binding("1", "pull('0')", "Pull 1", show=False),
        Binding("2", "pull('1')", "Pull 2", show=False),
        Binding("3", "pull('2')", "Pull 3", show=False),
        Binding("4", "pull('3')", "Pull 4", show=False),
        Binding("5", "pull('4')", "Pull 5", show=False),
        Binding("a", "send_in('0')", "Send A", show=False),
        Binding("b", "send_in('1')", "Send B", show=False),
        Binding("c", "send_in('2')", "Send C", show=False),
        Binding("d", "send_in('3')", "Send D", show=False),
        Binding("e", "send_in('4')", "Send E", show=False),
        Binding("f", "send_in('5')", "Send F", show=False),
        Binding("g", "send_in('6')", "Send G", show=False),
        Binding("h", "send_in('7')", "Send H", show=False),
    ]

    DEFAULT_CSS = """
    CoachSubScreen {
        layout: vertical;
    }
    CoachSubScreen .intro {
        height: auto;
        padding: 1 2;
    }
    CoachSubScreen .body {
        height: 1fr;
        padding: 1 2;
        border: solid $accent;
    }
    CoachSubScreen #sub-status {
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    """

    def __init__(
        self, game, home_name: str, away_name: str,
        sub_side: Side | None = None,
    ):
        super().__init__()
        from hoops.engine.interactive import InteractiveGame
        self.game: InteractiveGame = game
        self.home_name = home_name
        self.away_name = away_name
        self._sub_side = sub_side if sub_side is not None else self.game.human_side
        self._pull_idx: int | None = None
        self._subs_made: int = 0

    def compose(self) -> ComposeResult:
        side_label = "HOME" if self._sub_side is Side.HOME else "AWAY"
        yield Header(show_clock=False)
        yield Static(
            f"Your roster ({side_label})  ·  1-5 pull starter  ·  a-h send in  ·  Esc done",
            classes="intro",
        )
        yield Static(self._lineup_text(), classes="body", id="lineup-body")
        yield Static(self._status_text(), id="sub-status")
        yield Footer()

    def on_mount(self) -> None:
        self.app.title = "Substitutions"

    def _lineup_text(self) -> str:
        side = self._sub_side
        on_court = self.game.lineup.on_court(side)
        bench = self.game.lineup.bench(side)
        rows = ["On court:", ""]
        for idx, p in enumerate(on_court):
            marker = " *" if self._pull_idx == idx else "  "
            fatigue = self.game.fatigue.fatigue(p.player_id)
            fatigue_bar = "!" if fatigue > 0.7 else ""
            fouls = self.game.fatigue.fouls(p.player_id)
            rows.append(f"{marker}{idx + 1}. {p.name}  ({int(p.minutes)} min)  F:{fouls}  {fatigue_bar}")
        rows += ["", "Bench:", ""]
        for idx, p in enumerate(bench[:8]):
            letter = "abcdefgh"[idx]
            fatigue = self.game.fatigue.fatigue(p.player_id)
            rested = "rested" if fatigue < 0.3 else ""
            rows.append(f"  {letter}. {p.name}  ({int(p.minutes)} min)  {rested}")
        return "\n".join(rows)

    def _status_text(self) -> str:
        made = f"  ({self._subs_made} made)" if self._subs_made else ""
        if self._pull_idx is None:
            return f"Dead ball — pick a starter to pull (1-5).{made}  Esc when done."
        on_court = self.game.lineup.on_court(self._sub_side)
        starter = on_court[self._pull_idx]
        return f"Pulling {starter.name}. Press a-h to bring in a bench player.{made}"

    def _refresh(self) -> None:
        self.query_one("#lineup-body", Static).update(self._lineup_text())
        self.query_one("#sub-status", Static).update(self._status_text())

    def action_pull(self, idx: str) -> None:
        i = int(idx)
        on_court = self.game.lineup.on_court(self._sub_side)
        if 0 <= i < len(on_court):
            self._pull_idx = i
        self._refresh()

    def action_send_in(self, idx: str) -> None:
        if self._pull_idx is None:
            return
        i = int(idx)
        bench = self.game.lineup.bench(self._sub_side)[:8]
        if not (0 <= i < len(bench)):
            return
        on_court = self.game.lineup.on_court(self._sub_side)
        off_player = on_court[self._pull_idx]
        on_player = bench[i]
        # Apply immediately — this screen only opens at dead balls.
        self.game.substitute(self._sub_side, off_player.player_id, on_player.player_id)
        self._subs_made += 1
        self._pull_idx = None
        self._refresh()

    def action_close(self) -> None:
        self.dismiss(self._subs_made)


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
        Binding("slash", "start_search", "/ search", priority=True),
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
        return sorted(all_priors, key=lambda p: p.team_name.lower()), load_league_prior(League.WBB, season)

    def _rebuild_list(self, list_id: str, priors: list[TeamPriors], records: dict[int, str]) -> None:
        ol = self.query_one(f"#{list_id}", OptionList)
        ol.clear_options()
        ol.add_options([Option(self._team_label(p, records), id=str(p.team_id)) for p in priors])

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Pick teams (Tab/Enter)  ·  / search  ·  P play  ·  C coach side  ·  "
            "1/2 scheme  ·  3/4 foul-up-3  ·  5/6 season  ·  Q quit",
            classes="intro",
        )
        self._home_priors, self._home_league_prior = self._load_priors(self.home_season)
        self._away_priors, self._away_league_prior = self._load_priors(self.away_season)
        self._home_priors_by_id = {p.team_id: p for p in self._home_priors}
        self._away_priors_by_id = {p.team_id: p for p in self._away_priors}
        self._home_records = _load_team_records(League.WBB, self.home_season)
        self._away_records = _load_team_records(League.WBB, self.away_season)

        opts_home = [Option(self._team_label(p, self._home_records), id=str(p.team_id)) for p in self._home_priors]
        opts_away = [Option(self._team_label(p, self._away_records), id=str(p.team_id)) for p in self._away_priors]

        with Horizontal():
            with Vertical():
                yield Static(f"HOME ({_short_season(self.home_season)})", classes="column-header", id="home_header")
                yield OptionList(*opts_home, id="home_list")
                yield Static(
                    self._policy_text(self.home_policy),
                    id="home_policy", classes="policy-line",
                )
            with Vertical():
                yield Static(f"AWAY ({_short_season(self.away_season)})", classes="column-header", id="away_header")
                yield OptionList(*opts_away, id="away_list")
                yield Static(
                    self._policy_text(self.away_policy),
                    id="away_policy", classes="policy-line",
                )
        yield Input(placeholder="Type to jump to team…", id="search-bar")
        yield Static("(no teams selected)", id="status")
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
            self.app.title = f"Hoops 2026 — pick matchup ({_short_season(self.home_season)} vs {_short_season(self.away_season)})"

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

    def _coach_label(self) -> str:
        if self.coach_side == "h2h":
            return "H2H"
        if self.coach_side is None:
            return "WATCH"
        return "Coach HOME" if self.coach_side is Side.HOME else "Coach AWAY"

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
        text = f"HOME: {home_label}    AWAY: {away_label}    [{coach}]    {suffix}".rstrip()
        self.last_status_text = text
        self.query_one("#status", Static).update(text)

    # --- actions ----------------------------------------------------------

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
        self.query_one("#home_header", Static).update(f"HOME ({_short_season(self.home_season)})")
        self._update_title()
        self._refresh_status()

    def action_cycle_away_season(self) -> None:
        self.away_season = self._cycle_season(self.away_season)
        self._away_priors, self._away_league_prior = self._load_priors(self.away_season)
        self._away_priors_by_id = {p.team_id: p for p in self._away_priors}
        self._away_records = _load_team_records(League.WBB, self.away_season)
        self.away_id = None
        self._rebuild_list("away_list", self._away_priors, self._away_records)
        self.query_one("#away_header", Static).update(f"AWAY ({_short_season(self.away_season)})")
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
        home_label = f"{home.team_name} ({_short_season(self.home_season)})" if self.home_season != self.away_season else home.team_name
        away_label = f"{away.team_name} ({_short_season(self.away_season)})" if self.home_season != self.away_season else away.team_name

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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class HoopsApp(App):
    """Hoops 2026 single-game UI.

    Two construction modes:

    - With ``events``: skip the picker, go straight to the game (used by
      ``hoops play --home X --away Y`` and by the headless tests).
    - Without ``events`` (just ``season``): show the team picker first.
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    #top {
        height: 1fr;
    }
    """

    # priority=True so the binding fires regardless of which widget is
    # focused. Without it, OptionList's letter-jump navigation captures
    # 'q' before it reaches the screen.
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        events: list[Event] | None = None,
        home_name: str = "",
        away_name: str = "",
        season: str | None = None,
        seed: int | None = None,
        division_one_only: bool = True,
        season_explicit: bool = False,
        neutral_site: bool = False,
        **kw,
    ):
        super().__init__(**kw)
        self._events = events
        self._home_name = home_name
        self._away_name = away_name
        self._season = season
        self._seed = seed
        self._division_one_only = division_one_only
        self._season_explicit = season_explicit
        self._neutral_site = neutral_site

    def on_mount(self) -> None:
        if self._events is not None:
            self.push_screen(
                GameScreen(self._events, self._home_name, self._away_name)
            )
            return

        available = fitted_seasons(League.WBB)
        if not available:
            self.exit(message="No fitted seasons found. Run scripts/fit_distributions.py first.")
            return

        default = self._season if self._season in available else None

        if self._season_explicit and default:
            self.push_screen(TeamSelectScreen(
                seasons=[default],
                seed=self._seed,
                division_one_only=self._division_one_only,
                neutral_site=self._neutral_site,
            ))
        elif len(available) == 1:
            self.push_screen(TeamSelectScreen(
                seasons=available,
                seed=self._seed,
                division_one_only=self._division_one_only,
                neutral_site=self._neutral_site,
            ))
        elif len(available) > 1 and default:
            self.push_screen(TeamSelectScreen(
                seasons=available,
                default_season=default,
                seed=self._seed,
                division_one_only=self._division_one_only,
                neutral_site=self._neutral_site,
            ))
        else:
            self.push_screen(SeasonSelectScreen(
                seed=self._seed,
                division_one_only=self._division_one_only,
                neutral_site=self._neutral_site,
            ))
