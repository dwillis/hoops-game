"""Game and coaching screens: playback, subs, lineups, post-game."""

from __future__ import annotations

from collections.abc import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from hoops.data.rosters import Roster
from hoops.engine.events import Event
from hoops.engine.policy import CoachPolicies, DefensiveScheme, OffensiveScheme
from hoops.engine.sampling import make_rng
from hoops.engine.state import Side
from hoops.ui.lineup import LineupError, LineupState
from hoops.ui.playback import PlaybackState, PlayerBox
from hoops.ui.widgets import BoxScorePanel, PossessionLog, Scoreboard

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
        Binding("g", "show_team_stats", "Team stats"),
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
        if self.home_roster and self.away_roster:
            self.box.bind_rosters(self.home_roster, self.away_roster)
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

    def action_show_team_stats(self) -> None:
        self.box.show_team_view()

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
            title = (
                f"FINAL: {self.home_name} {p.home_score} — "
                f"{self.away_name} {p.away_score} (TIE)"
            )
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
        active = " <-- selected" if side is self._active_side else ""
        pending_tag = "  (PENDING)" if self.lineup.has_pending(side) else ""
        rows = [f"On court{active}{pending_tag}:", ""]
        for idx, p in enumerate(post_on_court):
            marker = " *" if (
                self._pull_idx == idx and side is self._active_side
            ) else "  "
            tag = "  (in →)" if p.player_id not in actual_ids else ""
            team_gp = self.lineup.roster(side).team_games
            mpg = p.mpg(team_gp)
            rows.append(f"{marker}{idx + 1}. {p.name}  ({mpg:.1f} mpg){tag}")
        rows += ["", "Bench:", ""]
        bench = self.lineup.pending_bench(side)[:8]
        for idx, p in enumerate(bench):
            letter = "abcdefgh"[idx]
            tag = "  (→ out)" if p.player_id in actual_ids else ""
            mpg = p.mpg(team_gp)
            rows.append(f"  {letter}. {p.name}  ({mpg:.1f} mpg){tag}")
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
        except LineupError as exc:
            self.app.notify(str(exc), severity="warning")
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
        roster: Roster,
        team_name: str,
        side: Side,
        callback: Callable[[Side, list[int]], None],
    ):
        super().__init__()
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
            "     PLAYER               POS   GP   MPG   PPG   RPG   APG  "
            "FG%   3P%   FT%"
        )
        sep = "     " + "─" * (len(header) - 5)
        rows = [header, sep]
        for i, p in enumerate(self._roster.players):
            marker = " ★ " if i in self._selected else "   "
            team_gp = self._roster.team_games
            gp = max(p.games_played, 1)
            ppg = p.points / gp
            rpg = (p.orb + p.drb) / gp
            apg = p.ast / gp
            mpg = p.mpg(team_gp)
            fg_pct = (p.fgm / p.fga * 100) if p.fga > 0 else 0.0
            fg3_pct = (p.fg3m / p.fg3a * 100) if p.fg3a > 0 else 0.0
            ft_pct = (p.ftm / p.fta * 100) if p.fta > 0 else 0.0
            pos = p.position or "—"
            name = p.name[:20]
            rows.append(
                f"{marker} {name:<20s} {pos:<4s} {gp:3d}  "
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
        Binding("g", "show_team_stats", "Team stats"),
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
        self._subs_allowed: bool = True
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
        self.box.bind_rosters(self.game.home_roster, self.game.away_roster)
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
        self._subs_allowed = True

        if self.h2h_mode and self._awaiting_away:
            # Away coach pressed Space -> advance possession
            self._awaiting_away = False
            self._active_coach = Side.HOME
            result = self.game.step_possession()
            self._sync_events(result.events)
            if self._is_stoppage(result) and not self.game.is_game_over:
                self._at_dead_ball = True
                self._subs_allowed = result.subs_allowed
                if self._subs_allowed:
                    self._open_subs_if_requested()
                else:
                    self._clear_sub_requests()
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
                self._subs_allowed = result.subs_allowed
                if self._subs_allowed:
                    self._open_subs_if_requested()
                else:
                    self._clear_sub_requests()
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

    def _clear_sub_requests(self) -> None:
        for side in (Side.HOME, Side.AWAY):
            self._sub_requested[side] = False

    def action_open_subs(self) -> None:
        if self.game.is_game_over:
            return
        if self._at_dead_ball and not self._subs_allowed:
            self.notify("No subs after made FG in last minute.", timeout=2.0)
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
        from hoops.engine.interactive import InteractiveGame
        from hoops.engine.save import has_save, load_save, save_path_for
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
        self.box.bind_fatigue(self.game.fatigue, self.game.lineup)
        self.box.bind_rosters(self.game.home_roster, self.game.away_roster)
        self.event_log.clear()
        for e in self.playback.events:
            self.event_log.append_event(e)
        self._refresh_panels()
        self._update_coach_bar()
        self.query_one("#coach-bar", Static).update("Game loaded!")
        self.set_timer(2.0, lambda: self._update_coach_bar())

    def action_toggle_box_detail(self) -> None:
        self.box.toggle_detail()

    def action_show_team_stats(self) -> None:
        self.box.show_team_view()

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
        team_gp = self.game.lineup.roster(side).team_games
        rows = ["On court:", ""]
        for idx, p in enumerate(on_court):
            marker = " *" if self._pull_idx == idx else "  "
            fatigue = self.game.fatigue.fatigue(p.player_id)
            fatigue_bar = "!" if fatigue > 0.7 else ""
            fouls = self.game.fatigue.fouls(p.player_id)
            mpg = p.mpg(team_gp)
            rows.append(f"{marker}{idx + 1}. {p.name}  ({mpg:.1f} mpg)  F:{fouls}  {fatigue_bar}")
        rows += ["", "Bench:", ""]
        for idx, p in enumerate(bench[:8]):
            letter = "abcdefgh"[idx]
            fatigue = self.game.fatigue.fatigue(p.player_id)
            rested = "rested" if fatigue < 0.3 else ""
            mpg = p.mpg(team_gp)
            rows.append(f"  {letter}. {p.name}  ({mpg:.1f} mpg)  {rested}")
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

