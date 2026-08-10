"""Reusable widgets for the game screens: scoreboard, log, box score."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import RichLog, Static

from hoops.engine.events import Event, fmt_clock, fmt_event
from hoops.engine.fatigue import WORKLOAD_GASSED, WORKLOAD_TIRED
from hoops.engine.state import Side
from hoops.ui.playback import PlaybackState, PlayerBox


def fatigue_tag(ratio: float) -> str:
    """Return the GASSED/TIRED markup tag for a workload ratio, or "" if
    neither threshold is met. *ratio* is fraction of the player's normal
    per-game workload (star-aware) — 1.0 means exactly their usual minutes.
    """
    if ratio >= WORKLOAD_GASSED:
        return "[blink bold red]*** GASSED ***[/]"
    if ratio >= WORKLOAD_TIRED:
        return "[yellow]* TIRED[/]"
    return ""


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
        if e.type in ("assist", "steal", "block"):
            # Rendered inline on the preceding shot/turnover line instead
            # (via assist_by / stolen_by / blocked_by). The standalone
            # events still feed the box score.
            return
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
        self._home_roster = None  # Roster, for season averages view
        self._away_roster = None
        self.last_text = ""
        self._view_mode = 0  # 0=team, 1=players, 2=roster averages

    def bind_playback(self, p: PlaybackState) -> None:
        self._playback = p
        self.refresh_view()

    def bind_fatigue(self, fatigue, lineup) -> None:
        self._fatigue = fatigue
        self._lineup = lineup

    def bind_rosters(self, home_roster, away_roster) -> None:
        self._home_roster = home_roster
        self._away_roster = away_roster

    def toggle_detail(self) -> None:
        max_modes = 3 if self._home_roster else 2
        self._view_mode = (self._view_mode + 1) % max_modes
        self.refresh_view()

    def show_team_view(self) -> None:
        self._view_mode = 0
        self.refresh_view()

    def _set(self, text: str) -> None:
        self.last_text = text
        self.update(text)

    def refresh_view(self) -> None:
        if self._playback is None:
            self._set("(no game loaded)")
            return
        if self._view_mode == 1:
            self._render_player_view()
        elif self._view_mode == 2:
            self._render_roster_view()
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
            "Team summary  [x] → Player box scores",
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
            p.name: self._fatigue.workload_ratio(p.player_id)
            for p in roster.players
            if self._fatigue.tracked(p.player_id)
        }

    @staticmethod
    def _fatigue_tag(ratio: float) -> str:
        tag = fatigue_tag(ratio)
        return f" {tag}" if tag else ""

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

        if self._home_roster:
            hint = "Player box scores  [x] → Season roster"
        else:
            hint = "Player box scores  [x] → Team summary"
        lines.append(hint)
        self._set("\n".join(lines))

    def _render_roster_view(self) -> None:
        lines: list[str] = []
        header = (
            f"{'Player':<22} {'POS':<4} {'GP':>3}  "
            f"{'MPG':>4}  {'PPG':>5} {'RPG':>5} {'APG':>5}  "
            f"{'FG%':>4}  {'3P%':>4}  {'FT%':>4}"
        )

        for label, roster in [
            (self.home_name, self._home_roster),
            (self.away_name, self._away_roster),
        ]:
            lines.append(f"── {label} (season averages) ──")
            lines.append(header)
            team_gp = roster.team_games
            for p in roster.players:
                gp = max(p.games_played, 1)
                ppg = p.points / gp
                rpg = (p.orb + p.drb) / gp
                apg = p.ast / gp
                mpg = p.mpg(team_gp)
                fg_pct = (p.fgm / p.fga * 100) if p.fga > 0 else 0.0
                fg3_pct = (p.fg3m / p.fg3a * 100) if p.fg3a > 0 else 0.0
                ft_pct = (p.ftm / p.fta * 100) if p.fta > 0 else 0.0
                pos = p.position or "—"
                name = p.name[:21]
                lines.append(
                    f"{name:<22} {pos:<4} {gp:3d}  "
                    f"{mpg:4.1f}  {ppg:5.1f} {rpg:5.1f} {apg:5.1f}  "
                    f"{fg_pct:4.1f}  {fg3_pct:4.1f}  {ft_pct:4.1f}"
                )
            lines.append("")

        lines.append("Season roster  [x] → Team summary")
        self._set("\n".join(lines))

