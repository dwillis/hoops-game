"""Textual UI screens for tournament bracket mode."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from hoops.engine.bracket import Bracket, BracketGame


def _game_line(game: BracketGame, show_score: bool = True) -> str:
    """Render one game as a compact text line."""
    def _team_str(slot) -> str:
        seed = f"({slot.seed})" if slot.seed is not None else "    "
        name = (slot.team_name or "TBD")[:18].ljust(18)
        return f"{seed:>4} {name}"

    home_win = game.winner_id == game.home.team_id if game.winner_id else False
    away_win = game.winner_id == game.away.team_id if game.winner_id else False
    home_str = _team_str(game.home)
    away_str = _team_str(game.away)

    if game.is_played and show_score:
        score = f"{game.home_score:>3}-{game.away_score:<3}"
        upset = " !" if game.is_upset else "  "
        wh = "*" if home_win else " "
        wa = "*" if away_win else " "
        return f"  {wh}{home_str}  {score}  {wa}{away_str}{upset}"
    else:
        return f"   {home_str}    vs    {away_str}"


def render_bracket_region(bracket: Bracket, region_idx: int, round_num: int) -> str:
    """Render all games for a region in a specific round."""
    games = [g for g in bracket.games_in_round(round_num) if g.region == region_idx]
    if not games:
        return ""
    region_name = (
        bracket.regions[region_idx] if region_idx < len(bracket.regions)
        else f"Region {region_idx + 1}"
    )
    lines = [f"  {region_name}", "  " + "-" * 50]
    for game in games:
        lines.append(_game_line(game))
    return "\n".join(lines)


def render_bracket_round(bracket: Bracket, round_num: int) -> str:
    """Render all games in a round, grouped by region (or flat for conf tournaments)."""
    round_name = bracket.round_name(round_num)
    lines = [f"{'=' * 20} {round_name} {'=' * 20}", ""]

    games = bracket.games_in_round(round_num)
    region_indices = sorted(set(
        g.region for g in games if g.region is not None
    ))

    if region_indices:
        # NCAA: group by region
        for ridx in region_indices:
            lines.append(render_bracket_region(bracket, ridx, round_num))
            lines.append("")
    else:
        # Conference or final rounds: flat list
        for game in games:
            lines.append(_game_line(game))
        lines.append("")

    upsets = bracket.upsets(round_num)
    if upsets:
        lines.append(f"  UPSETS: {len(upsets)} upset(s) this round!")
        for u in upsets:
            winner = u.home if u.winner_id == u.home.team_id else u.away
            loser = u.away if u.winner_id == u.home.team_id else u.home
            lines.append(
                f"    ({winner.seed}) {winner.team_name} over "
                f"({loser.seed}) {loser.team_name}  "
                f"{u.home_score}-{u.away_score}"
            )
    return "\n".join(lines)


class BracketViewScreen(Screen):
    """Between-round bracket view with scores and upset highlights."""

    BINDINGS = [
        Binding("enter", "continue_bracket", "Continue"),
        Binding("escape", "continue_bracket", "Continue"),
        Binding("q", "quit_app", "Quit"),
    ]

    DEFAULT_CSS = """
    BracketViewScreen {
        layout: vertical;
    }
    BracketViewScreen > Static.bracket-title {
        height: auto;
        padding: 1 2;
        text-align: center;
        text-style: bold;
        background: $accent;
        color: $background;
    }
    BracketViewScreen > Static.bracket-body {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }
    BracketViewScreen > Static.bracket-prompt {
        height: auto;
        padding: 1 2;
        text-align: center;
        background: $primary;
        color: $background;
    }
    """

    def __init__(self, bracket: Bracket, round_num: int, user_team_id: int,
                 user_eliminated: bool = False):
        super().__init__()
        self.bracket = bracket
        self.round_num = round_num
        self.user_team_id = user_team_id
        self.user_eliminated = user_eliminated

    def compose(self) -> ComposeResult:
        round_name = self.bracket.round_name(self.round_num)
        if self.bracket.conference_name:
            title = f"{self.bracket.conference_name} - {self.bracket.season} - {round_name} Results"
        else:
            title = f"NCAA Tournament - {self.bracket.season} - {round_name} Results"
        yield Header(show_clock=False)
        yield Static(title, classes="bracket-title")
        body = render_bracket_round(self.bracket, self.round_num)
        yield Static(body, classes="bracket-body")

        if self.user_eliminated:
            prompt = "Your tournament run is over. Press Enter to see final bracket."
        else:
            next_game = self.bracket.next_game_for(self.user_team_id)
            if next_game is not None:
                game = self.bracket.games[next_game]
                opp = game.away if game.home.team_id == self.user_team_id else game.home
                opp_name = opp.team_name or "TBD"
                opp_seed = f"({opp.seed}) " if opp.seed else ""
                prompt = f"Next up: vs {opp_seed}{opp_name}  -  Press Enter to play"
            else:
                prompt = "Press Enter to continue"
        yield Static(prompt, classes="bracket-prompt")
        yield Footer()

    def action_continue_bracket(self) -> None:
        self.dismiss(True)

    def action_quit_app(self) -> None:
        self.app.exit()


class ChampionScreen(Screen):
    """End-of-tournament screen."""

    BINDINGS = [
        Binding("enter", "done", "Done"),
        Binding("escape", "done", "Done"),
        Binding("q", "quit_app", "Quit"),
    ]

    DEFAULT_CSS = """
    ChampionScreen {
        layout: vertical;
        align: center middle;
    }
    ChampionScreen > Static.champion-title {
        height: auto;
        padding: 2 4;
        text-align: center;
        text-style: bold;
        background: $accent;
        color: $background;
    }
    ChampionScreen > Static.champion-body {
        height: auto;
        padding: 2 4;
        text-align: center;
    }
    ChampionScreen > Static.champion-hint {
        height: auto;
        padding: 1 2;
        text-align: center;
    }
    """

    def __init__(self, bracket: Bracket, user_team_id: int, user_won: bool,
                 user_team_name: str = "", eliminated_round: int | None = None):
        super().__init__()
        self.bracket = bracket
        self.user_team_id = user_team_id
        self.user_won = user_won
        self.user_team_name = user_team_name
        self.eliminated_round = eliminated_round

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        is_conf = bool(self.bracket.conference_name)
        tourney_name = self.bracket.conference_name or "NCAA"

        if self.user_won:
            seed = self.bracket.team_seed(self.user_team_id)
            seed_str = f"({seed}) " if seed else ""
            if is_conf:
                yield Static(f"CONFERENCE CHAMPIONS!\n\n{seed_str}{self.user_team_name}", classes="champion-title")
                yield Static(
                    f"Congratulations! You won the {self.bracket.season} {tourney_name}!",
                    classes="champion-body",
                )
            else:
                yield Static(f"NATIONAL CHAMPIONS!\n\n{seed_str}{self.user_team_name}", classes="champion-title")
                yield Static(
                    f"Congratulations! You coached {self.user_team_name} "
                    f"to the {self.bracket.season} NCAA Championship!",
                    classes="champion-body",
                )
        else:
            elim_name = self.bracket.round_name(self.eliminated_round) if self.eliminated_round else "the tournament"
            yield Static(f"TOURNAMENT OVER\n\n{self.user_team_name} eliminated in {elim_name}", classes="champion-title")
            champ_id = self.bracket.champion()
            if champ_id:
                seed = self.bracket.team_seed(champ_id)
                champ_name = ""
                for g in self.bracket.games:
                    if g.home.team_id == champ_id:
                        champ_name = g.home.team_name; break
                    if g.away.team_id == champ_id:
                        champ_name = g.away.team_name; break
                seed_str = f"({seed}) " if seed else ""
                yield Static(f"Champion: {seed_str}{champ_name}", classes="champion-body")
        yield Static("Press Enter or Q to exit", classes="champion-hint")
        yield Footer()

    def action_done(self) -> None:
        self.app.exit()

    def action_quit_app(self) -> None:
        self.app.exit()
