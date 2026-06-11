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

The widgets and screens live in :mod:`hoops.ui.widgets`,
:mod:`hoops.ui.game_screens`, and :mod:`hoops.ui.picker_screens`; they are
re-exported here for backward compatibility.
"""

from __future__ import annotations

from textual.app import App
from textual.binding import Binding

from hoops.data.paths import fitted_seasons
from hoops.engine.events import Event
from hoops.league import League

# Re-exports for backward compatibility — other modules and tests import
# these names from hoops.ui.app.
from hoops.ui.game_screens import (
    _AUTO_SPEEDS as _AUTO_SPEEDS,
)
from hoops.ui.game_screens import (
    _DEFAULT_SPEED_IDX as _DEFAULT_SPEED_IDX,
)
from hoops.ui.game_screens import (
    CoachGameScreen as CoachGameScreen,
)
from hoops.ui.game_screens import (
    CoachSubScreen as CoachSubScreen,
)
from hoops.ui.game_screens import (
    ConfirmQuitScreen as ConfirmQuitScreen,
)
from hoops.ui.game_screens import (
    GameScreen as GameScreen,
)
from hoops.ui.game_screens import (
    PostGameScreen as PostGameScreen,
)
from hoops.ui.game_screens import (
    StartingLineupScreen as StartingLineupScreen,
)
from hoops.ui.game_screens import (
    SubScreen as SubScreen,
)
from hoops.ui.picker_screens import (
    SeasonSelectScreen as SeasonSelectScreen,
)
from hoops.ui.picker_screens import (
    TeamSelectScreen as TeamSelectScreen,
)
from hoops.ui.picker_screens import (
    _load_team_records as _load_team_records,
)
from hoops.ui.picker_screens import (
    _short_season as _short_season,
)
from hoops.ui.widgets import (
    BoxScorePanel as BoxScorePanel,
)
from hoops.ui.widgets import (
    PossessionLog as PossessionLog,
)
from hoops.ui.widgets import (
    Scoreboard as Scoreboard,
)

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
