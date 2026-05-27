"""Per-side on-court lineup state for live (interactive) attribution.

The static :func:`hoops.engine.attribution.attribute_players` post-pass
samples from the *full* roster and is appropriate for batch sims and
non-interactive playback. When the user is coaching, we want events
attributed only to the players currently on the floor, and we want
substitutions to take immediate effect on subsequent attributions.

This module provides ``LineupState`` — an in-memory state object that
mirrors the static attribution rules but constrains sampling to a
mutable on-court roster of five per side. The Textual UI builds one of
these once per game and threads it through ``PlaybackState``.

Substitutions are made via :meth:`substitute(side, off_player_id, on_player_id)`
and take effect on the next :meth:`attribute` call. The pending free-throw
shooter logic is preserved so a shooter who's been pulled mid-trip still
takes their FTs (real-life rule: subs can't enter mid-FT-trip).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from hoops.data.rosters import Player, Roster
from hoops.engine.events import Event
from hoops.engine.state import Side


STARTING_LINEUP_SIZE = 5


class LineupError(ValueError):
    """Raised on illegal substitution requests (player not on bench, etc.)."""


@dataclass
class LineupState:
    home_roster: Roster
    away_roster: Roster
    home_on_court: list[Player]
    away_on_court: list[Player]
    rng: np.random.Generator
    # Pending FT shooter per side, preserved across mid-trip subs.
    pending_ft_shooter: dict[Side, str | None] = field(
        default_factory=lambda: {Side.HOME: None, Side.AWAY: None}
    )
    # "Shadow" lineups: the state the floor will have after the next dead
    # ball commits all queued substitutions. Always equals on_court when
    # nothing is pending. Mutated by ``request_substitution``; copied to
    # ``*_on_court`` by ``commit_pending_subs``. Real-basketball rule:
    # subs can only enter at a dead ball — see PlaybackState for the
    # foul / non-steal-turnover trigger.
    home_pending_on_court: list[Player] = field(default_factory=list)
    away_pending_on_court: list[Player] = field(default_factory=list)

    @classmethod
    def with_default_starters(
        cls, home_roster: Roster, away_roster: Roster, rng: np.random.Generator
    ) -> "LineupState":
        """Default starters = top-N by minutes from each roster, padded to 5
        with the next-best players if the roster is shorter."""
        starters_h = list(home_roster.players[:STARTING_LINEUP_SIZE])
        starters_a = list(away_roster.players[:STARTING_LINEUP_SIZE])
        return cls(
            home_roster=home_roster,
            away_roster=away_roster,
            home_on_court=starters_h,
            away_on_court=starters_a,
            rng=rng,
            home_pending_on_court=list(starters_h),
            away_pending_on_court=list(starters_a),
        )

    def __post_init__(self) -> None:
        # If the shadow lineups weren't initialized (e.g., construction via
        # ``LineupState(...)`` directly rather than via the classmethod),
        # mirror them from the actual on-court state.
        if not self.home_pending_on_court:
            self.home_pending_on_court = list(self.home_on_court)
        if not self.away_pending_on_court:
            self.away_pending_on_court = list(self.away_on_court)

    # --- introspection ----------------------------------------------------

    def on_court(self, side: Side) -> list[Player]:
        return self.home_on_court if side is Side.HOME else self.away_on_court

    def pending_on_court(self, side: Side) -> list[Player]:
        """The lineup that will be on the floor after the next dead ball.
        Equals ``on_court(side)`` when nothing is queued."""
        return (
            self.home_pending_on_court if side is Side.HOME
            else self.away_pending_on_court
        )

    def roster(self, side: Side) -> Roster:
        return self.home_roster if side is Side.HOME else self.away_roster

    def bench(self, side: Side) -> list[Player]:
        """Players not currently on the floor (regardless of pending subs)."""
        on = {p.player_id for p in self.on_court(side)}
        return [p for p in self.roster(side).players if p.player_id not in on]

    def pending_bench(self, side: Side) -> list[Player]:
        """Players not on the *post-commit* floor — the pool the user can
        send in next."""
        on = {p.player_id for p in self.pending_on_court(side)}
        return [p for p in self.roster(side).players if p.player_id not in on]

    def has_pending(self, side: Side) -> bool:
        return [p.player_id for p in self.on_court(side)] != [
            p.player_id for p in self.pending_on_court(side)
        ]

    # --- mutation ---------------------------------------------------------

    def substitute(self, side: Side, off_player_id: int, on_player_id: int) -> None:
        """Apply a substitution immediately (bypasses the dead-ball wait).

        Used by tests and by callers that already know the ball is dead.
        Interactive UI callers should use ``request_substitution`` instead;
        the engine commits queued subs on the next dead ball."""
        idx = self._find_in_actual(side, off_player_id)
        bench_player = next(
            (p for p in self.bench(side) if p.player_id == on_player_id), None
        )
        if bench_player is None:
            raise LineupError(
                f"player {on_player_id} not on the bench for {side.name}"
            )
        on_court = self.on_court(side)
        on_court[idx] = bench_player
        # Keep the shadow in sync so a request_substitution after a
        # direct substitute() still validates correctly.
        shadow = self.pending_on_court(side)
        shadow_idx = next(
            (i for i, p in enumerate(shadow) if p.player_id == off_player_id), None
        )
        if shadow_idx is not None:
            shadow[shadow_idx] = bench_player

    def _find_in_actual(self, side: Side, off_player_id: int) -> int:
        idx = next(
            (i for i, p in enumerate(self.on_court(side)) if p.player_id == off_player_id),
            None,
        )
        if idx is None:
            raise LineupError(
                f"player {off_player_id} not on the floor for {side.name}"
            )
        return idx

    def request_substitution(
        self, side: Side, off_player_id: int, on_player_id: int
    ) -> None:
        """Queue a substitution to take effect on the next dead ball.

        Validates against the *post-pending* state so chained subs are
        coherent (e.g. requesting P1→P6 then P6→P7 is allowed; the second
        sub effectively sends P7 in for P1)."""
        shadow = self.pending_on_court(side)
        idx = next(
            (i for i, p in enumerate(shadow) if p.player_id == off_player_id), None
        )
        if idx is None:
            raise LineupError(
                f"player {off_player_id} would not be on the floor for {side.name}"
            )
        on_player = next(
            (p for p in self.roster(side).players if p.player_id == on_player_id),
            None,
        )
        if on_player is None:
            raise LineupError(
                f"player {on_player_id} not on the {side.name} roster"
            )
        if on_player.player_id in {p.player_id for p in shadow}:
            raise LineupError(
                f"player {on_player_id} would already be on the floor"
            )
        shadow[idx] = on_player

    def discard_pending_subs(self, side: Side | None = None) -> None:
        """Drop queued subs without committing. ``side=None`` clears both."""
        if side is None or side is Side.HOME:
            self.home_pending_on_court = list(self.home_on_court)
        if side is None or side is Side.AWAY:
            self.away_pending_on_court = list(self.away_on_court)

    def commit_pending_subs(self) -> int:
        """Make queued subs live. Returns the number of changes applied."""
        changed = 0
        if self.has_pending(Side.HOME):
            self.home_on_court = list(self.home_pending_on_court)
            changed += 1
        if self.has_pending(Side.AWAY):
            self.away_on_court = list(self.away_pending_on_court)
            changed += 1
        return changed

    # --- attribution ------------------------------------------------------

    def attribute(self, e: Event) -> Event:
        """Return a copy of ``e`` with ``player`` filled in. Side effects
        on internal state (pending FT shooter) so subsequent FTs chain."""
        if e.team is None or e.type in (
            "tip_off", "quarter_end", "overtime_start", "game_end"
        ):
            return e

        side = e.team
        attacker = self._adhoc(side)
        defender = self._adhoc(side.other)

        if e.type == "shot_made" or e.type == "shot_missed":
            if e.player is not None:
                self.pending_ft_shooter[side] = e.player
                return e
            shooter = (
                attacker.three_point_shooter(self.rng)
                if e.detail == "three"
                else attacker.shooter(self.rng)
            )
            self.pending_ft_shooter[side] = shooter.name
            return dataclasses.replace(e, player=shooter.name)

        if e.type == "free_throw_made" or e.type == "free_throw_missed":
            if e.player is not None:
                self.pending_ft_shooter[side] = e.player
                return e
            name = self.pending_ft_shooter.get(side)
            if name is None:
                shooter = attacker.ft_shooter(self.rng)
                name = shooter.name
                self.pending_ft_shooter[side] = name
            return dataclasses.replace(e, player=name)

        if e.type == "foul_personal" or e.type == "foul_shooting":
            if e.player is not None:
                return e
            fouler = self._adhoc(e.team).fouler(self.rng)
            return dataclasses.replace(e, player=fouler.name)

        if e.type == "rebound_off":
            if e.player is not None:
                self.pending_ft_shooter[e.team] = None
                return e
            reb = self._adhoc(e.team).rebounder_off(self.rng)
            self.pending_ft_shooter[e.team] = None
            return dataclasses.replace(e, player=reb.name)

        if e.type == "rebound_def":
            if e.player is not None:
                self.pending_ft_shooter[Side.HOME] = None
                self.pending_ft_shooter[Side.AWAY] = None
                return e
            reb = self._adhoc(e.team).rebounder_def(self.rng)
            self.pending_ft_shooter[Side.HOME] = None
            self.pending_ft_shooter[Side.AWAY] = None
            return dataclasses.replace(e, player=reb.name)

        if e.type == "turnover":
            if e.player is not None:
                self.pending_ft_shooter[Side.HOME] = None
                self.pending_ft_shooter[Side.AWAY] = None
                return e
            t = self._adhoc(e.team).turnover(self.rng)
            self.pending_ft_shooter[Side.HOME] = None
            self.pending_ft_shooter[Side.AWAY] = None
            return dataclasses.replace(e, player=t.name)

        # Credit events (assist / block / steal) — sample from the right
        # side's on-court 5. Don't use the same player as a recently-named
        # actor: this is a cosmetic detail (would need to track 'last shooter'
        # to fully prevent self-assists, but the static post-pass already
        # handles that). Here we just sample weighted by the relevant stat.
        if e.type == "assist":
            if e.player is not None:
                return e
            p = self._adhoc(e.team).assister(self.rng)
            return dataclasses.replace(e, player=p.name)
        if e.type == "block":
            if e.player is not None:
                return e
            p = self._adhoc(e.team).blocker(self.rng)
            return dataclasses.replace(e, player=p.name)
        if e.type == "steal":
            if e.player is not None:
                return e
            p = self._adhoc(e.team).stealer(self.rng)
            return dataclasses.replace(e, player=p.name)

        return e

    def _adhoc(self, side: Side) -> Roster:
        """Build a Roster restricted to the side's on-court 5 so existing
        weighted samplers work without modification."""
        roster = self.roster(side)
        return Roster(
            team_id=roster.team_id,
            team_name=roster.team_name,
            players=tuple(self.on_court(side)),
        )
