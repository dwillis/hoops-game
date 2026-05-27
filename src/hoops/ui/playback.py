"""Pure-data playback state: walks an Event sequence and derives a box score.

The Textual app holds an instance of ``PlaybackState`` and asks it to advance
one event or one possession at a time. Nothing here imports Textual — the
class is fully unit-testable without a terminal.

The doc's UX requirements (§6) are entirely derivable from the event log:

- Quarter scoreboard and running scores: from event.home_score / away_score.
- Team fouls per quarter: count foul_personal / foul_shooting events in
  the current period for each side (resets at quarter rollover).
- Bonus indicator: same threshold rule as engine/fouls.py — the team
  facing >= 5 opponent fouls this quarter is in the bonus.
- Per-team box score: derive from shot_made / shot_missed / rebound_off /
  rebound_def / free_throw_* / turnover events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hoops.engine.events import Event
from hoops.engine.fouls import BONUS_THRESHOLD_PER_QUARTER
from hoops.engine.state import Side

# Forward-only type reference; LineupState lives in hoops.ui.lineup but
# is only imported lazily by callers to avoid circular imports.


@dataclass
class PlayerBox:
    name: str = ""
    seconds: float = 0.0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    orb: int = 0
    drb: int = 0
    tov: int = 0
    pf: int = 0
    points: int = 0
    ast: int = 0
    blk: int = 0
    stl: int = 0

    @property
    def reb(self) -> int:
        return self.orb + self.drb

    @property
    def minutes_display(self) -> str:
        total_sec = int(self.seconds)
        return f"{total_sec // 60}:{total_sec % 60:02d}"


@dataclass
class TeamBox:
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    orb: int = 0
    drb: int = 0
    tov: int = 0
    pf: int = 0
    points: int = 0
    ast: int = 0
    blk: int = 0
    stl: int = 0


@dataclass
class QuarterScore:
    home: int = 0
    away: int = 0


@dataclass
class PlaybackState:
    events: tuple[Event, ...]
    pointer: int = 0  # index of the next unprocessed event

    home_score: int = 0
    away_score: int = 0
    quarter: int = 1
    seconds_left: int = 600  # set from the first tip_off event
    home_team_fouls_q: int = 0
    away_team_fouls_q: int = 0
    home_box: TeamBox = field(default_factory=TeamBox)
    away_box: TeamBox = field(default_factory=TeamBox)
    home_players: dict[str, PlayerBox] = field(default_factory=dict)
    away_players: dict[str, PlayerBox] = field(default_factory=dict)

    # Per-quarter scores, indexed [Q1, Q2, Q3, Q4, OT1, OT2, ...].
    quarter_scores: list[QuarterScore] = field(default_factory=lambda: [QuarterScore()])

    # Optional live-attribution layer. When set, every event read out
    # of step_one() is run through ``lineup.attribute(e)`` before being
    # applied to box stats and returned. Substitutions on the lineup
    # affect *future* events only, since past events have already been
    # written to the box score.
    lineup: object | None = None  # LineupState; typed loose to avoid circular import

    # True iff the most-recently-applied event placed the game in a dead
    # ball — currently a foul or a non-steal turnover. The UI uses this
    # to decide whether substitutions can be queued: real basketball only
    # allows entry on a dead ball, so `b` no-ops elsewhere.
    is_dead_ball: bool = False

    is_dead_ball: bool = False

    _prev_quarter: int = 1
    _prev_seconds_left: int = 600

    @classmethod
    def from_events(cls, events: list[Event], lineup: object | None = None) -> "PlaybackState":
        if not events:
            raise ValueError("events list is empty; nothing to play back")
        s = cls(events=tuple(events), lineup=lineup)
        s.seconds_left = events[0].seconds_left
        s._prev_seconds_left = events[0].seconds_left
        s._prev_quarter = events[0].quarter
        return s

    @property
    def is_done(self) -> bool:
        return self.pointer >= len(self.events)

    @property
    def current(self) -> Event | None:
        """The most-recently-applied event, or None if nothing has played."""
        if self.pointer == 0:
            return None
        return self.events[self.pointer - 1]

    def in_bonus(self, shooting_side: Side) -> bool:
        defender_fouls = (
            self.home_team_fouls_q if shooting_side is Side.AWAY else self.away_team_fouls_q
        )
        return defender_fouls >= BONUS_THRESHOLD_PER_QUARTER

    # --- advancement --------------------------------------------------------

    def step_one(self) -> Event | None:
        """Apply the next event. Returns it (or None if at end).

        If a ``lineup`` is bound:

        - The event is attributed to the current on-court 5 first (and
          the in-place tuple is updated so subsequent UI lookups see
          the attributed event).
        - After the event applies, queued substitutions commit if the
          event was a dead ball — currently a foul (personal or
          shooting) or a non-steal turnover. A turnover paired with a
          ``steal`` credit event is a live ball; in that case the queue
          waits for the next dead ball."""
        if self.is_done:
            return None
        e = self.events[self.pointer]
        next_e = (
            self.events[self.pointer + 1]
            if self.pointer + 1 < len(self.events)
            else None
        )
        if self.lineup is not None and e.player is None:
            e = self.lineup.attribute(e)
            new_events = list(self.events)
            new_events[self.pointer] = e
            self.events = tuple(new_events)
        self._apply(e)
        self.is_dead_ball = self._is_dead_ball(e, next_e)
        if self.is_dead_ball and self.lineup is not None:
            self.lineup.commit_pending_subs()
        self.pointer += 1
        return e

    @staticmethod
    def _is_dead_ball(e: Event, next_e: Event | None) -> bool:
        """Return True iff applying ``e`` leaves the game in a dead-ball
        state. Spec: a foul, or a turnover not followed by a steal credit
        event (a steal turnover is a live-ball fast break)."""
        if e.type in ("foul_personal", "foul_shooting"):
            return True
        if e.type == "turnover":
            return next_e is None or next_e.type != "steal"
        return False

    def step_to_next_score_change(self, max_steps: int = 50) -> list[Event]:
        """Apply events until the score changes or a possession-ending event fires.

        Returns the list of events applied. ``max_steps`` is a safety cap.
        """
        applied: list[Event] = []
        starting_score = (self.home_score, self.away_score)
        for _ in range(max_steps):
            if self.is_done:
                break
            e = self.step_one()
            if e is None:
                break
            applied.append(e)
            if (self.home_score, self.away_score) != starting_score:
                break
            if e.type in ("turnover", "rebound_def", "rebound_off", "quarter_end"):
                break
        return applied

    def step_to_end_of_quarter(self) -> list[Event]:
        applied: list[Event] = []
        target = self.quarter
        while not self.is_done:
            e = self.step_one()
            if e is None:
                break
            applied.append(e)
            if e.type == "quarter_end" and target == e.quarter:
                break
        return applied

    def step_to_end(self) -> list[Event]:
        applied: list[Event] = []
        while not self.is_done:
            e = self.step_one()
            if e is None:
                break
            applied.append(e)
        return applied

    # --- internals ----------------------------------------------------------

    def _player_box(self, name: str, side: Side) -> PlayerBox:
        players = self.home_players if side is Side.HOME else self.away_players
        if name not in players:
            players[name] = PlayerBox(name=name)
        return players[name]

    def _credit_minutes(self, e: Event) -> None:
        if self.lineup is None:
            return
        if e.quarter == self._prev_quarter:
            elapsed = max(0, self._prev_seconds_left - e.seconds_left)
        else:
            elapsed = max(0, self._prev_seconds_left)
        if elapsed > 0:
            from hoops.engine.state import Side as _Side
            for side in (_Side.HOME, _Side.AWAY):
                players = self.home_players if side is _Side.HOME else self.away_players
                on_court_names = {p.name for p in self.lineup.on_court(side)}
                for name in on_court_names:
                    if name not in players:
                        players[name] = PlayerBox(name=name)
                    players[name].seconds += elapsed
                # Credit the event's player too if they were already subbed
                # out but the event was attributed while they were still in.
                if (
                    e.player
                    and e.team is side
                    and e.player not in on_court_names
                ):
                    if e.player not in players:
                        players[e.player] = PlayerBox(name=e.player)
                    players[e.player].seconds += elapsed
        self._prev_quarter = e.quarter
        self._prev_seconds_left = e.seconds_left

    def _apply(self, e: Event) -> None:
        self.home_score = e.home_score
        self.away_score = e.away_score
        self.seconds_left = e.seconds_left
        self._credit_minutes(e)

        if e.type == "quarter_end":
            self._record_quarter_end()
            self.home_team_fouls_q = 0
            self.away_team_fouls_q = 0
            return
        if e.type == "overtime_start":
            self.quarter = e.quarter
            self.seconds_left = e.seconds_left
            self.home_team_fouls_q = 0
            self.away_team_fouls_q = 0
            self.quarter_scores.append(QuarterScore(home=self.home_score, away=self.away_score))
            return
        if e.type == "game_end":
            return

        if e.quarter > self.quarter:
            self.quarter = e.quarter
            self.home_team_fouls_q = 0
            self.away_team_fouls_q = 0
            while len(self.quarter_scores) < e.quarter:
                self.quarter_scores.append(
                    QuarterScore(home=self.home_score, away=self.away_score)
                )

        # Keep the current quarter's cumulative score up to date so the
        # scoreboard shows correct per-quarter values mid-quarter.
        idx = self.quarter - 1
        while len(self.quarter_scores) <= idx:
            self.quarter_scores.append(QuarterScore())
        self.quarter_scores[idx] = QuarterScore(
            home=self.home_score, away=self.away_score,
        )

        if e.type == "substitution" and self.lineup is not None:
            self._apply_substitution(e)
            return

        if e.team is None:
            return
        team_box = self.home_box if e.team is Side.HOME else self.away_box
        pb = self._player_box(e.player, e.team) if e.player else None

        if e.type == "shot_made":
            team_box.fga += 1
            team_box.fgm += 1
            if pb:
                pb.fga += 1
                pb.fgm += 1
            if e.detail == "three":
                team_box.fg3a += 1
                team_box.fg3m += 1
                team_box.points += 3
                if pb:
                    pb.fg3a += 1
                    pb.fg3m += 1
                    pb.points += 3
            else:
                team_box.points += 2
                if pb:
                    pb.points += 2
        elif e.type == "shot_missed":
            team_box.fga += 1
            if pb:
                pb.fga += 1
            if e.detail == "three":
                team_box.fg3a += 1
                if pb:
                    pb.fg3a += 1
        elif e.type == "free_throw_made":
            team_box.fta += 1
            team_box.ftm += 1
            team_box.points += 1
            if pb:
                pb.fta += 1
                pb.ftm += 1
                pb.points += 1
        elif e.type == "free_throw_missed":
            team_box.fta += 1
            if pb:
                pb.fta += 1
        elif e.type == "rebound_off":
            team_box.orb += 1
            if pb:
                pb.orb += 1
        elif e.type == "rebound_def":
            team_box.drb += 1
            if pb:
                pb.drb += 1
        elif e.type == "turnover":
            team_box.tov += 1
            if pb:
                pb.tov += 1
        elif e.type == "assist":
            team_box.ast += 1
            if pb:
                pb.ast += 1
        elif e.type == "block":
            team_box.blk += 1
            if pb:
                pb.blk += 1
        elif e.type == "steal":
            team_box.stl += 1
            if pb:
                pb.stl += 1
        elif e.type == "foul_personal" or e.type == "foul_shooting":
            team_box.pf += 1
            if pb:
                pb.pf += 1
            if e.team is Side.HOME:
                self.home_team_fouls_q += 1
            else:
                self.away_team_fouls_q += 1

    def _apply_substitution(self, e: Event) -> None:
        parts = e.detail.split(" in for ")
        if len(parts) != 2:
            return
        on_name, off_name = parts
        side = e.team
        roster = self.lineup.roster(side)
        on_player = next((p for p in roster.players if p.name == on_name), None)
        off_player = next(
            (p for p in self.lineup.on_court(side) if p.name == off_name), None
        )
        if on_player and off_player:
            self.lineup.substitute(side, off_player.player_id, on_player.player_id)

    def _record_quarter_end(self) -> None:
        # Update the current quarter's running score (last value seen).
        # quarter_scores[i] holds (home, away) at end of quarter i+1.
        idx = self.quarter - 1
        while len(self.quarter_scores) <= idx:
            self.quarter_scores.append(QuarterScore())
        self.quarter_scores[idx] = QuarterScore(
            home=self.home_score, away=self.away_score
        )
