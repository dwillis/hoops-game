"""Game state, deliberately small and immutable.

The engine threads ``GameState`` through every operation; transformations
return a new state rather than mutating. This is what makes seeded
reproducibility tractable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

from hoops.rules import Rules


class Side(IntEnum):
    HOME = 0
    AWAY = 1

    @property
    def other(self) -> "Side":
        return Side.AWAY if self is Side.HOME else Side.HOME


@dataclass(frozen=True)
class GameState:
    rules: Rules

    quarter: int  # 1..4 regulation, 5+ for OT
    seconds_left: int  # in current period

    home_score: int
    away_score: int

    possession: Side

    # Team fouls accumulated within the current quarter (the field that
    # resets at every quarter rollover; doc §1.2). For OT, fouls reset
    # at the start of OT and stay for the duration of that OT period.
    home_team_fouls_q: int
    away_team_fouls_q: int

    # Total possessions ended by each team (for diagnostics / pace fits).
    home_possessions: int
    away_possessions: int

    # Tip-off winner; used to determine quarter-start possession via the
    # alternating-possession rule. Stored so end_period can compute it
    # without consulting the event log.
    opening_possession: Side

    @classmethod
    def initial(cls, rules: Rules, opening_possession: Side = Side.HOME) -> "GameState":
        if rules.structure != "quarters":
            raise ValueError(f"engine v0 only supports 'quarters' rules; got {rules.structure}")
        assert rules.quarter_minutes is not None
        return cls(
            rules=rules,
            quarter=1,
            seconds_left=rules.quarter_minutes * 60,
            home_score=0,
            away_score=0,
            possession=opening_possession,
            home_team_fouls_q=0,
            away_team_fouls_q=0,
            home_possessions=0,
            away_possessions=0,
            opening_possession=opening_possession,
        )

    @property
    def is_regulation(self) -> bool:
        return self.quarter <= 4

    @property
    def is_tied(self) -> bool:
        return self.home_score == self.away_score

    @property
    def is_final(self) -> bool:
        """Game is over if regulation ended and someone led, or OT ended and someone led."""
        if self.seconds_left > 0:
            return False
        if self.is_regulation and self.quarter < 4:
            return False
        return not self.is_tied

    def fouls_for(self, side: Side) -> int:
        return self.home_team_fouls_q if side is Side.HOME else self.away_team_fouls_q

    def score_for(self, side: Side) -> int:
        return self.home_score if side is Side.HOME else self.away_score

    def add_score(self, side: Side, points: int) -> "GameState":
        if side is Side.HOME:
            return replace(self, home_score=self.home_score + points)
        return replace(self, away_score=self.away_score + points)

    def add_team_foul(self, side: Side) -> "GameState":
        if side is Side.HOME:
            return replace(self, home_team_fouls_q=self.home_team_fouls_q + 1)
        return replace(self, away_team_fouls_q=self.away_team_fouls_q + 1)

    def with_possession(self, side: Side) -> "GameState":
        return replace(self, possession=side)

    def advance_clock(self, seconds: int) -> "GameState":
        if seconds < 0:
            raise ValueError(f"can't advance clock by negative seconds: {seconds}")
        return replace(self, seconds_left=max(0, self.seconds_left - seconds))

    def end_possession(self, by: Side) -> "GameState":
        if by is Side.HOME:
            return replace(self, home_possessions=self.home_possessions + 1)
        return replace(self, away_possessions=self.away_possessions + 1)
