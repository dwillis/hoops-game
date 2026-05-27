"""Coaching policy: every per-possession decision the engine consults.

The engine is a pure simulator of basketball mechanics. Anything that
varies by *coaching judgment* — defensive scheme, end-of-quarter timing,
late-game fouling — lives behind this interface so AI and human coaches
share the same surface (doc §6 / Phase 6 plan).

v0 ships static policies: each ``CoachPolicy`` is set once before tip-off
and read on every possession. A future refinement can have the policy
respond to the live state (e.g. switch from man to zone after the third
foul on the opposing center). The engine's API already passes the full
``GameState`` to the relevant decision points, so that's a data-only
extension, not a structural change.

Substitutions / lineups are *not* in this phase: the engine has no
per-player entities yet (priors are team-level). The plan flags this
as the natural pairing with per-player rate fitting; until that lands,
only roster-independent controls live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hoops.engine.state import Side


class DefensiveScheme(str, Enum):
    """Doc §3.4: man / zone / press is the granularity our coaching data
    supports. Per-possession switching is a Phase 7 concern."""

    MAN = "man"
    ZONE = "zone"
    PRESS = "press"


class OffensiveScheme(str, Enum):
    """Offensive tempo/shot-selection scheme."""

    NORMAL = "normal"
    HURRY_UP = "hurry_up"
    SLOW_DOWN = "slow_down"
    THREE_POINT = "three_point"


@dataclass
class CoachPolicy:
    """One side's coaching dispositions. All fields have sensible defaults
    so callers can override only what they care about."""

    scheme: DefensiveScheme = DefensiveScheme.MAN
    off_scheme: OffensiveScheme = OffensiveScheme.NORMAL

    # End-of-quarter timing
    two_for_one: bool = True
    """If True, when offense holds the ball with ~35-50s left in a quarter,
    target a quick possession (~15-18s) so the team gets the last shot of
    the quarter as well."""

    hold_for_last: bool = True
    """If True, when offense has the ball with ≤30s in the quarter, hold
    the ball into the final seconds rather than shooting early."""

    # Late-game decision rules
    foul_when_down_3: bool = False
    """Down 3 with little time remaining and the opponent has the ball:
    foul to send them to the line, hoping they miss one or both, then
    get the ball back for a tying 3."""

    intentional_foul_in_bonus_when_trailing: bool = False
    """Trailing late, opponent in bonus: foul off-ball to stop the clock."""

    timeouts_remaining: int = 4
    """Per the rules table; decrements each call_timeout. v0 has no
    engine effect (no momentum modeling) but the count is tracked so
    the UI can display it."""


@dataclass
class CoachPolicies:
    """Pair of policies, one per side. The engine looks up the right
    policy via :meth:`for_side`."""

    home: CoachPolicy = field(default_factory=CoachPolicy)
    away: CoachPolicy = field(default_factory=CoachPolicy)

    def for_side(self, side: Side) -> CoachPolicy:
        return self.home if side is Side.HOME else self.away
