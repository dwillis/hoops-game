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
    """Doc §3.4: man / zone / press is the base granularity. Effort level
    (safe / normal / tight) is an orthogonal knob — see DefensiveIntensity."""

    MAN = "man"
    ZONE = "zone"
    PRESS = "press"


class DefensiveIntensity(str, Enum):
    """How hard the defense pressures, orthogonal to the scheme.

    NORMAL is the identity baseline (no adjustment). TIGHT trades better
    shot contests for more fouls; SAFE gives up cleaner looks but fouls
    less and gambles less. Applied on top of the scheme in apply_scheme."""

    SAFE = "safe"
    NORMAL = "normal"
    TIGHT = "tight"


class OffensiveScheme(str, Enum):
    """Offensive tempo/shot-selection scheme.

    SEMISTALL is the former SLOW_DOWN (renamed for the DOS-style stall
    family); the legacy ``"slow_down"`` string still deserializes to it via
    :meth:`_missing_`. STALL is the full clock-bleed set."""

    NORMAL = "normal"
    HURRY_UP = "hurry_up"
    SEMISTALL = "semistall"
    STALL = "stall"
    THREE_POINT = "three_point"

    @classmethod
    def _missing_(cls, value: object) -> OffensiveScheme | None:
        # Backward compatibility: saves written before the stall family used
        # "slow_down" for what is now SEMISTALL.
        if value == "slow_down":
            return cls.SEMISTALL
        return None


@dataclass
class CoachPolicy:
    """One side's coaching dispositions. All fields have sensible defaults
    so callers can override only what they care about."""

    scheme: DefensiveScheme = DefensiveScheme.MAN
    intensity: DefensiveIntensity = DefensiveIntensity.NORMAL
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
    """Per Rules.timeouts_per_team; decrements each call_timeout. v0 has
    no engine effect (no momentum modeling) but the count is tracked so
    the UI can display it."""

    # Per-player coaching (roster-dependent; None/0 = identity behavior)
    shot_distribution: dict[int, float] | None = None
    """Optional coach-set shot shares, ``player_id -> desired share`` (0..1),
    for the on-court five. ``None`` (the default) uses natural usage. Shares
    for benched players are ignored; unset on-court players split the leftover
    by usage. Pushing a player above their natural usage taxes their eFG."""

    double_team_pct: float = 0.0
    """Probability [0..1] of double-teaming the opponent's top scoring option
    each possession. 0.0 (default) never doubles. Doubling forces more
    turnovers and steals but opens up the doubled player's teammates."""

    man_assignments: dict[int, int] | None = None
    """Optional man-to-man matchups, ``defender_id -> opponent_id``. Only
    used when ``scheme`` is MAN. ``None`` (default) uses the auto-generated
    matchups, which are exactly neutral. Effects are computed as deltas versus
    that default, so a coach swap onto a star is a real tradeoff."""


@dataclass
class CoachPolicies:
    """Pair of policies, one per side. The engine looks up the right
    policy via :meth:`for_side`."""

    home: CoachPolicy = field(default_factory=CoachPolicy)
    away: CoachPolicy = field(default_factory=CoachPolicy)

    def for_side(self, side: Side) -> CoachPolicy:
        return self.home if side is Side.HOME else self.away
