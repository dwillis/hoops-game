"""Per-quarter bonus rules for WBB 2015-16+.

The relevant rule (doc §1.2): a team enters the bonus on the *opponent's*
5th team foul of the quarter, which awards two free throws (no 1-and-1).
The team-foul counter resets at every quarter rollover.

Encoded as a single function so the engine never inspects rule strings
directly. ``GameState.fouls_for(side)`` is the count *currently committed*
by ``side``; the bonus accrues to the *other* team.
"""

from __future__ import annotations

from hoops.engine.state import GameState, Side

BONUS_THRESHOLD_PER_QUARTER = 5


def is_in_bonus(state: GameState, shooting_side: Side) -> bool:
    """True if ``shooting_side``'s opponents have committed >= 5 fouls this quarter."""
    if state.rules.bonus != "per_quarter_5th_foul_two_shots":
        raise ValueError(f"engine v0 only supports per-quarter bonus; got {state.rules.bonus!r}")
    defender_fouls = state.fouls_for(shooting_side.other)
    return defender_fouls >= BONUS_THRESHOLD_PER_QUARTER
