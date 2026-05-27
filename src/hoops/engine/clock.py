"""Quarter-rollover and overtime logic.

The doc emphasizes (§4) that end-of-quarter logic happens four times per
game, not twice — and that the per-quarter foul reset materially changes
late-game decision trees. v0 implements the structural piece; the
strategic 2-for-1 / hold-for-last layer is a Phase 6 hook.
"""

from __future__ import annotations

from dataclasses import replace

from hoops.engine.events import Event
from hoops.engine.state import GameState, Side


def _quarter_start_possession(opening: Side, next_quarter: int) -> Side:
    """NCAA alternating-possession rule for quarter starts.

    Tip winner gets Q1; opponent gets Q2; tip winner Q3; opponent Q4. For
    OT periods (>= 5) we alternate from the previous quarter, treating each
    OT as another rung on the alternation ladder. (Real NCAA OT starts
    with a jump ball; alternating is a reasonable v0 stand-in.)
    """
    return opening if next_quarter % 2 == 1 else opening.other


def end_period(state: GameState) -> tuple[GameState, list[Event]]:
    """Wrap up the current quarter/OT and start the next, if any.

    Returns the new state and any structural events emitted.
    """
    events: list[Event] = []
    events.append(Event(
        quarter=state.quarter,
        seconds_left=0,
        type="quarter_end",
        team=None,
        home_score=state.home_score,
        away_score=state.away_score,
    ))

    # Game ends if we're at the end of regulation (Q4) or any OT and not tied.
    if state.quarter >= 4 and not state.is_tied:
        events.append(Event(
            quarter=state.quarter,
            seconds_left=0,
            type="game_end",
            team=None,
            home_score=state.home_score,
            away_score=state.away_score,
        ))
        return state, events

    # Otherwise advance to next period and reset team fouls.
    next_quarter = state.quarter + 1
    period_seconds = (
        state.rules.quarter_minutes * 60
        if next_quarter <= 4
        else state.rules.ot_minutes * 60
    )
    next_state = replace(
        state,
        quarter=next_quarter,
        seconds_left=period_seconds,
        home_team_fouls_q=0,
        away_team_fouls_q=0,
        possession=_quarter_start_possession(state.opening_possession, next_quarter),
    )
    if next_quarter > 4:
        events.append(Event(
            quarter=next_quarter,
            seconds_left=period_seconds,
            type="overtime_start",
            team=None,
            home_score=state.home_score,
            away_score=state.away_score,
        ))
    return next_state, events
