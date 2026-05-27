"""Post-simulation pass that attaches a plausible player to each event.

The engine emits events without player attribution (it only knows team-
level rates). This module walks the event log, samples a roster member
for each event using event-type-appropriate weights, and returns a new
event list with ``player`` filled in.

Coupling rules:

- **Shot events** sample a shooter from the offensive roster (3-pointers
  use the ``three_point_shooter`` weight; 2-pointers use ``shooter``).
  The same shooter is remembered as the *current shooter* for subsequent
  free throws and the and-1 case.
- **Free throws** without a preceding shot (e.g. after an intentional
  foul-up-3) sample an FT shooter from the offensive roster's FTA
  weights, and that player is remembered for any chained FTs.
- **Fouls** sample from the defensive roster's foul weights.
- **Rebounds** sample from the appropriate side's ORB / DRB weights.
- **Turnovers** sample from the offensive roster's TOV weights.
- **Structural events** (tip_off / quarter_end / overtime_start /
  game_end) get no player.

The "current shooter" memory resets whenever a possession ends
(turnover, defensive rebound, quarter rollover, made shot without an
and-1).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from hoops.data.rosters import Roster
from hoops.engine.events import Event
from hoops.engine.state import Side

# Probability that a credit-style event accompanies its parent. These are
# WBB league-mean approximations (assists ~55% of FGM, steals ~50% of TOV,
# blocks ~6% of missed FGAs). Per-team rates would be a refinement; v0
# uses one rate league-wide because the credit event is cosmetic and the
# *attribution* (who) already varies by team.
_BLOCK_PROB = 0.06
_STEAL_PROB = 0.50
_ASSIST_PROB = 0.55


def _credit_event(parent: Event, type_, team: Side, player: str) -> Event:
    """Copy a parent's clock/score onto a synthetic credit event."""
    return Event(
        quarter=parent.quarter,
        seconds_left=parent.seconds_left,
        type=type_,
        team=team,
        player=player,
        home_score=parent.home_score,
        away_score=parent.away_score,
    )


def attribute_players(
    events: list[Event],
    home_roster: Roster,
    away_roster: Roster,
    rng: np.random.Generator,
) -> list[Event]:
    """Return a new event list with ``player`` filled in and supplementary
    credit events (assists / blocks / steals) inserted after the relevant
    primary events."""
    rosters: dict[Side, Roster] = {Side.HOME: home_roster, Side.AWAY: away_roster}
    out: list[Event] = []

    # Track the player currently expected to shoot any pending free throws.
    pending_ft_shooter: dict[Side, str | None] = {Side.HOME: None, Side.AWAY: None}

    for i, e in enumerate(events):
        next_e = events[i + 1] if i + 1 < len(events) else None

        if e.team is None or e.type in ("tip_off", "quarter_end", "overtime_start", "game_end"):
            out.append(e)
            continue

        attacker_side = e.team
        attacker_roster = rosters[attacker_side]
        defender_side = attacker_side.other

        if e.type == "shot_made":
            if e.player is not None:
                shooter_name = e.player
            else:
                shooter = (
                    attacker_roster.three_point_shooter(rng)
                    if e.detail == "three"
                    else attacker_roster.shooter(rng)
                )
                shooter_name = shooter.name
            pending_ft_shooter[attacker_side] = shooter_name
            out.append(dataclasses.replace(e, player=shooter_name))
            if rng.random() < _ASSIST_PROB:
                assister = attacker_roster.assister(rng, exclude=shooter_name)
                out.append(_credit_event(e, "assist", attacker_side, assister.name))
            continue

        if e.type == "shot_missed":
            if e.player is not None:
                shooter_name = e.player
            else:
                shooter = (
                    attacker_roster.three_point_shooter(rng)
                    if e.detail == "three"
                    else attacker_roster.shooter(rng)
                )
                shooter_name = shooter.name
            pending_ft_shooter[attacker_side] = shooter_name
            out.append(dataclasses.replace(e, player=shooter_name))
            fouled = next_e is not None and next_e.type == "foul_shooting"
            if not fouled and rng.random() < _BLOCK_PROB:
                blocker = rosters[defender_side].blocker(rng)
                out.append(_credit_event(e, "block", defender_side, blocker.name))
            continue

        if e.type == "free_throw_made" or e.type == "free_throw_missed":
            if e.player is not None:
                name = e.player
                pending_ft_shooter[attacker_side] = name
            else:
                name = pending_ft_shooter.get(attacker_side)
                if name is None:
                    shooter = attacker_roster.ft_shooter(rng)
                    name = shooter.name
                    pending_ft_shooter[attacker_side] = name
            out.append(dataclasses.replace(e, player=name))
            continue

        if e.type == "foul_personal" or e.type == "foul_shooting":
            fouler = rosters[e.team].fouler(rng)
            out.append(dataclasses.replace(e, player=fouler.name))
            continue

        if e.type == "rebound_off":
            reb = rosters[e.team].rebounder_off(rng)
            out.append(dataclasses.replace(e, player=reb.name))
            pending_ft_shooter[e.team] = None
            continue

        if e.type == "rebound_def":
            reb = rosters[e.team].rebounder_def(rng)
            out.append(dataclasses.replace(e, player=reb.name))
            pending_ft_shooter[Side.HOME] = None
            pending_ft_shooter[Side.AWAY] = None
            continue

        if e.type == "turnover":
            t = rosters[e.team].turnover(rng)
            out.append(dataclasses.replace(e, player=t.name))
            pending_ft_shooter[Side.HOME] = None
            pending_ft_shooter[Side.AWAY] = None
            # Maybe credit a steal to a defender.
            if rng.random() < _STEAL_PROB:
                stealer = rosters[defender_side].stealer(rng)
                out.append(_credit_event(e, "steal", defender_side, stealer.name))
            continue

        out.append(e)

    return out
