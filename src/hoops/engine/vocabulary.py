"""Play-by-play flavor text — DOS-Hoops-style sentence variety.

This is a *presentation* layer. It never touches the engine RNG: variant
selection is a deterministic ``zlib.crc32`` hash of the event's identity
fields, so the same event always renders the same line (watch mode,
interactive play, and a reloaded save all agree) and seed reproducibility
and calibration are untouched.

``phrase_for(event)`` returns a flavored natural-language phrase, or
``None`` when the event has no attributed player or isn't a type we vary —
in which case the caller (:func:`hoops.engine.events._phrase`) falls back
to its terse phrasing.

Templates are written for WBB (she/her) but avoid asserting facts the
attribution pass doesn't guarantee (e.g. whose miss a rebound followed).
Dunks / alley-oops are intentionally excluded — rare in WBB.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hoops.engine.events import Event

# --- Template pools -------------------------------------------------------
#
# Each pool is a list of (predicate, weight). The predicate follows the
# actor's name ("{actor} <predicate>"), except assist phrasings which are
# appended after a made-shot line. ``{dist}`` is replaced with a
# hash-derived mid-range distance; ``{a}``/``{s}`` with the assister /
# stealer name.

_MADE = {
    "rim": [
        ("drives and lays it in", 5),
        ("drives the lane and lays it in", 4),
        ("puts it up and in", 4),
        ("finishes in traffic", 3),
        ("scores on a putback", 2),
        ("scores on a goaltend", 1),  # ~1%
    ],
    "mid": [
        ("hits from {dist}", 5),
        ("sinks a baseline jumper", 4),
        ("hits a turnaround jumper", 3),
        ("hits from the elbow", 3),
        ("knocks down the mid-range jumper", 3),
    ],
    "three": [
        ("sinks a three-pointer", 5),
        ("hits from the corner", 4),
        ("connects from deep", 3),
        ("buries one from the wing", 3),
        ("drains a three from the top of the key", 2),
    ],
}

_MISSED = {
    "rim": [
        ("misses a layup", 5),
        ("misses in traffic", 3),
        ("can't finish at the rim", 3),
        ("has the layup roll off", 2),
    ],
    "mid": [
        ("misses from {dist}", 5),
        ("misses a baseline jumper", 3),
        ("misses a turnaround jumper", 3),
        ("misfires from the elbow", 3),
    ],
    "three": [
        ("misses from three-point range", 5),
        ("rims out from the corner", 3),
        ("misses from deep", 3),
        ("is off the mark from the wing", 2),
    ],
}

_ASSISTS = [
    ("(good pass from {a})", 3),
    ("({a} with the assist)", 3),
    ("(good feed by {a})", 2),
    ("(fine pass from {a})", 2),
    ("({a} with the feed)", 2),
]

# Dead-ball turnovers. "travels" is weighted comfortably above 3x
# "double-dribble" (8 vs 2 => 4x expected) so that even with hash-selection
# variance, travels actually occurs at least 3x as often. See TOV_TRAVEL_MIN.
_TOV_DEAD = [
    ("travels", 8),
    ("throws the ball out of bounds", 4),
    ("loses control", 3),
    ("is called for a carry", 2),
    ("commits an over-and-back violation", 2),
    ("is called for double-dribble", 2),
]

#: Required minimum ratio of "travels" to "double-dribble" occurrence.
TOV_TRAVEL_MIN_RATIO = 3

# Live-ball turnovers, used when a steal was credited. Each incorporates
# the stealer name ``{s}`` so it reads as one line.
_TOV_LIVE = [
    ("loses the ball to {s}", 4),
    ("has the ball knocked away by {s}", 3),
    ("has the pass picked off by {s}", 3),
    ("makes a bad pass, stolen by {s}", 3),
]

_REB_DEF = [
    ("pulls down the rebound", 5),
    ("grabs the board", 3),
    ("controls the defensive rebound", 3),
    ("clears the glass", 2),
]

_REB_OFF = [
    ("grabs the offensive rebound", 5),
    ("keeps the trip alive with the offensive board", 3),
    ("cleans up the offensive glass", 3),
    ("battles for the offensive board", 2),
]

# Non-shooting personal fouls. Empty suffix keeps some plain "FOUL X" lines.
_FOUL_SUFFIX = [
    ("", 4),
    (" (reach-in)", 3),
    (" (holding)", 2),
    (" (blocking)", 2),
    (" (over the back)", 1),
]


def _event_key(e: Event) -> str:
    """Stable identity string for one event (drives variant selection)."""
    team = int(e.team) if e.team is not None else -1
    return "|".join(str(x) for x in (
        e.quarter, e.seconds_left, e.type, team, e.player,
        e.home_score, e.away_score, e.detail,
    ))


def _pick(pool: list[tuple[str, int]], key: str, salt: str) -> str:
    """Deterministic weighted choice from ``pool`` for this event+channel."""
    total = sum(w for _, w in pool)
    h = zlib.crc32(f"{key}:{salt}".encode()) % total
    upto = 0
    for text, w in pool:
        upto += w
        if h < upto:
            return text
    return pool[-1][0]


def _sub_dist(predicate: str, key: str) -> str:
    """Fill ``{dist}`` with a hash-derived mid-range distance (12-19 ft)."""
    if "{dist}" not in predicate:
        return predicate
    d = 12 + (zlib.crc32(f"{key}:dist".encode()) % 8)
    return predicate.replace("{dist}", str(d))


def _zone(e: Event) -> str:
    return e.detail if e.detail in ("rim", "mid", "three") else "mid"


def phrase_for(e: Event) -> str | None:
    """Flavored phrase for an event, or ``None`` to use terse fallback."""
    if e.player is None:
        return None

    key = _event_key(e)
    actor = e.player
    t = e.type

    if t == "shot_made":
        verb = _sub_dist(_pick(_MADE[_zone(e)], key, "verb"), key)
        line = f"{actor} {verb}"
        if e.assist_by:
            line = f"{line} {_pick(_ASSISTS, key, 'assist').format(a=e.assist_by)}"
        return line

    if t == "shot_missed":
        if e.blocked_by:
            return f"{actor} shoots; it's blocked by {e.blocked_by}"
        verb = _sub_dist(_pick(_MISSED[_zone(e)], key, "verb"), key)
        return f"{actor} {verb}"

    if t == "turnover":
        if e.stolen_by:
            return f"{actor} {_pick(_TOV_LIVE, key, 'tov').format(s=e.stolen_by)}"
        return f"{actor} {_pick(_TOV_DEAD, key, 'tov')}"

    if t == "rebound_off":
        return f"{actor} {_pick(_REB_OFF, key, 'reb')}"

    if t == "rebound_def":
        return f"{actor} {_pick(_REB_DEF, key, 'reb')}"

    if t == "foul_personal":
        # Intentional fouls keep their terse rendering (handled by caller).
        if "intentional" in (e.detail or ""):
            return None
        return f"FOUL {actor}{_pick(_FOUL_SUFFIX, key, 'foul')}"

    # Everything else (free throws, tip-off, structural, standalone credit
    # events, substitutions, timeouts, shooting fouls) uses terse phrasing.
    return None
