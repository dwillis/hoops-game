"""Structured event log emitted by the engine.

The UI and validation harness consume events; they never inspect engine
internals. Adding fields is fine; renaming is breaking for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hoops.engine.state import Side

EventType = Literal[
    "tip_off",
    "shot_made",
    "shot_missed",
    "rebound_off",
    "rebound_def",
    "turnover",
    "block",
    "steal",
    "assist",
    "foul_personal",
    "foul_shooting",
    "free_throw_made",
    "free_throw_missed",
    "quarter_end",
    "overtime_start",
    "game_end",
    "substitution",
    "timeout",
    "media_timeout",
]


@dataclass(frozen=True)
class Event:
    quarter: int
    seconds_left: int
    type: EventType
    team: Side | None  # None for clock/structural events
    detail: str = ""
    home_score: int = 0
    away_score: int = 0
    player: str | None = None
    """Plain-text player name attached by the post-sim attribution pass.
    The engine itself emits events without players; ``attribute_players``
    fills this in by sampling roster weights."""


def fmt_clock(seconds_left: int) -> str:
    m, s = divmod(max(0, seconds_left), 60)
    return f"{m:02d}:{s:02d}"


_SHOT_LABEL = {
    "rim": "layup",
    "mid": "jumper",
    "three": "3-pointer",
}


def _team_tag(e: Event, home_short: str, away_short: str) -> str:
    """Short team label for the event, or empty for structural events."""
    if e.team is Side.HOME:
        return home_short
    if e.team is Side.AWAY:
        return away_short
    return ""


def _phrase(e: Event, team_label: str = "") -> str:
    """Natural-language description of one event.

    ``team_label`` is the short team name used as fallback actor when no
    player is attributed (e.g. "MD defensive rebound").
    """
    actor = e.player or team_label
    if e.type == "tip_off":
        return "Tip-off"
    if e.type == "shot_made":
        shot = _SHOT_LABEL.get(e.detail, "shot")
        return f"{actor} made {shot}".strip()
    if e.type == "shot_missed":
        shot = _SHOT_LABEL.get(e.detail, "shot")
        return f"{actor} missed {shot}".strip()
    if e.type == "rebound_off":
        return f"{actor} offensive rebound".strip()
    if e.type == "rebound_def":
        return f"{actor} defensive rebound".strip()
    if e.type == "turnover":
        return f"{actor} turnover".strip()
    if e.type == "block":
        return f"{actor} block".strip()
    if e.type == "steal":
        return f"{actor} steal".strip()
    if e.type == "assist":
        return f"{actor} assist".strip()
    if e.type == "foul_personal":
        if "intentional" in (e.detail or ""):
            return f"FOUL {actor} intentional".strip()
        return f"FOUL {actor}".strip()
    if e.type == "foul_shooting":
        return f"FOUL {actor} (shooting)".strip()
    if e.type == "free_throw_made":
        tag = " (and-1)" if e.detail and "and-1" in e.detail else ""
        return f"{actor} free throw good{tag}".strip()
    if e.type == "free_throw_missed":
        tag = " (and-1)" if e.detail and "and-1" in e.detail else ""
        return f"{actor} free throw missed{tag}".strip()
    if e.type == "quarter_end":
        return f"End of Q{e.quarter}"
    if e.type == "overtime_start":
        ot_idx = e.quarter - 4
        return f"Overtime {ot_idx} begins"
    if e.type == "game_end":
        return "Final"
    if e.type == "substitution":
        return f"SUB {e.detail}"
    if e.type == "timeout":
        remaining = f" ({e.detail})" if e.detail else ""
        return f"TIMEOUT{remaining}"
    if e.type == "media_timeout":
        return "MEDIA TIMEOUT"
    return e.type  # fallback, shouldn't happen


def fmt_event(e: Event, home_short: str = "Home", away_short: str = "Away") -> str:
    """Single-line render for the UI scroll panel.

    Format: ``Q1 09:41   0-0   Maryland  Bree Hall shooting foul``
    The team tag column is sized to the longer team name so both sides
    align consistently.
    """
    score = f"{e.home_score:>3}-{e.away_score:<3}"
    tag = _team_tag(e, home_short, away_short)
    tag_width = max(len(home_short), len(away_short))
    # When the tag column already shows the team name, don't repeat it
    # inside the phrase — pass empty team_label so _phrase uses just the
    # verb (e.g. "defensive rebound" not "Maryland defensive rebound").
    phrase = _phrase(e, team_label="" if tag else "")
    if tag:
        return f"Q{e.quarter} {fmt_clock(e.seconds_left)}  {score}  {tag:<{tag_width}s} {phrase}"
    pad = " " * tag_width
    return f"Q{e.quarter} {fmt_clock(e.seconds_left)}  {score}  {pad} {phrase}"
