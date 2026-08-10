"""Tests for the DOS-style play-by-play vocabulary layer."""

from __future__ import annotations

from collections import Counter

import pytest

from hoops.engine import vocabulary as v
from hoops.engine.events import Event, fmt_event
from hoops.engine.state import Side

ACTOR = "Bree Hall"


def _mk(type_, *, detail="", **kw) -> Event:
    return Event(
        quarter=kw.pop("quarter", 1),
        seconds_left=kw.pop("seconds_left", 300),
        type=type_,
        team=kw.pop("team", Side.HOME),
        detail=detail,
        player=kw.pop("player", ACTOR),
        **kw,
    )


def _predicate(e: Event) -> str:
    """Phrase with the leading actor name stripped."""
    phrase = v.phrase_for(e)
    assert phrase is not None and phrase.startswith(e.player)
    return phrase[len(e.player) + 1:]


# --- Fallback behavior ----------------------------------------------------

def test_no_player_falls_back_to_none():
    for t in ("shot_made", "shot_missed", "turnover", "rebound_def", "foul_personal"):
        assert v.phrase_for(_mk(t, player=None)) is None


def test_unvaried_types_return_none():
    # Free throws, structural, standalone credit, subs stay terse.
    for t in ("free_throw_made", "tip_off", "assist", "steal", "block",
              "foul_shooting", "quarter_end", "substitution", "timeout"):
        assert v.phrase_for(_mk(t)) is None


def test_intentional_foul_stays_terse():
    assert v.phrase_for(_mk("foul_personal", detail="intentional (down 3)")) is None


# --- Determinism ----------------------------------------------------------

def test_selection_is_deterministic():
    for t, detail in [("shot_made", "mid"), ("shot_missed", "three"),
                      ("turnover", ""), ("rebound_off", "")]:
        e = _mk(t, detail=detail)
        assert v.phrase_for(e) == v.phrase_for(e)


def test_distinct_events_vary():
    # Different clocks/players should not all collapse to one phrase.
    seen = {v.phrase_for(_mk("shot_made", detail="mid", seconds_left=s))
            for s in range(0, 400)}
    assert len(seen) > 1


# --- Pool membership ------------------------------------------------------

@pytest.mark.parametrize("zone", ["rim", "mid", "three"])
def test_made_shot_in_pool(zone):
    pred = _predicate(_mk("shot_made", detail=zone))
    expected = {p for p, _ in v._MADE[zone]}
    # mid pool has a {dist} template; accept any filled distance.
    assert pred in expected or pred.startswith("hits from ")


@pytest.mark.parametrize("zone", ["rim", "mid", "three"])
def test_missed_shot_in_pool(zone):
    pred = _predicate(_mk("shot_missed", detail=zone))
    expected = {p for p, _ in v._MISSED[zone]}
    assert pred in expected or pred.startswith("misses from ")


def test_unknown_zone_defaults_to_mid():
    # An empty/odd detail shouldn't crash; treated as mid.
    assert v.phrase_for(_mk("shot_made", detail="")) is not None


def test_mid_distance_is_in_range():
    # Force a made shot whose selected verb is the distance template by
    # scanning clocks until we hit it.
    for s in range(2000):
        ev = _mk("shot_made", detail="mid", seconds_left=s)
        pred = _predicate(ev)
        if pred.startswith("hits from "):
            dist = int(pred.rsplit(" ", 1)[1])
            assert 12 <= dist <= 19
            return
    pytest.fail("distance template never selected")


# --- Assist weaving -------------------------------------------------------

def test_made_shot_weaves_assist():
    e = _mk("shot_made", detail="rim", assist_by="K. Smikle")
    phrase = v.phrase_for(e)
    assert "K. Smikle" in phrase
    assert phrase.startswith(ACTOR)


# --- Turnover pools -------------------------------------------------------

def test_dead_ball_turnover_pool():
    pred = _predicate(_mk("turnover"))
    assert pred in {p for p, _ in v._TOV_DEAD}


def test_live_ball_turnover_names_stealer():
    e = _mk("turnover", stolen_by="Defender Y")
    phrase = v.phrase_for(e)
    assert "Defender Y" in phrase
    # It should read as a live-ball loss, not a dead-ball violation.
    assert not phrase.endswith("travels")


def test_travels_at_least_3x_double_dribble_by_weight():
    weights = dict(v._TOV_DEAD)
    assert weights["travels"] >= v.TOV_TRAVEL_MIN_RATIO * weights["is called for double-dribble"]


def test_travels_at_least_3x_double_dribble_empirically():
    c = Counter()
    for q in range(1, 5):
        for s in range(600):
            for pid in range(6):
                pred = _predicate(_mk("turnover", quarter=q, seconds_left=s,
                                      player=f"P{pid}"))
                c[pred] += 1
    assert c["travels"] >= v.TOV_TRAVEL_MIN_RATIO * c["is called for double-dribble"]


# --- Missed shot with block -----------------------------------------------

def test_blocked_shot_names_blocker():
    e = _mk("shot_missed", detail="rim", blocked_by="A. Sellers")
    phrase = v.phrase_for(e)
    assert "blocked by A. Sellers" in phrase


# --- Rebounds & fouls -----------------------------------------------------

def test_rebound_pools():
    assert _predicate(_mk("rebound_def")) in {p for p, _ in v._REB_DEF}
    assert _predicate(_mk("rebound_off")) in {p for p, _ in v._REB_OFF}


def test_non_shooting_foul_suffix():
    # Foul lines read "FOUL {actor}" plus an optional suffix like "(reach-in)".
    full = v.phrase_for(_mk("foul_personal"))
    assert full.startswith(f"FOUL {ACTOR}")
    suffix = full[len(f"FOUL {ACTOR}"):]
    assert suffix in {s for s, _ in v._FOUL_SUFFIX}


# --- End-to-end through fmt_event -----------------------------------------

def test_fmt_event_renders_flavored_line():
    e = _mk("shot_made", detail="three", player="Caitlin Clark",
            home_score=3, away_score=0)
    out = fmt_event(e, "Iowa", "Maryland")
    assert "Caitlin Clark" in out
    assert "_" not in out.split("  ", 2)[-1]
