"""Tests for hoops.engine.scheme_affinity."""

from __future__ import annotations

from hoops.data.rosters import Player
from hoops.engine.policy import DefensiveScheme
from hoops.engine.scheme_affinity import detect_archetype, scheme_affinity


def _player(pid, name, **kw):
    base = dict(
        player_id=pid, name=name, minutes=200.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
        min_share=0.25,
    )
    base.update(kw)
    return Player(**base)


# ---- archetype detection ---------------------------------------------------

def test_rim_protector_detected():
    p = _player(1, "Big Block", blk_pct=3.0, drb_pct=14.0)
    assert detect_archetype(p) == "rim_protector"


def test_perimeter_stopper_detected():
    p = _player(2, "Quick Hands", stl_pct=5.0, blk_pct=1.0)
    assert detect_archetype(p) == "perimeter_stopper"


def test_ball_handler_detected():
    p = _player(3, "Floor General", usage_pct=0.28, ast_pct=10.0)
    assert detect_archetype(p) == "ball_handler"


def test_floor_spacer_detected():
    p = _player(4, "Sharpshooter", fg3a_share=0.50, stl_pct=2.0)
    assert detect_archetype(p) == "floor_spacer"


def test_default_archetype():
    p = _player(5, "Role Player")
    assert detect_archetype(p) == "default"


# ---- scheme affinity values ------------------------------------------------

def test_rim_protector_zone_affinity():
    p = _player(1, "Big Block", blk_pct=3.0, drb_pct=14.0)
    aff = scheme_affinity(p)
    assert aff[DefensiveScheme.ZONE] > aff[DefensiveScheme.MAN]
    assert aff[DefensiveScheme.ZONE] > 1.0


def test_perimeter_stopper_press_affinity():
    p = _player(2, "Quick Hands", stl_pct=5.0, blk_pct=1.0)
    aff = scheme_affinity(p)
    assert aff[DefensiveScheme.PRESS] > aff[DefensiveScheme.ZONE]


def test_default_archetype_neutral():
    p = _player(5, "Role Player")
    aff = scheme_affinity(p)
    assert aff[DefensiveScheme.MAN] == 1.0
    assert aff[DefensiveScheme.ZONE] == 1.0
    assert aff[DefensiveScheme.PRESS] == 1.0


def test_scheme_affinity_handles_none_rates():
    p = Player(
        player_id=99, name="Bare Min", minutes=100.0,
        fga=50, fg3a=10, fta=20, orb=5, drb=20,
        fouls=10, tov=8, ast=12, blk=2, stl=3,
    )
    aff = scheme_affinity(p)
    assert aff[DefensiveScheme.MAN] == 1.0
    assert aff[DefensiveScheme.ZONE] == 1.0
    assert aff[DefensiveScheme.PRESS] == 1.0
