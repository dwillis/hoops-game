"""Tests for save/load file I/O."""
from __future__ import annotations

from hoops.engine.events import Event
from hoops.engine.interactive import (
    _deserialize_event,
    _deserialize_policy,
    _serialize_event,
    _serialize_policy,
)
from hoops.engine.policy import (
    CoachPolicy,
    DefensiveIntensity,
    DefensiveScheme,
    OffensiveScheme,
)
from hoops.engine.save import has_save, load_save, save_game, save_path_for, saves_dir
from hoops.engine.state import Side


def test_policy_roundtrip_preserves_all_coaching_fields():
    p = CoachPolicy(
        scheme=DefensiveScheme.PRESS,
        intensity=DefensiveIntensity.TIGHT,
        off_scheme=OffensiveScheme.STALL,
        shot_distribution={1: 0.4, 2: 0.3, 3: 0.3},
        double_team_pct=0.6,
        man_assignments={10: 21, 11: 22},
    )
    out = _deserialize_policy(_serialize_policy(p))
    assert out == p
    # Dict keys must survive as ints (JSON stringifies them).
    assert all(isinstance(k, int) for k in out.shot_distribution)
    assert all(isinstance(k, int) and isinstance(v, int)
               for k, v in out.man_assignments.items())


def test_old_save_defaults_new_perplayer_fields():
    d = {"scheme": "man", "off_scheme": "normal"}
    p = _deserialize_policy(d)
    assert p.shot_distribution is None
    assert p.double_team_pct == 0.0
    assert p.man_assignments is None


def test_old_save_policy_defaults_intensity_normal():
    # A pre-intensity save dict lacks the "intensity" key.
    d = {"scheme": "man", "off_scheme": "normal"}
    p = _deserialize_policy(d)
    assert p.intensity is DefensiveIntensity.NORMAL


def test_old_save_slow_down_aliases_to_semistall():
    d = {"scheme": "man", "off_scheme": "slow_down"}
    p = _deserialize_policy(d)
    assert p.off_scheme is OffensiveScheme.SEMISTALL


def test_event_roundtrip_preserves_inline_credit_fields():
    made = Event(
        quarter=3, seconds_left=210, type="shot_made", team=Side.HOME,
        detail="rim", home_score=2, away_score=0,
        player="Bree Hall", assist_by="K. Smikle",
    )
    tov = Event(
        quarter=1, seconds_left=100, type="turnover", team=Side.AWAY,
        player="P", stolen_by="Defender",
    )
    miss = Event(
        quarter=2, seconds_left=50, type="shot_missed", team=Side.HOME,
        detail="three", player="Shooter", blocked_by="Blocker",
    )
    for e in (made, tov, miss):
        assert _deserialize_event(_serialize_event(e)) == e


def test_deserialize_old_save_defaults_new_fields():
    # A version-1 save dict lacking the new keys must load cleanly.
    d = {
        "quarter": 1, "seconds_left": 300, "type": "turnover",
        "team": 0, "detail": "", "home_score": 0, "away_score": 0,
        "player": "P",
    }
    e = _deserialize_event(d)
    assert e.assist_by is None
    assert e.stolen_by is None
    assert e.blocked_by is None


def test_saves_dir_returns_hoops_saves_path():
    d = saves_dir()
    assert d.name == "saves"
    assert ".hoops" in str(d)


def test_save_path_for_generates_filename():
    p = save_path_for("Maryland", "South Carolina", "2023-24")
    assert p.name == "Maryland_vs_South_Carolina_2023-24.json"


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("hoops.engine.save._SAVES_DIR", tmp_path)
    fake_save = {
        "version": 1,
        "home_team_id": 1,
        "away_team_id": 2,
        "human_side": 0,
        "game_state": {"quarter": 2},
        "events": [],
    }
    save_game(fake_save, "Home", "Away", "2023-24")
    expected = tmp_path / "Home_vs_Away_2023-24.json"
    assert expected.exists()
    loaded = load_save(expected)
    assert loaded["version"] == 1
    assert loaded["home_team_id"] == 1


def test_has_save_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("hoops.engine.save._SAVES_DIR", tmp_path)
    assert not has_save("Foo", "Bar", "2023-24")


def test_has_save_true_after_save(tmp_path, monkeypatch):
    monkeypatch.setattr("hoops.engine.save._SAVES_DIR", tmp_path)
    save_game({"version": 1}, "Foo", "Bar", "2023-24")
    assert has_save("Foo", "Bar", "2023-24")
