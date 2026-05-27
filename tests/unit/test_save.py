"""Tests for save/load file I/O."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hoops.engine.save import save_game, load_save, has_save, saves_dir, save_path_for


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
