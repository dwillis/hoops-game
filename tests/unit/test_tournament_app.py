"""Tests for TournamentApp orchestration."""

from __future__ import annotations
import pytest
from hoops.ui.tournament_app import TournamentApp


def test_tournament_app_instantiation():
    """TournamentApp can be constructed with basic params."""
    app = TournamentApp(
        season="2023-24",
        user_team_id=2579,
        user_team_name="Gamecocks",
        seed=42,
    )
    assert app._season == "2023-24"
    assert app._user_team_id == 2579
    assert app._user_team_name == "Gamecocks"


def test_tournament_app_with_bracket_path():
    from pathlib import Path
    app = TournamentApp(
        season="2023-24",
        user_team_id=2579,
        user_team_name="Gamecocks",
        seed=42,
        bracket_path_override=Path("/fake/path.json"),
    )
    assert app._bracket_path_override == Path("/fake/path.json")
