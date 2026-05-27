"""CLI smoke tests for `hoops play`.

The full Textual app launch is verified in tests/unit/test_ui_app.py via
Pilot. Here we cover the CLI-side glue: team resolution and the parts of
the play handler that run *before* HoopsApp is instantiated.
"""

from __future__ import annotations

import pytest

from typer.testing import CliRunner

from hoops.cli import _resolve_team, app
from hoops.data.paths import teams_path
from hoops.league import League

runner = CliRunner()


def _data_present() -> bool:
    return teams_path(League.WBB, "2023-24").exists()


pytestmark = pytest.mark.skipif(
    not _data_present(),
    reason="canonical team-season parquet missing",
)


def test_resolve_team_by_name_substring():
    team_id, name = _resolve_team("south-carolina", "2023-24")
    assert name == "South Carolina"
    assert team_id == 2579


def test_resolve_team_case_insensitive():
    team_id, name = _resolve_team("IOWA", "2023-24")
    assert name == "Iowa"


def test_resolve_team_by_full_slug():
    team_id, name = _resolve_team("uconn-huskies", "2023-24")
    assert name == "UConn"


def test_resolve_team_ambiguous_raises():
    import typer

    with pytest.raises(typer.BadParameter, match="ambiguous"):
        _resolve_team("state", "2023-24")  # many "X State" teams


def test_resolve_team_unknown_raises():
    import typer

    with pytest.raises(typer.BadParameter, match="no team matching"):
        _resolve_team("definitely-not-a-team", "2023-24")


def test_seasons_command_lists_fitted():
    result = runner.invoke(app, ["seasons"])
    assert result.exit_code == 0
    assert "2023-24" in result.output
