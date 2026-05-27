"""Tests for fitted_seasons() and SeasonSelectScreen."""

from __future__ import annotations

from pathlib import Path

import pytest

from hoops.data.paths import fitted_seasons
from hoops.league import League


def test_fitted_seasons_returns_only_seasons_with_priors(tmp_path, monkeypatch):
    monkeypatch.setattr("hoops.data.paths.DATA_ROOT", tmp_path)

    dist_root = tmp_path / "pbp_distributions" / "wbb"
    dist_root.mkdir(parents=True)

    (dist_root / "2020-21").mkdir()
    (dist_root / "2020-21" / "team_priors.parquet").touch()

    (dist_root / "2021-22").mkdir()
    (dist_root / "2021-22" / "team_priors.parquet").touch()

    # Season with directory but no priors file should be excluded.
    (dist_root / "2022-23").mkdir()

    result = fitted_seasons(League.WBB)
    assert result == ["2020-21", "2021-22"]


def test_fitted_seasons_empty_when_no_dist_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("hoops.data.paths.DATA_ROOT", tmp_path)
    assert fitted_seasons(League.WBB) == []


def test_fitted_seasons_sorted(tmp_path, monkeypatch):
    monkeypatch.setattr("hoops.data.paths.DATA_ROOT", tmp_path)

    dist_root = tmp_path / "pbp_distributions" / "wbb"
    dist_root.mkdir(parents=True)

    for s in ["2024-25", "2015-16", "2019-20"]:
        (dist_root / s).mkdir()
        (dist_root / s / "team_priors.parquet").touch()

    result = fitted_seasons(League.WBB)
    assert result == ["2015-16", "2019-20", "2024-25"]


def _data_present() -> bool:
    from hoops.data.paths import distributions_dir
    return (distributions_dir(League.WBB, "2023-24") / "team_priors.parquet").exists()


@pytest.mark.skipif(not _data_present(), reason="no fitted data on disk")
def test_fitted_seasons_includes_real_data():
    result = fitted_seasons(League.WBB)
    assert "2023-24" in result
    assert len(result) >= 1


@pytest.mark.skipif(not _data_present(), reason="no fitted data on disk")
@pytest.mark.asyncio
async def test_season_picker_mounts_with_seasons():
    from hoops.ui.app import HoopsApp, SeasonSelectScreen

    app = HoopsApp(season=None, season_explicit=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SeasonSelectScreen)
