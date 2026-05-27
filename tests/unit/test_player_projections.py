"""Tests for player-season projection pipeline."""

from __future__ import annotations

import pytest
import polars as pl
import numpy as np

from hoops.data.schemas import PlayerSeason
from hoops.data.paths import players_path, raw_dir, teams_path
from hoops.data.player_projections import project_player_season
from hoops.league import League


SEASON = "2023-24"


def _data_present() -> bool:
    return (raw_dir(League.WBB, SEASON) / "player_box.parquet").exists()


def test_player_season_schema_has_all_fields():
    p = PlayerSeason(
        league=League.WBB,
        season="2023-24",
        team_id=2579,
        player_id=12345,
        player_name="Test Player",
        position="G",
        games_played=30,
        games_started=28,
        minutes=900.0,
        fgm=200, fga=450, fg3m=50, fg3a=120,
        ftm=80, fta=100,
        orb=30, drb=150, ast=120, stl=45, blk=10,
        tov=80, fouls=60, points=530,
        min_share=0.28,
        usage_pct=0.25,
        ts_pct=0.55,
        fg3a_share=0.267,
        ft_pct=0.80,
        ast_pct=8.5,
        tov_pct=0.14,
        orb_pct=2.1,
        drb_pct=10.6,
        stl_pct=3.2,
        blk_pct=0.7,
        foul_rate=4.2,
        ppg_mean=17.7,
        ppg_std=6.2,
    )
    assert p.player_name == "Test Player"
    assert p.usage_pct == 0.25
    assert p.ppg_std == 6.2


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_project_returns_dataframe_with_expected_columns():
    df = project_player_season(SEASON)
    expected_cols = {
        "league", "season", "team_id", "player_id", "player_name", "position",
        "games_played", "games_started", "minutes",
        "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "orb", "drb", "ast", "stl", "blk", "tov", "fouls", "points",
        "min_share", "usage_pct", "ts_pct", "fg3a_share", "ft_pct",
        "ast_pct", "tov_pct", "orb_pct", "drb_pct",
        "stl_pct", "blk_pct", "foul_rate",
        "ppg_mean", "ppg_std",
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"missing: {expected_cols - set(df.columns)}"
    )


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_project_filters_low_minute_players():
    df = project_player_season(SEASON)
    assert df.filter(pl.col("minutes") < 10).height == 0


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_project_caitlin_clark_usage():
    df = project_player_season(SEASON)
    clark = df.filter(pl.col("player_name") == "Caitlin Clark")
    assert clark.height == 1
    usage = clark["usage_pct"].item()
    assert 0.28 <= usage <= 0.50, f"Clark usage {usage} outside expected range"


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_project_rate_bounds():
    df = project_player_season(SEASON)
    assert df.filter(pl.col("usage_pct") < 0).height == 0
    assert df.filter(pl.col("usage_pct") > 0.65).height == 0
    assert df.filter(pl.col("ts_pct") < 0).height == 0
    assert df.filter(pl.col("ts_pct") > 1.5).height == 0
    assert df.filter(pl.col("min_share") < 0).height == 0


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_project_team_fga_cross_check():
    df = project_player_season(SEASON)
    teams = pl.read_parquet(teams_path(League.WBB, SEASON))
    player_fga = df.group_by("team_id").agg(pl.col("fga").sum().alias("player_fga"))
    merged = teams.select("team_id", "fga").join(player_fga, on="team_id", how="inner")
    ratio = (merged["player_fga"] / merged["fga"]).mean()
    assert 0.85 <= ratio <= 1.02, f"player/team FGA ratio {ratio} outside expected range"


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_project_coverage_all_teams_have_players():
    df = project_player_season(SEASON)
    teams = pl.read_parquet(teams_path(League.WBB, SEASON))
    active_teams = teams.filter(pl.col("games") > 10)["team_id"].to_list()
    player_counts = df.group_by("team_id").len().rename({"len": "n_players"})
    for tid in active_teams:
        count = player_counts.filter(pl.col("team_id") == tid)
        assert count.height > 0, f"team {tid} has no projected players"
        assert count[0, "n_players"] >= 8, (
            f"team {tid} has only {count[0, 'n_players']} projected players"
        )


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_project_minutes_sanity():
    df = project_player_season(SEASON)
    teams = pl.read_parquet(teams_path(League.WBB, SEASON))
    player_mins = df.group_by("team_id").agg(
        pl.col("minutes").sum().alias("player_min_total")
    )
    merged = teams.select("team_id", "games").join(player_mins, on="team_id", how="inner")
    merged = merged.with_columns(
        expected_min=(pl.col("games").cast(pl.Float64) * 200.0)
    )
    ratio = (merged["player_min_total"] / merged["expected_min"]).mean()
    assert 0.85 <= ratio <= 1.05, f"player/team minutes ratio {ratio} out of range"


@pytest.mark.skipif(not _data_present(), reason="raw data missing")
def test_load_roster_prefers_projected_data():
    from hoops.data.rosters import load_roster

    pp = players_path(League.WBB, SEASON)
    if not pp.exists():
        pytest.skip("projected data not on disk yet")
    sc_id = 2579  # South Carolina
    roster = load_roster(sc_id, SEASON, top_n=8)
    assert roster.team_id == sc_id
    assert len(roster.players) == 8
    assert all(p.minutes > 0 for p in roster.players)
    names = {p.name for p in roster.players}
    assert "Kamilla Cardoso" in names
