"""Project per-player advanced stats from raw box scores.

Reads raw player_box.parquet and team-season aggregates to derive
per-100-possession rates for each (player, team, season). Output
lands at data/players/wbb/<season>.parquet.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from hoops.data.paths import players_path, raw_dir, teams_path
from hoops.league import League

_MIN_MINUTES = 10


def project_player_season(
    season: str, league: League = League.WBB
) -> pl.DataFrame:
    pb = pl.read_parquet(raw_dir(league, season) / "player_box.parquet")

    pb = pb.filter(
        (pl.col("did_not_play") != True)
        & pl.col("minutes").is_not_null()
        & (pl.col("minutes") > 0)
    )

    variance = (
        pb.group_by(["athlete_id", "team_id"])
        .agg(
            pl.col("points").mean().alias("ppg_mean"),
            pl.col("points").std().alias("ppg_std"),
        )
    )

    agg = (
        pb.group_by(["athlete_id", "team_id"])
        .agg(
            pl.col("athlete_display_name").first().alias("player_name"),
            pl.col("athlete_position_abbreviation").first().alias("position"),
            pl.col("game_id").n_unique().alias("games_played"),
            pl.col("starter").sum().cast(pl.Int32).alias("games_started"),
            pl.col("minutes").sum().alias("minutes"),
            pl.col("field_goals_made").sum().alias("fgm"),
            pl.col("field_goals_attempted").sum().alias("fga"),
            pl.col("three_point_field_goals_made").sum().alias("fg3m"),
            pl.col("three_point_field_goals_attempted").sum().alias("fg3a"),
            pl.col("free_throws_made").sum().alias("ftm"),
            pl.col("free_throws_attempted").sum().alias("fta"),
            pl.col("offensive_rebounds").sum().alias("orb"),
            pl.col("defensive_rebounds").sum().alias("drb"),
            pl.col("assists").sum().alias("ast"),
            pl.col("steals").sum().alias("stl"),
            pl.col("blocks").sum().alias("blk"),
            pl.col("turnovers").sum().alias("tov"),
            pl.col("fouls").sum().alias("fouls"),
            pl.col("points").sum().alias("points"),
        )
    )

    agg = agg.join(variance, on=["athlete_id", "team_id"], how="left")
    agg = agg.filter(pl.col("minutes") >= _MIN_MINUTES)

    teams = pl.read_parquet(teams_path(league, season)).select(
        "team_id", "poss", "games"
    )
    agg = agg.join(teams, on="team_id", how="left")

    team_min_pool = pl.col("games").cast(pl.Float64) * 200.0
    min_share = pl.col("minutes") / team_min_pool
    # Player's share of team possessions: scale total poss by fraction
    # of game-minutes (40 min/game), not player-minutes (200 min/game).
    team_game_min = pl.col("games").cast(pl.Float64) * 40.0
    player_poss = pl.col("poss").cast(pl.Float64) * (
        pl.col("minutes") / team_game_min
    )

    shot_volume = (
        pl.col("fga").cast(pl.Float64)
        + 0.44 * pl.col("fta").cast(pl.Float64)
        + pl.col("tov").cast(pl.Float64)
    )

    ts_denom = 2.0 * (
        pl.col("fga").cast(pl.Float64) + 0.44 * pl.col("fta").cast(pl.Float64)
    )

    agg = agg.with_columns(
        league=pl.lit(league.value),
        season=pl.lit(season),
        min_share=min_share,
        usage_pct=pl.when(player_poss > 0)
            .then(shot_volume / player_poss)
            .otherwise(None),
        ts_pct=pl.when(ts_denom > 0)
            .then(pl.col("points").cast(pl.Float64) / ts_denom)
            .otherwise(None),
        fg3a_share=pl.when(pl.col("fga") > 0)
            .then(pl.col("fg3a").cast(pl.Float64) / pl.col("fga").cast(pl.Float64))
            .otherwise(None),
        ft_pct=pl.when(pl.col("fta") > 0)
            .then(pl.col("ftm").cast(pl.Float64) / pl.col("fta").cast(pl.Float64))
            .otherwise(None),
        ast_pct=pl.when(player_poss > 0)
            .then(pl.col("ast").cast(pl.Float64) / player_poss * 100.0)
            .otherwise(None),
        tov_pct=pl.when(shot_volume > 0)
            .then(pl.col("tov").cast(pl.Float64) / shot_volume)
            .otherwise(None),
        orb_pct=pl.when(player_poss > 0)
            .then(pl.col("orb").cast(pl.Float64) / player_poss * 100.0)
            .otherwise(None),
        drb_pct=pl.when(player_poss > 0)
            .then(pl.col("drb").cast(pl.Float64) / player_poss * 100.0)
            .otherwise(None),
        stl_pct=pl.when(player_poss > 0)
            .then(pl.col("stl").cast(pl.Float64) / player_poss * 100.0)
            .otherwise(None),
        blk_pct=pl.when(player_poss > 0)
            .then(pl.col("blk").cast(pl.Float64) / player_poss * 100.0)
            .otherwise(None),
        foul_rate=pl.when(player_poss > 0)
            .then(pl.col("fouls").cast(pl.Float64) / player_poss * 100.0)
            .otherwise(None),
    )

    agg = agg.rename({"athlete_id": "player_id"})

    return agg.select([
        "league", "season", "team_id", "player_id", "player_name", "position",
        "games_played", "games_started", "minutes",
        "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "orb", "drb", "ast", "stl", "blk", "tov", "fouls", "points",
        "min_share", "usage_pct", "ts_pct", "fg3a_share", "ft_pct",
        "ast_pct", "tov_pct", "orb_pct", "drb_pct",
        "stl_pct", "blk_pct", "foul_rate",
        "ppg_mean", "ppg_std",
    ]).sort("team_id", "minutes", descending=[False, True])


def write_player_projections(
    season: str, league: League = League.WBB
) -> Path:
    df = project_player_season(season, league)
    dst = players_path(league, season)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst)
    return dst
