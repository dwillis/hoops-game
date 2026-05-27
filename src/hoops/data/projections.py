"""Project raw sportsdataverse-data Parquet into canonical schemas.

Inputs live under ``data/raw/wbb/<season>/`` (see scripts/ingest_wehoop.py).
Outputs land at the paths returned by ``hoops.data.paths``: one Parquet per
season per kind. Downstream phases (fitter, engine, UI) read only the
canonical files and never touch the raw frames.
"""

from __future__ import annotations

import polars as pl

from hoops.data.paths import games_path, raw_dir, teams_path
from hoops.data.sos import adjust as sos_adjust, compute_schedule
from hoops.league import League


# --- team-season aggregation --------------------------------------------------

def _possessions_expr() -> pl.Expr:
    """Dean Oliver's possession estimator from box-score totals."""
    return (
        pl.col("field_goals_attempted")
        + 0.44 * pl.col("free_throws_attempted")
        - pl.col("offensive_rebounds")
        + pl.col("total_turnovers")
    )


def project_team_seasons(season: str, league: League = League.WBB) -> pl.DataFrame:
    """Aggregate per-game team box scores into one row per team-season.

    The output schema is stable; the columns are exactly what the fitter and
    engine need. Adding columns is fine; renaming/removing is breaking.
    """
    src = raw_dir(league, season) / "team_box.parquet"
    box = pl.read_parquet(src)

    # Some columns are nullable in the source; cast for safe arithmetic.
    numeric_cols = [
        "field_goals_made", "field_goals_attempted",
        "three_point_field_goals_made", "three_point_field_goals_attempted",
        "free_throws_made", "free_throws_attempted",
        "offensive_rebounds", "defensive_rebounds",
        "total_turnovers", "fouls",
        "team_score", "opponent_team_score",
    ]
    box = box.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in numeric_cols])

    by_team = box.group_by("team_id").agg(
        pl.col("team_slug").first(),
        pl.col("team_short_display_name").first().alias("team_name"),
        pl.col("team_display_name").first(),
        pl.len().alias("games"),
        pl.col("team_winner").sum().alias("wins"),
        (pl.len() - pl.col("team_winner").sum()).alias("losses"),
        pl.col("field_goals_made").sum().alias("fgm"),
        pl.col("field_goals_attempted").sum().alias("fga"),
        pl.col("three_point_field_goals_made").sum().alias("fg3m"),
        pl.col("three_point_field_goals_attempted").sum().alias("fg3a"),
        pl.col("free_throws_made").sum().alias("ftm"),
        pl.col("free_throws_attempted").sum().alias("fta"),
        pl.col("offensive_rebounds").sum().alias("orb"),
        pl.col("defensive_rebounds").sum().alias("drb"),
        pl.col("total_turnovers").sum().alias("tov"),
        pl.col("fouls").sum().alias("personal_fouls"),
        pl.col("team_score").sum().alias("points_for"),
        pl.col("opponent_team_score").sum().alias("points_against"),
        _possessions_expr().sum().alias("poss"),
    )

    # Opponent four factors (per-team season totals of opponents-of-this-team)
    # come from joining the box on game_id with the opposing row.
    opp = box.select([
        pl.col("game_id"),
        pl.col("team_id").alias("opponent_team_id"),
        pl.col("field_goals_attempted").alias("opp_fga"),
        pl.col("free_throws_attempted").alias("opp_fta"),
        pl.col("offensive_rebounds").alias("opp_orb"),
        pl.col("defensive_rebounds").alias("opp_drb"),
        pl.col("total_turnovers").alias("opp_tov"),
        pl.col("field_goals_made").alias("opp_fgm"),
        pl.col("three_point_field_goals_made").alias("opp_fg3m"),
        pl.col("three_point_field_goals_attempted").alias("opp_fg3a"),
    ])
    paired = box.join(opp, on=["game_id", "opponent_team_id"], how="inner")

    by_opp = paired.group_by("team_id").agg(
        pl.col("opp_fga").sum().alias("def_fga"),
        pl.col("opp_fgm").sum().alias("def_fgm"),
        pl.col("opp_fg3m").sum().alias("def_fg3m"),
        pl.col("opp_fg3a").sum().alias("def_fg3a"),
        pl.col("opp_fta").sum().alias("def_fta"),
        pl.col("opp_orb").sum().alias("def_orb"),
        pl.col("opp_drb").sum().alias("def_drb"),
        pl.col("opp_tov").sum().alias("def_tov"),
        # average of own + opponent possessions per game = pace estimator
        ((_possessions_expr()
          + pl.col("opp_fga") + 0.44 * pl.col("opp_fta")
          - pl.col("opp_orb") + pl.col("opp_tov")) / 2).mean().alias("pace_per_game"),
    )

    out = by_team.join(by_opp, on="team_id", how="inner").with_columns(
        league=pl.lit(league.value),
        season=pl.lit(season),
        # offensive four factors
        off_efg=(pl.col("fgm") + 0.5 * pl.col("fg3m")) / pl.col("fga"),
        off_tov_pct=pl.col("tov") / (pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov")),
        off_orb_pct=pl.col("orb") / (pl.col("orb") + pl.col("def_drb")),
        off_fta_rate=pl.col("fta") / pl.col("fga"),
        off_3pt_rate=pl.col("fg3a") / pl.col("fga"),
        off_ft_pct=pl.when(pl.col("fta") > 0)
        .then(pl.col("ftm") / pl.col("fta"))
        .otherwise(None),
        # defensive four factors
        def_efg=(pl.col("def_fgm") + 0.5 * pl.col("def_fg3m")) / pl.col("def_fga"),
        def_tov_pct=pl.col("def_tov") / (pl.col("def_fga") + 0.44 * pl.col("def_fta") + pl.col("def_tov")),
        def_orb_pct=pl.col("def_orb") / (pl.col("def_orb") + pl.col("drb")),
        def_fta_rate=pl.col("def_fta") / pl.col("def_fga"),
        # pace = poss per 40, treating each game as 40 regulation minutes
        # (OT inflation washes out at season scale; flagged for revisit)
        pace=pl.col("pace_per_game"),
    )

    out = out.sort("team_id")

    # SoS-adjust the four factors and pace.
    schedule = compute_schedule(box)
    out = _add_sos_adjusted_columns(out, schedule)
    return out


_SOS_FACTORS = [
    # (raw_off_col, raw_def_col, adj_off_col, adj_def_col)
    ("off_efg", "def_efg", "off_efg_adj", "def_efg_adj"),
    ("off_tov_pct", "def_tov_pct", "off_tov_pct_adj", "def_tov_pct_adj"),
    ("off_orb_pct", "def_orb_pct", "off_orb_pct_adj", "def_orb_pct_adj"),
    ("off_fta_rate", "def_fta_rate", "off_fta_rate_adj", "def_fta_rate_adj"),
]


def _add_sos_adjusted_columns(out: pl.DataFrame, schedule: dict) -> pl.DataFrame:
    """Run iterative SoS on each four-factor stat and append _adj columns.

    Pace gets a single-stat treatment: a fast team plays fast both ways, so
    we adjust pace symmetrically (both adj_off_pace == adj_def_pace) by
    folding the team's raw pace as both off and def.
    """
    team_ids = out["team_id"].to_list()

    for raw_off_c, raw_def_c, adj_off_c, adj_def_c in _SOS_FACTORS:
        raw_off = dict(zip(team_ids, out[raw_off_c].to_list(), strict=True))
        raw_def = dict(zip(team_ids, out[raw_def_c].to_list(), strict=True))
        # Fill any nulls with league mean; SoS adjustment is robust to a few.
        league_mean = float(
            sum(v for v in raw_off.values() if v is not None)
            / sum(1 for v in raw_off.values() if v is not None)
        )
        raw_off = {k: (v if v is not None else league_mean) for k, v in raw_off.items()}
        raw_def = {k: (v if v is not None else league_mean) for k, v in raw_def.items()}
        adj_off, adj_def = sos_adjust(raw_off, raw_def, schedule, league_mean)
        out = out.with_columns(
            pl.Series(adj_off_c, [adj_off[t] for t in team_ids]),
            pl.Series(adj_def_c, [adj_def[t] for t in team_ids]),
        )

    # Pace adjustment: fold pace as both off and def for each team. The
    # league mean of pace is unchanged; the per-team adjusted pace tells
    # you what tempo the team would play at against a league-average
    # opponent's pace preference.
    raw_pace = dict(zip(team_ids, out["pace"].to_list(), strict=True))
    pace_mean = float(sum(raw_pace.values()) / len(raw_pace))
    adj_pace_off, adj_pace_def = sos_adjust(raw_pace, raw_pace, schedule, pace_mean)
    # Combine off/def into one pace-rating per team: their tempo preference.
    adj_pace = {t: 0.5 * (adj_pace_off[t] + adj_pace_def[t]) for t in team_ids}
    out = out.with_columns(pl.Series("pace_adj", [adj_pace[t] for t in team_ids]))

    return out


def project_games(season: str, league: League = League.WBB) -> pl.DataFrame:
    """One row per game with the canonical Game fields."""
    src = raw_dir(league, season) / "team_box.parquet"
    box = pl.read_parquet(src)

    # team_box has two rows per game (one per team). We collapse into one.
    home = box.filter(pl.col("team_home_away") == "home").select([
        pl.col("game_id"),
        pl.col("season"),
        pl.col("season_type"),
        pl.col("game_date"),
        pl.col("team_id").alias("home_team_id"),
        pl.col("team_score").alias("home_score"),
        pl.col("opponent_team_id").alias("away_team_id"),
        pl.col("opponent_team_score").alias("away_score"),
    ])
    return home.with_columns(
        league=pl.lit(league.value),
        season=pl.lit(season),
        # Postseason in ESPN is season_type == 3.
        postseason=pl.col("season_type") == 3,
        # Phase 2 leaves all games tagged "good"; pbp-derived quality
        # flagging happens in the fitter when we look at per-game event
        # consistency. Default keeps schema stable.
        lineup_quality=pl.lit("good"),
    ).sort("game_date", "game_id")


def write_canonical(season: str, league: League = League.WBB) -> dict[str, str]:
    """Run all projections and persist them. Returns paths written."""
    out: dict[str, str] = {}
    teams = project_team_seasons(season, league)
    teams_dst = teams_path(league, season)
    teams_dst.parent.mkdir(parents=True, exist_ok=True)
    teams.write_parquet(teams_dst)
    out["teams"] = str(teams_dst)

    games = project_games(season, league)
    games_dst = games_path(league, season)
    games_dst.parent.mkdir(parents=True, exist_ok=True)
    games.write_parquet(games_dst)
    out["games"] = str(games_dst)
    return out
