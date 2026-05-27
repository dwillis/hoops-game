"""Fit per-team priors from canonical team-season aggregates and raw pbp.

Inputs:
- ``data/teams/wbb/<season>.parquet`` — produced by projections.py
- ``data/raw/wbb/<season>/pbp.parquet`` — produced by ingest_wehoop.py

Outputs:
- ``data/pbp_distributions/wbb/<season>/team_priors.parquet`` — one row/team
- ``data/pbp_distributions/wbb/<season>/league_prior.parquet`` — one row

Shot-zone classification uses pbp ``type_text`` and ``score_value``:
- rim:   LayUpShot, DunkShot, TipShot
- three: any shot row with score_value == 3 (mostly JumpShot)
- mid:   the remaining 2-point shot rows (mostly JumpShot with score_value == 2)
"""

from __future__ import annotations

import polars as pl

from hoops.data.paths import distributions_dir, raw_dir, teams_path
from hoops.league import League

_RIM_TYPES = ("LayUpShot", "DunkShot", "TipShot")


def _classify_shot_zone() -> pl.Expr:
    """Return an expr that maps a pbp row to 'rim' / 'mid' / 'three' / null."""
    return (
        pl.when(pl.col("type_text").is_in(_RIM_TYPES)).then(pl.lit("rim"))
        .when(pl.col("score_value") == 3).then(pl.lit("three"))
        .when((pl.col("type_text") == "JumpShot") & (pl.col("score_value") == 2))
        .then(pl.lit("mid"))
        .otherwise(None)
    )


def _shot_priors_from_pbp(season: str, league: League) -> pl.DataFrame:
    pbp = pl.read_parquet(raw_dir(league, season) / "pbp.parquet")
    shots = (
        pbp.filter(pl.col("shooting_play") == True)  # noqa: E712
        .with_columns(zone=_classify_shot_zone())
        .filter(pl.col("zone").is_not_null())
        .filter(pl.col("team_id").is_not_null())
        .with_columns(made=pl.col("scoring_play").cast(pl.Int8))
    )

    by_team_zone = shots.group_by(["team_id", "zone"]).agg(
        pl.len().alias("attempts"),
        pl.col("made").sum().alias("makes"),
    )
    pivoted = by_team_zone.pivot(
        on="zone", index="team_id", values=["attempts", "makes"], aggregate_function="first"
    )

    # Pivot column names are like "attempts_rim" / "makes_rim". Coerce nulls
    # to 0 so a team that never attempted a zone still gets a prior.
    for col in (
        "attempts_rim", "attempts_mid", "attempts_three",
        "makes_rim", "makes_mid", "makes_three",
    ):
        if col not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0).alias(col))
    pivoted = pivoted.with_columns([pl.col(c).fill_null(0) for c in (
        "attempts_rim", "attempts_mid", "attempts_three",
        "makes_rim", "makes_mid", "makes_three",
    )])

    total_att = pl.col("attempts_rim") + pl.col("attempts_mid") + pl.col("attempts_three")
    return pivoted.with_columns(
        mix_rim=pl.col("attempts_rim") / total_att,
        mix_mid=pl.col("attempts_mid") / total_att,
        mix_three=pl.col("attempts_three") / total_att,
        efg_rim=pl.when(pl.col("attempts_rim") > 0)
        .then(pl.col("makes_rim") / pl.col("attempts_rim"))
        .otherwise(0.0),
        efg_mid=pl.when(pl.col("attempts_mid") > 0)
        .then(pl.col("makes_mid") / pl.col("attempts_mid"))
        .otherwise(0.0),
        efg_three=pl.when(pl.col("attempts_three") > 0)
        .then(pl.col("makes_three") / pl.col("attempts_three"))
        .otherwise(0.0),
    ).select([
        "team_id", "mix_rim", "mix_mid", "mix_three",
        "efg_rim", "efg_mid", "efg_three",
    ])


def fit_team_priors(season: str, league: League = League.WBB) -> pl.DataFrame:
    """Combine team-season aggregates with pbp shot priors. Returns one row/team."""
    teams = pl.read_parquet(teams_path(league, season))
    shots = _shot_priors_from_pbp(season, league)

    # Foul rate: PF / opponent possessions, per 100. Opponent possessions
    # equal the team's defensive opportunities. We approximate with own poss.
    foul_rate = (pl.col("personal_fouls") / pl.col("poss") * 100).alias("foul_rate_per_100")

    return (
        teams.join(shots, on="team_id", how="inner")
        .with_columns(foul_rate)
        # The engine consumes the SoS-adjusted four factors and pace;
        # raw values stay in the team-season parquet for diagnostics
        # but are not surfaced through the priors interface.
        .select([
            "league", "season", "team_id", "team_name",
            pl.col("pace_adj").alias("pace"),
            pl.col("off_efg_adj").alias("off_efg"),
            pl.col("off_tov_pct_adj").alias("off_tov_pct"),
            pl.col("off_orb_pct_adj").alias("off_orb_pct"),
            pl.col("off_fta_rate_adj").alias("off_fta_rate"),
            "off_3pt_rate",
            "off_ft_pct",
            pl.col("def_efg_adj").alias("def_efg"),
            pl.col("def_tov_pct_adj").alias("def_tov_pct"),
            pl.col("def_orb_pct_adj").alias("def_orb_pct"),
            pl.col("def_fta_rate_adj").alias("def_fta_rate"),
            "mix_rim", "mix_mid", "mix_three",
            "efg_rim", "efg_mid", "efg_three",
            "foul_rate_per_100",
        ])
        .sort("team_id")
    )


def fit_league_prior(team_priors: pl.DataFrame) -> pl.DataFrame:
    return team_priors.select([
        pl.col("league").first(),
        pl.col("season").first(),
        pl.len().alias("n_teams"),
        pl.col("pace").mean(),
        pl.col("off_efg").mean(),
        pl.col("off_tov_pct").mean(),
        pl.col("off_orb_pct").mean(),
        pl.col("off_fta_rate").mean(),
        pl.col("off_3pt_rate").mean(),
        pl.col("off_ft_pct").mean(),
        pl.col("mix_rim").mean(),
        pl.col("mix_mid").mean(),
        pl.col("mix_three").mean(),
        pl.col("efg_rim").mean(),
        pl.col("efg_mid").mean(),
        pl.col("efg_three").mean(),
        pl.col("foul_rate_per_100").mean(),
    ])


def write_priors(season: str, league: League = League.WBB) -> dict[str, str]:
    out_dir = distributions_dir(league, season)
    out_dir.mkdir(parents=True, exist_ok=True)

    team_priors = fit_team_priors(season, league)
    team_path = out_dir / "team_priors.parquet"
    team_priors.write_parquet(team_path)

    league_prior = fit_league_prior(team_priors)
    league_path = out_dir / "league_prior.parquet"
    league_prior.write_parquet(league_path)

    return {"team_priors": str(team_path), "league_prior": str(league_path)}
