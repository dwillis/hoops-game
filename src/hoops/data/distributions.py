"""Loaders for fitted team-season priors.

These are the distributions the engine samples from at each state-machine
node. The fitter (scripts/fit_distributions.py) writes the parquet files;
this module is the read-side and defines the canonical Pydantic shape.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from hoops.data.paths import distributions_dir, teams_path
from hoops.league import League


class ShotMix(BaseModel):
    """Per-team shot zone shares; sums to 1.0."""

    model_config = ConfigDict(frozen=True)
    rim: float = Field(ge=0.0, le=1.0)
    mid: float = Field(ge=0.0, le=1.0)
    three: float = Field(ge=0.0, le=1.0)


class ZoneEFG(BaseModel):
    """Per-team make rate by zone (raw FG%, not eFG)."""

    model_config = ConfigDict(frozen=True)
    rim: float = Field(ge=0.0, le=1.0)
    mid: float = Field(ge=0.0, le=1.0)
    three: float = Field(ge=0.0, le=1.0)


class TeamPriors(BaseModel):
    """One team's fitted priors for a given season.

    The engine samples from these at each possession. The four-factor
    fields are season totals; ``shot_mix`` and ``zone_efg`` come from pbp.
    """

    model_config = ConfigDict(frozen=True)

    league: League
    season: str
    team_id: int
    team_name: str

    pace: float
    off_efg: float
    off_tov_pct: float
    off_orb_pct: float
    off_fta_rate: float
    off_3pt_rate: float
    off_ft_pct: float = 0.70  # league-ish fallback for teams with 0 FTA

    def_efg: float
    def_tov_pct: float
    def_orb_pct: float
    def_fta_rate: float

    shot_mix: ShotMix
    zone_efg: ZoneEFG

    foul_rate_per_100: float
    """Personal fouls committed per 100 opponent possessions."""


class LeaguePrior(BaseModel):
    """Mean of per-team priors. Used as a fallback / shrinkage target."""

    model_config = ConfigDict(frozen=True)

    league: League
    season: str
    n_teams: int
    pace: float
    off_efg: float
    off_tov_pct: float
    off_orb_pct: float
    off_fta_rate: float
    off_3pt_rate: float
    off_ft_pct: float
    shot_mix: ShotMix
    zone_efg: ZoneEFG
    foul_rate_per_100: float


# --- read API ----------------------------------------------------------------


def _team_priors_path(league: League, season: str) -> Path:
    return distributions_dir(league, season) / "team_priors.parquet"


def _league_prior_path(league: League, season: str) -> Path:
    return distributions_dir(league, season) / "league_prior.parquet"


def load_team_priors(league: League, season: str) -> list[TeamPriors]:
    df = pl.read_parquet(_team_priors_path(league, season))
    return [_row_to_team_priors(r) for r in df.iter_rows(named=True)]


def load_team_prior(league: League, season: str, team_id: int) -> TeamPriors:
    df = pl.read_parquet(_team_priors_path(league, season)).filter(
        pl.col("team_id") == team_id
    )
    if df.is_empty():
        raise KeyError(f"no priors for team_id={team_id} in {league.value} {season}")
    return _row_to_team_priors(next(df.iter_rows(named=True)))


def load_league_prior(league: League, season: str) -> LeaguePrior:
    df = pl.read_parquet(_league_prior_path(league, season))
    return _row_to_league_prior(next(df.iter_rows(named=True)))


# Minimum games-played to count a team as D-I for picker / season-replay
# purposes. The 2023-24 distribution is bimodal: ~360 teams play 25+ games
# (full D-I season), the rest play 1-2 (non-D-I opponents that appear in
# the dataset because a D-I team played them once). 20 is the cleanest
# cutoff between the two clusters.
DIVISION_ONE_MIN_GAMES = 20


def division_one_team_ids(league: League, season: str) -> set[int]:
    """Return the set of team_ids whose games-played puts them in D-I."""
    df = pl.read_parquet(teams_path(league, season))
    return set(
        df.filter(pl.col("games") >= DIVISION_ONE_MIN_GAMES)["team_id"].to_list()
    )


def _row_to_team_priors(r: dict) -> TeamPriors:
    return TeamPriors(
        league=r["league"],
        season=r["season"],
        team_id=r["team_id"],
        team_name=r["team_name"],
        pace=r["pace"],
        off_efg=r["off_efg"],
        off_tov_pct=r["off_tov_pct"],
        off_orb_pct=r["off_orb_pct"],
        off_fta_rate=r["off_fta_rate"],
        off_3pt_rate=r["off_3pt_rate"],
        off_ft_pct=r["off_ft_pct"] if r["off_ft_pct"] is not None else 0.70,
        def_efg=r["def_efg"],
        def_tov_pct=r["def_tov_pct"],
        def_orb_pct=r["def_orb_pct"],
        def_fta_rate=r["def_fta_rate"],
        shot_mix=ShotMix(rim=r["mix_rim"], mid=r["mix_mid"], three=r["mix_three"]),
        zone_efg=ZoneEFG(rim=r["efg_rim"], mid=r["efg_mid"], three=r["efg_three"]),
        foul_rate_per_100=r["foul_rate_per_100"],
    )


def _row_to_league_prior(r: dict) -> LeaguePrior:
    return LeaguePrior(
        league=r["league"],
        season=r["season"],
        n_teams=r["n_teams"],
        pace=r["pace"],
        off_efg=r["off_efg"],
        off_tov_pct=r["off_tov_pct"],
        off_orb_pct=r["off_orb_pct"],
        off_fta_rate=r["off_fta_rate"],
        off_3pt_rate=r["off_3pt_rate"],
        off_ft_pct=r["off_ft_pct"],
        shot_mix=ShotMix(rim=r["mix_rim"], mid=r["mix_mid"], three=r["mix_three"]),
        zone_efg=ZoneEFG(rim=r["efg_rim"], mid=r["efg_mid"], three=r["efg_three"]),
        foul_rate_per_100=r["foul_rate_per_100"],
    )
