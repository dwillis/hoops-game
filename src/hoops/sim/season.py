"""Season replay: simulate every game on a real schedule N times.

This is the substrate the §5 validation harness runs on. Phase 4's W-L
and four-factor checks call into here; Phase 8 batch sims will too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hoops.data.distributions import LeaguePrior, TeamPriors, load_league_prior, load_team_priors
from hoops.data.paths import games_path
from hoops.engine.machine import simulate_game
from hoops.engine.sampling import make_rng
from hoops.engine.state import Side
from hoops.league import League
from hoops.rules import Rules, rules_for


@dataclass
class GameResult:
    game_id: int
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int

    @property
    def home_won(self) -> bool:
        return self.home_score > self.away_score


def _load_schedule(season: str, league: League = League.WBB) -> pl.DataFrame:
    """Canonical games for the season. Falls back to projecting raw if needed."""
    p = games_path(league, season)
    if not p.exists():
        from hoops.data.projections import write_canonical
        write_canonical(season, league)
    return pl.read_parquet(p)


def _priors_index(season: str, league: League) -> dict[int, TeamPriors]:
    return {p.team_id: p for p in load_team_priors(league, season)}


def _eligible_games(
    games: pl.DataFrame, priors: dict[int, TeamPriors]
) -> pl.DataFrame:
    """Drop games where either team has no fitted prior (D-II opponents, etc.)."""
    have = list(priors.keys())
    return games.filter(
        pl.col("home_team_id").is_in(have) & pl.col("away_team_id").is_in(have)
    )


def simulate_one_game(
    home_priors: TeamPriors,
    away_priors: TeamPriors,
    rules: Rules,
    league_prior: LeaguePrior,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Run a single game; return (home_score, away_score)."""
    state, _ = simulate_game(
        home_priors, away_priors, rules, rng,
        opening_possession=Side.HOME, league=league_prior,
    )
    return state.home_score, state.away_score


def simulate_season(
    season: str,
    n_runs: int,
    league: League = League.WBB,
    base_seed: int = 0,
) -> pl.DataFrame:
    """Simulate every eligible game on the schedule ``n_runs`` times.

    Returns a long-format frame with one row per (team_id, run_index) and
    that team's simulated win count under that run. Aggregating downstream
    is the caller's job.
    """
    rules = rules_for(league, season)
    league_prior = load_league_prior(league, season)
    priors = _priors_index(season, league)
    games = _eligible_games(_load_schedule(season, league), priors)

    home_ids = games["home_team_id"].to_list()
    away_ids = games["away_team_id"].to_list()
    n_games = len(home_ids)

    # Pre-resolve to TeamPriors per game.
    home_p = [priors[t] for t in home_ids]
    away_p = [priors[t] for t in away_ids]

    rng_master = make_rng(base_seed)
    seeds = rng_master.integers(0, 2**31 - 1, size=(n_runs, n_games))

    # Aggregate wins per team per run. Pre-allocate dict of {team_id: ndarray}
    team_ids = sorted(priors.keys())
    win_counts = {tid: np.zeros(n_runs, dtype=np.int32) for tid in team_ids}

    for run in range(n_runs):
        for g in range(n_games):
            game_rng = make_rng(int(seeds[run, g]))
            hs, as_ = simulate_one_game(home_p[g], away_p[g], rules, league_prior, game_rng)
            winner = home_ids[g] if hs > as_ else away_ids[g]
            win_counts[winner][run] += 1

    rows = []
    for tid in team_ids:
        for run in range(n_runs):
            rows.append({"team_id": tid, "run": run, "wins": int(win_counts[tid][run])})
    return pl.DataFrame(rows)


def actual_wins(season: str, league: League = League.WBB) -> pl.DataFrame:
    """Pull per-team actual W-L from the canonical schedule."""
    games = _load_schedule(season, league)
    home = games.select([
        pl.col("home_team_id").alias("team_id"),
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int32).alias("win"),
    ])
    away = games.select([
        pl.col("away_team_id").alias("team_id"),
        (pl.col("away_score") > pl.col("home_score")).cast(pl.Int32).alias("win"),
    ])
    return (
        pl.concat([home, away])
        .group_by("team_id")
        .agg(pl.col("win").sum().alias("actual_wins"), pl.len().alias("actual_games"))
        .sort("team_id")
    )


def simulate_team_schedule(
    team_id: int,
    season: str,
    n_runs: int,
    league: League = League.WBB,
    base_seed: int = 0,
) -> np.ndarray:
    """Sim a single team's actual schedule ``n_runs`` times. Returns wins per run.

    Used for the §5.2 top-team-specificity check (e.g. SC's actual 38 games).
    """
    rules = rules_for(league, season)
    league_prior = load_league_prior(league, season)
    priors = _priors_index(season, league)
    if team_id not in priors:
        raise KeyError(f"team {team_id} not in {league.value} {season} priors")

    games = _load_schedule(season, league)
    schedule = games.filter(
        (pl.col("home_team_id") == team_id) | (pl.col("away_team_id") == team_id)
    ).filter(
        pl.col("home_team_id").is_in(list(priors.keys()))
        & pl.col("away_team_id").is_in(list(priors.keys()))
    )

    home_ids = schedule["home_team_id"].to_list()
    away_ids = schedule["away_team_id"].to_list()
    n_games = len(home_ids)

    rng_master = make_rng(base_seed)
    seeds = rng_master.integers(0, 2**31 - 1, size=(n_runs, n_games))

    wins = np.zeros(n_runs, dtype=np.int32)
    for run in range(n_runs):
        for g in range(n_games):
            game_rng = make_rng(int(seeds[run, g]))
            hp = priors[home_ids[g]]
            ap = priors[away_ids[g]]
            hs, as_ = simulate_one_game(hp, ap, rules, league_prior, game_rng)
            if (home_ids[g] == team_id and hs > as_) or (away_ids[g] == team_id and as_ > hs):
                wins[run] += 1
    return wins
