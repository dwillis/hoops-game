"""Per-team rosters for narrating play-by-play.

The engine simulates from team-level rates; per-player rate fitting is
deferred. This module provides just enough player data to *attribute*
events naturally in the play-by-play (and eventually in the box score),
without driving the simulation by player.

For each (team, season) we build a roster of the top-N players by total
minutes, with their season totals for the stats we use as sampling
weights:

- ``fga``: who's shooting
- ``fta``: who's at the free-throw line
- ``orb`` / ``drb``: who's rebounding
- ``tov``: who's turning the ball over
- ``fouls``: who's committing fouls
- ``ast``: who's assisting (currently unused; kept for future)

Sampling falls back to uniform-over-roster if the relevant weight is zero
(early-season backups, walk-ons, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hoops.data.paths import players_path, raw_dir, teams_path
from hoops.league import League


@dataclass(frozen=True)
class Player:
    player_id: int
    name: str
    minutes: float
    fga: int
    fg3a: int
    fta: int
    orb: int
    drb: int
    fouls: int
    tov: int
    ast: int
    blk: int = 0
    stl: int = 0
    games_played: int = 0
    points: int = 0
    fgm: int = 0
    fg3m: int = 0
    ftm: int = 0
    position: str = ""
    usage_pct: float | None = None
    ts_pct: float | None = None
    fg3a_share: float | None = None
    ft_pct: float | None = None
    tov_pct: float | None = None
    orb_pct: float | None = None
    drb_pct: float | None = None
    stl_pct: float | None = None
    blk_pct: float | None = None
    foul_rate: float | None = None
    min_share: float | None = None
    ast_pct: float | None = None


@dataclass
class Roster:
    team_id: int
    team_name: str
    players: tuple[Player, ...]

    def _weighted_sample(
        self, weights: np.ndarray, rng: np.random.Generator
    ) -> Player:
        if weights.sum() <= 0 or len(self.players) == 0:
            return self.players[int(rng.integers(len(self.players)))]
        p = weights / weights.sum()
        idx = int(rng.choice(len(self.players), p=p))
        return self.players[idx]

    def shooter(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.fga for p in self.players], dtype=float), rng
        )

    def three_point_shooter(self, rng: np.random.Generator) -> Player:
        # Some rosters concentrate 3pt attempts on specific shooters; if
        # that's the case we honor it. Falls back to overall fga.
        weights = np.array([p.fg3a for p in self.players], dtype=float)
        if weights.sum() == 0:
            return self.shooter(rng)
        return self._weighted_sample(weights, rng)

    def ft_shooter(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.fta for p in self.players], dtype=float), rng
        )

    def rebounder_off(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.orb for p in self.players], dtype=float), rng
        )

    def rebounder_def(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.drb for p in self.players], dtype=float), rng
        )

    def turnover(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.tov for p in self.players], dtype=float), rng
        )

    def fouler(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.fouls for p in self.players], dtype=float), rng
        )

    def blocker(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.blk for p in self.players], dtype=float), rng
        )

    def stealer(self, rng: np.random.Generator) -> Player:
        return self._weighted_sample(
            np.array([p.stl for p in self.players], dtype=float), rng
        )

    def assister(
        self, rng: np.random.Generator, exclude: str | None = None
    ) -> Player:
        """Pick an assist credit. Excludes the shooter from the pool when
        provided so a player doesn't assist their own basket."""
        candidates = [p for p in self.players if p.name != exclude]
        if not candidates:
            candidates = list(self.players)
        weights = np.array([p.ast for p in candidates], dtype=float)
        if weights.sum() <= 0:
            return candidates[int(rng.integers(len(candidates)))]
        weights = weights / weights.sum()
        return candidates[int(rng.choice(len(candidates), p=weights))]


def load_roster(
    team_id: int,
    season: str,
    league: League = League.WBB,
    top_n: int = 12,
) -> Roster:
    """Build a Roster for ``(team_id, season)``.

    Prefers projected player data when available; falls back to raw
    player_box aggregation.
    """
    projected = players_path(league, season)
    if projected.exists():
        return _roster_from_projected(projected, league, season, team_id, top_n)
    return _roster_from_raw(league, season, team_id, top_n)


def _team_name_from_teams(league: League, season: str, team_id: int) -> str:
    tp = teams_path(league, season)
    if not tp.exists():
        return f"Team {team_id}"
    teams_df = pl.read_parquet(tp)
    row = teams_df.filter(pl.col("team_id") == team_id)
    return row[0, "team_name"] if row.height else f"Team {team_id}"


def _roster_from_projected(
    path: str, league: League, season: str, team_id: int, top_n: int
) -> Roster:
    df = pl.read_parquet(path)
    rows = (
        df.filter(pl.col("team_id") == team_id)
        .sort("minutes", descending=True)
        .head(top_n)
    )
    team_name = _team_name_from_teams(league, season, team_id)
    players = tuple(
        Player(
            player_id=int(r["player_id"]),
            name=r["player_name"] or f"Player {r['player_id']}",
            minutes=float(r["minutes"] or 0),
            fga=int(r["fga"] or 0),
            fg3a=int(r["fg3a"] or 0),
            fta=int(r["fta"] or 0),
            orb=int(r["orb"] or 0),
            drb=int(r["drb"] or 0),
            fouls=int(r["fouls"] or 0),
            tov=int(r["tov"] or 0),
            ast=int(r["ast"] or 0),
            blk=int(r["blk"] or 0),
            stl=int(r["stl"] or 0),
            games_played=int(r["games_played"]) if r.get("games_played") is not None else 0,
            points=int(r["points"]) if r.get("points") is not None else 0,
            fgm=int(r["fgm"]) if r.get("fgm") is not None else 0,
            fg3m=int(r["fg3m"]) if r.get("fg3m") is not None else 0,
            ftm=int(r["ftm"]) if r.get("ftm") is not None else 0,
            position=r.get("position") or "",
            usage_pct=float(r["usage_pct"]) if r.get("usage_pct") is not None else None,
            ts_pct=float(r["ts_pct"]) if r.get("ts_pct") is not None else None,
            fg3a_share=float(r["fg3a_share"]) if r.get("fg3a_share") is not None else None,
            ft_pct=float(r["ft_pct"]) if r.get("ft_pct") is not None else None,
            tov_pct=float(r["tov_pct"]) if r.get("tov_pct") is not None else None,
            orb_pct=float(r["orb_pct"]) if r.get("orb_pct") is not None else None,
            drb_pct=float(r["drb_pct"]) if r.get("drb_pct") is not None else None,
            stl_pct=float(r["stl_pct"]) if r.get("stl_pct") is not None else None,
            blk_pct=float(r["blk_pct"]) if r.get("blk_pct") is not None else None,
            foul_rate=float(r["foul_rate"]) if r.get("foul_rate") is not None else None,
            min_share=float(r["min_share"]) if r.get("min_share") is not None else None,
            ast_pct=float(r["ast_pct"]) if r.get("ast_pct") is not None else None,
        )
        for r in rows.iter_rows(named=True)
    )
    return Roster(team_id=team_id, team_name=team_name, players=players)


def _roster_from_raw(
    league: League, season: str, team_id: int, top_n: int
) -> Roster:
    pb = pl.read_parquet(raw_dir(league, season) / "player_box.parquet")
    rows = (
        pb.filter(pl.col("team_id") == team_id)
        .group_by(["athlete_id", "athlete_display_name"])
        .agg(
            pl.col("minutes").sum().alias("minutes"),
            pl.col("field_goals_attempted").sum().alias("fga"),
            pl.col("field_goals_made").sum().alias("fgm"),
            pl.col("three_point_field_goals_attempted").sum().alias("fg3a"),
            pl.col("three_point_field_goals_made").sum().alias("fg3m"),
            pl.col("free_throws_attempted").sum().alias("fta"),
            pl.col("free_throws_made").sum().alias("ftm"),
            pl.col("offensive_rebounds").sum().alias("orb"),
            pl.col("defensive_rebounds").sum().alias("drb"),
            pl.col("fouls").sum().alias("fouls"),
            pl.col("turnovers").sum().alias("tov"),
            pl.col("assists").sum().alias("ast"),
            pl.col("blocks").sum().alias("blk"),
            pl.col("steals").sum().alias("stl"),
            pl.col("points").sum().alias("points"),
            pl.col("game_id").n_unique().alias("games_played"),
            pl.col("athlete_position_abbreviation").first().alias("position"),
        )
        .sort("minutes", descending=True)
        .head(top_n)
    )
    team_name_row = (
        pb.filter(pl.col("team_id") == team_id)
        .select("team_short_display_name")
        .head(1)
    )
    team_name = (
        team_name_row.item(0, 0) if team_name_row.height else f"Team {team_id}"
    )
    players = tuple(
        Player(
            player_id=int(r["athlete_id"]) if r["athlete_id"] is not None else 0,
            name=r["athlete_display_name"] or f"Player {r['athlete_id']}",
            minutes=float(r["minutes"] or 0),
            fga=int(r["fga"] or 0),
            fg3a=int(r["fg3a"] or 0),
            fta=int(r["fta"] or 0),
            orb=int(r["orb"] or 0),
            drb=int(r["drb"] or 0),
            fouls=int(r["fouls"] or 0),
            tov=int(r["tov"] or 0),
            ast=int(r["ast"] or 0),
            blk=int(r["blk"] or 0),
            stl=int(r["stl"] or 0),
            games_played=int(r["games_played"] or 0),
            points=int(r["points"] or 0),
            fgm=int(r["fgm"] or 0),
            fg3m=int(r["fg3m"] or 0),
            ftm=int(r["ftm"] or 0),
            position=r["position"] or "",
        )
        for r in rows.iter_rows(named=True)
    )
    return Roster(team_id=team_id, team_name=team_name, players=players)
