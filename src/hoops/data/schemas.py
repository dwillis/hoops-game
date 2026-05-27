"""Canonical schemas for ingested data, persisted as Parquet.

These are the v1 columns. Adding a column is fine; renaming or dropping one
is a breaking change for downstream phases (distribution fitting, engine).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hoops.league import League

LineupQuality = Literal["good", "partial", "box_only"]
"""Per-game pbp quality flag.

- ``good``: shot coordinates and event ordering look fine.
- ``partial``: some shot coords missing or events out of order; pbp usable
  with caveats.
- ``box_only``: pbp unusable; only box-score-derived priors should be fit
  from this game.
"""


class TeamSeason(BaseModel):
    model_config = ConfigDict(frozen=True)

    league: League
    season: str
    team_id: int
    team_slug: str
    team_name: str
    conference: str | None = None
    wins: int
    losses: int
    pace: float | None = None  # poss / 40
    off_efg: float | None = None
    def_efg: float | None = None
    off_tov_pct: float | None = None
    def_tov_pct: float | None = None
    off_orb_pct: float | None = None
    def_orb_pct: float | None = None
    off_fta_rate: float | None = None
    def_fta_rate: float | None = None


class PlayerSeason(BaseModel):
    model_config = ConfigDict(frozen=True)

    league: League
    season: str
    team_id: int
    player_id: int
    player_name: str
    position: str | None = None
    height_in: int | None = None

    # Raw aggregates (season totals)
    games_played: int = 0
    games_started: int = 0
    minutes: float = 0.0
    fgm: int = 0
    fga: int = 0
    fg3m: int = 0
    fg3a: int = 0
    ftm: int = 0
    fta: int = 0
    orb: int = 0
    drb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    fouls: int = 0
    points: int = 0

    # Advanced rates
    min_share: float | None = None
    usage_pct: float | None = None
    ts_pct: float | None = None
    fg3a_share: float | None = None
    ft_pct: float | None = None
    ast_pct: float | None = None
    tov_pct: float | None = None
    orb_pct: float | None = None
    drb_pct: float | None = None
    stl_pct: float | None = None
    blk_pct: float | None = None
    foul_rate: float | None = None

    # Variance
    ppg_mean: float | None = None
    ppg_std: float | None = None


class Game(BaseModel):
    model_config = ConfigDict(frozen=True)

    league: League
    season: str
    game_id: int
    game_date: date
    home_team_id: int
    away_team_id: int
    home_score: int | None = None
    away_score: int | None = None
    neutral_site: bool = False
    postseason: bool = False
    lineup_quality: LineupQuality = "good"


class Event(BaseModel):
    """One row in the canonical pbp table.

    Sourced from sportsdataverse `load_wbb_pbp`. We keep the fields the
    distribution-fitter actually needs and discard the rest at ingest time.
    """

    model_config = ConfigDict(frozen=True)

    game_id: int
    sequence: int  # event order within the game
    quarter: int  # 1..4 regulation, 5+ for OT
    clock_seconds: int  # seconds remaining in the quarter
    event_type: str  # "made_2", "miss_3", "drb", "orb", "foul", "tov", "sub", ...
    team_id: int | None = None
    player_id: int | None = None
    secondary_player_id: int | None = None  # assist / blocker / fouled-by
    shot_x: float | None = None
    shot_y: float | None = None
    score_home: int = 0
    score_away: int = 0
    raw_text: str | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
