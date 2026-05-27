"""Centralized data paths so callers don't hardcode them."""

from __future__ import annotations

import sys
from pathlib import Path

from hoops.league import League


def _find_data_root() -> Path:
    if getattr(sys, "_MEIPASS", None):
        return Path(sys._MEIPASS) / "data"
    return Path(__file__).resolve().parents[3] / "data"


DATA_ROOT = _find_data_root()


def raw_dir(league: League, season: str) -> Path:
    return DATA_ROOT / "raw" / league.value / season


def teams_path(league: League, season: str) -> Path:
    return DATA_ROOT / "teams" / league.value / f"{season}.parquet"


def players_path(league: League, season: str) -> Path:
    return DATA_ROOT / "players" / league.value / f"{season}.parquet"


def games_path(league: League, season: str) -> Path:
    return DATA_ROOT / "games" / league.value / f"{season}.parquet"


def distributions_dir(league: League, season: str) -> Path:
    return DATA_ROOT / "pbp_distributions" / league.value / season


def fitted_seasons(league: League) -> list[str]:
    """Return sorted list of seasons that have fitted priors on disk."""
    dist_root = DATA_ROOT / "pbp_distributions" / league.value
    if not dist_root.exists():
        return []
    return sorted(
        child.name
        for child in dist_root.iterdir()
        if child.is_dir() and (child / "team_priors.parquet").exists()
    )


def bracket_path(league: League, season: str) -> Path:
    return DATA_ROOT / "brackets" / league.value / f"{season}.json"


def bracket_seasons(league: League) -> list[str]:
    """Return sorted list of seasons with extracted bracket data."""
    bracket_root = DATA_ROOT / "brackets" / league.value
    if not bracket_root.exists():
        return []
    return sorted(
        p.stem
        for p in bracket_root.iterdir()
        if p.suffix == ".json"
    )


def conf_tournament_dir(league: League, season: str) -> Path:
    return DATA_ROOT / "conf_tournaments" / league.value / season


def conf_tournament_path(league: League, season: str, tournament_id: int) -> Path:
    return DATA_ROOT / "conf_tournaments" / league.value / season / f"{tournament_id}.json"


def list_conf_tournaments(league: League, season: str) -> list[dict]:
    """Return list of conference tournament metadata dicts for a season.

    Each dict has: tournament_id, conference_name, num_teams, num_games.
    Returns empty list if no data available.
    """
    index_path = conf_tournament_dir(league, season) / "index.json"
    if not index_path.exists():
        return []
    import json
    with open(index_path) as f:
        data = json.load(f)
    return data.get("conferences", [])
