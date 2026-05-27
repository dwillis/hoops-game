"""Ingest one season of WBB data from the sportsdataverse-data release feed.

Phase 1 deliverable: pull raw frames (schedule, team box, player box, pbp)
and write them to ``data/raw/wbb/<season>/`` as Parquet. A separate
canonical-projection step will read these and emit schema-typed Parquet
under ``data/teams/wbb/`` etc.

We hit the sportsdataverse-data GitHub Releases directly rather than
importing sportsdataverse-py, because that package's init transitively
loads xgboost (which needs libomp on macOS) and pkg_resources (removed
in setuptools 81). Reading the same Parquet artifacts with polars is
simpler and has no system dependencies.

Usage:
    uv run python scripts/ingest_wehoop.py --season 2023-24
    uv run python scripts/ingest_wehoop.py --all-seasons
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hoops.data.seasons import season_end_year  # noqa: E402
from hoops.league import League  # noqa: E402
from hoops.rules import available_seasons  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "wbb"
SDV_RELEASES = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"

URLS = {
    "schedule": SDV_RELEASES
    + "espn_womens_college_basketball_schedules/wbb_schedule_{year}.parquet",
    "team_box": SDV_RELEASES
    + "espn_womens_college_basketball_team_boxscores/team_box_{year}.parquet",
    "player_box": SDV_RELEASES
    + "espn_womens_college_basketball_player_boxscores/player_box_{year}.parquet",
    "pbp": SDV_RELEASES
    + "espn_womens_college_basketball_pbp/play_by_play_{year}.parquet",
}


def ingest_season(season: str, kinds: Iterable[str] = URLS.keys()) -> dict[str, Path]:
    """Pull the requested frames for ``season`` and write Parquet."""
    end_year = season_end_year(season)
    out_dir = RAW_DIR / season
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for kind in kinds:
        url = URLS[kind].format(year=end_year)
        out_path = out_dir / f"{kind}.parquet"
        print(f"[{season}] {kind} <- {url}")
        df = pl.read_parquet(url)
        df.write_parquet(out_path)
        print(f"[{season}] {kind} -> {out_path} ({df.height} rows)")
        written[kind] = out_path
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", help='Season string, e.g. "2023-24"')
    g.add_argument(
        "--all-seasons",
        action="store_true",
        help="Ingest every season listed in data/rules/wbb.yaml",
    )
    p.add_argument(
        "--kinds",
        nargs="+",
        choices=list(URLS),
        default=list(URLS),
        help="Subset of frames to pull (default: all four)",
    )
    args = p.parse_args()

    seasons = available_seasons(League.WBB) if args.all_seasons else [args.season]
    for s in seasons:
        try:
            ingest_season(s, kinds=args.kinds)
        except Exception as e:
            print(f"[{s}] SKIPPED: {e}", file=sys.stderr)
            if not args.all_seasons:
                raise


if __name__ == "__main__":
    main()
