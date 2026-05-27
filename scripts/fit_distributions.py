"""Fit per-team and league-average priors for a WBB season.

Reads:
    data/teams/wbb/<season>.parquet      (canonical team-season aggregates)
    data/raw/wbb/<season>/pbp.parquet    (raw play-by-play)

Writes:
    data/pbp_distributions/wbb/<season>/team_priors.parquet
    data/pbp_distributions/wbb/<season>/league_prior.parquet

If the canonical team-season Parquet does not exist, the projection step
runs first.

Usage:
    uv run python scripts/fit_distributions.py --season 2023-24
    uv run python scripts/fit_distributions.py --all-seasons
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hoops.data.fit import write_priors  # noqa: E402
from hoops.data.paths import teams_path  # noqa: E402
from hoops.data.projections import write_canonical  # noqa: E402
from hoops.league import League  # noqa: E402
from hoops.rules import available_seasons  # noqa: E402


def fit_one(season: str) -> None:
    if not teams_path(League.WBB, season).exists():
        print(f"[{season}] no canonical team-season parquet; running projection")
        write_canonical(season)
    print(f"[{season}] fitting team + league priors")
    paths = write_priors(season)
    for k, p in paths.items():
        print(f"[{season}] {k} -> {p}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--season")
    g.add_argument("--all-seasons", action="store_true")
    args = p.parse_args()
    seasons = available_seasons(League.WBB) if args.all_seasons else [args.season]
    for s in seasons:
        try:
            fit_one(s)
        except Exception as e:
            print(f"[{s}] SKIPPED: {e}", file=sys.stderr)
            if not args.all_seasons:
                raise


if __name__ == "__main__":
    main()
