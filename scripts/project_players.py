"""Project per-player advanced stats for one or all WBB seasons.

Reads:
    data/raw/wbb/<season>/player_box.parquet
    data/teams/wbb/<season>.parquet

Writes:
    data/players/wbb/<season>.parquet

Usage:
    uv run python scripts/project_players.py --season 2023-24
    uv run python scripts/project_players.py --all-seasons
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hoops.data.player_projections import write_player_projections  # noqa: E402
from hoops.league import League  # noqa: E402
from hoops.rules import available_seasons  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--season", help='Season string, e.g. "2023-24"')
    g.add_argument(
        "--all-seasons",
        action="store_true",
        help="Project every season listed in data/rules/wbb.yaml",
    )
    args = p.parse_args()

    seasons = available_seasons(League.WBB) if args.all_seasons else [args.season]
    for s in seasons:
        try:
            print(f"[{s}] projecting player stats")
            path = write_player_projections(s)
            print(f"[{s}] -> {path}")
        except Exception as e:
            print(f"[{s}] SKIPPED: {e}", file=sys.stderr)
            if not args.all_seasons:
                raise


if __name__ == "__main__":
    main()
