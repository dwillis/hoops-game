"""Run the Phase 4 / §5 validation harness on demand.

Wraps the same season-replay infrastructure the pytest validation suite
uses, but with progress reporting and tunable N. Useful when iterating
on the engine and you want quick numbers without writing a new test.

Usage:
    uv run python scripts/validate_engine.py sc-specificity --runs 1000
    uv run python scripts/validate_engine.py season-wl --runs 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from hoops.data.distributions import load_team_priors  # noqa: E402
from hoops.league import League  # noqa: E402
from hoops.sim.season import (  # noqa: E402
    actual_wins,
    simulate_season,
    simulate_team_schedule,
)


def cmd_sc_specificity(args: argparse.Namespace) -> int:
    priors = load_team_priors(League.WBB, args.season)
    sc = next((p for p in priors if p.team_name == "South Carolina"), None)
    if sc is None:
        print("South Carolina not in priors", file=sys.stderr)
        return 1
    actual = actual_wins(args.season).filter(pl.col("team_id") == sc.team_id).item(0, "actual_wins")
    sched = actual_wins(args.season).filter(pl.col("team_id") == sc.team_id).item(0, "actual_games")

    t0 = time.time()
    wins = simulate_team_schedule(sc.team_id, args.season, n_runs=args.runs, base_seed=args.seed)
    elapsed = time.time() - t0

    print(f"South Carolina specificity ({args.runs} runs in {elapsed:.1f}s)")
    print(f"  actual: {actual}-{sched - actual}")
    print(f"  mean:   {wins.mean():.2f}")
    print(f"  median: {np.median(wins):.1f}")
    print(f"  range:  [{wins.min()}, {wins.max()}]")
    print(f"  stdev:  {wins.std():.2f}")
    print(f"  P(undefeated): {(wins == sched).mean():.3%}")
    print(f"  P(>= 36 wins): {(wins >= 36).mean():.3%}")
    return 0


def cmd_season_wl(args: argparse.Namespace) -> int:
    t0 = time.time()
    df = simulate_season(args.season, n_runs=args.runs, base_seed=args.seed)
    elapsed = time.time() - t0

    aggs = df.group_by("team_id").agg(
        pl.col("wins").mean().alias("mean_wins"),
        pl.col("wins").std().alias("std_wins"),
    )
    actual = actual_wins(args.season)
    joined = (
        aggs.join(actual, on="team_id")
        .with_columns(diff=pl.col("mean_wins") - pl.col("actual_wins"))
        .filter(pl.col("actual_games") >= 25)
        .sort("actual_wins", descending=True)
    )

    top = joined.head(20)
    print(f"Season W-L replay ({args.runs} runs in {elapsed:.1f}s)")
    print(f"  teams scored:    {joined.height}")
    print(f"  mean |diff|:     {joined['diff'].abs().mean():.2f}")
    print(f"  max |diff|:      {joined['diff'].abs().max():.2f}")
    print(f"  RMSE:            {(joined['diff'] ** 2).mean() ** 0.5:.2f}")
    print()
    print("Top-20 teams (by actual wins):")
    # Add team name for readability
    priors = {p.team_id: p.team_name for p in load_team_priors(League.WBB, args.season)}
    name_col = pl.Series("team", [priors.get(t, "?") for t in top["team_id"].to_list()])
    print(top.with_columns(name_col).select([
        "team", "actual_wins", "actual_games", "mean_wins", "std_wins", "diff"
    ]).to_pandas().to_string(index=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2023-24")
    p.add_argument("--seed", type=int, default=2024)
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("sc-specificity", help="§5.2: South Carolina N-run specificity")
    sc.add_argument("--runs", type=int, default=1000)
    sc.set_defaults(func=cmd_sc_specificity)

    sw = sub.add_parser("season-wl", help="§5.1: full season W-L replay")
    sw.add_argument("--runs", type=int, default=20)
    sw.set_defaults(func=cmd_season_wl)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
