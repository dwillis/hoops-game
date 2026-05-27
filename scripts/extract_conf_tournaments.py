"""Extract conference tournament brackets from raw schedule data.

Reads data/raw/wbb/<season>/schedule.parquet, identifies conference tournaments,
parses round structure from notes_headline, checks data sufficiency against
fitted priors, and writes per-conference JSON to data/conf_tournaments/wbb/<season>/.

Usage:
    uv run python scripts/extract_conf_tournaments.py                  # all seasons
    uv run python scripts/extract_conf_tournaments.py --season 2023-24 # one season
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Tournament IDs to exclude (not conference tournaments)
# 22=NCAA, 37=NIT, 38=WBI, 62=WBIT
EXCLUDE_IDS = {22, 37, 38, 62}

ROUND_NAME_ORDER = [
    "play-in",
    "opening round",
    "1st round",
    "2nd round",
    "3rd round",
    "4th round",
    "quarterfinal",
    "semifinal",
    "final",
]


def _parse_conf_headline(headline: str) -> tuple[str | None, str | None]:
    """Parse conference name and round from headline.

    Returns (conference_name, round_name_lowercase).
    Example: "SEC Tournament - Quarterfinal" -> ("SEC Tournament", "quarterfinal")
    """
    if not headline or " - " not in headline:
        return None, None
    parts = headline.rsplit(" - ", 1)
    conf_name = parts[0].strip()
    round_name = parts[1].strip().lower()
    return conf_name, round_name


def _fitted_team_ids(season: str) -> set[int]:
    """Return set of team_ids that have fitted priors."""
    priors_path = DATA_ROOT / "pbp_distributions" / "wbb" / season / "team_priors.parquet"
    if not priors_path.exists():
        return set()
    df = pl.read_parquet(priors_path)
    return set(df["team_id"].to_list())


def _extract_season(season: str) -> int:
    """Extract all conference tournaments for one season. Returns count extracted."""
    raw_path = DATA_ROOT / "raw" / "wbb" / season / "schedule.parquet"
    if not raw_path.exists():
        print(f"[{season}] SKIPPED: no raw schedule data", file=sys.stderr)
        return 0

    df = pl.read_parquet(raw_path)
    fitted_ids = _fitted_team_ids(season)
    if not fitted_ids:
        print(f"[{season}] SKIPPED: no fitted priors", file=sys.stderr)
        return 0

    # Conference tournaments happen in March (some start late Feb).
    # Season "2023-24" -> March 2024.
    end_year = int(season.split("-")[0]) + 1
    march_start = date(end_year, 2, 20)
    march_end = date(end_year, 3, 20)

    conf_games = df.filter(
        pl.col("tournament_id").is_not_null()
        & ~pl.col("tournament_id").is_in(list(EXCLUDE_IDS))
        & pl.col("notes_headline").is_not_null()
        & (pl.col("game_date") >= march_start)
        & (pl.col("game_date") <= march_end)
    )

    if conf_games.height == 0:
        print(f"[{season}] SKIPPED: no conference tournament games found", file=sys.stderr)
        return 0

    # Group by tournament_id
    tournament_ids = conf_games["tournament_id"].unique().to_list()

    out_dir = DATA_ROOT / "conf_tournaments" / "wbb" / season
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    extracted = 0

    for tid in sorted(tournament_ids):
        t_games = conf_games.filter(pl.col("tournament_id") == tid).sort("game_date", "game_id")

        if t_games.height == 0:
            continue

        # Get conference name from first headline
        first_hl = t_games[0, "notes_headline"]
        conf_name, _ = _parse_conf_headline(first_hl)
        if conf_name is None:
            continue

        # Collect all team IDs and build games list
        team_ids_in_tourney: set[int] = set()
        games_out: list[dict] = []

        # Determine round numbering from chronological game dates.
        # Group by date to find distinct rounds, assign 1, 2, 3...
        round_names_seen: list[str] = []
        for row in t_games.iter_rows(named=True):
            _, rn = _parse_conf_headline(row["notes_headline"])
            if rn is not None and rn not in round_names_seen:
                round_names_seen.append(rn)

        # Assign round numbers in chronological order (games are already sorted by date)
        round_num_map = {name: idx + 1 for idx, name in enumerate(round_names_seen)}

        for row in t_games.iter_rows(named=True):
            _, round_name = _parse_conf_headline(row["notes_headline"])
            if round_name is None:
                continue
            round_num = round_num_map.get(round_name)
            if round_num is None:
                continue

            home_id = int(row["home_id"])
            away_id = int(row["away_id"])
            team_ids_in_tourney.add(home_id)
            team_ids_in_tourney.add(away_id)

            games_out.append({
                "game_id": int(row["game_id"]),
                "round": round_num,
                "region": None,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_seed": None,
                "away_seed": None,
                "home_score": None,
                "away_score": None,
            })

        if not games_out:
            continue

        # Check data sufficiency: all teams must have fitted priors
        missing = team_ids_in_tourney - fitted_ids
        if missing:
            print(
                f"[{season}] SKIPPED {conf_name}: {len(missing)} teams without priors",
                file=sys.stderr,
            )
            continue

        bracket = {
            "season": season,
            "tournament_id": tid,
            "conference_name": conf_name,
            "num_teams": len(team_ids_in_tourney),
            "regions": [],
            "games": games_out,
        }

        out_path = out_dir / f"{tid}.json"
        with open(out_path, "w") as f:
            json.dump(bracket, f, indent=2)

        index.append({
            "tournament_id": tid,
            "conference_name": conf_name,
            "num_teams": len(team_ids_in_tourney),
            "num_games": len(games_out),
        })
        extracted += 1
        print(f"[{season}] {conf_name}: {len(games_out)} games, {len(team_ids_in_tourney)} teams")

    # Write index
    if index:
        with open(out_dir / "index.json", "w") as f:
            json.dump({"season": season, "conferences": index}, f, indent=2)

    print(f"[{season}] Extracted {extracted} conference tournaments")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract conference tournament brackets")
    parser.add_argument("--season", help="Single season to extract (e.g. 2023-24)")
    args = parser.parse_args()

    if args.season:
        seasons = [args.season]
    else:
        raw_root = DATA_ROOT / "raw" / "wbb"
        if not raw_root.exists():
            print("No raw data found", file=sys.stderr)
            sys.exit(1)
        seasons = sorted(d.name for d in raw_root.iterdir() if d.is_dir())

    total = 0
    for season in seasons:
        total += _extract_season(season)
    print(f"\nTotal: {total} conference tournaments across {len(seasons)} seasons")


if __name__ == "__main__":
    main()
