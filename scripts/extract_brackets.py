"""Extract NCAA tournament bracket structure from raw schedule data into JSON files.

Reads schedule.parquet for each season, filters to tournament_id == 22 (NCAA tournament),
parses the notes_headline to determine region and round, infers seeds from rankings,
and writes structured bracket JSON to data/brackets/wbb/<season>.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_ROOT = DATA_ROOT / "raw" / "wbb"
OUT_ROOT = DATA_ROOT / "brackets" / "wbb"

# Round name -> round number mapping
ROUND_MAP: dict[str, int] = {
    "1ST ROUND": 1,
    "2ND ROUND": 2,
    "SWEET 16": 3,
    "ELITE 8": 4,
    "SEMIFINAL": 3,   # older seasons use SEMIFINAL for Sweet 16
    "FINAL": 4,        # older seasons use FINAL for Elite 8
    "FINAL FOUR": 5,
    "NATIONAL CHAMPIONSHIP": 6,
}


def parse_headline(headline: str) -> tuple[str | None, int | None]:
    """Parse a notes_headline into (region_name, round_number).

    Returns (None, None) for First Four games.
    Returns (None, round_number) for Final Four / National Championship.
    Returns (region_name, round_number) for regional games.
    """
    h = headline.upper().strip()

    # Split by ' - ' to get parts after the tournament name prefix
    parts = [p.strip() for p in h.split(" - ")]
    meaningful = parts[1:]  # skip tournament name

    if not meaningful:
        return None, None

    combined = " ".join(meaningful)

    # Skip First Four
    if "FIRST FOUR" in combined:
        return None, None

    # Final Four / National Championship (no region)
    if "FINAL FOUR" in combined:
        return None, ROUND_MAP["FINAL FOUR"]
    if "NATIONAL CHAMPIONSHIP" in combined:
        return None, ROUND_MAP["NATIONAL CHAMPIONSHIP"]

    # Two-part format: "REGION_NAME - ROUND_NAME"
    if len(meaningful) >= 2:
        region = meaningful[0]
        round_text = meaningful[-1].strip()
        round_num = ROUND_MAP.get(round_text)
        if round_num is not None:
            return region, round_num
        # Shouldn't happen, but warn
        print(f"  WARNING: unknown round text '{round_text}' in '{headline}'")
        return region, None

    # Single-part format: "REGION_NAME SEMIFINAL" or "REGION_NAME FINAL"
    # (used in some 2016-17, 2017-18 seasons)
    text = meaningful[0]
    for round_name in ["SEMIFINAL", "FINAL"]:
        if text.endswith(round_name):
            region = text[: -len(round_name)].strip()
            return region, ROUND_MAP[round_name]

    print(f"  WARNING: could not parse '{headline}'")
    return None, None


def normalize_region(region: str) -> str:
    """Normalize region name for consistent display.

    Converts 'BRIDGEPORT REGION' -> 'Bridgeport',
    'REGIONAL 1 IN ALBANY' -> 'Albany 1',
    'GREENVILLE REGIONAL 1' -> 'Greenville 1', etc.
    """
    r = region.strip()

    # Format: "REGIONAL N IN CITY" -> "City N"
    m = re.match(r"REGIONAL\s+(\d+)\s+IN\s+(.+)", r, re.IGNORECASE)
    if m:
        num, city = m.group(1), m.group(2)
        return f"{city.title()} {num}"

    # Format: "CITY REGIONAL N" -> "City N"
    m = re.match(r"(.+?)\s+REGIONAL\s+(\d+)", r, re.IGNORECASE)
    if m:
        city, num = m.group(1), m.group(2)
        return f"{city.title()} {num}"

    # Format: "CITY REGION" -> "City"
    m = re.match(r"(.+?)\s+REGION$", r, re.IGNORECASE)
    if m:
        return m.group(1).title()

    # Fallback
    return r.title()


def find_main_regions(
    games_with_parsed: list[dict],
) -> dict[str, str]:
    """For seasons with sub-host sites (e.g., 2018-19), map sub-region names
    to the main region they feed into.

    Returns a mapping of raw_region -> main_raw_region.
    """
    # Identify main regions: those that have round 3 or 4 games
    main_regions: set[str] = set()
    sub_regions: set[str] = set()
    all_regions: set[str] = set()

    for g in games_with_parsed:
        if g["raw_region"] is not None and g["round"] is not None:
            all_regions.add(g["raw_region"])
            if g["round"] >= 3:
                main_regions.add(g["raw_region"])

    sub_regions = all_regions - main_regions

    if not sub_regions:
        # No sub-regions; identity mapping
        return {r: r for r in all_regions}

    # Build team -> sub_region mapping from 1st/2nd round games
    # Then find which main region those teams appear in for round 3+
    team_to_sub: dict[int, str] = {}
    for g in games_with_parsed:
        if g["raw_region"] in sub_regions and g["round"] in (1, 2):
            team_to_sub[g["home_team_id"]] = g["raw_region"]
            team_to_sub[g["away_team_id"]] = g["raw_region"]

    sub_to_main: dict[str, str] = {}
    for g in games_with_parsed:
        if g["raw_region"] in main_regions and g["round"] in (3, 4):
            for tid in (g["home_team_id"], g["away_team_id"]):
                if tid in team_to_sub:
                    sub = team_to_sub[tid]
                    if sub not in sub_to_main:
                        sub_to_main[sub] = g["raw_region"]

    # Build full mapping
    mapping = {r: r for r in main_regions}
    for sub, main in sub_to_main.items():
        mapping[sub] = main

    # Any unmapped sub-regions: try to map via 2nd-round winners advancing
    for sub in sub_regions:
        if sub not in mapping:
            # Find 2nd-round winners from this sub-region
            for g in games_with_parsed:
                if g["raw_region"] == sub and g["round"] == 2:
                    winner_id = (
                        g["home_team_id"] if g["home_winner"] else g["away_team_id"]
                    )
                    # Find this team in round 3+ games
                    for g2 in games_with_parsed:
                        if g2["round"] and g2["round"] >= 3 and g2["raw_region"] in main_regions:
                            if winner_id in (g2["home_team_id"], g2["away_team_id"]):
                                mapping[sub] = g2["raw_region"]
                                break
                    if sub in mapping:
                        break

    return mapping


def extract_season(season: str) -> dict | None:
    """Extract bracket data for a single season."""
    parquet_path = RAW_ROOT / season / "schedule.parquet"
    if not parquet_path.exists():
        print(f"  Skipping {season}: no schedule.parquet")
        return None

    df = pl.read_parquet(parquet_path)
    tourney = df.filter(pl.col("tournament_id") == 22)

    if len(tourney) == 0:
        print(f"  Skipping {season}: no tournament games (tournament_id == 22)")
        return None

    has_ranks = "home_current_rank" in tourney.columns

    # Parse all games
    parsed_games: list[dict] = []
    for row in tourney.sort("game_date", "game_id").iter_rows(named=True):
        headline = row["notes_headline"] or ""
        raw_region, round_num = parse_headline(headline)

        if round_num is None:
            # First Four or unparseable -> skip
            continue

        parsed_games.append(
            {
                "game_id": row["game_id"],
                "round": round_num,
                "raw_region": raw_region,
                "home_team_id": row["home_id"],
                "away_team_id": row["away_id"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "home_winner": row["home_winner"],
                "away_winner": row.get("away_winner", not row["home_winner"]),
                "home_current_rank": row["home_current_rank"] if has_ranks else None,
                "away_current_rank": row["away_current_rank"] if has_ranks else None,
                "game_date": str(row["game_date"]),
            }
        )

    if len(parsed_games) != 63:
        print(
            f"  WARNING: {season} has {len(parsed_games)} main bracket games "
            f"(expected 63)"
        )

    # Handle sub-host regions
    region_mapping = find_main_regions(parsed_games)

    # Apply region mapping
    for g in parsed_games:
        if g["raw_region"] is not None:
            g["raw_region"] = region_mapping.get(g["raw_region"], g["raw_region"])

    # Build ordered list of 4 main region names
    # Use order of first appearance in round 1 games sorted by date/game_id
    seen_regions: list[str] = []
    for g in parsed_games:
        if g["round"] == 1 and g["raw_region"] is not None:
            if g["raw_region"] not in seen_regions:
                seen_regions.append(g["raw_region"])

    # Map raw region names to indices
    region_to_idx = {r: i for i, r in enumerate(seen_regions)}
    normalized_regions = [normalize_region(r) for r in seen_regions]

    # Build seed lookup: team_id -> seed (from round 1 games)
    seed_map: dict[int, int] = {}
    for g in parsed_games:
        if g["round"] == 1:
            home_rank = g["home_current_rank"]
            away_rank = g["away_current_rank"]
            if home_rank is not None and home_rank == home_rank:  # not NaN
                seed_map[g["home_team_id"]] = int(home_rank)
            if away_rank is not None and away_rank == away_rank:  # not NaN
                seed_map[g["away_team_id"]] = int(away_rank)

    # Build output games list
    output_games: list[dict] = []
    for g in parsed_games:
        region_idx: int | None = None
        if g["raw_region"] is not None:
            region_idx = region_to_idx.get(g["raw_region"])

        home_seed = seed_map.get(g["home_team_id"])
        away_seed = seed_map.get(g["away_team_id"])

        output_games.append(
            {
                "game_id": g["game_id"],
                "round": g["round"],
                "region": region_idx,
                "home_team_id": g["home_team_id"],
                "away_team_id": g["away_team_id"],
                "home_seed": home_seed,
                "away_seed": away_seed,
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "game_date": g["game_date"],
            }
        )

    return {
        "season": season,
        "regions": normalized_regions,
        "num_games": len(output_games),
        "games": output_games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract NCAA WBB tournament bracket data from schedule parquet files."
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Single season to extract (e.g., '2023-24'). Default: all seasons.",
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.season:
        seasons = [args.season]
    else:
        seasons = sorted(
            d.name for d in RAW_ROOT.iterdir() if d.is_dir() and "-" in d.name
        )

    for season in seasons:
        print(f"Processing {season}...")
        result = extract_season(season)
        if result is None:
            continue

        out_path = OUT_ROOT / f"{season}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"  Wrote {out_path.name}: {result['num_games']} games, "
            f"{len(result['regions'])} regions: {result['regions']}"
        )


if __name__ == "__main__":
    main()
