"""Tournament auto-simulation: sim non-user games in a bracket round."""

from __future__ import annotations

import numpy as np

from hoops.data.distributions import TeamPriors
from hoops.engine.bracket import Bracket
from hoops.engine.machine import simulate_game
from hoops.engine.state import Side
from hoops.rules import Rules


def auto_sim_round(
    bracket: Bracket,
    round_num: int,
    user_team_id: int,
    priors: dict[int, TeamPriors],
    rules: Rules,
    rng: np.random.Generator,
    league=None,
    rosters: dict | None = None,
) -> list[dict]:
    """Auto-simulate all non-user, unplayed games in a bracket round.

    Returns a list of result dicts for display:
    [{"game_idx": int, "home_name": str, "away_name": str,
      "home_score": int, "away_score": int, "home_seed": int, "away_seed": int,
      "is_upset": bool}]
    """
    results = []
    round_games = bracket.games_in_round(round_num)

    for game in round_games:
        # Skip user's game
        if game.home.team_id == user_team_id or game.away.team_id == user_team_id:
            continue
        # Skip already played
        if game.is_played:
            continue
        # Skip if teams not yet determined
        if game.home.team_id is None or game.away.team_id is None:
            continue

        home_priors = priors.get(game.home.team_id)
        away_priors = priors.get(game.away.team_id)
        if home_priors is None or away_priors is None:
            continue

        home_roster = rosters.get(game.home.team_id) if rosters else None
        away_roster = rosters.get(game.away.team_id) if rosters else None

        final_state, _events = simulate_game(
            home_priors, away_priors, rules, rng,
            opening_possession=Side.HOME,
            league=league,
            home_roster=home_roster,
            away_roster=away_roster,
            neutral_site=True,
        )

        winner_id = (
            game.home.team_id if final_state.home_score > final_state.away_score
            else game.away.team_id
        )
        bracket.advance(
            game.game_idx,
            winner_id=winner_id,
            home_score=final_state.home_score,
            away_score=final_state.away_score,
        )

        results.append({
            "game_idx": game.game_idx,
            "home_name": game.home.team_name,
            "away_name": game.away.team_name,
            "home_score": final_state.home_score,
            "away_score": final_state.away_score,
            "home_seed": game.home.seed,
            "away_seed": game.away.seed,
            "is_upset": game.is_upset,
        })

    return results
