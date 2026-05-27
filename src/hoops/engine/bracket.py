"""Tournament bracket engine for NCAA tournament simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BracketSlot:
    team_id: int | None = None
    seed: int | None = None
    team_name: str = ""


@dataclass
class BracketGame:
    game_idx: int
    round: int
    region: int | None = None
    home: BracketSlot = field(default_factory=BracketSlot)
    away: BracketSlot = field(default_factory=BracketSlot)
    home_score: int | None = None
    away_score: int | None = None
    winner_id: int | None = None
    next_game_idx: int | None = None
    next_slot: str | None = None  # "home" or "away"

    @property
    def is_played(self) -> bool:
        return self.winner_id is not None

    @property
    def is_upset(self) -> bool:
        if not self.is_played:
            return False
        if self.home.seed is None or self.away.seed is None:
            return False
        # Higher seed number = lower-seeded team
        if self.winner_id == self.home.team_id:
            return self.home.seed > self.away.seed
        return self.away.seed > self.home.seed


ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}

CONF_ROUND_NAMES_BY_MAX = {
    2: {1: "Semifinal", 2: "Final"},
    3: {1: "Quarterfinal", 2: "Semifinal", 3: "Final"},
    4: {1: "1st Round", 2: "Quarterfinal", 3: "Semifinal", 4: "Final"},
    5: {1: "1st Round", 2: "2nd Round", 3: "Quarterfinal", 4: "Semifinal", 5: "Final"},
}


def _conf_round_names(max_round: int) -> dict[int, str]:
    if max_round in CONF_ROUND_NAMES_BY_MAX:
        return CONF_ROUND_NAMES_BY_MAX[max_round]
    names = {}
    for r in range(1, max_round + 1):
        if r == max_round:
            names[r] = "Final"
        elif r == max_round - 1:
            names[r] = "Semifinal"
        elif r == max_round - 2:
            names[r] = "Quarterfinal"
        else:
            names[r] = f"Round {r}"
    return names


@dataclass
class Bracket:
    season: str
    regions: list[str]
    games: list[BracketGame] = field(default_factory=list)
    max_round: int = 0
    conference_name: str = ""  # non-empty for conference tournaments
    _round_names: dict[int, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, data: dict) -> Bracket:
        season = data["season"]
        regions = data["regions"]
        conference_name = data.get("conference_name", "")
        games: list[BracketGame] = []

        for i, g in enumerate(data["games"]):
            game = BracketGame(
                game_idx=i,
                round=g["round"],
                region=g.get("region"),
                home=BracketSlot(
                    team_id=g.get("home_team_id"),
                    seed=g.get("home_seed"),
                    team_name=g.get("home_team_name", ""),
                ),
                away=BracketSlot(
                    team_id=g.get("away_team_id"),
                    seed=g.get("away_seed"),
                    team_name=g.get("away_team_name", ""),
                ),
                home_score=g.get("home_score"),
                away_score=g.get("away_score"),
                winner_id=g.get("winner_id"),
            )
            games.append(game)

        # Sort by (round, region, game_idx) to ensure consistent ordering
        games.sort(key=lambda g: (g.round, g.region if g.region is not None else -1, g.game_idx))
        # Re-index after sort
        for i, game in enumerate(games):
            game.game_idx = i

        max_round = max(g.round for g in games) if games else 0
        bracket = cls(season=season, regions=regions, games=games, max_round=max_round, conference_name=conference_name)

        if conference_name:
            bracket._round_names = _conf_round_names(max_round)
        else:
            bracket._round_names = dict(ROUND_NAMES)

        bracket._link_advancement()
        return bracket

    def _link_advancement(self) -> None:
        """Link each game to the next-round game the winner advances to."""
        if not self.games:
            return

        has_regions = any(g.region is not None for g in self.games)

        for rnd in range(1, self.max_round):
            current_round = [g for g in self.games if g.round == rnd]
            next_round = [g for g in self.games if g.round == rnd + 1]

            if not has_regions:
                # Conference tournament: no region grouping, pair consecutively
                self._link_pairs(current_round, next_round)
            elif rnd <= 3:
                # NCAA regional rounds: group by region
                regions_in_round = sorted(set(g.region for g in current_round if g.region is not None))
                for reg in regions_in_round:
                    cur = [g for g in current_round if g.region == reg]
                    nxt = [g for g in next_round if g.region == reg]
                    self._link_pairs(cur, nxt)
            else:
                # NCAA inter-regional rounds
                self._link_pairs(current_round, next_round)

    @staticmethod
    def _link_pairs(current: list[BracketGame], nxt: list[BracketGame]) -> None:
        """Link current-round games to next-round games.

        If current has twice as many games as next, consecutive pairs feed
        into each next game (home/away slots).  If current and next have
        equal length, each current game feeds into the away slot of the
        corresponding next game (bye / 1-to-1 advancement).
        """
        if len(current) <= len(nxt):
            # 1:1 mapping — each current winner fills the away slot
            for i, cur_game in enumerate(current):
                if i < len(nxt):
                    cur_game.next_game_idx = nxt[i].game_idx
                    cur_game.next_slot = "away"
        else:
            # 2:1 pairing — consecutive pairs feed into each next game
            for i, next_game in enumerate(nxt):
                pair_start = i * 2
                if pair_start < len(current):
                    current[pair_start].next_game_idx = next_game.game_idx
                    current[pair_start].next_slot = "home"
                if pair_start + 1 < len(current):
                    current[pair_start + 1].next_game_idx = next_game.game_idx
                    current[pair_start + 1].next_slot = "away"

    @classmethod
    def load(cls, bracket_path: Path) -> Bracket:
        with open(bracket_path) as f:
            data = json.load(f)
        return cls.from_json(data)

    def advance(self, game_idx: int, winner_id: int, home_score: int, away_score: int) -> None:
        """Record a game result and advance the winner to the next game."""
        game = self.games[game_idx]
        game.winner_id = winner_id
        game.home_score = home_score
        game.away_score = away_score

        if game.next_game_idx is not None:
            next_game = self.games[game.next_game_idx]
            # Determine winner's seed
            if winner_id == game.home.team_id:
                winner_seed = game.home.seed
                winner_name = game.home.team_name
            else:
                winner_seed = game.away.seed
                winner_name = game.away.team_name

            if game.next_slot == "home":
                next_game.home.team_id = winner_id
                next_game.home.seed = winner_seed
                next_game.home.team_name = winner_name
            else:
                next_game.away.team_id = winner_id
                next_game.away.seed = winner_seed
                next_game.away.team_name = winner_name

    def round_complete(self, round_num: int) -> bool:
        """Return True if all games in the given round have been played."""
        round_games = [g for g in self.games if g.round == round_num]
        return all(g.is_played for g in round_games)

    def upsets(self, round_num: int) -> list[BracketGame]:
        """Return list of upset games in the given round."""
        return [g for g in self.games if g.round == round_num and g.is_upset]

    def champion(self) -> int | None:
        """Return the tournament champion's team_id, or None if not yet decided."""
        final_games = [g for g in self.games if g.round == self.max_round]
        if final_games and final_games[0].is_played:
            return final_games[0].winner_id
        return None

    def next_game_for(self, team_id: int) -> int | None:
        """Return the game_idx of the next unplayed game for a team, or None."""
        for g in self.games:
            if not g.is_played:
                if g.home.team_id == team_id or g.away.team_id == team_id:
                    return g.game_idx
        return None

    def team_ids(self) -> set[int]:
        """Return set of all team IDs in the bracket."""
        ids: set[int] = set()
        for g in self.games:
            if g.home.team_id is not None:
                ids.add(g.home.team_id)
            if g.away.team_id is not None:
                ids.add(g.away.team_id)
        return ids

    def games_in_round(self, round_num: int) -> list[BracketGame]:
        """Return all games in the given round."""
        return [g for g in self.games if g.round == round_num]

    def team_seed(self, team_id: int) -> int | None:
        """Return the seed for a given team_id."""
        for g in self.games:
            if g.home.team_id == team_id:
                return g.home.seed
            if g.away.team_id == team_id:
                return g.away.seed
        return None

    def team_region(self, team_id: int) -> str | None:
        """Return the region name for a given team_id."""
        for g in self.games:
            if g.home.team_id == team_id or g.away.team_id == team_id:
                if g.region is not None and g.region < len(self.regions):
                    return self.regions[g.region]
        return None

    def round_name(self, round_num: int) -> str:
        """Return the human-readable name for a round number."""
        return self._round_names.get(round_num, f"Round {round_num}")

    def populate_names(self, names: dict[int, str]) -> None:
        """Fill in team_name from a {team_id: name} mapping."""
        for g in self.games:
            if g.home.team_id is not None and g.home.team_id in names:
                g.home.team_name = names[g.home.team_id]
            if g.away.team_id is not None and g.away.team_id in names:
                g.away.team_name = names[g.away.team_id]

    def to_dict(self) -> dict:
        """Serialize bracket to a dictionary."""
        return {
            "season": self.season,
            "regions": self.regions,
            "num_games": len(self.games),
            "games": [
                {
                    "game_id": g.game_idx,
                    "round": g.round,
                    "region": g.region,
                    "home_team_id": g.home.team_id,
                    "away_team_id": g.away.team_id,
                    "home_seed": g.home.seed,
                    "away_seed": g.away.seed,
                    "home_team_name": g.home.team_name,
                    "away_team_name": g.away.team_name,
                    "home_score": g.home_score,
                    "away_score": g.away_score,
                    "winner_id": g.winner_id,
                }
                for g in self.games
            ],
        }
