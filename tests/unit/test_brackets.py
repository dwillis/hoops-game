"""Tests for bracket path helpers and bracket engine."""

from __future__ import annotations

from pathlib import Path

import pytest

import json

from hoops.data.paths import (
    DATA_ROOT,
    bracket_path,
    bracket_seasons,
    conf_tournament_dir,
    conf_tournament_path,
    list_conf_tournaments,
)
from hoops.engine.bracket import Bracket
from hoops.league import League


def _mini_bracket_json() -> dict:
    """A 4-team, 2-round mini bracket for testing."""
    return {
        "season": "test",
        "regions": ["TestRegion"],
        "num_games": 3,
        "games": [
            {
                "game_id": 1,
                "round": 1,
                "region": 0,
                "home_team_id": 101,
                "away_team_id": 104,
                "home_seed": 1,
                "away_seed": 4,
                "home_score": None,
                "away_score": None,
            },
            {
                "game_id": 2,
                "round": 1,
                "region": 0,
                "home_team_id": 102,
                "away_team_id": 103,
                "home_seed": 2,
                "away_seed": 3,
                "home_score": None,
                "away_score": None,
            },
            {
                "game_id": 3,
                "round": 2,
                "region": 0,
                "home_team_id": None,
                "away_team_id": None,
                "home_seed": None,
                "away_seed": None,
                "home_score": None,
                "away_score": None,
            },
        ],
    }


class TestBracketPath:
    def test_bracket_path_returns_correct_json_path(self):
        result = bracket_path(League.WBB, "2025")
        expected = DATA_ROOT / "brackets" / "wbb" / "2025.json"
        assert result == expected

    def test_bracket_seasons_scans_disk(self, tmp_path, monkeypatch):
        import hoops.data.paths as paths_mod

        bracket_dir = tmp_path / "brackets" / "wbb"
        bracket_dir.mkdir(parents=True)
        (bracket_dir / "2024.json").write_text("{}")
        (bracket_dir / "2025.json").write_text("{}")
        (bracket_dir / "ignore.txt").write_text("")

        monkeypatch.setattr(paths_mod, "DATA_ROOT", tmp_path)
        result = bracket_seasons(League.WBB)
        assert result == ["2024", "2025"]

    def test_bracket_seasons_empty_when_no_dir(self, tmp_path, monkeypatch):
        import hoops.data.paths as paths_mod

        monkeypatch.setattr(paths_mod, "DATA_ROOT", tmp_path)
        result = bracket_seasons(League.WBB)
        assert result == []


class TestBracketFromJson:
    def test_from_json_builds_bracket(self):
        data = _mini_bracket_json()
        bracket = Bracket.from_json(data)

        assert bracket.season == "test"
        assert bracket.regions == ["TestRegion"]
        assert len(bracket.games) == 3
        assert bracket.max_round == 2

    def test_from_json_links_advancement(self):
        data = _mini_bracket_json()
        bracket = Bracket.from_json(data)

        # Round 1 games should link to round 2 game
        r1_games = bracket.games_in_round(1)
        assert len(r1_games) == 2
        r2_games = bracket.games_in_round(2)
        assert len(r2_games) == 1

        # Both round 1 games should point to the round 2 game
        r2_idx = r2_games[0].game_idx
        assert r1_games[0].next_game_idx == r2_idx
        assert r1_games[0].next_slot == "home"
        assert r1_games[1].next_game_idx == r2_idx
        assert r1_games[1].next_slot == "away"


class TestBracketAdvance:
    def test_advance_records_result(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        g0_idx = r1[0].game_idx

        bracket.advance(g0_idx, winner_id=101, home_score=70, away_score=60)

        game = bracket.games[g0_idx]
        assert game.is_played
        assert game.winner_id == 101
        assert game.home_score == 70
        assert game.away_score == 60

    def test_advance_moves_winner_to_next_game(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        g0_idx = r1[0].game_idx

        bracket.advance(g0_idx, winner_id=101, home_score=70, away_score=60)

        r2 = bracket.games_in_round(2)
        assert r2[0].home.team_id == 101
        assert r2[0].home.seed == 1


class TestRoundComplete:
    def test_round_not_complete(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        assert bracket.round_complete(1) is False

    def test_round_complete_after_all_played(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        bracket.advance(r1[0].game_idx, winner_id=101, home_score=70, away_score=60)
        bracket.advance(r1[1].game_idx, winner_id=102, home_score=65, away_score=55)
        assert bracket.round_complete(1) is True


class TestUpsets:
    def test_no_upsets_when_favorites_win(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        bracket.advance(r1[0].game_idx, winner_id=101, home_score=70, away_score=60)
        assert bracket.upsets(1) == []

    def test_upset_detected(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        # 4-seed beats 1-seed = upset
        bracket.advance(r1[0].game_idx, winner_id=104, home_score=60, away_score=70)
        upsets = bracket.upsets(1)
        assert len(upsets) == 1
        assert upsets[0].winner_id == 104


class TestChampion:
    def test_no_champion_initially(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        assert bracket.champion() is None

    def test_champion_after_final(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        bracket.advance(r1[0].game_idx, winner_id=101, home_score=70, away_score=60)
        bracket.advance(r1[1].game_idx, winner_id=102, home_score=65, away_score=55)

        r2 = bracket.games_in_round(2)
        bracket.advance(r2[0].game_idx, winner_id=101, home_score=80, away_score=70)
        assert bracket.champion() == 101


class TestNextGameFor:
    def test_next_game_for_existing_team(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        idx = bracket.next_game_for(101)
        assert idx == r1[0].game_idx

    def test_next_game_for_after_advance(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        bracket.advance(r1[0].game_idx, winner_id=101, home_score=70, away_score=60)

        # Winner should now have a next game in round 2
        r2 = bracket.games_in_round(2)
        assert bracket.next_game_for(101) == r2[0].game_idx

    def test_next_game_for_eliminated_team(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        r1 = bracket.games_in_round(1)
        bracket.advance(r1[0].game_idx, winner_id=101, home_score=70, away_score=60)
        # Team 104 lost, no next game
        assert bracket.next_game_for(104) is None


class TestTeamIds:
    def test_team_ids_returns_all_round1_teams(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        ids = bracket.team_ids()
        assert ids == {101, 102, 103, 104}


class TestPopulateNames:
    def test_populate_names_fills_names(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        names = {101: "Team A", 102: "Team B", 103: "Team C", 104: "Team D"}
        bracket.populate_names(names)

        r1 = bracket.games_in_round(1)
        assert r1[0].home.team_name == "Team A"
        assert r1[0].away.team_name == "Team D"
        assert r1[1].home.team_name == "Team B"
        assert r1[1].away.team_name == "Team C"

    def test_populate_names_skips_none_ids(self):
        bracket = Bracket.from_json(_mini_bracket_json())
        names = {101: "Team A"}
        bracket.populate_names(names)

        r2 = bracket.games_in_round(2)
        # Round 2 has None team_ids, should not crash
        assert r2[0].home.team_name == ""


# ---------------------------------------------------------------------------
# Integration tests using real bracket data
# ---------------------------------------------------------------------------


def test_full_bracket_load_from_real_data():
    """Integration: load a real extracted bracket and verify structure."""
    from hoops.data.paths import bracket_path, bracket_seasons
    from hoops.league import League

    seasons = bracket_seasons(League.WBB)
    if not seasons:
        pytest.skip("No bracket data available")

    season = seasons[-1]  # Most recent
    bp = bracket_path(League.WBB, season)
    b = Bracket.load(bp)

    # Should have ~63 games
    assert len(b.games) >= 60
    assert len(b.games) <= 67

    # Should have 4 regions
    assert len(b.regions) == 4

    # Round 1 should have 32 games
    r1 = b.games_in_round(1)
    assert len(r1) == 32

    # All round 1 games should have team IDs
    for g in r1:
        assert g.home.team_id is not None
        assert g.away.team_id is not None

    # Should have 64 unique teams
    assert len(b.team_ids()) == 64

    # Championship should be the max round
    finals = b.games_in_round(b.max_round)
    assert len(finals) == 1


def test_auto_sim_with_real_data():
    """Integration: auto-sim a round with real team priors."""
    from hoops.data.distributions import load_league_prior, load_team_priors
    from hoops.data.paths import bracket_path, bracket_seasons
    from hoops.engine.sampling import make_rng
    from hoops.engine.tournament import auto_sim_round
    from hoops.league import League
    from hoops.rules import rules_for

    seasons = bracket_seasons(League.WBB)
    if not seasons:
        pytest.skip("No bracket data available")

    season = seasons[-1]
    bp = bracket_path(League.WBB, season)
    b = Bracket.load(bp)

    all_priors = load_team_priors(League.WBB, season)
    priors = {p.team_id: p for p in all_priors}
    league = load_league_prior(League.WBB, season)
    rules = rules_for(League.WBB, season)
    rng = make_rng(seed=99)

    user_id = list(b.team_ids())[0]

    names = {p.team_id: p.team_name for p in all_priors}
    b.populate_names(names)

    results = auto_sim_round(b, 1, user_team_id=user_id, priors=priors,
                              rules=rules, rng=rng, league=league)

    # Should have simmed 31 games (32 minus user's 1)
    assert len(results) == 31

    for r in results:
        assert r["home_score"] is not None
        assert r["away_score"] is not None
        assert r["home_score"] != r["away_score"]  # No ties in basketball


# ---------------------------------------------------------------------------
# Conference tournament path helpers
# ---------------------------------------------------------------------------


def test_conf_tournament_dir():
    p = conf_tournament_dir(League.WBB, "2023-24")
    assert "conf_tournaments" in str(p)
    assert "wbb" in str(p)
    assert "2023-24" in str(p)


def test_conf_tournament_path():
    p = conf_tournament_path(League.WBB, "2023-24", 28)
    assert p.name == "28.json"


def test_list_conf_tournaments(tmp_path, monkeypatch):
    import hoops.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "DATA_ROOT", tmp_path)
    d = tmp_path / "conf_tournaments" / "wbb" / "2023-24"
    d.mkdir(parents=True)
    index = {
        "season": "2023-24",
        "conferences": [
            {"tournament_id": 28, "conference_name": "SEC Tournament", "num_teams": 14, "num_games": 13},
            {"tournament_id": 9, "conference_name": "Big Ten Tournament", "num_teams": 14, "num_games": 13},
        ],
    }
    (d / "index.json").write_text(json.dumps(index))
    result = list_conf_tournaments(League.WBB, "2023-24")
    assert len(result) == 2
    assert result[0]["conference_name"] == "SEC Tournament"


def test_list_conf_tournaments_empty(tmp_path, monkeypatch):
    import hoops.data.paths as paths_mod

    monkeypatch.setattr(paths_mod, "DATA_ROOT", tmp_path)
    result = list_conf_tournaments(League.WBB, "2023-24")
    assert result == []


# ---------------------------------------------------------------------------
# Conference tournament bracket tests
# ---------------------------------------------------------------------------


def _conf_bracket_json() -> dict:
    """A 6-team conference tournament with byes.

    Seeds 1-2 get byes to the semifinal (round 2).
    Seeds 3-6 play in round 1 (quarterfinal).
    """
    return {
        "season": "test",
        "conference_name": "Test Conference",
        "regions": [],
        "games": [
            # Round 1: Quarterfinals (3v6, 4v5)
            {"game_id": 1, "round": 1, "region": None,
             "home_team_id": 203, "away_team_id": 206,
             "home_seed": 3, "away_seed": 6,
             "home_score": None, "away_score": None},
            {"game_id": 2, "round": 1, "region": None,
             "home_team_id": 204, "away_team_id": 205,
             "home_seed": 4, "away_seed": 5,
             "home_score": None, "away_score": None},
            # Round 2: Semifinals (1 vs winner of 3v6, 2 vs winner of 4v5)
            {"game_id": 3, "round": 2, "region": None,
             "home_team_id": 201, "away_team_id": None,
             "home_seed": 1, "away_seed": None,
             "home_score": None, "away_score": None},
            {"game_id": 4, "round": 2, "region": None,
             "home_team_id": 202, "away_team_id": None,
             "home_seed": 2, "away_seed": None,
             "home_score": None, "away_score": None},
            # Round 3: Final
            {"game_id": 5, "round": 3, "region": None,
             "home_team_id": None, "away_team_id": None,
             "home_seed": None, "away_seed": None,
             "home_score": None, "away_score": None},
        ],
    }


class TestConferenceBracket:
    def test_from_json_no_regions(self):
        b = Bracket.from_json(_conf_bracket_json())
        assert len(b.games) == 5
        assert b.max_round == 3
        assert b.regions == []
        assert b.conference_name == "Test Conference"

    def test_advancement_links_without_regions(self):
        b = Bracket.from_json(_conf_bracket_json())
        # Round 1 games should link to round 2 games
        r1 = b.games_in_round(1)
        assert r1[0].next_game_idx is not None
        assert r1[1].next_game_idx is not None
        # Round 2 games should link to final
        r2 = b.games_in_round(2)
        assert r2[0].next_game_idx is not None
        assert r2[1].next_game_idx is not None

    def test_byes_advance_correctly(self):
        b = Bracket.from_json(_conf_bracket_json())
        # Advance round 1 winners
        b.advance(0, winner_id=203, home_score=70, away_score=60)
        b.advance(1, winner_id=205, home_score=55, away_score=65)
        # Round 2: seed 1 vs winner(3v6)=203, seed 2 vs winner(4v5)=205
        r2 = b.games_in_round(2)
        has_203 = any(g.away.team_id == 203 for g in r2)
        has_205 = any(g.away.team_id == 205 for g in r2)
        assert has_203
        assert has_205

    def test_team_ids_includes_bye_teams(self):
        b = Bracket.from_json(_conf_bracket_json())
        ids = b.team_ids()
        # Should include bye teams 201, 202 even though they're not in round 1
        assert 201 in ids
        assert 202 in ids
        assert 203 in ids
        assert 206 in ids

    def test_round_name_conf_tournament(self):
        b = Bracket.from_json(_conf_bracket_json())
        # Conference tournament with 3 rounds: Quarterfinal, Semifinal, Final
        assert b.round_name(1) == "Quarterfinal"
        assert b.round_name(2) == "Semifinal"
        assert b.round_name(3) == "Final"


def test_conf_tournament_load_from_real_data():
    """Integration: load a real conference tournament and verify structure."""
    from hoops.data.paths import list_conf_tournaments, conf_tournament_path
    from hoops.league import League

    seasons = bracket_seasons(League.WBB)
    if not seasons:
        pytest.skip("No bracket data available")

    season = seasons[-1]
    conferences = list_conf_tournaments(League.WBB, season)
    if not conferences:
        pytest.skip("No conference tournament data")

    # Load the first available conference
    c = conferences[0]
    bp = conf_tournament_path(League.WBB, season, c["tournament_id"])
    b = Bracket.load(bp)

    assert len(b.games) >= 3
    assert b.max_round >= 2
    assert b.conference_name != ""
    assert b.regions == []

    # All teams should be present
    assert len(b.team_ids()) >= 4

    # Final round should have exactly 1 game
    finals = b.games_in_round(b.max_round)
    assert len(finals) == 1
