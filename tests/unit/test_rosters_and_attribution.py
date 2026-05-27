"""Tests for the player-attribution layer."""

from __future__ import annotations

import numpy as np
import pytest

from hoops.data.paths import raw_dir
from hoops.data.rosters import Player, Roster, load_roster
from hoops.engine.attribution import attribute_players
from hoops.engine.events import Event, fmt_event
from hoops.engine.state import Side
from hoops.league import League


SEASON = "2023-24"


def _data_present() -> bool:
    return (raw_dir(League.WBB, SEASON) / "player_box.parquet").exists()


# ---------- Roster loader (live data) ---------------------------------------


@pytest.mark.skipif(not _data_present(), reason="player_box parquet missing")
def test_load_roster_returns_top_players():
    sc_id = 2579  # South Carolina
    roster = load_roster(sc_id, SEASON, top_n=8)
    assert roster.team_id == sc_id
    assert "South Carolina" in roster.team_name
    names = [p.name for p in roster.players]
    # 2023-24 SC starters / rotation: a few should appear.
    expected = {"Kamilla Cardoso", "Te-Hina Paopao", "Bree Hall", "Ashlyn Watkins"}
    assert expected.issubset(set(names)), (names, expected)
    assert all(p.minutes > 0 for p in roster.players)


# ---------- Sampling helpers (synthetic) ------------------------------------


def _synthetic_roster() -> Roster:
    return Roster(
        team_id=99, team_name="Synth",
        players=(
            Player(player_id=1, name="A", minutes=1000, fga=400, fg3a=80,
                   fta=100, orb=50, drb=200, fouls=70, tov=80, ast=120),
            Player(player_id=2, name="B", minutes=900, fga=300, fg3a=200,
                   fta=60, orb=20, drb=120, fouls=60, tov=70, ast=140),
            Player(player_id=3, name="C", minutes=800, fga=100, fg3a=10,
                   fta=120, orb=200, drb=180, fouls=90, tov=60, ast=20),
        ),
    )


def test_three_point_shooter_prefers_high_3pa_player():
    r = _synthetic_roster()
    rng = np.random.default_rng(42)
    counts = {p.name: 0 for p in r.players}
    for _ in range(2000):
        counts[r.three_point_shooter(rng).name] += 1
    # B has the most 3pa (200/290), should win the plurality.
    assert counts["B"] > counts["A"] > counts["C"]


def test_rebounder_off_prefers_high_orb():
    r = _synthetic_roster()
    rng = np.random.default_rng(42)
    counts = {p.name: 0 for p in r.players}
    for _ in range(2000):
        counts[r.rebounder_off(rng).name] += 1
    # C has the most ORB (200), wins.
    assert counts["C"] > counts["A"] > counts["B"]


def test_player_advanced_rate_fields_default_to_none():
    """Player dataclass has optional advanced rate fields."""
    p = Player(
        player_id=1, name="Test", minutes=100.0,
        fga=50, fg3a=20, fta=30, orb=10, drb=40,
        fouls=15, tov=12, ast=25,
    )
    assert p.usage_pct is None
    assert p.ts_pct is None
    assert p.fg3a_share is None
    assert p.ft_pct is None
    assert p.tov_pct is None
    assert p.orb_pct is None
    assert p.drb_pct is None
    assert p.stl_pct is None
    assert p.blk_pct is None
    assert p.foul_rate is None


def test_uniform_fallback_when_weight_is_zero():
    r = Roster(
        team_id=1, team_name="Z",
        players=(
            Player(1, "X", 0, 0, 0, 0, 0, 0, 0, 0, 0),
            Player(2, "Y", 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
    )
    rng = np.random.default_rng(0)
    # Should still return *one* of the two, never crash.
    seen = {r.shooter(rng).name for _ in range(50)}
    assert seen == {"X", "Y"}


# ---------- Attribution pass ------------------------------------------------


def _ev(quarter, seconds, type_, team=None, **kw):
    return Event(
        quarter=quarter, seconds_left=seconds, type=type_, team=team,
        home_score=kw.get("home_score", 0), away_score=kw.get("away_score", 0),
        detail=kw.get("detail", ""),
    )


def test_attribution_assigns_player_to_shot_and_chains_ft():
    """Made shot → and-1 FT should attribute the same player."""
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(7)
    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_made", Side.HOME, home_score=2, detail="rim"),
        _ev(1, 580, "foul_shooting", Side.AWAY, detail="on shot (rim)"),
        _ev(1, 580, "free_throw_made", Side.HOME, home_score=3, detail="and-1"),
    ]
    out = attribute_players(events, home, away, rng)
    # tip_off has no player.
    assert out[0].player is None
    # shot_made has a HOME shooter.
    shooter = out[1].player
    assert shooter in {p.name for p in home.players}
    # foul_shooting attributes to AWAY (the defender).
    assert out[2].player in {p.name for p in away.players}
    # And-1 FT goes to the same shooter as the made shot.
    assert out[3].player == shooter


def test_attribution_chains_two_free_throws_to_same_shooter():
    """Missed shot with shooting foul → 2 FTs go to the same shooter."""
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(7)
    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_missed", Side.HOME, detail="mid"),
        _ev(1, 580, "foul_shooting", Side.AWAY, detail="on shot (mid)"),
        _ev(1, 580, "free_throw_made", Side.HOME, home_score=1),
        _ev(1, 580, "free_throw_missed", Side.HOME, home_score=1),
    ]
    out = attribute_players(events, home, away, rng)
    assert out[3].player == out[1].player == out[4].player


def test_attribution_resets_on_defensive_rebound():
    """A DRB ends the offense's possession; pending FT shooter clears."""
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(7)
    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_missed", Side.HOME, detail="rim"),
        _ev(1, 580, "rebound_def", Side.AWAY),
        # Now AWAY's shot:
        _ev(1, 560, "shot_made", Side.AWAY, away_score=2, detail="three"),
    ]
    out = attribute_players(events, home, away, rng)
    assert out[1].player in {p.name for p in home.players}
    assert out[2].player in {p.name for p in away.players}
    assert out[3].player in {p.name for p in away.players}


def test_attribution_handles_intentional_foul_with_no_preceding_shot():
    """foul-up-3 → FT trip with no shot before; attribute a fresh FT shooter."""
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(7)
    events = [
        _ev(4, 10, "foul_personal", Side.HOME, detail="intentional (down 3, foul-up-3)"),
        _ev(4, 10, "free_throw_made", Side.AWAY, away_score=1),
        _ev(4, 10, "free_throw_made", Side.AWAY, away_score=2),
    ]
    out = attribute_players(events, home, away, rng)
    # Fouler comes from HOME's roster.
    assert out[0].player in {p.name for p in home.players}
    # Both FTs go to the same AWAY player.
    assert out[1].player == out[2].player
    assert out[1].player in {p.name for p in away.players}


# ---------- Natural-language formatting -------------------------------------


def test_fmt_event_uses_natural_language():
    e = Event(
        quarter=1, seconds_left=540, type="shot_made", team=Side.HOME,
        detail="rim", home_score=2, away_score=0, player="Kamilla Cardoso",
    )
    out = fmt_event(e)
    assert "Kamilla Cardoso made layup" in out
    # No underscores in the rendered phrase.
    phrase = out.split("  ", 2)[-1]
    assert "_" not in phrase


def test_fmt_event_renders_three_pointer():
    e = Event(
        quarter=2, seconds_left=120, type="shot_missed", team=Side.AWAY,
        detail="three", player="Caitlin Clark",
    )
    assert "Caitlin Clark missed 3-pointer" in fmt_event(e)


def test_fmt_event_renders_free_throw():
    made = Event(quarter=1, seconds_left=60, type="free_throw_made",
                 team=Side.HOME, player="P", detail="and-1")
    miss = Event(quarter=1, seconds_left=60, type="free_throw_missed",
                 team=Side.HOME, player="P")
    assert "P free throw good" in fmt_event(made)
    assert "and-1" in fmt_event(made)
    assert "P free throw missed" in fmt_event(miss)


def test_fmt_event_falls_back_to_team_label_without_player():
    e = Event(
        quarter=1, seconds_left=540, type="rebound_def", team=Side.AWAY,
    )
    out = fmt_event(e, "South Carolina", "Iowa")
    # Team name appears in the tag column, phrase says "defensive rebound"
    # without repeating the team name.
    assert "Iowa" in out
    assert "defensive rebound" in out
    # Should NOT duplicate: "Iowa Iowa defensive rebound"
    assert "Iowa  Iowa" not in out


def test_fmt_event_renders_block_steal_assist():
    block_e = Event(quarter=2, seconds_left=300, type="block",
                    team=Side.AWAY, player="Cardoso")
    steal_e = Event(quarter=2, seconds_left=300, type="steal",
                    team=Side.HOME, player="Bueckers")
    assist_e = Event(quarter=2, seconds_left=300, type="assist",
                     team=Side.HOME, player="Paopao")
    assert "Cardoso block" in fmt_event(block_e)
    assert "Bueckers steal" in fmt_event(steal_e)
    assert "Paopao assist" in fmt_event(assist_e)


# ---------- Credit-event insertion (assists / blocks / steals) -------------


def test_made_shot_sometimes_inserts_assist():
    """With ASSIST_PROB ~0.55, a long sequence of made shots should produce
    assists ~half the time, attributed to teammates of the shooter."""
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(0)
    events = [Event(
        quarter=1, seconds_left=600 - i, type="shot_made",
        team=Side.HOME, detail="rim",
    ) for i in range(200)]
    out = attribute_players(events, home, away, rng)
    assists = [e for e in out if e.type == "assist"]
    # Expect ~110 assists; allow wide tolerance.
    assert 70 <= len(assists) <= 150
    # Each assist comes from HOME and is not the shooter on the prior event.
    for i, e in enumerate(out):
        if e.type == "assist":
            prev = out[i - 1]
            assert prev.type == "shot_made"
            assert e.team is Side.HOME
            assert e.player != prev.player


def test_missed_shot_sometimes_inserts_block_attributed_to_defense():
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(1)
    # All clean misses (no following foul) → blocks possible.
    events = []
    for i in range(500):
        events.append(Event(
            quarter=1, seconds_left=600 - i, type="shot_missed",
            team=Side.HOME, detail="rim",
        ))
        # Add a defensive rebound to make each one a complete possession.
        events.append(Event(
            quarter=1, seconds_left=600 - i, type="rebound_def",
            team=Side.AWAY,
        ))
    out = attribute_players(events, home, away, rng)
    blocks = [e for e in out if e.type == "block"]
    # ~6% of 500 misses ≈ 30; allow wide tolerance.
    assert 10 <= len(blocks) <= 70
    for e in blocks:
        assert e.team is Side.AWAY  # defender's team
        assert e.player in {p.name for p in away.players}


def test_block_does_not_fire_on_fouled_shot():
    """A shot_missed immediately followed by a foul_shooting must never
    have a block inserted — you can't block a shot you fouled on."""
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(2)
    # 500 fouled-shot sequences.
    events = []
    for i in range(500):
        events.append(Event(quarter=1, seconds_left=600 - i, type="shot_missed",
                            team=Side.HOME, detail="rim"))
        events.append(Event(quarter=1, seconds_left=600 - i, type="foul_shooting",
                            team=Side.AWAY))
    out = attribute_players(events, home, away, rng)
    assert all(e.type != "block" for e in out)


def test_turnover_sometimes_inserts_steal():
    home = _synthetic_roster()
    away = _synthetic_roster()
    rng = np.random.default_rng(3)
    events = [Event(
        quarter=1, seconds_left=600 - i, type="turnover",
        team=Side.HOME,
    ) for i in range(300)]
    out = attribute_players(events, home, away, rng)
    steals = [e for e in out if e.type == "steal"]
    # ~50% should produce a steal.
    assert 100 <= len(steals) <= 200
    for e in steals:
        assert e.team is Side.AWAY
        assert e.player in {p.name for p in away.players}


def test_fmt_event_renders_structural_events():
    qend = Event(quarter=1, seconds_left=0, type="quarter_end", team=None)
    assert "End of Q1" in fmt_event(qend)
    final = Event(quarter=4, seconds_left=0, type="game_end", team=None,
                  home_score=80, away_score=70)
    assert "Final" in fmt_event(final)


def test_fmt_event_renders_timeout():
    from hoops.engine.events import fmt_event, Event
    from hoops.engine.state import Side

    e = Event(
        quarter=2, seconds_left=292, type="timeout", team=Side.HOME,
        detail="3 remaining", home_score=28, away_score=22,
    )
    out = fmt_event(e, "MD", "SC")
    assert "MD" in out
    assert "TIMEOUT" in out
    assert "3 remaining" in out


def test_fmt_event_renders_media_timeout():
    from hoops.engine.events import fmt_event, Event

    e = Event(
        quarter=1, seconds_left=295, type="media_timeout", team=None,
        home_score=15, away_score=12,
    )
    out = fmt_event(e, "MD", "SC")
    assert "MEDIA TIMEOUT" in out
