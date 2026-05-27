"""LineupState: live attribution + substitution mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from hoops.data.rosters import Player, Roster
from hoops.engine.events import Event
from hoops.engine.state import Side
from hoops.ui.lineup import LineupError, LineupState


def _player(pid: int, name: str, **stats) -> Player:
    base = dict(
        minutes=600.0, fga=200, fg3a=80, fta=60, orb=20, drb=80,
        fouls=40, tov=30, ast=40, blk=10, stl=15,
    )
    base.update(stats)
    return Player(player_id=pid, name=name, **base)


def _roster(team_id: int, name: str, n: int = 8) -> Roster:
    return Roster(
        team_id=team_id, team_name=name,
        players=tuple(_player(i, f"{name}_P{i}") for i in range(1, n + 1)),
    )


# --- defaults -----------------------------------------------------------------

def test_default_starters_top_5_by_roster_order():
    h = _roster(1, "Home")
    a = _roster(2, "Away")
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    assert [p.player_id for p in ls.on_court(Side.HOME)] == [1, 2, 3, 4, 5]
    assert [p.player_id for p in ls.on_court(Side.AWAY)] == [1, 2, 3, 4, 5]


def test_bench_is_complement_of_on_court():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    bench_ids = {p.player_id for p in ls.bench(Side.HOME)}
    assert bench_ids == {6, 7, 8}


# --- substitution mechanics ---------------------------------------------------

def test_substitute_swaps_on_and_off():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.substitute(Side.HOME, off_player_id=1, on_player_id=6)
    assert 1 not in {p.player_id for p in ls.on_court(Side.HOME)}
    assert 6 in {p.player_id for p in ls.on_court(Side.HOME)}
    assert 1 in {p.player_id for p in ls.bench(Side.HOME)}


def test_substitute_rejects_player_not_on_court():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    with pytest.raises(LineupError):
        ls.substitute(Side.HOME, off_player_id=99, on_player_id=6)


def test_substitute_rejects_player_not_on_bench():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    with pytest.raises(LineupError):
        # Player 2 is already on the floor, not on the bench.
        ls.substitute(Side.HOME, off_player_id=1, on_player_id=2)


# --- live attribution ---------------------------------------------------------

def _ev(quarter, seconds, type_, team=None, **kw):
    return Event(
        quarter=quarter, seconds_left=seconds, type=type_, team=team,
        home_score=kw.get("home_score", 0), away_score=kw.get("away_score", 0),
        detail=kw.get("detail", ""), player=kw.get("player"),
    )


def test_attribute_picks_only_on_court_players():
    """A pulled bench player should never be attributed events."""
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(42)
    ls = LineupState.with_default_starters(h, a, rng)

    on_court_names = {p.name for p in ls.on_court(Side.HOME)}
    bench_names = {p.name for p in ls.bench(Side.HOME)}

    seen_attr = set()
    for _ in range(200):
        e = _ev(1, 580, "shot_made", Side.HOME, detail="rim")
        a_e = ls.attribute(e)
        seen_attr.add(a_e.player)
    assert seen_attr.issubset(on_court_names)
    assert not (seen_attr & bench_names)


def test_substitution_changes_who_is_attributed():
    """After subbing player 1 out for player 6, only the new starter 6
    can be attributed (until further subs)."""
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    # Build a roster where only player_id == 6 has nonzero FGA so
    # weighted sampling deterministically picks them after the sub.
    h6 = Roster(
        team_id=10, team_name="HomeOnly6",
        players=tuple(
            _player(i, f"HomeOnly6_P{i}", fga=(1000 if i == 6 else 0))
            for i in range(1, 9)
        ),
    )
    ls = LineupState.with_default_starters(h6, a, rng)
    ls.substitute(Side.HOME, off_player_id=1, on_player_id=6)
    # Now only HomeOnly6_P6 should be attributed shot events.
    for _ in range(50):
        e = _ev(1, 580, "shot_made", Side.HOME, detail="rim")
        attributed = ls.attribute(e)
        assert attributed.player == "HomeOnly6_P6"


def test_attribute_chains_ft_to_pending_shooter():
    """Made shot → and-1 should see the same player on the FT."""
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(7)
    ls = LineupState.with_default_starters(h, a, rng)
    shot = ls.attribute(_ev(1, 580, "shot_made", Side.HOME, detail="rim"))
    ft = ls.attribute(_ev(1, 580, "free_throw_made", Side.HOME, detail="and-1"))
    assert ft.player == shot.player


def test_attribute_resets_pending_ft_on_drb():
    """Defensive rebound clears pending FT shooter for both sides."""
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(7)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.attribute(_ev(1, 580, "shot_missed", Side.HOME, detail="three"))
    assert ls.pending_ft_shooter[Side.HOME] is not None
    ls.attribute(_ev(1, 580, "rebound_def", Side.AWAY))
    assert ls.pending_ft_shooter[Side.HOME] is None
    assert ls.pending_ft_shooter[Side.AWAY] is None


def test_attribute_passes_through_structural_events():
    h = _roster(1, "Home")
    a = _roster(2, "Away")
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    for type_ in ("tip_off", "quarter_end", "overtime_start", "game_end"):
        out = ls.attribute(_ev(1, 0, type_, team=None))
        assert out.player is None


# --- integration with PlaybackState -------------------------------------------

def test_playback_with_lineup_attributes_events_live():
    from hoops.ui.playback import PlaybackState

    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_made", Side.HOME, home_score=2, detail="rim"),
        _ev(1, 0, "quarter_end"),
    ]
    pb = PlaybackState.from_events(events, lineup=ls)
    pb.step_one()  # tip
    pb.step_one()  # shot
    # The applied event in the playback's events tuple should now have a player.
    assert pb.events[1].player is not None
    assert pb.events[1].player.startswith("Home_P")


def test_playback_without_lineup_leaves_events_untouched():
    """Backwards compat: existing tests pass events without a lineup."""
    from hoops.ui.playback import PlaybackState

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_made", Side.HOME, home_score=2, detail="rim"),
    ]
    pb = PlaybackState.from_events(events)  # no lineup
    pb.step_one()
    pb.step_one()
    assert pb.events[1].player is None  # unchanged


# --- box-score: assist / block / steal accumulation --------------------------

# --- pending substitutions + dead-ball commit ------------------------------


def test_request_substitution_queues_without_changing_actual():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.request_substitution(Side.HOME, off_player_id=1, on_player_id=6)
    assert ls.has_pending(Side.HOME)
    actual = {p.player_id for p in ls.on_court(Side.HOME)}
    assert 1 in actual and 6 not in actual
    pending = {p.player_id for p in ls.pending_on_court(Side.HOME)}
    assert 1 not in pending and 6 in pending


def test_request_substitution_chains_pending_subs():
    """P1 → P6, then P6 → P7 should resolve to P7 on the floor (and 1, 6 off)."""
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.request_substitution(Side.HOME, off_player_id=1, on_player_id=6)
    ls.request_substitution(Side.HOME, off_player_id=6, on_player_id=7)
    pending = {p.player_id for p in ls.pending_on_court(Side.HOME)}
    assert 7 in pending and 1 not in pending and 6 not in pending


def test_request_rejects_player_not_in_pending():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    with pytest.raises(LineupError):
        ls.request_substitution(Side.HOME, off_player_id=99, on_player_id=6)


def test_commit_pending_subs_applies_queue():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.request_substitution(Side.HOME, 1, 6)
    ls.commit_pending_subs()
    assert not ls.has_pending(Side.HOME)
    actual = {p.player_id for p in ls.on_court(Side.HOME)}
    assert 1 not in actual and 6 in actual


def test_discard_pending_subs():
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.request_substitution(Side.HOME, 1, 6)
    ls.discard_pending_subs(Side.HOME)
    assert not ls.has_pending(Side.HOME)
    pending = [p.player_id for p in ls.pending_on_court(Side.HOME)]
    assert pending == [1, 2, 3, 4, 5]


def test_playback_commits_pending_subs_on_foul():
    from hoops.ui.playback import PlaybackState

    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.request_substitution(Side.HOME, 1, 6)

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "foul_personal", Side.AWAY),
    ]
    pb = PlaybackState.from_events(events, lineup=ls)
    pb.step_one()  # tip
    assert ls.has_pending(Side.HOME)  # not yet committed
    pb.step_one()  # foul → dead ball
    assert not ls.has_pending(Side.HOME)
    on_court = {p.player_id for p in ls.on_court(Side.HOME)}
    assert 1 not in on_court and 6 in on_court


def test_playback_commits_on_non_steal_turnover():
    from hoops.ui.playback import PlaybackState

    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.request_substitution(Side.HOME, 1, 6)

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "turnover", Side.HOME),
        # No 'steal' follow-up — bad pass / out of bounds, dead ball.
        _ev(1, 540, "shot_made", Side.AWAY, away_score=2, detail="rim"),
    ]
    pb = PlaybackState.from_events(events, lineup=ls)
    pb.step_one()  # tip
    pb.step_one()  # turnover (no steal next)
    assert not ls.has_pending(Side.HOME)


def test_playback_holds_pending_on_steal_turnover():
    """Steal turnover is a live ball; pending subs must wait."""
    from hoops.ui.playback import PlaybackState

    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.request_substitution(Side.HOME, 1, 6)

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "turnover", Side.HOME),
        _ev(1, 580, "steal", Side.AWAY),
        _ev(1, 560, "shot_made", Side.AWAY, away_score=2, detail="rim"),
        _ev(1, 540, "foul_personal", Side.HOME),  # eventual dead ball
    ]
    pb = PlaybackState.from_events(events, lineup=ls)
    pb.step_one()  # tip
    pb.step_one()  # turnover (live ball — steal next)
    assert ls.has_pending(Side.HOME)
    pb.step_one()  # steal credit
    assert ls.has_pending(Side.HOME)
    pb.step_one()  # made shot — also live ball per current spec
    assert ls.has_pending(Side.HOME)
    pb.step_one()  # foul → dead ball
    assert not ls.has_pending(Side.HOME)


def test_is_dead_ball_after_foul():
    from hoops.ui.playback import PlaybackState

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "foul_personal", Side.AWAY),
    ]
    pb = PlaybackState.from_events(events)
    pb.step_one()  # tip
    assert pb.is_dead_ball is False
    pb.step_one()  # foul
    assert pb.is_dead_ball is True


def test_is_dead_ball_after_non_steal_turnover():
    from hoops.ui.playback import PlaybackState

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "turnover", Side.HOME),
        _ev(1, 540, "shot_made", Side.AWAY, away_score=2, detail="rim"),
    ]
    pb = PlaybackState.from_events(events)
    pb.step_one()  # tip
    pb.step_one()  # turnover (next is a made shot, not a steal)
    assert pb.is_dead_ball is True


def test_is_dead_ball_clears_on_next_event():
    """After the dead ball, the next applied event returns the game to a
    live state. b shouldn't keep working forever after one foul."""
    from hoops.ui.playback import PlaybackState

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "foul_personal", Side.AWAY),
        _ev(1, 580, "free_throw_made", Side.HOME, home_score=1),
    ]
    pb = PlaybackState.from_events(events)
    pb.step_one()  # tip
    pb.step_one()  # foul (dead ball)
    assert pb.is_dead_ball is True
    pb.step_one()  # FT (not a dead ball event)
    assert pb.is_dead_ball is False


def test_is_dead_ball_false_after_steal_turnover():
    from hoops.ui.playback import PlaybackState

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "turnover", Side.HOME),
        _ev(1, 580, "steal", Side.AWAY),
    ]
    pb = PlaybackState.from_events(events)
    pb.step_one()  # tip
    pb.step_one()  # turnover (next is a steal credit → live ball)
    assert pb.is_dead_ball is False


def test_substitute_immediate_keeps_shadow_in_sync():
    """Direct ``substitute()`` (immediate apply) shouldn't desync the shadow."""
    h = _roster(1, "Home", n=8)
    a = _roster(2, "Away", n=8)
    rng = np.random.default_rng(0)
    ls = LineupState.with_default_starters(h, a, rng)
    ls.substitute(Side.HOME, 1, 6)
    assert not ls.has_pending(Side.HOME)
    actual = {p.player_id for p in ls.on_court(Side.HOME)}
    pending = {p.player_id for p in ls.pending_on_court(Side.HOME)}
    assert actual == pending


def test_box_score_tracks_credit_events():
    from hoops.ui.playback import PlaybackState

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_made", Side.HOME, detail="rim", home_score=2),
        _ev(1, 580, "assist", Side.HOME),
        _ev(1, 560, "shot_missed", Side.AWAY, detail="three"),
        _ev(1, 560, "block", Side.HOME),
        _ev(1, 540, "turnover", Side.AWAY),
        _ev(1, 540, "steal", Side.HOME),
    ]
    pb = PlaybackState.from_events(events)
    while not pb.is_done:
        pb.step_one()
    assert pb.home_box.ast == 1
    assert pb.home_box.blk == 1
    assert pb.home_box.stl == 1


def test_player_box_score_accumulates():
    from hoops.ui.playback import PlaybackState

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 580, "shot_made", Side.HOME, detail="three", home_score=3, player="Alice"),
        _ev(1, 580, "assist", Side.HOME, player="Beth"),
        _ev(1, 560, "shot_missed", Side.AWAY, detail="rim", player="Cora"),
        _ev(1, 555, "rebound_def", Side.HOME, player="Alice"),
        _ev(1, 540, "shot_made", Side.HOME, detail="rim", home_score=5, player="Alice"),
        _ev(1, 530, "foul_shooting", Side.AWAY, player="Cora"),
        _ev(1, 530, "free_throw_made", Side.HOME, home_score=6, player="Alice"),
        _ev(1, 530, "free_throw_missed", Side.HOME, player="Alice"),
        _ev(1, 520, "turnover", Side.HOME, player="Beth"),
        _ev(1, 520, "steal", Side.AWAY, player="Dana"),
        _ev(1, 0, "quarter_end", None, home_score=6, away_score=0),
    ]
    pb = PlaybackState.from_events(events)
    while not pb.is_done:
        pb.step_one()

    alice = pb.home_players["Alice"]
    assert alice.points == 6
    assert alice.fgm == 2 and alice.fga == 2
    assert alice.fg3m == 1 and alice.fg3a == 1
    assert alice.ftm == 1 and alice.fta == 2
    assert alice.drb == 1
    assert alice.reb == 1

    beth = pb.home_players["Beth"]
    assert beth.ast == 1
    assert beth.tov == 1
    assert beth.points == 0

    cora = pb.away_players["Cora"]
    assert cora.fga == 1 and cora.fgm == 0
    assert cora.pf == 1

    dana = pb.away_players["Dana"]
    assert dana.stl == 1


def test_player_minutes_tracked_with_lineup():
    from hoops.ui.playback import PlaybackState

    hr = _roster(1, "Home")
    ar = _roster(2, "Away")
    rng = np.random.default_rng(42)
    lineup = LineupState.with_default_starters(hr, ar, rng)
    starters = [p.name for p in lineup.on_court(Side.HOME)]

    events = [
        _ev(1, 600, "tip_off", Side.HOME),
        _ev(1, 500, "shot_made", Side.HOME, detail="rim", home_score=2, player=starters[0]),
        _ev(1, 400, "shot_missed", Side.AWAY, detail="rim"),
        _ev(1, 300, "rebound_def", Side.HOME),
        _ev(1, 0, "quarter_end", None, home_score=2, away_score=0),
    ]
    pb = PlaybackState.from_events(events, lineup=lineup)
    while not pb.is_done:
        pb.step_one()

    for name in starters:
        assert pb.home_players[name].seconds == pytest.approx(600, abs=1)

    bench_names = [p.name for p in hr.players[5:]]
    for name in bench_names:
        if name in pb.home_players:
            assert pb.home_players[name].seconds == 0.0
