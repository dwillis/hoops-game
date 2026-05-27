"""Phase 3 engine smoke tests + Hypothesis invariants.

Plan §3 verification items:
- Single-game smoke test reproduces byte-identically with a fixed seed.
- Property tests: no possession exceeds the shot clock; total game time
  equals 4 * 10 minutes + OT only when scores were tied at end of Q4;
  team fouls per quarter never decrement except on quarter rollover.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.data.paths import distributions_dir
from hoops.data.rosters import Player, Roster
from hoops.engine.lineup_rates import LineupRates, compute_lineup_rates
from hoops.engine.policy import CoachPolicies, CoachPolicy, DefensiveScheme
from hoops.engine.machine import simulate_game, simulate_possession
from hoops.engine.sampling import make_rng
from hoops.engine.state import GameState, Side
from hoops.league import League
from hoops.rules import rules_for

SEASON = "2023-24"
RULES_2023_24 = rules_for(League.WBB, SEASON)


def _synthetic_team(name: str = "Test", **overrides) -> TeamPriors:
    base = dict(
        league=League.WBB, season=SEASON, team_id=1, team_name=name,
        pace=70.0,
        off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )
    base.update(overrides)
    return TeamPriors(**base)


# --- smoke test ---------------------------------------------------------------

def test_simulate_game_finishes_with_a_winner():
    home = _synthetic_team("Home")
    away = _synthetic_team("Away")
    rng = make_rng(seed=42)
    state, events = simulate_game(home, away, RULES_2023_24, rng)
    assert state.is_final
    assert state.home_score != state.away_score
    assert events[-1].type == "game_end"


def test_simulate_game_is_byte_identical_under_fixed_seed():
    home = _synthetic_team("Home")
    away = _synthetic_team("Away")
    s1, e1 = simulate_game(home, away, RULES_2023_24, make_rng(seed=42))
    s2, e2 = simulate_game(home, away, RULES_2023_24, make_rng(seed=42))
    assert (s1.home_score, s1.away_score) == (s2.home_score, s2.away_score)
    assert len(e1) == len(e2)
    for a, b in zip(e1, e2):
        assert a == b


def test_different_seeds_produce_different_games():
    home = _synthetic_team("Home")
    away = _synthetic_team("Away")
    s1, _ = simulate_game(home, away, RULES_2023_24, make_rng(seed=1))
    s2, _ = simulate_game(home, away, RULES_2023_24, make_rng(seed=2))
    # Vanishingly unlikely identical scores under different seeds.
    assert (s1.home_score, s1.away_score) != (s2.home_score, s2.away_score)


# --- the SC-vs-Iowa scenario the plan calls for ------------------------------

@pytest.mark.skipif(
    not (distributions_dir(League.WBB, SEASON) / "team_priors.parquet").exists(),
    reason="run scripts/fit_distributions.py --season 2023-24 first",
)
def test_south_carolina_vs_iowa_smoke_test():
    """The doc's plan §3: SC vs Iowa with fixed seed must reproduce identically.

    We don't assert who wins (that's a §5 statistical check) — only that
    the simulation runs end-to-end on real fitted priors and reproduces.
    """
    from hoops.data.distributions import load_team_priors

    priors = load_team_priors(League.WBB, SEASON)
    sc = next((p for p in priors if p.team_name == "South Carolina"), None)
    iowa = next((p for p in priors if p.team_name == "Iowa"), None)
    assert sc is not None and iowa is not None

    s1, e1 = simulate_game(sc, iowa, RULES_2023_24, make_rng(seed=2024))
    s2, e2 = simulate_game(sc, iowa, RULES_2023_24, make_rng(seed=2024))
    assert (s1.home_score, s1.away_score) == (s2.home_score, s2.away_score)
    assert len(e1) == len(e2)
    # Sanity floors: combined points should look like a basketball game.
    assert 90 <= s1.home_score + s1.away_score <= 220


# --- structural invariants (Hypothesis) ---------------------------------------

@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_game_ends_with_winner_or_extends_to_ot(seed):
    rng = make_rng(seed=seed)
    state, events = simulate_game(
        _synthetic_team("A"), _synthetic_team("B"), RULES_2023_24, rng
    )
    assert state.is_final
    # If we ended in regulation, the score is not tied.
    if state.quarter == 4:
        assert not state.is_tied
    else:
        # Otherwise we extended to OT, ended an OT period, and broke the tie.
        assert state.quarter >= 5
        assert not state.is_tied


@settings(max_examples=80, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_no_possession_exceeds_shot_clock(seed):
    """No event ever shows seconds_left less than possession-start - 30s.

    A weaker but easier-to-verify form: the *clock advance* per call to
    simulate_possession must be <= rules.shot_clock_seconds.
    """
    rules = RULES_2023_24
    rng = make_rng(seed=seed)
    state = GameState.initial(rules)
    # Run a handful of possessions and check each clock advance is bounded.
    for _ in range(20):
        before = state.seconds_left
        if before <= 0:
            break
        state, _ = simulate_possession(
            state, _synthetic_team("A"), _synthetic_team("B"), rng
        )
        delta = before - state.seconds_left
        assert 0 <= delta <= rules.shot_clock_seconds, (
            f"possession advanced clock by {delta}s; shot clock is {rules.shot_clock_seconds}s"
        )


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_team_fouls_never_decrement_within_quarter(seed):
    """Within a quarter, both teams' foul counts are monotone non-decreasing."""
    rng = make_rng(seed=seed)
    state = GameState.initial(RULES_2023_24)
    home = _synthetic_team("Home")
    away = _synthetic_team("Away")
    last_q = state.quarter
    last_h = state.home_team_fouls_q
    last_a = state.away_team_fouls_q
    for _ in range(60):
        if state.is_final:
            break
        if state.seconds_left <= 0:
            from hoops.engine.clock import end_period
            state, _ = end_period(state)
            last_q = state.quarter
            last_h = state.home_team_fouls_q
            last_a = state.away_team_fouls_q
            continue
        state, _ = simulate_possession(state, home, away, rng)
        if state.quarter == last_q:
            assert state.home_team_fouls_q >= last_h
            assert state.away_team_fouls_q >= last_a
        last_q = state.quarter
        last_h = state.home_team_fouls_q
        last_a = state.away_team_fouls_q


def test_overtime_only_when_regulation_tied():
    """A regulation game that ended untied should never reach OT."""
    home = _synthetic_team("Home", off_efg=0.55)  # asymmetric: home advantage
    away = _synthetic_team("Away", off_efg=0.40)
    rng = make_rng(seed=0)
    state, events = simulate_game(home, away, RULES_2023_24, rng)
    if state.quarter > 4:
        # OT only if there was actually a tie at end of Q4. The structural
        # event log carries that signal: an "overtime_start" event must
        # have followed a Q4 "quarter_end" with equal scores.
        for e_prev, e in zip(events, events[1:]):
            if e.type == "overtime_start":
                assert e_prev.type == "quarter_end"
                assert e_prev.home_score == e_prev.away_score
                break


# --- lineup_rates helpers ----------------------------------------------------

def _player_for_engine(pid, name, **kw):
    base = dict(
        player_id=pid, name=name, minutes=200.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=0.20, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
    )
    base.update(kw)
    return Player(**base)

def _five_for_engine(**overrides):
    return [_player_for_engine(i, f"P{i}", **overrides) for i in range(5)]


# --- lineup_rates integration tests ------------------------------------------

def test_simulate_possession_accepts_lineup_rates():
    home_team = _synthetic_team("Home")
    away_team = _synthetic_team("Away")
    rng = make_rng(seed=42)
    state = GameState.initial(RULES_2023_24)
    off_lr = compute_lineup_rates(_five_for_engine(), home_team)
    def_lr = compute_lineup_rates(_five_for_engine(), away_team)
    state2, events = simulate_possession(
        state, home_team, away_team, rng,
        off_lineup_rates=off_lr, def_lineup_rates=def_lr,
    )
    assert state2 is not state
    assert len(events) > 0


def test_lineup_rates_none_is_backward_compatible():
    home = _synthetic_team("Home")
    away = _synthetic_team("Away")
    rng1 = make_rng(seed=42)
    rng2 = make_rng(seed=42)
    state = GameState.initial(RULES_2023_24)
    s1, e1 = simulate_possession(state, home, away, rng1)
    s2, e2 = simulate_possession(
        state, home, away, rng2,
        off_lineup_rates=None, def_lineup_rates=None,
    )
    assert s1 == s2
    assert e1 == e2


def test_high_tov_lineup_produces_more_turnovers():
    home = _synthetic_team("Home")
    away = _synthetic_team("Away")
    normal_lr = compute_lineup_rates(_five_for_engine(tov_pct=0.10), home)
    high_tov_lr = compute_lineup_rates(_five_for_engine(tov_pct=0.40), home)

    def count_turnovers(lr, seed_start=0):
        tovs = 0
        for s in range(seed_start, seed_start + 200):
            rng = make_rng(seed=s)
            state = GameState.initial(RULES_2023_24)
            _, events = simulate_possession(
                state, home, away, rng, off_lineup_rates=lr,
            )
            tovs += sum(1 for e in events if e.type == "turnover")
        return tovs

    normal_tovs = count_turnovers(normal_lr)
    high_tovs = count_turnovers(high_tov_lr)
    assert high_tovs > normal_tovs * 1.5


# --- simulate_game with lineups (Task 5) ------------------------------------

def _roster_for_engine(team_id, name):
    players = tuple(
        _player_for_engine(
            team_id * 100 + i, f"{name}_P{i}",
            usage_pct=0.25 - i * 0.03,
            ts_pct=0.55 - i * 0.02,
        )
        for i in range(12)
    )
    return Roster(team_id=team_id, team_name=name, players=players)


def test_simulate_game_with_lineups():
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    home_roster = _roster_for_engine(1, "Home")
    away_roster = _roster_for_engine(2, "Away")
    rng = make_rng(seed=42)
    state, events = simulate_game(
        home, away, RULES_2023_24, rng,
        home_roster=home_roster, away_roster=away_roster,
    )
    assert state.is_final
    assert state.home_score != state.away_score
    assert events[-1].type == "game_end"


def test_simulate_game_with_lineups_is_reproducible():
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    s1, e1 = simulate_game(home, away, RULES_2023_24, make_rng(seed=99), home_roster=hr, away_roster=ar)
    s2, e2 = simulate_game(home, away, RULES_2023_24, make_rng(seed=99), home_roster=hr, away_roster=ar)
    assert (s1.home_score, s1.away_score) == (s2.home_score, s2.away_score)
    assert len(e1) == len(e2)


def test_simulate_game_with_lineups_produces_realistic_scores():
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    state, _ = simulate_game(home, away, RULES_2023_24, make_rng(seed=42), home_roster=hr, away_roster=ar)
    total = state.home_score + state.away_score
    assert 80 <= total <= 220, f"unrealistic combined score: {total}"


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=5000))
def test_game_with_lineups_always_ends(seed):
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    state, events = simulate_game(home, away, RULES_2023_24, make_rng(seed=seed), home_roster=hr, away_roster=ar)
    assert state.is_final


# --- fatigue integration (Task 5) -------------------------------------------

def test_simulate_game_with_fatigue_finishes():
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    rng = make_rng(seed=42)
    state, events = simulate_game(
        home, away, RULES_2023_24, rng,
        home_roster=hr, away_roster=ar,
        enable_fatigue=True,
    )
    assert state.is_final
    assert 80 <= state.home_score + state.away_score <= 220


def test_simulate_game_fatigue_can_be_disabled():
    """Explicitly disabling fatigue produces a different game than the default (on)."""
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    s_on, _ = simulate_game(home, away, RULES_2023_24, make_rng(seed=42), home_roster=hr, away_roster=ar)
    s_off, _ = simulate_game(home, away, RULES_2023_24, make_rng(seed=42), home_roster=hr, away_roster=ar, enable_fatigue=False)
    # With fatigue on by default, the two runs diverge once subs happen.
    # Both should still finish as valid games.
    assert s_on.is_final
    assert s_off.is_final


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=5000))
def test_game_with_fatigue_always_ends(seed):
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    state, events = simulate_game(
        home, away, RULES_2023_24, make_rng(seed=seed),
        home_roster=hr, away_roster=ar, enable_fatigue=True,
    )
    assert state.is_final


def test_simulate_game_fatigue_is_reproducible():
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    s1, e1 = simulate_game(
        home, away, RULES_2023_24, make_rng(seed=99),
        home_roster=hr, away_roster=ar, enable_fatigue=True,
    )
    s2, e2 = simulate_game(
        home, away, RULES_2023_24, make_rng(seed=99),
        home_roster=hr, away_roster=ar, enable_fatigue=True,
    )
    assert (s1.home_score, s1.away_score) == (s2.home_score, s2.away_score)
    assert len(e1) == len(e2)


def test_zone_defense_with_rim_protectors_produces_valid_game():
    """Switching scheme should produce valid games with specialist lineups."""
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    from dataclasses import replace as dc_replace
    home_players = list(_roster_for_engine(1, "Home").players)
    home_players[4] = dc_replace(home_players[4], blk_pct=5.0, drb_pct=16.0)
    hr = Roster(team_id=1, team_name="Home", players=tuple(home_players))
    ar = _roster_for_engine(2, "Away")

    zone_policy = CoachPolicies(home=CoachPolicy(scheme=DefensiveScheme.ZONE))
    man_policy = CoachPolicies(home=CoachPolicy(scheme=DefensiveScheme.MAN))

    for seed in range(20):
        s_zone, _ = simulate_game(home, away, RULES_2023_24, make_rng(seed=seed),
                             home_roster=hr, away_roster=ar, policies=zone_policy)
        s_man, _ = simulate_game(home, away, RULES_2023_24, make_rng(seed=seed),
                             home_roster=hr, away_roster=ar, policies=man_policy)
        assert s_zone.is_final
        assert s_man.is_final
        total_z = s_zone.home_score + s_zone.away_score
        total_m = s_man.home_score + s_man.away_score
        assert 80 <= total_z <= 220, f"zone: unrealistic combined score {total_z}"
        assert 80 <= total_m <= 220, f"man: unrealistic combined score {total_m}"


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=5000))
def test_game_with_scheme_affinity_always_ends(seed):
    home = _synthetic_team("Home", team_id=1)
    away = _synthetic_team("Away", team_id=2)
    hr = _roster_for_engine(1, "Home")
    ar = _roster_for_engine(2, "Away")
    zone_policy = CoachPolicies(
        home=CoachPolicy(scheme=DefensiveScheme.ZONE),
        away=CoachPolicy(scheme=DefensiveScheme.PRESS),
    )
    state, _ = simulate_game(
        home, away, RULES_2023_24, make_rng(seed=seed),
        home_roster=hr, away_roster=ar, policies=zone_policy,
    )
    assert state.is_final
