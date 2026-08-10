"""Tests for the double-team knob."""

from __future__ import annotations

from hoops.data.distributions import ShotMix, TeamPriors, ZoneEFG
from hoops.data.rosters import Player, Roster
from hoops.engine.attribution import attribute_players
from hoops.engine.lineup_rates import compute_lineup_rates
from hoops.engine.machine import simulate_possession
from hoops.engine.policy import CoachPolicies, CoachPolicy
from hoops.engine.sampling import make_rng
from hoops.engine.state import GameState, Side
from hoops.league import League
from hoops.rules import rules_for

RULES = rules_for(League.WBB, "2023-24")


def _team() -> TeamPriors:
    return TeamPriors(
        league=League.WBB, season="2023-24", team_id=1, team_name="T",
        pace=70, off_efg=0.45, off_tov_pct=0.18, off_orb_pct=0.30,
        off_fta_rate=0.30, off_3pt_rate=0.30, off_ft_pct=0.70,
        def_efg=0.45, def_tov_pct=0.18, def_orb_pct=0.30, def_fta_rate=0.30,
        shot_mix=ShotMix(rim=0.35, mid=0.30, three=0.35),
        zone_efg=ZoneEFG(rim=0.55, mid=0.35, three=0.32),
        foul_rate_per_100=20.0,
    )


def _player(pid, usage=0.20):
    return Player(
        player_id=pid, name=f"P{pid}", minutes=400.0,
        fga=100, fg3a=30, fta=40, orb=15, drb=50,
        fouls=20, tov=15, ast=30, blk=5, stl=10,
        usage_pct=usage, ts_pct=0.52, fg3a_share=0.30,
        ft_pct=0.75, tov_pct=0.15, orb_pct=2.0,
        drb_pct=8.0, stl_pct=2.5, blk_pct=0.8, foul_rate=3.0,
    )


def _five_with_star():
    # P1 is the clear top option.
    return [_player(1, usage=0.40)] + [_player(i, usage=0.15) for i in range(2, 6)]


def _run(double_team_pct: float, n=800):
    """Run n one-possession sims with HOME on offense; return event lists."""
    team = _team()
    players = _five_with_star()
    off_lr = compute_lineup_rates(players, team)
    results = []
    for seed in range(n):
        rng = make_rng(seed)
        state = GameState.initial(RULES, opening_possession=Side.HOME)
        policies = CoachPolicies(
            home=CoachPolicy(),
            away=CoachPolicy(double_team_pct=double_team_pct),  # defense doubles
        )
        _st, evs = simulate_possession(
            state, team, team, rng, policies=policies,
            off_lineup_rates=off_lr, def_lineup_rates=off_lr,
        )
        results.append(evs)
    return results


def test_no_rng_draw_at_zero_pct():
    # With double_team_pct=0.0, RNG state after a possession must match a run
    # with the field entirely absent (no extra draw consumed).
    team = _team()
    players = _five_with_star()
    off_lr = compute_lineup_rates(players, team)

    def run_once(policies):
        rng = make_rng(42)
        state = GameState.initial(RULES, opening_possession=Side.HOME)
        simulate_possession(
            state, team, team, rng, policies=policies,
            off_lineup_rates=off_lr, def_lineup_rates=off_lr,
        )
        return rng.bit_generator.state

    s_zero = run_once(CoachPolicies(home=CoachPolicy(),
                                    away=CoachPolicy(double_team_pct=0.0)))
    s_default = run_once(CoachPolicies())
    assert s_zero == s_default


def test_fired_double_team_tags_turnover_detail():
    results = _run(1.0)  # always double
    tovs = [e for evs in results for e in evs if e.type == "turnover"]
    assert tovs, "expected some turnovers"
    assert all(e.detail == "double_team" for e in tovs)


def test_double_team_raises_turnovers():
    base = _run(0.0)
    doubled = _run(1.0)
    base_tovs = sum(1 for evs in base for e in evs if e.type == "turnover")
    dt_tovs = sum(1 for evs in doubled for e in evs if e.type == "turnover")
    assert dt_tovs > base_tovs


def test_double_team_shifts_shots_off_the_star():
    base = _run(0.0)
    doubled = _run(1.0)

    # Attribute both runs so we can see who shot.
    def star_fraction(results):
        roster = Roster(team_id=1, team_name="T",
                        players=tuple(_five_with_star()))
        star = 0
        total = 0
        for evs in results:
            rng = make_rng(1234)
            attributed = attribute_players(evs, roster, roster, rng)
            for e in attributed:
                if e.type in ("shot_made", "shot_missed") and e.team is Side.HOME:
                    total += 1
                    if e.player == "P1":
                        star += 1
        return star / total if total else 0.0

    assert star_fraction(doubled) < star_fraction(base)


def test_attribution_raises_steal_prob_on_double_team():
    from hoops.engine.events import Event

    roster = Roster(team_id=1, team_name="T", players=tuple(_five_with_star()))
    # Many double-team turnovers; count how often a steal is credited.
    steals = 0
    n = 2000
    for seed in range(n):
        rng = make_rng(seed)
        ev = Event(quarter=1, seconds_left=300, type="turnover",
                   team=Side.HOME, detail="double_team")
        out = attribute_players([ev], roster, roster, rng)
        if any(e.type == "steal" for e in out):
            steals += 1
    # Should be ~0.75, clearly above the base 0.50.
    assert steals / n > 0.65


def test_noop_without_lineup_rates():
    # Team-prior-only path: double_team_pct set but no lineup -> no crash,
    # no double-team tag.
    team = _team()
    rng = make_rng(3)
    state = GameState.initial(RULES, opening_possession=Side.HOME)
    policies = CoachPolicies(home=CoachPolicy(),
                             away=CoachPolicy(double_team_pct=1.0))
    _st, evs = simulate_possession(state, team, team, rng, policies=policies)
    for e in evs:
        if e.type == "turnover":
            assert e.detail != "double_team"
