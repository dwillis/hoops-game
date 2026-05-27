"""Possession state machine and game loop.

Each call to :func:`simulate_possession` advances the clock by a sampled
duration, draws a possession outcome from the offensive team's priors,
and (where applicable) draws follow-up events: rebounds after misses,
free throws after fouls. The function returns a new ``GameState`` and a
flat list of ``Event``s describing what happened.

The model is deliberately small for engine v0:

- Possession ends in turnover, free-throw trip, or shot attempt.
- Shot attempts pick a zone from the team's ``shot_mix`` and roll a
  Bernoulli on ``zone_efg``.
- A miss draws an offensive vs defensive rebound from offensive ``orb_pct``;
  ORBs do *not* recurse in v0 — they bump score-state only via the count
  of FGAs, and the next possession-shaped event is sampled fresh. (This
  produces the right pace and four-factor marginals; possession-extending
  putbacks are a v1 refinement.)
- Free-throw trips draw 2 FTs at the team's ``off_ft_pct``. 1-and-1 is
  unsupported by the WBB 2015-16+ ruleset (doc §1.2).
- A defensive foul that's not on a shot increments the opposing team's
  per-quarter fouls; if it pushes them into the bonus, that fact will
  affect *future* possessions through ``is_in_bonus``.
"""

from __future__ import annotations

import numpy as np

from hoops.data.distributions import LeaguePrior, TeamPriors
from hoops.data.rosters import Roster
from hoops.engine.clock import end_period
from hoops.engine.fatigue import FatigueTracker, check_substitutions
from hoops.engine.events import Event
from hoops.engine.fouls import is_in_bonus
from hoops.engine.lineup_rates import (
    LineupRates, compute_lineup_rates,
    sample_shooter, player_shot_zone, player_zone_make_prob,
)
from hoops.engine.matchup import adjust_offense, apply_hca, apply_off_scheme, apply_scheme
from hoops.engine.policy import CoachPolicies, CoachPolicy
from hoops.engine.state import GameState, Side
from hoops.rules import Rules
from hoops.ui.lineup import LineupState

def _star_player_ids(roster: Roster) -> set[int]:
    from hoops.engine.fatigue import player_importance
    ranked = sorted(roster.players, key=player_importance, reverse=True)
    return {p.player_id for p in ranked[:2]}


def _player_name(
    player_id: int, side: Side,
    home_roster: Roster | None, away_roster: Roster | None,
) -> str:
    roster = home_roster if side is Side.HOME else away_roster
    if roster is not None:
        for p in roster.players:
            if p.player_id == player_id:
                return p.name
    return f"#{player_id}"


# Mean offensive possession length (seconds) at pace=70 over 40 minutes.
# 40min * 60 / (2 * pace) gives mean per offensive possession assuming
# both teams alternate. We sample uniform around this mean so the shot
# clock is rarely exhausted but never exceeded.
_MIN_POSS_SECONDS = 4
_MAX_POSS_SECONDS = 30  # the WBB shot clock

# End-of-quarter strategy thresholds (seconds left in the quarter).
_TWO_FOR_ONE_WINDOW = (35, 50)  # if poss starts here, target ~17s shot
_TWO_FOR_ONE_TARGET = 17
_HOLD_FOR_LAST_THRESHOLD = 30   # if ≤ this much left, hold the ball
_FOUL_UP_3_WINDOW = 12          # last N seconds when foul-up-3 fires
_INTENTIONAL_FOUL_BONUS_WINDOW = 25


def _sample_possession_seconds(
    off: TeamPriors,
    def_: TeamPriors,
    state: GameState,
    off_policy: CoachPolicy,
    rng: np.random.Generator,
    pace_adj: float = 0.0,
) -> int:
    """Sample a possession length, biased by end-of-quarter strategy.

    The base draw is a triangular around the matchup's mean possession
    time. Two policy hooks adjust it:

    - **two_for_one**: if the possession starts in the 35-50s window of
      the quarter, target ~17s so the offense can also get a return
      possession before the buzzer.
    - **hold_for_last**: if ≤30s remain, take the clock down to the
      shot-clock limit (or buzzer, whichever is sooner) — this is the
      "no return possession" final shot.
    """
    pace = 0.5 * (off.pace + def_.pace) + pace_adj
    mean = 40 * 60 / (2 * pace)
    lo = max(_MIN_POSS_SECONDS, mean - 6)
    hi = min(_MAX_POSS_SECONDS, mean + 8)
    base = int(rng.triangular(lo, mean, hi))

    secs = state.seconds_left
    if (
        off_policy.two_for_one
        and _TWO_FOR_ONE_WINDOW[0] <= secs <= _TWO_FOR_ONE_WINDOW[1]
    ):
        # Compress to roughly the target so we get the ball back.
        compressed = max(_MIN_POSS_SECONDS, min(base, _TWO_FOR_ONE_TARGET))
        return compressed
    if off_policy.hold_for_last and secs <= _HOLD_FOR_LAST_THRESHOLD:
        # Burn the clock to (just under) the shot clock or the buzzer.
        return min(secs, _MAX_POSS_SECONDS - 2)
    return max(_MIN_POSS_SECONDS, min(_MAX_POSS_SECONDS, base))


def _sample_zone(off: TeamPriors, rng: np.random.Generator) -> str:
    p = [off.shot_mix.rim, off.shot_mix.mid, off.shot_mix.three]
    # Defensive normalization: parquet rounding can leave the sum ~1±1e-9.
    total = sum(p)
    p = [x / total for x in p]
    return ["rim", "mid", "three"][rng.choice(3, p=p)]


def _zone_points(zone: str) -> int:
    return 3 if zone == "three" else 2


def _zone_make_prob(off: TeamPriors, zone: str) -> float:
    return {"rim": off.zone_efg.rim, "mid": off.zone_efg.mid, "three": off.zone_efg.three}[zone]


def _shot_foul_prob(off: TeamPriors) -> float:
    """Per-shot probability of a shooting foul on the offense's attempt.

    With v0's "all FTAs come from shooting fouls" simplification, expected
    FTAs per shot attempt should match the team's FTR. Each foul produces:
    - 1 FT on a make (and-1)
    - 2 FT on a missed 2-point attempt
    - 3 FT on a missed 3-point attempt

    Mix-weighted, this averages to ~1.6-1.8 FTs per foul. We divide by the
    *team-specific* expectation so the simulated FTR matches the input.
    """
    e_per_foul = (
        off.shot_mix.rim * (2 - off.zone_efg.rim)
        + off.shot_mix.mid * (2 - off.zone_efg.mid)
        + off.shot_mix.three * (3 - 2 * off.zone_efg.three)
    )
    if e_per_foul <= 0:
        return 0.0
    return min(0.25, off.off_fta_rate / e_per_foul)


def _sample_two_free_throws(off: TeamPriors, rng: np.random.Generator) -> tuple[int, list[bool]]:
    makes = (rng.random(2) < off.off_ft_pct).tolist()
    return sum(makes), makes


def _should_intentionally_foul(
    state: GameState,
    off_side: Side,
    def_policy: CoachPolicy,
) -> str | None:
    """Decide whether the defense fouls before the offense can score.

    Returns the rationale string (used as the event detail) or None.
    Two cases:

    - **foul_when_down_3**: defense is down by 3 in Q4 with ≤12s left.
      Foul to send the leading offense to the line, hope they miss, get
      the ball back for a tying 3.
    - **intentional_foul_in_bonus_when_trailing**: defense is trailing
      late in Q4 and the offense is already in the bonus; foul off-ball
      to stop the clock.
    """
    if state.quarter < 4:
        return None
    score_diff = state.score_for(off_side) - state.score_for(off_side.other)
    if (
        def_policy.foul_when_down_3
        and score_diff == 3
        and 0 < state.seconds_left <= _FOUL_UP_3_WINDOW
    ):
        return "down 3, foul-up-3"
    if (
        def_policy.intentional_foul_in_bonus_when_trailing
        and score_diff > 0
        and 0 < state.seconds_left <= _INTENTIONAL_FOUL_BONUS_WINDOW
        and is_in_bonus(state, off_side)
    ):
        return "trailing in bonus, intentional foul"
    return None


def simulate_possession(
    state: GameState,
    home: TeamPriors,
    away: TeamPriors,
    rng: np.random.Generator,
    policies: CoachPolicies | None = None,
    off_lineup_rates: LineupRates | None = None,
    def_lineup_rates: LineupRates | None = None,
) -> tuple[GameState, list[Event]]:
    off_side = state.possession
    off = home if off_side is Side.HOME else away
    def_ = away if off_side is Side.HOME else home

    if policies is None:
        policies = CoachPolicies()
    off_policy = policies.for_side(off_side)
    def_policy = policies.for_side(off_side.other)

    # Apply schemes to the offense's effective priors.
    off = apply_scheme(off, def_policy.scheme)
    off = apply_off_scheme(off, off_policy.off_scheme)

    # Per-possession shooter (set when lineup_rates is provided).
    _shooter = None

    def _effective_ft_pct(shooter_player=None):
        if off_lineup_rates is not None and shooter_player is not None and shooter_player.ft_pct is not None:
            return shooter_player.ft_pct
        if off_lineup_rates is not None:
            return off_lineup_rates.ft_pct
        return off.off_ft_pct

    events: list[Event] = []

    # Late-game intentional foul check: fires *before* the clock advances,
    # since the foul happens immediately on the inbound.
    foul_reason = _should_intentionally_foul(state, off_side, def_policy)
    if foul_reason is not None:
        # Burn ~1 second on the inbound + foul.
        state = state.advance_clock(1)
        state = state.add_team_foul(off_side.other)
        events.append(Event(
            quarter=state.quarter, seconds_left=state.seconds_left,
            type="foul_personal", team=off_side.other,
            detail=f"intentional ({foul_reason}); team_fouls_q={state.fouls_for(off_side.other)}",
            home_score=state.home_score, away_score=state.away_score,
        ))
        # If the offense is in the bonus (always true once the defense's
        # 5th foul of the quarter is committed), shoot two free throws
        # and change possession. Otherwise the offense keeps the ball,
        # which is the wrong outcome strategically — but the engine v0
        # treats this as the foul side-effect having "worked": defense
        # got the foul on the books, offense inbounds.
        if is_in_bonus(state, off_side):
            for _ in range(2):
                made = rng.random() < _effective_ft_pct()
                if made:
                    state = state.add_score(off_side, 1)
                events.append(Event(
                    quarter=state.quarter, seconds_left=state.seconds_left,
                    type="free_throw_made" if made else "free_throw_missed",
                    team=off_side, detail=foul_reason,
                    home_score=state.home_score, away_score=state.away_score,
                ))
            state = state.end_possession(off_side).with_possession(off_side.other)
            return state, events
        # Not in bonus: offense keeps the ball; possession continues
        # below as normal but the foul is on the books for next time.

    # Cap possession time at remaining clock so shot-clock invariant holds.
    _pace_adj = off_lineup_rates.pace_adj if off_lineup_rates is not None else 0.0
    duration = min(
        _sample_possession_seconds(off, def_, state, off_policy, rng, pace_adj=_pace_adj),
        state.seconds_left,
    )
    state = state.advance_clock(duration)

    # Outcome: turnover or shot attempt. v0 routes all FTAs through
    # shooting fouls (see _shot_foul_prob); off-ball intentional fouls
    # are a Phase 6 hook that only matters once a CoachPolicy exists.
    if off_lineup_rates is not None:
        p_tov = max(0.0, min(0.5, off_lineup_rates.tov_pct))
    else:
        p_tov = max(0.0, min(0.5, off.off_tov_pct))

    roll = rng.random()
    if roll < p_tov:
        events.append(Event(
            quarter=state.quarter, seconds_left=state.seconds_left,
            type="turnover", team=off_side,
            home_score=state.home_score, away_score=state.away_score,
        ))
        state = state.end_possession(off_side).with_possession(off_side.other)
        return state, events

    # Shot attempt path.
    if off_lineup_rates is not None:
        _shooter = sample_shooter(off_lineup_rates, rng)
        zone = player_shot_zone(_shooter, off, rng)
    else:
        _shooter = None
        zone = _sample_zone(off, rng)
    points = _zone_points(zone)

    # Shooting foul check before the make/miss roll. If the defense fouls
    # on the shot, the shot still happens; if it goes in, the offense
    # gets one extra free throw, otherwise they shoot ``points``.
    shot_foul = rng.random() < _shot_foul_prob(off)
    if shot_foul:
        # Shooting foul accrues to the defense's per-quarter team fouls.
        state = state.add_team_foul(off_side.other)
        events.append(Event(
            quarter=state.quarter, seconds_left=state.seconds_left,
            type="foul_shooting", team=off_side.other,
            detail=f"on shot ({zone}); team_fouls_q={state.fouls_for(off_side.other)}",
            home_score=state.home_score, away_score=state.away_score,
        ))
    if off_lineup_rates is not None and _shooter is not None:
        made = rng.random() < player_zone_make_prob(_shooter, zone, off)
    else:
        made = rng.random() < _zone_make_prob(off, zone)

    _shooter_name = _shooter.name if _shooter is not None else None

    if made:
        state = state.add_score(off_side, points)
        events.append(Event(
            quarter=state.quarter, seconds_left=state.seconds_left,
            type="shot_made", team=off_side, detail=zone,
            home_score=state.home_score, away_score=state.away_score,
            player=_shooter_name,
        ))
        if shot_foul:
            and1 = rng.random() < _effective_ft_pct(_shooter)
            if and1:
                state = state.add_score(off_side, 1)
            events.append(Event(
                quarter=state.quarter, seconds_left=state.seconds_left,
                type="free_throw_made" if and1 else "free_throw_missed",
                team=off_side, detail="and-1",
                home_score=state.home_score, away_score=state.away_score,
                player=_shooter_name,
            ))
        state = state.end_possession(off_side).with_possession(off_side.other)
        return state, events

    # Miss.
    events.append(Event(
        quarter=state.quarter, seconds_left=state.seconds_left,
        type="shot_missed", team=off_side, detail=zone,
        home_score=state.home_score, away_score=state.away_score,
        player=_shooter_name,
    ))
    if shot_foul:
        # Fouled on a missed shot: free throws (2 for 2pt, 3 for 3pt).
        n_fts = 3 if zone == "three" else 2
        for _ in range(n_fts):
            r = rng.random() < _effective_ft_pct(_shooter)
            if r:
                state = state.add_score(off_side, 1)
            events.append(Event(
                quarter=state.quarter, seconds_left=state.seconds_left,
                type="free_throw_made" if r else "free_throw_missed",
                team=off_side,
                home_score=state.home_score, away_score=state.away_score,
                player=_shooter_name,
            ))
        state = state.end_possession(off_side).with_possession(off_side.other)
        return state, events

    # No foul: rebound roll.
    if off_lineup_rates is not None:
        _orb_p = off_lineup_rates.orb_pct
        if _orb_p > 1.0:
            _orb_p = _orb_p / 100.0
    else:
        _orb_p = off.off_orb_pct
    if rng.random() < _orb_p:
        events.append(Event(
            quarter=state.quarter, seconds_left=state.seconds_left,
            type="rebound_off", team=off_side,
            home_score=state.home_score, away_score=state.away_score,
        ))
        # In v0 the ORB ends the current possession but offense keeps the
        # ball: we count a possession (so pace is faithful to Dean Oliver's
        # FGA-based estimator) and re-enter on the next iteration.
        state = state.end_possession(off_side).with_possession(off_side)
    else:
        events.append(Event(
            quarter=state.quarter, seconds_left=state.seconds_left,
            type="rebound_def", team=off_side.other,
            home_score=state.home_score, away_score=state.away_score,
        ))
        state = state.end_possession(off_side).with_possession(off_side.other)
    return state, events


def simulate_game(
    home: TeamPriors,
    away: TeamPriors,
    rules: Rules,
    rng: np.random.Generator,
    opening_possession: Side = Side.HOME,
    league: LeaguePrior | None = None,
    policies: CoachPolicies | None = None,
    max_iters: int = 2000,
    home_roster: Roster | None = None,
    away_roster: Roster | None = None,
    lineup_state: LineupState | None = None,
    enable_fatigue: bool = True,
    neutral_site: bool = False,
) -> tuple[GameState, list[Event]]:
    """Run a full game and return the final state plus the event log.

    If ``league`` is supplied, both teams' offensive priors are matchup-
    adjusted against the opponent's defensive priors before the simulation
    starts. Without ``league``, raw priors are used (suitable for synthetic
    unit tests; not appropriate for real-team head-to-head).

    ``policies`` (Phase 6): one ``CoachPolicy`` per side. The defensive
    scheme is applied per-possession; end-of-quarter timing (2-for-1,
    hold-for-last) and late-game intentional fouls fire when the
    relevant policy flag is set.
    """
    if league is not None:
        home = adjust_offense(home, away, league)
        away = adjust_offense(away, home, league)
    if not neutral_site:
        away = apply_hca(away)
    state = GameState.initial(rules, opening_possession=opening_possession)
    events: list[Event] = [Event(
        quarter=1, seconds_left=state.seconds_left, type="tip_off",
        team=opening_possession,
        home_score=0, away_score=0,
    )]

    # Lineup-aware rate computation (Phase A).
    home_lineup_rates: LineupRates | None = None
    away_lineup_rates: LineupRates | None = None

    if lineup_state is None and home_roster is not None and away_roster is not None:
        lineup_state = LineupState.with_default_starters(home_roster, away_roster, rng)

    fatigue_tracker: FatigueTracker | None = None
    if enable_fatigue and lineup_state is not None and home_roster is not None and away_roster is not None:
        fatigue_tracker = FatigueTracker(home_roster, away_roster)

    _home_scheme = policies.for_side(Side.HOME).scheme if policies else None
    _away_scheme = policies.for_side(Side.AWAY).scheme if policies else None

    if lineup_state is not None:
        home_lineup_rates = compute_lineup_rates(
            lineup_state.on_court(Side.HOME), home,
            fatigue_tracker=fatigue_tracker, scheme=_home_scheme,
        )
        away_lineup_rates = compute_lineup_rates(
            lineup_state.on_court(Side.AWAY), away,
            fatigue_tracker=fatigue_tracker, scheme=_away_scheme,
        )

    iters = 0
    # Loop invariant: end_period is the *only* way to advance past a zero
    # clock or emit game_end. We keep looping until end_period reports the
    # game is over, even if simulate_possession just exhausted the clock.
    while iters < max_iters:
        if state.seconds_left <= 0:
            state, evs = end_period(state)
            events.extend(evs)
            if state.is_final:
                return state, events
            continue
        if home_lineup_rates is not None and away_lineup_rates is not None:
            if state.possession is Side.HOME:
                _off_lr, _def_lr = home_lineup_rates, away_lineup_rates
            else:
                _off_lr, _def_lr = away_lineup_rates, home_lineup_rates
        else:
            _off_lr, _def_lr = None, None

        state, evs = simulate_possession(
            state, home, away, rng, policies=policies,
            off_lineup_rates=_off_lr, def_lineup_rates=_def_lr,
        )
        events.extend(evs)
        iters += 1

        # --- Phase B: fatigue tick + auto-subs at dead balls ---
        if fatigue_tracker is not None and lineup_state is not None:
            poss_duration = 17.0  # average WBB possession length

            home_on = [p.player_id for p in lineup_state.on_court(Side.HOME)]
            away_on = [p.player_id for p in lineup_state.on_court(Side.AWAY)]
            fatigue_tracker.tick(home_on + away_on, poss_duration)

            home_bench = [p.player_id for p in lineup_state.bench(Side.HOME)]
            away_bench = [p.player_id for p in lineup_state.bench(Side.AWAY)]
            fatigue_tracker.rest(home_bench + away_bench, poss_duration)
            fatigue_tracker.tick_cooldowns()

            for e in evs:
                if e.type in ("foul_personal", "foul_shooting") and e.team is not None:
                    on_court_foul = lineup_state.on_court(e.team)
                    if on_court_foul:
                        fouler = max(on_court_foul, key=lambda p: p.foul_rate or 0)
                        fatigue_tracker.add_foul(fouler.player_id)

            is_dead_ball = any(
                e.type in ("shot_made", "foul_personal", "foul_shooting",
                           "free_throw_made", "free_throw_missed")
                for e in evs
            )
            if is_dead_ball:
                lineup_changed = False
                for side in (Side.HOME, Side.AWAY):
                    sub_events = check_substitutions(
                        lineup_state, fatigue_tracker, state.quarter, side
                    )
                    roster = home_roster if side is Side.HOME else away_roster
                    star_ids = _star_player_ids(roster) if roster else set()
                    for se in sub_events:
                        off_name = _player_name(se.off_player_id, se.side, home_roster, away_roster)
                        on_name = _player_name(se.on_player_id, se.side, home_roster, away_roster)
                        lineup_state.substitute(se.side, se.off_player_id, se.on_player_id)
                        lineup_changed = True
                        fatigue_tracker.start_cooldown(se.off_player_id, se.off_player_id in star_ids)
                        fatigue_tracker.start_cooldown(se.on_player_id, se.on_player_id in star_ids)
                        events.append(Event(
                            quarter=state.quarter,
                            seconds_left=state.seconds_left,
                            type="substitution",
                            team=se.side,
                            detail=f"{on_name} in for {off_name}",
                            home_score=state.home_score,
                            away_score=state.away_score,
                            player=on_name,
                        ))

                if lineup_changed:
                    home_lineup_rates = compute_lineup_rates(
                        lineup_state.on_court(Side.HOME), home,
                        fatigue_tracker=fatigue_tracker, scheme=_home_scheme,
                    )
                    away_lineup_rates = compute_lineup_rates(
                        lineup_state.on_court(Side.AWAY), away,
                        fatigue_tracker=fatigue_tracker, scheme=_away_scheme,
                    )

    raise RuntimeError(
        f"game did not terminate within {max_iters} iterations; state={state!r}"
    )
