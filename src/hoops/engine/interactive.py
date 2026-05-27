"""Interactive (possession-by-possession) game engine.

Unlike :func:`machine.simulate_game` which runs the entire game at once,
:class:`InteractiveGame` advances one possession at a time and yields
control back to the caller between possessions. This lets a human coach
make substitutions and scheme changes that affect the *actual simulation*,
not just post-hoc attribution.

The CPU-coached side runs auto-subs via :func:`fatigue.check_substitutions`
at dead balls. The human side only subs when explicitly told to.
"""

from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass

import numpy as np

from hoops.data.distributions import LeaguePrior, TeamPriors
from hoops.data.rosters import Roster
from hoops.engine.clock import end_period
from hoops.engine.events import Event
from hoops.engine.fatigue import (
    FatigueTracker, check_substitutions, player_importance,
)
from hoops.engine.lineup_rates import LineupRates, compute_lineup_rates
from hoops.engine.machine import simulate_possession, _player_name, _star_player_ids
from hoops.engine.matchup import adjust_offense, apply_hca
from hoops.engine.policy import CoachPolicies, CoachPolicy, DefensiveScheme, OffensiveScheme
from hoops.engine.state import GameState, Side
from hoops.rules import Rules
from hoops.engine.cpu_coach import (
    CpuCoach, CpuPersonality, PossessionSummary, TrendTracker, assign_personality,
)
from hoops.engine.scheme_affinity import detect_archetype
from hoops.engine.attribution import _ASSIST_PROB, _BLOCK_PROB, _STEAL_PROB, _credit_event
from hoops.ui.lineup import LineupState


def _attribute_possession(
    evs: list[Event],
    lineup: LineupState,
    rng: np.random.Generator,
    fatigue: FatigueTracker,
) -> list[Event]:
    """Attribute all events to on-court players and generate credit events.

    Replaces the old foul-only attribution with full player attribution
    plus assist / steal / block generation — matching what
    ``attribution.attribute_players`` does for the batch path, but using
    the on-court five instead of the full roster.
    """
    out: list[Event] = []
    for i, e in enumerate(evs):
        next_e = evs[i + 1] if i + 1 < len(evs) else None
        e = lineup.attribute(e)

        if e.type in ("foul_personal", "foul_shooting") and e.team is not None:
            on_court = lineup.on_court(e.team)
            for p in on_court:
                if p.name == e.player:
                    fatigue.add_foul(p.player_id)
                    break

        out.append(e)

        if e.type == "shot_made" and e.team is not None:
            if rng.random() < _ASSIST_PROB:
                adhoc = lineup._adhoc(e.team)
                assister = adhoc.assister(rng, exclude=e.player)
                out.append(_credit_event(e, "assist", e.team, assister.name))

        if e.type == "shot_missed" and e.team is not None:
            fouled = next_e is not None and next_e.type == "foul_shooting"
            if not fouled and rng.random() < _BLOCK_PROB:
                def_side = e.team.other
                adhoc = lineup._adhoc(def_side)
                blocker = adhoc.blocker(rng)
                out.append(_credit_event(e, "block", def_side, blocker.name))

        if e.type == "turnover" and e.team is not None:
            if rng.random() < _STEAL_PROB:
                def_side = e.team.other
                adhoc = lineup._adhoc(def_side)
                stealer = adhoc.stealer(rng)
                out.append(_credit_event(e, "steal", def_side, stealer.name))

    return out


def _serialize_event(e: Event) -> dict:
    return {
        "quarter": e.quarter,
        "seconds_left": e.seconds_left,
        "type": e.type,
        "team": int(e.team) if e.team is not None else None,
        "detail": e.detail,
        "home_score": e.home_score,
        "away_score": e.away_score,
        "player": e.player,
    }


def _serialize_policy(p: CoachPolicy) -> dict:
    return {
        "scheme": p.scheme.value,
        "off_scheme": p.off_scheme.value,
        "two_for_one": p.two_for_one,
        "hold_for_last": p.hold_for_last,
        "foul_when_down_3": p.foul_when_down_3,
        "intentional_foul_in_bonus_when_trailing": p.intentional_foul_in_bonus_when_trailing,
        "timeouts_remaining": p.timeouts_remaining,
    }


def _deserialize_event(d: dict) -> Event:
    return Event(
        quarter=d["quarter"],
        seconds_left=d["seconds_left"],
        type=d["type"],
        team=Side(d["team"]) if d["team"] is not None else None,
        detail=d.get("detail", ""),
        home_score=d.get("home_score", 0),
        away_score=d.get("away_score", 0),
        player=d.get("player"),
    )


def _deserialize_policy(d: dict) -> CoachPolicy:
    return CoachPolicy(
        scheme=DefensiveScheme(d["scheme"]),
        off_scheme=OffensiveScheme(d.get("off_scheme", "normal")),
        two_for_one=d.get("two_for_one", True),
        hold_for_last=d.get("hold_for_last", True),
        foul_when_down_3=d.get("foul_when_down_3", False),
        intentional_foul_in_bonus_when_trailing=d.get("intentional_foul_in_bonus_when_trailing", False),
        timeouts_remaining=d.get("timeouts_remaining", 4),
    )


def _deserialize_rng(d: dict) -> np.random.Generator:
    """Restore numpy RNG from serialized state."""
    rng = np.random.default_rng(0)
    rng.bit_generator.state = {
        "bit_generator": d["bit_generator"],
        "state": {k: v for k, v in d["state"].items()},
        "has_uint32": d["has_uint32"],
        "uinteger": d["uinteger"],
    }
    return rng


def _serialize_rng(rng: np.random.Generator) -> dict:
    raw = rng.bit_generator.state
    state_inner = raw["state"]
    return {
        "bit_generator": raw["bit_generator"],
        "state": {k: int(v) for k, v in state_inner.items()},
        "has_uint32": int(raw["has_uint32"]),
        "uinteger": int(raw["uinteger"]),
    }


@dataclass
class PossessionResult:
    """What happened on the last possession."""
    events: list[Event]
    is_dead_ball: bool
    is_game_over: bool


class InteractiveGame:
    """Possession-by-possession game engine with human vs CPU coaching."""

    def __init__(
        self,
        home_priors: TeamPriors,
        away_priors: TeamPriors,
        rules: Rules,
        rng: np.random.Generator,
        home_roster: Roster,
        away_roster: Roster,
        human_side: Side | None,
        policies: CoachPolicies | None = None,
        league: LeaguePrior | None = None,
        opening_possession: Side = Side.HOME,
        neutral_site: bool = False,
        home_starters: list[int] | None = None,
        away_starters: list[int] | None = None,
    ):
        if league is not None:
            home_priors = adjust_offense(home_priors, away_priors, league)
            away_priors = adjust_offense(away_priors, home_priors, league)
        if not neutral_site:
            away_priors = apply_hca(away_priors)

        self.home_priors = home_priors
        self.away_priors = away_priors
        self.rules = rules
        self.rng = rng
        self.home_roster = home_roster
        self.away_roster = away_roster
        self.human_side = human_side
        self.policies = policies or CoachPolicies()

        self.state = GameState.initial(rules, opening_possession=opening_possession)
        if home_starters or away_starters:
            pid_map_h = {p.player_id: p for p in home_roster.players}
            pid_map_a = {p.player_id: p for p in away_roster.players}
            starters_h = (
                [pid_map_h[pid] for pid in home_starters]
                if home_starters
                else list(home_roster.players[:5])
            )
            starters_a = (
                [pid_map_a[pid] for pid in away_starters]
                if away_starters
                else list(away_roster.players[:5])
            )
            self.lineup = LineupState(
                home_roster=home_roster, away_roster=away_roster,
                home_on_court=starters_h, away_on_court=starters_a,
                rng=rng,
            )
        else:
            self.lineup = LineupState.with_default_starters(
                home_roster, away_roster, rng,
            )
        self.fatigue = FatigueTracker(home_roster, away_roster)

        self._home_stars = _star_player_ids(home_roster)
        self._away_stars = _star_player_ids(away_roster)

        # CPU coaching brain (only in single-player mode).
        if human_side is not None:
            cpu_priors = self.away_priors if human_side is Side.HOME else self.home_priors
            self.cpu_coach = CpuCoach(
                cpu_side=self.cpu_side,
                personality=assign_personality(
                    pace=cpu_priors.pace,
                    off_tov_pct=cpu_priors.off_tov_pct,
                    def_efg=cpu_priors.def_efg,
                ),
            )
        else:
            self.cpu_coach = None

        self._recompute_lineup_rates()

        self.all_events: list[Event] = [Event(
            quarter=1, seconds_left=self.state.seconds_left, type="tip_off",
            team=opening_possession, home_score=0, away_score=0,
        )]
        self._media_to_used: set[int] = set()
        self._cpu_opp_score_at_last_check: int = 0
        self._cpu_own_score_at_last_check: int = 0

    @property
    def is_game_over(self) -> bool:
        return self.state.is_final

    @property
    def cpu_side(self) -> Side | None:
        return self.human_side.other if self.human_side is not None else None

    def human_policy(self) -> CoachPolicy:
        assert self.human_side is not None, "use policies.for_side() in H2H mode"
        return self.policies.for_side(self.human_side)

    def cpu_policy(self) -> CoachPolicy:
        assert self.cpu_side is not None, "use policies.for_side() in H2H mode"
        return self.policies.for_side(self.cpu_side)

    def _replace_policy(self, side: Side, new_policy: CoachPolicy) -> None:
        if side is Side.HOME:
            self.policies = CoachPolicies(home=new_policy, away=self.policies.away)
        else:
            self.policies = CoachPolicies(home=self.policies.home, away=new_policy)

    def set_scheme(self, side: Side, scheme: DefensiveScheme) -> None:
        """Set defensive scheme for the given side (works in any mode)."""
        old = self.policies.for_side(side)
        new_policy = CoachPolicy(
            scheme=scheme,
            off_scheme=old.off_scheme,
            two_for_one=old.two_for_one,
            hold_for_last=old.hold_for_last,
            foul_when_down_3=old.foul_when_down_3,
            intentional_foul_in_bonus_when_trailing=old.intentional_foul_in_bonus_when_trailing,
            timeouts_remaining=old.timeouts_remaining,
        )
        self._replace_policy(side, new_policy)
        self._recompute_lineup_rates()

    def set_human_scheme(self, scheme: DefensiveScheme) -> None:
        assert self.human_side is not None, "use set_scheme() in H2H mode"
        self.set_scheme(self.human_side, scheme)

    def set_off_scheme(self, side: Side, scheme: OffensiveScheme) -> None:
        """Set offensive scheme for the given side (works in any mode)."""
        old = self.policies.for_side(side)
        new_policy = CoachPolicy(
            scheme=old.scheme,
            off_scheme=scheme,
            two_for_one=old.two_for_one,
            hold_for_last=old.hold_for_last,
            foul_when_down_3=old.foul_when_down_3,
            intentional_foul_in_bonus_when_trailing=old.intentional_foul_in_bonus_when_trailing,
            timeouts_remaining=old.timeouts_remaining,
        )
        self._replace_policy(side, new_policy)

    def set_human_off_scheme(self, scheme: OffensiveScheme) -> None:
        assert self.human_side is not None, "use set_off_scheme() in H2H mode"
        self.set_off_scheme(self.human_side, scheme)

    def _set_cpu_scheme(self, scheme: DefensiveScheme) -> None:
        """Change the CPU side's defensive scheme."""
        old = self.cpu_policy()
        new_policy = CoachPolicy(
            scheme=scheme,
            off_scheme=old.off_scheme,
            two_for_one=old.two_for_one,
            hold_for_last=old.hold_for_last,
            foul_when_down_3=old.foul_when_down_3,
            intentional_foul_in_bonus_when_trailing=old.intentional_foul_in_bonus_when_trailing,
            timeouts_remaining=old.timeouts_remaining,
        )
        self._replace_policy(self.cpu_side, new_policy)
        self._recompute_lineup_rates()

    _TIMEOUT_REST_SECONDS = 60.0

    def call_timeout(self, side: Side) -> list[Event]:
        """Call a timeout for *side*. Decrements count, grants fatigue
        recovery to all on-court players, emits event, and runs CPU auto-subs."""
        policy = self.policies.for_side(side)
        if policy.timeouts_remaining <= 0:
            raise ValueError("no timeouts remaining")
        if self.state.is_final:
            raise ValueError("game is over")

        # Decrement timeout count.
        new_policy = CoachPolicy(
            scheme=policy.scheme,
            two_for_one=policy.two_for_one,
            hold_for_last=policy.hold_for_last,
            foul_when_down_3=policy.foul_when_down_3,
            intentional_foul_in_bonus_when_trailing=policy.intentional_foul_in_bonus_when_trailing,
            timeouts_remaining=policy.timeouts_remaining - 1,
        )
        if side is Side.HOME:
            self.policies = CoachPolicies(home=new_policy, away=self.policies.away)
        else:
            self.policies = CoachPolicies(home=self.policies.home, away=new_policy)

        # Fatigue recovery for ALL on-court players (both teams benefit).
        home_on = [p.player_id for p in self.lineup.on_court(Side.HOME)]
        away_on = [p.player_id for p in self.lineup.on_court(Side.AWAY)]
        self.fatigue.rest(home_on + away_on, self._TIMEOUT_REST_SECONDS)

        remaining = new_policy.timeouts_remaining
        ev = Event(
            quarter=self.state.quarter,
            seconds_left=self.state.seconds_left,
            type="timeout",
            team=side,
            detail=f"{remaining} remaining",
            home_score=self.state.home_score,
            away_score=self.state.away_score,
        )
        result_events = [ev]
        self.all_events.append(ev)

        # CPU auto-subs at the dead ball created by the timeout.
        if self.cpu_coach is not None:
            cpu_sub_events = self._cpu_auto_subs()
            result_events.extend(cpu_sub_events)
            self.all_events.extend(cpu_sub_events)

        self._recompute_lineup_rates()
        return result_events

    _MEDIA_TO_THRESHOLD = 300  # 5:00 minutes

    def _check_media_timeout(self) -> list[Event]:
        """Fire a media timeout if this quarter hasn't had one and clock < 5:00."""
        q = self.state.quarter
        if q in self._media_to_used:
            return []
        if self.state.seconds_left >= self._MEDIA_TO_THRESHOLD:
            return []

        self._media_to_used.add(q)

        # Fatigue recovery for all on-court players.
        home_on = [p.player_id for p in self.lineup.on_court(Side.HOME)]
        away_on = [p.player_id for p in self.lineup.on_court(Side.AWAY)]
        self.fatigue.rest(home_on + away_on, self._TIMEOUT_REST_SECONDS)

        ev = Event(
            quarter=q,
            seconds_left=self.state.seconds_left,
            type="media_timeout",
            team=None,
            home_score=self.state.home_score,
            away_score=self.state.away_score,
        )
        self.all_events.append(ev)
        return [ev]

    _RUN_THRESHOLD = 8        # opponent unanswered points to trigger TO
    _LATE_GAME_SECONDS = 120  # reserve 1 TO for final 2 minutes of Q4

    def _cpu_should_call_timeout(self) -> bool:
        """Decide if the CPU coach should call a timeout at this dead ball."""
        if self.human_side is None:
            return False
        if self.state.is_final:
            return False
        policy = self.cpu_policy()
        if policy.timeouts_remaining <= 0:
            return False

        opp_side = self.human_side
        opp_score = self.state.score_for(opp_side)
        cpu_score = self.state.score_for(self.cpu_side)

        opp_run = opp_score - self._cpu_opp_score_at_last_check
        cpu_run = cpu_score - self._cpu_own_score_at_last_check

        # Late-game conservation: save last timeout for final 2 min of Q4+.
        if (
            policy.timeouts_remaining <= 1
            and self.state.quarter >= 4
            and self.state.seconds_left > self._LATE_GAME_SECONDS
        ):
            return False

        # Scoring run: opponent scored 8+ unanswered.
        if opp_run >= self._RUN_THRESHOLD and cpu_run == 0:
            return True

        # Final possession setup: trailing with <30s in Q4/OT.
        if (
            self.state.quarter >= 4
            and self.state.seconds_left <= 30
            and cpu_score < opp_score
            and self.state.possession is self.cpu_side
            and policy.timeouts_remaining >= 1
        ):
            return True

        return False

    def _cpu_call_timeout(self) -> list[Event]:
        """Have the CPU call a timeout and reset the run tracker."""
        events = self.call_timeout(self.cpu_side)
        self._reset_run_tracker()
        return events

    def _reset_run_tracker(self) -> None:
        if self.human_side is None:
            return
        self._cpu_opp_score_at_last_check = self.state.score_for(self.human_side)
        self._cpu_own_score_at_last_check = self.state.score_for(self.cpu_side)

    def substitute(self, side: Side, off_player_id: int, on_player_id: int) -> None:
        """Substitute a player for the given side (works in any mode)."""
        self.lineup.substitute(side, off_player_id, on_player_id)
        star_ids = self._home_stars if side is Side.HOME else self._away_stars
        self.fatigue.start_cooldown(off_player_id, off_player_id in star_ids)
        self.fatigue.start_cooldown(on_player_id, on_player_id in star_ids)
        self._recompute_lineup_rates()

    def human_substitute(self, off_player_id: int, on_player_id: int) -> None:
        assert self.human_side is not None, "use substitute() in H2H mode"
        self.substitute(self.human_side, off_player_id, on_player_id)

    def step_possession(self) -> PossessionResult:
        """Simulate one possession and return the result.

        After calling this, the caller should check ``is_dead_ball`` to
        decide whether to offer substitution opportunities. CPU auto-subs
        are already applied; human subs must be made via
        ``human_substitute()`` before calling ``step_possession()`` again.
        """
        if self.state.is_final:
            return PossessionResult(events=[], is_dead_ball=False, is_game_over=True)

        result_events: list[Event] = []

        # Handle period end if clock is at zero.
        if self.state.seconds_left <= 0:
            self.state, evs = end_period(self.state)
            result_events.extend(evs)
            self.all_events.extend(evs)
            if self.state.is_final:
                return PossessionResult(
                    events=result_events, is_dead_ball=False, is_game_over=True,
                )
            return PossessionResult(
                events=result_events, is_dead_ball=False, is_game_over=False,
            )

        # Determine lineup rates for this possession.
        if self.state.possession is Side.HOME:
            off_lr, def_lr = self._home_lr, self._away_lr
        else:
            off_lr, def_lr = self._away_lr, self._home_lr

        self.state, evs = simulate_possession(
            self.state, self.home_priors, self.away_priors, self.rng,
            policies=self.policies,
            off_lineup_rates=off_lr, def_lineup_rates=def_lr,
        )

        # Full attribution: assign players to all events and generate
        # credit events (assists, steals, blocks) using on-court rosters.
        evs = _attribute_possession(evs, self.lineup, self.rng, self.fatigue)

        result_events.extend(evs)
        self.all_events.extend(evs)

        # Feed CPU trend tracker.
        # After simulate_possession, self.state.possession has flipped to the
        # NEXT offense, so the side that just had the ball is .other.
        poss_side = self.state.possession.other
        scored = any(e.type in ("shot_made", "free_throw_made") for e in evs)
        pts = sum(
            (3 if e.detail == "three" else 2) for e in evs if e.type == "shot_made"
        ) + sum(1 for e in evs if e.type == "free_throw_made")
        shot_zone = None
        for e in evs:
            if e.type in ("shot_made", "shot_missed"):
                shot_zone = e.detail
                break
        had_tov = any(e.type == "turnover" for e in evs)
        had_foul = any(e.type in ("foul_personal", "foul_shooting") for e in evs)
        scorer: str | None = None
        for e in evs:
            if e.type == "shot_made" and e.player:
                scorer = e.player
                break
        if self.cpu_coach is not None:
            self.cpu_coach.trend.record(PossessionSummary(
                side=poss_side, scored=scored, points=pts,
                zone=shot_zone, turnover=had_tov, foul=had_foul,
                player=scorer,
            ))

        # Fatigue tick
        poss_duration = 17.0
        home_on = [p.player_id for p in self.lineup.on_court(Side.HOME)]
        away_on = [p.player_id for p in self.lineup.on_court(Side.AWAY)]
        self.fatigue.tick(home_on + away_on, poss_duration)
        home_bench = [p.player_id for p in self.lineup.bench(Side.HOME)]
        away_bench = [p.player_id for p in self.lineup.bench(Side.AWAY)]
        self.fatigue.rest(home_bench + away_bench, poss_duration)
        self.fatigue.tick_cooldowns()

        is_dead = any(
            e.type in ("shot_made", "foul_personal", "foul_shooting",
                       "free_throw_made", "free_throw_missed")
            for e in evs
        )

        # Auto-subs at dead balls
        if is_dead:
            if self.cpu_coach is not None:
                # CPU timeout decision (before regular auto-subs since TO triggers its own).
                if self._cpu_should_call_timeout():
                    cpu_to_events = self._cpu_call_timeout()
                    result_events.extend(cpu_to_events)
                else:
                    # Update run tracker: reset if CPU scored (run broken).
                    cpu_score = self.state.score_for(self.cpu_side)
                    if cpu_score > self._cpu_own_score_at_last_check:
                        self._reset_run_tracker()

                # Late-game policy updates (before subs).
                cpu_policy = self.policies.for_side(self.cpu_side)
                self.cpu_coach.update_late_game_policy(
                    cpu_policy,
                    quarter=self.state.quarter,
                    seconds_left=self.state.seconds_left,
                    cpu_score=self.state.score_for(self.cpu_side),
                    opp_score=self.state.score_for(self.human_side),
                )

                # Foul trouble subs (before matchup and fatigue subs).
                cpu_on_court_ft = self.lineup.on_court(self.cpu_side)
                cpu_bench_ft = self.lineup.bench(self.cpu_side)
                fouls_map = {p.player_id: self.fatigue.fouls(p.player_id) for p in cpu_on_court_ft}
                foul_trouble_subs = self.cpu_coach.should_foul_trouble_sub(
                    cpu_on_court_ft, cpu_bench_ft, fouls_map,
                    quarter=self.state.quarter,
                    seconds_left=self.state.seconds_left,
                )
                star_ids_ft = self._home_stars if self.cpu_side is Side.HOME else self._away_stars
                for off_id, on_id in foul_trouble_subs:
                    off_name = _player_name(off_id, self.cpu_side, self.home_roster, self.away_roster)
                    on_name = _player_name(on_id, self.cpu_side, self.home_roster, self.away_roster)
                    self.lineup.substitute(self.cpu_side, off_id, on_id)
                    self.fatigue.start_cooldown(off_id, off_id in star_ids_ft)
                    self.fatigue.start_cooldown(on_id, on_id in star_ids_ft)
                    ev = Event(
                        quarter=self.state.quarter,
                        seconds_left=self.state.seconds_left,
                        type="substitution",
                        team=self.cpu_side,
                        detail=f"{on_name} in for {off_name}",
                        home_score=self.state.home_score,
                        away_score=self.state.away_score,
                        player=on_name,
                    )
                    result_events.append(ev)
                    self.all_events.append(ev)

                # CPU scheme-switch evaluation.
                total_poss = self.state.home_possessions + self.state.away_possessions
                opp_on_court = self.lineup.on_court(self.human_side)
                opp_archetypes = [detect_archetype(p) for p in opp_on_court]
                new_scheme = self.cpu_coach.should_switch_scheme(
                    quarter=self.state.quarter,
                    seconds_left=self.state.seconds_left,
                    cpu_score=self.state.score_for(self.cpu_side),
                    opp_score=self.state.score_for(self.human_side),
                    opp_lineup_archetypes=opp_archetypes,
                    total_possessions=total_poss,
                )
                if new_scheme is not None:
                    self._set_cpu_scheme(new_scheme)
                    self.cpu_coach.apply_scheme_switch(new_scheme, total_poss)

                # CPU offensive scheme evaluation.
                opp_def_scheme = self.policies.for_side(self.cpu_side.other).scheme
                new_off_scheme = self.cpu_coach.should_switch_off_scheme(
                    quarter=self.state.quarter,
                    seconds_left=self.state.seconds_left,
                    cpu_score=self.state.score_for(self.cpu_side),
                    opp_score=self.state.score_for(self.cpu_side.other),
                    opp_def_scheme=opp_def_scheme,
                    total_possessions=total_poss,
                )
                if new_off_scheme is not None:
                    self.set_off_scheme(self.cpu_side, new_off_scheme)
                    self.cpu_coach.apply_off_scheme_switch(new_off_scheme, total_poss)

                # CPU matchup-based subs.
                cpu_on_court = self.lineup.on_court(self.cpu_side)
                cpu_bench = self.lineup.bench(self.cpu_side)
                matchup_subs = self.cpu_coach.should_matchup_sub(cpu_on_court, cpu_bench)
                star_ids = self._home_stars if self.cpu_side is Side.HOME else self._away_stars
                for off_id, on_id in matchup_subs:
                    off_name = _player_name(off_id, self.cpu_side, self.home_roster, self.away_roster)
                    on_name = _player_name(on_id, self.cpu_side, self.home_roster, self.away_roster)
                    self.lineup.substitute(self.cpu_side, off_id, on_id)
                    self.fatigue.start_cooldown(off_id, off_id in star_ids)
                    self.fatigue.start_cooldown(on_id, on_id in star_ids)
                    ev = Event(
                        quarter=self.state.quarter,
                        seconds_left=self.state.seconds_left,
                        type="substitution",
                        team=self.cpu_side,
                        detail=f"{on_name} in for {off_name}",
                        home_score=self.state.home_score,
                        away_score=self.state.away_score,
                        player=on_name,
                    )
                    result_events.append(ev)
                    self.all_events.append(ev)

                cpu_sub_events = self._cpu_auto_subs()
                result_events.extend(cpu_sub_events)
                self.all_events.extend(cpu_sub_events)
            else:
                # H2H mode: fatigue auto-subs for BOTH sides (no CPU coaching).
                for side in (Side.HOME, Side.AWAY):
                    sub_events_data = check_substitutions(
                        self.lineup, self.fatigue, self.state.quarter, side,
                    )
                    if sub_events_data:
                        star_ids = self._home_stars if side is Side.HOME else self._away_stars
                        for se in sub_events_data:
                            off_name = _player_name(se.off_player_id, se.side, self.home_roster, self.away_roster)
                            on_name = _player_name(se.on_player_id, se.side, self.home_roster, self.away_roster)
                            self.lineup.substitute(se.side, se.off_player_id, se.on_player_id)
                            self.fatigue.start_cooldown(se.off_player_id, se.off_player_id in star_ids)
                            self.fatigue.start_cooldown(se.on_player_id, se.on_player_id in star_ids)
                            ev = Event(
                                quarter=self.state.quarter,
                                seconds_left=self.state.seconds_left,
                                type="substitution",
                                team=se.side,
                                detail=f"{on_name} in for {off_name}",
                                home_score=self.state.home_score,
                                away_score=self.state.away_score,
                                player=on_name,
                            )
                            result_events.append(ev)
                            self.all_events.append(ev)
                        self._recompute_lineup_rates()

            # Media timeout check at dead balls (both modes).
            media_events = self._check_media_timeout()
            result_events.extend(media_events)

        return PossessionResult(
            events=result_events,
            is_dead_ball=is_dead,
            is_game_over=False,
        )

    def _cpu_auto_subs(self) -> list[Event]:
        """Run check_substitutions for the CPU side and apply them."""
        sub_events_data = check_substitutions(
            self.lineup, self.fatigue, self.state.quarter, self.cpu_side,
        )
        if not sub_events_data:
            return []

        roster = self.home_roster if self.cpu_side is Side.HOME else self.away_roster
        star_ids = self._home_stars if self.cpu_side is Side.HOME else self._away_stars
        events: list[Event] = []

        for se in sub_events_data:
            off_name = _player_name(se.off_player_id, se.side, self.home_roster, self.away_roster)

            # Hot-hand veto: skip fatigue sub if player is hot.
            if self.cpu_coach is not None:
                fatigue_val = self.fatigue.fatigue(se.off_player_id)
                fouls_val = self.fatigue.fouls(se.off_player_id)
                # Determine if this is a foul-related sub (mandatory, not vetoable).
                is_foul_sub = fouls_val >= 5 or (
                    (self.state.quarter <= 2 and fouls_val >= 3)
                    or (self.state.quarter > 2 and fouls_val >= 4)
                )
                if not is_foul_sub and self.cpu_coach.should_veto_fatigue_sub(off_name, fatigue_val):
                    continue

            on_name = _player_name(se.on_player_id, se.side, self.home_roster, self.away_roster)
            self.lineup.substitute(se.side, se.off_player_id, se.on_player_id)
            self.fatigue.start_cooldown(se.off_player_id, se.off_player_id in star_ids)
            self.fatigue.start_cooldown(se.on_player_id, se.on_player_id in star_ids)
            events.append(Event(
                quarter=self.state.quarter,
                seconds_left=self.state.seconds_left,
                type="substitution",
                team=se.side,
                detail=f"{on_name} in for {off_name}",
                home_score=self.state.home_score,
                away_score=self.state.away_score,
                player=on_name,
            ))

        if events:
            self._recompute_lineup_rates()
        return events

    def _recompute_lineup_rates(self) -> None:
        home_scheme = self.policies.for_side(Side.HOME).scheme
        away_scheme = self.policies.for_side(Side.AWAY).scheme
        self._home_lr = compute_lineup_rates(
            self.lineup.on_court(Side.HOME), self.home_priors,
            fatigue_tracker=self.fatigue, scheme=home_scheme,
        )
        self._away_lr = compute_lineup_rates(
            self.lineup.on_court(Side.AWAY), self.away_priors,
            fatigue_tracker=self.fatigue, scheme=away_scheme,
        )

    def to_save_dict(self) -> dict:
        """Serialize the full game state to a JSON-compatible dict."""
        return {
            "version": 1,
            "saved_at": datetime.datetime.now().isoformat(),
            "home_season": self.home_priors.season,
            "away_season": self.away_priors.season,
            "home_team_id": self.home_priors.team_id,
            "away_team_id": self.away_priors.team_id,
            "human_side": int(self.human_side) if self.human_side is not None else None,
            "game_state": {
                "quarter": self.state.quarter,
                "seconds_left": self.state.seconds_left,
                "home_score": self.state.home_score,
                "away_score": self.state.away_score,
                "possession": int(self.state.possession),
                "home_team_fouls_q": self.state.home_team_fouls_q,
                "away_team_fouls_q": self.state.away_team_fouls_q,
                "home_possessions": self.state.home_possessions,
                "away_possessions": self.state.away_possessions,
                "opening_possession": int(self.state.opening_possession),
            },
            "policies": {
                "home": _serialize_policy(self.policies.home),
                "away": _serialize_policy(self.policies.away),
            },
            "fatigue": {
                "fatigue": {str(k): v for k, v in self.fatigue._fatigue.items()},
                "fouls": {str(k): v for k, v in self.fatigue._fouls.items()},
                "cooldowns": {str(k): v for k, v in self.fatigue._cooldown.items()},
            },
            "lineup": {
                "home_on_court": [p.player_id for p in self.lineup.on_court(Side.HOME)],
                "away_on_court": [p.player_id for p in self.lineup.on_court(Side.AWAY)],
            },
            "events": [_serialize_event(e) for e in self.all_events],
            "rng_state": _serialize_rng(self.rng),
            "media_to_used": sorted(self._media_to_used),
            "cpu_opp_score_at_last_check": self._cpu_opp_score_at_last_check,
            "cpu_own_score_at_last_check": self._cpu_own_score_at_last_check,
            "cpu_coach": {
                "personality": self.cpu_coach.personality.value,
                "current_scheme": self.cpu_coach.current_scheme.value,
                "current_off_scheme": self.cpu_coach.current_off_scheme.value,
                "last_scheme_poss": self.cpu_coach._last_scheme_poss,
                "last_off_scheme_poss": self.cpu_coach._last_off_scheme_poss,
                "trend": self.cpu_coach.trend.to_list(),
            } if self.cpu_coach is not None else None,
        }

    @classmethod
    def from_save_dict(
        cls,
        d: dict,
        *,
        _preloaded: tuple | None = None,
    ) -> "InteractiveGame":
        """Reconstruct an InteractiveGame from a save dict.

        If ``_preloaded`` is given as ``(home_priors, away_priors, rules,
        home_roster, away_roster)``, those are used directly instead of
        loading from disk (useful for tests with synthetic data).
        """
        if _preloaded is not None:
            home_priors, away_priors, rules, home_roster, away_roster = _preloaded
        else:
            from hoops.data.distributions import load_team_priors
            from hoops.data.rosters import load_roster
            from hoops.league import League
            from hoops.rules import rules_for

            home_season = d.get("home_season", d.get("season", "2023-24"))
            away_season = d.get("away_season", home_season)
            home_team_id = d["home_team_id"]
            away_team_id = d["away_team_id"]

            all_home = load_team_priors(League.WBB, home_season)
            home_priors = next(p for p in all_home if p.team_id == home_team_id)
            all_away = load_team_priors(League.WBB, away_season)
            away_priors = next(p for p in all_away if p.team_id == away_team_id)
            game_season = max(home_season, away_season)
            rules = rules_for(League.WBB, game_season)
            home_roster = load_roster(home_team_id, home_season)
            away_roster = load_roster(away_team_id, away_season)

        rng = _deserialize_rng(d["rng_state"])
        human_side = Side(d["human_side"]) if d["human_side"] is not None else None
        policies = CoachPolicies(
            home=_deserialize_policy(d["policies"]["home"]),
            away=_deserialize_policy(d["policies"]["away"]),
        )

        # Build without calling __init__.
        game = object.__new__(cls)
        game.home_priors = home_priors
        game.away_priors = away_priors
        game.rules = rules
        game.rng = rng
        game.home_roster = home_roster
        game.away_roster = away_roster
        game.human_side = human_side
        game.policies = policies

        # Restore GameState.
        gs = d["game_state"]
        game.state = GameState(
            rules=rules,
            quarter=gs["quarter"],
            seconds_left=gs["seconds_left"],
            home_score=gs["home_score"],
            away_score=gs["away_score"],
            possession=Side(gs["possession"]),
            home_team_fouls_q=gs["home_team_fouls_q"],
            away_team_fouls_q=gs["away_team_fouls_q"],
            home_possessions=gs["home_possessions"],
            away_possessions=gs["away_possessions"],
            opening_possession=Side(gs["opening_possession"]),
        )

        # Restore lineup.
        all_players = {p.player_id: p for p in home_roster.players + away_roster.players}
        home_on = [all_players[pid] for pid in d["lineup"]["home_on_court"]]
        away_on = [all_players[pid] for pid in d["lineup"]["away_on_court"]]
        game.lineup = LineupState(
            home_roster=home_roster,
            away_roster=away_roster,
            home_on_court=home_on,
            away_on_court=away_on,
            rng=rng,
        )

        # Restore fatigue.
        game.fatigue = FatigueTracker(home_roster, away_roster)
        for pid_str, val in d["fatigue"]["fatigue"].items():
            game.fatigue._fatigue[int(pid_str)] = val
        for pid_str, val in d["fatigue"]["fouls"].items():
            game.fatigue._fouls[int(pid_str)] = val
        for pid_str, val in d["fatigue"].get("cooldowns", {}).items():
            game.fatigue._cooldown[int(pid_str)] = val

        # Restore events.
        game.all_events = [_deserialize_event(e) for e in d["events"]]

        # Recompute derived state.
        game._home_stars = _star_player_ids(home_roster)
        game._away_stars = _star_player_ids(away_roster)
        game._recompute_lineup_rates()

        # Restore timeout tracking.
        game._media_to_used = set(d.get("media_to_used", []))
        game._cpu_opp_score_at_last_check = d.get("cpu_opp_score_at_last_check", 0)
        game._cpu_own_score_at_last_check = d.get("cpu_own_score_at_last_check", 0)

        # Restore CPU coaching brain.
        if human_side is not None:
            cpu_priors = away_priors if human_side is Side.HOME else home_priors
            if "cpu_coach" in d and d["cpu_coach"] is not None:
                cc = d["cpu_coach"]
                game.cpu_coach = CpuCoach(
                    cpu_side=game.cpu_side,
                    personality=CpuPersonality(cc["personality"]),
                    current_scheme=DefensiveScheme(cc["current_scheme"]),
                    current_off_scheme=OffensiveScheme(cc.get("current_off_scheme", "normal")),
                )
                game.cpu_coach._last_scheme_poss = cc.get("last_scheme_poss", 0)
                game.cpu_coach._last_off_scheme_poss = cc.get("last_off_scheme_poss", 0)
                game.cpu_coach.trend = TrendTracker.from_list(cc.get("trend", []))
            else:
                # Backwards compat: old saves without cpu_coach.
                game.cpu_coach = CpuCoach(
                    cpu_side=game.cpu_side,
                    personality=assign_personality(
                        pace=cpu_priors.pace,
                        off_tov_pct=cpu_priors.off_tov_pct,
                        def_efg=cpu_priors.def_efg,
                    ),
                )
        else:
            game.cpu_coach = None

        return game
