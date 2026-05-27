"""CPU coaching intelligence: adaptive scheme switching, matchup subs, timeouts.

The CpuCoach evaluates a set of coaching rules at each dead ball, using
a rolling window of recent possession outcomes to detect patterns and
react. Personality (derived from team priors) shifts trigger thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from hoops.data.rosters import Player
from hoops.engine.fatigue import player_importance
from hoops.engine.policy import DefensiveScheme, OffensiveScheme
from hoops.engine.scheme_affinity import detect_archetype
from hoops.engine.state import Side


@dataclass(frozen=True)
class PossessionSummary:
    """Lightweight record of one possession's outcome for trend tracking."""
    side: Side           # who had the ball
    scored: bool
    points: int          # 0, 2, or 3
    zone: str | None     # "rim", "mid", "three", or None (turnover/foul)
    turnover: bool
    foul: bool
    player: str | None = None


class TrendTracker:
    """Rolling window of recent possession summaries."""

    def __init__(self, window: int = 10) -> None:
        self._recent: deque[PossessionSummary] = deque(maxlen=window)

    @property
    def recent(self) -> list[PossessionSummary]:
        return list(self._recent)

    def record(self, summary: PossessionSummary) -> None:
        self._recent.append(summary)

    def made_threes_by(self, side: Side) -> int:
        return sum(
            1 for s in self._recent
            if s.side is side and s.scored and s.zone == "three"
        )

    def rim_makes_by(self, side: Side) -> int:
        return sum(
            1 for s in self._recent
            if s.side is side and s.scored and s.zone == "rim"
        )

    def scored_possessions_by(self, side: Side) -> int:
        return sum(1 for s in self._recent if s.side is side and s.scored)

    def points_by_player(self, player_name: str) -> int:
        """Sum points from scored possessions where player matches."""
        return sum(
            s.points for s in self._recent
            if s.scored and s.player == player_name
        )

    def to_list(self) -> list[dict]:
        """Serialize for save/load."""
        return [
            {
                "side": int(s.side),
                "scored": s.scored,
                "points": s.points,
                "zone": s.zone,
                "turnover": s.turnover,
                "foul": s.foul,
                "player": s.player,
            }
            for s in self._recent
        ]

    @classmethod
    def from_list(cls, data: list[dict], window: int = 10) -> "TrendTracker":
        t = cls(window=window)
        for d in data:
            t.record(PossessionSummary(
                side=Side(d["side"]),
                scored=d["scored"],
                points=d["points"],
                zone=d.get("zone"),
                turnover=d["turnover"],
                foul=d["foul"],
                player=d.get("player"),
            ))
        return t


class CpuPersonality(str, Enum):
    """CPU coaching personality — shifts trigger thresholds."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


def assign_personality(
    pace: float,
    off_tov_pct: float,
    def_efg: float,
) -> CpuPersonality:
    """Derive CPU personality from team priors.

    - AGGRESSIVE: fast pace (>72) or low turnover rate (<0.15)
    - CONSERVATIVE: slow pace (<65) or elite defense (def_efg <0.40)
    - BALANCED: everything else
    """
    if pace > 72.0 or off_tov_pct < 0.15:
        return CpuPersonality.AGGRESSIVE
    if pace < 65.0 or def_efg < 0.40:
        return CpuPersonality.CONSERVATIVE
    return CpuPersonality.BALANCED


_SCHEME_COOLDOWN = 6  # possessions between scheme switches
_HOT_HAND_HARD_CEILING = 0.85


class CpuCoach:
    """Reactive CPU coaching brain.

    Evaluates scheme-switch and matchup-sub rules at each dead ball.
    """

    def __init__(
        self,
        cpu_side: Side,
        personality: CpuPersonality,
        current_scheme: DefensiveScheme = DefensiveScheme.MAN,
        current_off_scheme: OffensiveScheme = OffensiveScheme.NORMAL,
        window: int = 10,
    ) -> None:
        self.cpu_side = cpu_side
        self.personality = personality
        self.current_scheme = current_scheme
        self.current_off_scheme = current_off_scheme
        self.trend = TrendTracker(window=window)
        self._last_scheme_poss: int = 0
        self._last_off_scheme_poss: int = 0

    def should_switch_scheme(
        self,
        quarter: int,
        seconds_left: int,
        cpu_score: int,
        opp_score: int,
        opp_lineup_archetypes: list[str],
        total_possessions: int,
    ) -> DefensiveScheme | None:
        """Return a new scheme if rules trigger, else None."""
        # Cooldown check.
        if total_possessions - self._last_scheme_poss < _SCHEME_COOLDOWN:
            return None

        deficit = opp_score - cpu_score
        opp_side = self.cpu_side.other

        # --- PRESS triggers ---
        if self.personality is CpuPersonality.AGGRESSIVE:
            if quarter >= 3 and deficit >= 8:
                return DefensiveScheme.PRESS
        if quarter >= 4 and deficit >= 10:
            return DefensiveScheme.PRESS

        # --- ZONE triggers ---
        three_count = self.trend.made_threes_by(opp_side)
        threshold_3 = 2 if self.personality is CpuPersonality.AGGRESSIVE else 3
        if three_count >= threshold_3 and self.current_scheme is not DefensiveScheme.ZONE:
            return DefensiveScheme.ZONE

        # Opponent lineup is floor-spacer heavy (3+ spacers/wings).
        spacer_count = sum(
            1 for a in opp_lineup_archetypes
            if a in ("floor_spacer", "versatile_wing")
        )
        if spacer_count >= 3 and self.current_scheme is not DefensiveScheme.ZONE:
            return DefensiveScheme.ZONE

        # --- Revert to MAN triggers ---
        if self.current_scheme is not DefensiveScheme.MAN:
            recent = self.trend.recent
            opp_recent = [s for s in recent if s.side is opp_side][-6:]
            scored_count = sum(1 for s in opp_recent if s.scored)
            fail_threshold = 5 if self.personality is CpuPersonality.CONSERVATIVE else 4
            if len(opp_recent) >= 6 and scored_count >= fail_threshold:
                return DefensiveScheme.MAN

            # Zone-specific: opponent dominating rim.
            if self.current_scheme is DefensiveScheme.ZONE:
                rim_count = self.trend.rim_makes_by(opp_side)
                if rim_count >= 3:
                    return DefensiveScheme.MAN

        return None

    def apply_scheme_switch(self, scheme: DefensiveScheme, total_possessions: int) -> None:
        """Record that a scheme switch happened."""
        self.current_scheme = scheme
        self._last_scheme_poss = total_possessions

    def should_switch_off_scheme(
        self,
        quarter: int,
        seconds_left: int,
        cpu_score: int,
        opp_score: int,
        opp_def_scheme: DefensiveScheme,
        total_possessions: int,
    ) -> OffensiveScheme | None:
        """Return a new offensive scheme if rules trigger, else None."""
        # Cooldown check.
        if total_possessions - self._last_off_scheme_poss < _SCHEME_COOLDOWN:
            return None

        deficit = opp_score - cpu_score
        lead = cpu_score - opp_score

        # --- Revert triggers (check first) ---
        if self.current_off_scheme is OffensiveScheme.HURRY_UP:
            if deficit <= 3:
                return OffensiveScheme.NORMAL

        if self.current_off_scheme is OffensiveScheme.SLOW_DOWN:
            if lead <= 2:
                return OffensiveScheme.NORMAL

        if self.current_off_scheme is OffensiveScheme.THREE_POINT:
            # Missed 4+ consecutive threes in last 4 CPU possessions
            cpu_recent = [s for s in self.trend.recent if s.side is self.cpu_side][-4:]
            if len(cpu_recent) >= 4 and all(
                not s.scored and s.zone == "three" for s in cpu_recent
            ):
                return OffensiveScheme.NORMAL

        # --- HURRY_UP: Q4+, trailing by 8+, <=120s (AGGRESSIVE: <=180s) ---
        time_cutoff = 180 if self.personality is CpuPersonality.AGGRESSIVE else 120
        if quarter >= 4 and deficit >= 8 and seconds_left <= time_cutoff:
            if self.current_off_scheme is not OffensiveScheme.HURRY_UP:
                return OffensiveScheme.HURRY_UP

        # --- SLOW_DOWN: Q4+, leading by 5+, <=120s ---
        if quarter >= 4 and lead >= 5 and seconds_left <= 120:
            if self.current_off_scheme is not OffensiveScheme.SLOW_DOWN:
                return OffensiveScheme.SLOW_DOWN

        # --- THREE_POINT: opponent in ZONE; OR scored 0 in last 4 CPU possessions while NORMAL ---
        if opp_def_scheme is DefensiveScheme.ZONE:
            if self.current_off_scheme is not OffensiveScheme.THREE_POINT:
                return OffensiveScheme.THREE_POINT

        if self.current_off_scheme is OffensiveScheme.NORMAL:
            cpu_recent = [s for s in self.trend.recent if s.side is self.cpu_side][-4:]
            if len(cpu_recent) >= 4 and all(not s.scored for s in cpu_recent):
                return OffensiveScheme.THREE_POINT

        return None

    def apply_off_scheme_switch(self, scheme: OffensiveScheme, total_possessions: int) -> None:
        """Record that an offensive scheme switch happened."""
        self.current_off_scheme = scheme
        self._last_off_scheme_poss = total_possessions

    def should_matchup_sub(
        self,
        on_court: list[Player],
        bench: list[Player],
    ) -> list[tuple[int, int]]:
        """Return (off_player_id, on_player_id) pairs for matchup-driven subs.

        Checks: opponent hot from 3 → perimeter_stopper,
        opponent dominating paint → rim_protector,
        running press → ensure ball_handler.
        """
        if not bench:
            return []

        opp_side = self.cpu_side.other
        on_archetypes = [detect_archetype(p) for p in on_court]
        bench_archetypes = [(p, detect_archetype(p)) for p in bench]

        needs: list[str] = []

        # Hot from 3 → need perimeter_stopper.
        if self.trend.made_threes_by(opp_side) >= 3:
            if "perimeter_stopper" not in on_archetypes:
                needs.append("perimeter_stopper")

        # Dominating paint → need rim_protector.
        if self.trend.rim_makes_by(opp_side) >= 3:
            if "rim_protector" not in on_archetypes:
                needs.append("rim_protector")

        # Running press → need ball_handler.
        if self.current_scheme is DefensiveScheme.PRESS:
            if "ball_handler" not in on_archetypes:
                needs.append("ball_handler")

        subs: list[tuple[int, int]] = []
        used_bench: set[int] = set()
        used_court: set[int] = set()

        for needed_archetype in needs:
            # Find best bench player with the needed archetype.
            candidate = None
            for bp, arch in bench_archetypes:
                if arch == needed_archetype and bp.player_id not in used_bench:
                    candidate = bp
                    break
            if candidate is None:
                continue

            # Pull the least important on-court player.
            pullable = sorted(
                [p for p in on_court if p.player_id not in used_court],
                key=player_importance,
            )
            if not pullable:
                continue

            pull = pullable[0]
            subs.append((pull.player_id, candidate.player_id))
            used_bench.add(candidate.player_id)
            used_court.add(pull.player_id)

        return subs

    def should_veto_fatigue_sub(self, player_name: str, fatigue: float) -> bool:
        """Return True if a hot hand should override a fatigue-triggered sub.

        The CPU coach keeps a tired-but-hot player on court if they scored
        enough in the recent window and haven't hit the hard fatigue ceiling.
        """
        if fatigue >= _HOT_HAND_HARD_CEILING:
            return False

        if self.personality is CpuPersonality.AGGRESSIVE:
            threshold = 4
        elif self.personality is CpuPersonality.CONSERVATIVE:
            threshold = 8
        else:
            threshold = 6

        return self.trend.points_by_player(player_name) >= threshold

    def update_late_game_policy(
        self,
        policy: "CoachPolicy",
        quarter: int,
        seconds_left: int,
        cpu_score: int,
        opp_score: int,
    ) -> None:
        """Mutate *policy* in place with late-game strategy decisions."""
        deficit = opp_score - cpu_score
        leading = cpu_score > opp_score
        tied = cpu_score == opp_score
        lead = cpu_score - opp_score

        # --- Intentional fouling ---
        # Always reset both foul flags first.
        policy.intentional_foul_in_bonus_when_trailing = False
        policy.foul_when_down_3 = False

        if quarter >= 4 and 1 <= deficit <= 8:
            foul_cutoffs = {
                CpuPersonality.AGGRESSIVE: 90,
                CpuPersonality.BALANCED: 60,
                CpuPersonality.CONSERVATIVE: 45,
            }
            cutoff = foul_cutoffs[self.personality]
            if seconds_left <= cutoff:
                policy.intentional_foul_in_bonus_when_trailing = True
                if deficit == 3 and seconds_left <= 30:
                    policy.foul_when_down_3 = True

        # --- Clock management ---
        # Hold for last shot: Q4+, leading or tied, <=35 seconds.
        if quarter >= 4 and (leading or tied) and seconds_left <= 35:
            policy.hold_for_last = True

        # Two-for-one: <=40 seconds left in ANY quarter.
        if seconds_left <= 40:
            policy.two_for_one = True

        # Run clock with lead: Q4+, leading by 5+, <=120 seconds.
        if quarter >= 4 and lead >= 5 and seconds_left <= 120:
            policy.hold_for_last = True

    def should_foul_trouble_sub(
        self,
        on_court: list[Player],
        bench: list[Player],
        fouls: dict[int, int],
        quarter: int,
        seconds_left: int,
    ) -> list[tuple[int, int]]:
        """Return (off_id, on_id) pairs for foul-trouble subs.

        Rules:
        - First half (Q1-Q2): pull any on-court player with 3+ fouls.
        - Second half (Q3-Q4): pull any on-court player with 4+ fouls.
        - Crunch-time exception (personality-dependent): don't pull in Q4 late.
          - AGGRESSIVE: crunch-time exception at <=4:00 (240s)
          - BALANCED: crunch-time exception at <=2:00 (120s)
          - CONSERVATIVE: no crunch-time exception (crunch_cutoff = 0)
        """
        # Crunch-time exception: personality-dependent cutoff in Q4+.
        crunch_cutoffs = {
            CpuPersonality.AGGRESSIVE: 240,
            CpuPersonality.BALANCED: 120,
            CpuPersonality.CONSERVATIVE: 0,
        }
        crunch_cutoff = crunch_cutoffs[self.personality]
        if quarter >= 4 and crunch_cutoff > 0 and seconds_left <= crunch_cutoff:
            return []

        # Foul limit depends on half.
        foul_limit = 3 if quarter <= 2 else 4

        # Find on-court players in foul trouble.
        troubled = [
            p for p in on_court
            if fouls.get(p.player_id, 0) >= foul_limit
        ]
        if not troubled or not bench:
            return []

        # Sort bench by importance (best first).
        sorted_bench = sorted(bench, key=player_importance, reverse=True)

        subs: list[tuple[int, int]] = []
        bench_idx = 0
        for p in troubled:
            if bench_idx >= len(sorted_bench):
                break
            subs.append((p.player_id, sorted_bench[bench_idx].player_id))
            bench_idx += 1

        return subs
