"""Fatigue tracking and recovery for player minutes management."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from hoops.data.rosters import Player, Roster
from hoops.engine.state import Side

if TYPE_CHECKING:
    from hoops.ui.lineup import LineupState

# Calibrated so a 40-minute player reaches ~0.85 fatigue.
MAX_STAMINA: float = 2824.0

# --- Substitution thresholds ------------------------------------------------
_FATIGUE_THRESHOLD_HIGH = 0.85   # top 2 importance
_FATIGUE_THRESHOLD_MED = 0.70    # 3rd-5th
_FATIGUE_THRESHOLD_LOW = 0.55    # 6th+
_FOUL_TROUBLE_FIRST_HALF = 2    # Q1/Q2: 2+ fouls for non-stars
_FOUL_TROUBLE_SECOND_HALF = 4   # Q3/Q4: 4+ fouls
_FOULED_OUT = 5                  # WBB disqualification
_STAR_BONUS = 1.20               # Stars are 20% harder to pull
_SUB_COOLDOWN = 2                # possessions before a subbed player can change status again
_SUB_COOLDOWN_STAR = 1           # stars can re-enter faster


@dataclasses.dataclass(frozen=True)
class SubEvent:
    """A substitution decision: *off_player_id* leaves, *on_player_id* enters."""
    side: Side
    off_player_id: int
    on_player_id: int


def player_importance(p: Player) -> float:
    """Return a blended importance score for *p*.

    Uses ``min_share * 0.4 + usage_pct * 0.6``, defaulting either
    component to 0.15 when the underlying field is ``None``.
    """
    ms = p.min_share if p.min_share is not None else 0.15
    usg = p.usage_pct if p.usage_pct is not None else 0.15
    return ms * 0.4 + usg * 0.6


def apply_fatigue(player: Player, fatigue: float) -> Player:
    """Return a copy of *player* with shooting rates degraded by *fatigue*.

    Multiplier curve: ``effectiveness = 1.0 - 0.15 * fatigue**2``.

    * ``ts_pct`` and ``ft_pct`` are multiplied by effectiveness (degraded).
    * ``tov_pct`` is divided by effectiveness (increased), capped at 0.50.
    * All other fields are unchanged.
    * If a rate is ``None``, it stays ``None``.
    """
    if fatigue <= 0.0:
        return player

    effectiveness = 1.0 - 0.15 * fatigue ** 2

    replacements: dict[str, float | None] = {}

    # Degrade shooting rates
    if player.ts_pct is not None:
        replacements["ts_pct"] = player.ts_pct * effectiveness
    if player.ft_pct is not None:
        replacements["ft_pct"] = player.ft_pct * effectiveness

    # Increase turnover rate (inverse)
    if player.tov_pct is not None:
        replacements["tov_pct"] = min(player.tov_pct / effectiveness, 0.50)

    return dataclasses.replace(player, **replacements)


class FatigueTracker:
    """Tracks per-player fatigue (0-1 float) and personal fouls."""

    def __init__(self, home_roster: Roster, away_roster: Roster) -> None:
        self._fatigue: dict[int, float] = {}
        self._fouls: dict[int, int] = {}
        self._cooldown: dict[int, int] = {}
        for roster in (home_roster, away_roster):
            for p in roster.players:
                self._fatigue[p.player_id] = 0.0
                self._fouls[p.player_id] = 0

    def fatigue(self, player_id: int) -> float:
        """Return current fatigue level for *player_id*."""
        return self._fatigue[player_id]

    def fouls(self, player_id: int) -> int:
        """Return current foul count for *player_id*."""
        return self._fouls[player_id]

    def tick(self, on_court_ids: list[int], duration_seconds: float) -> None:
        """Accumulate fatigue for players currently on the court."""
        increment = duration_seconds / MAX_STAMINA
        for pid in on_court_ids:
            self._fatigue[pid] += increment

    def rest(self, bench_ids: list[int], duration_seconds: float) -> None:
        """Recover fatigue for benched players (2x recovery rate)."""
        decrement = (duration_seconds / MAX_STAMINA) * 2.0
        for pid in bench_ids:
            self._fatigue[pid] = max(0.0, self._fatigue[pid] - decrement)

    def add_foul(self, player_id: int) -> None:
        """Increment foul count for *player_id*."""
        self._fouls[player_id] += 1

    def start_cooldown(self, player_id: int, is_star: bool = False) -> None:
        """Begin a substitution cooldown for *player_id*."""
        self._cooldown[player_id] = _SUB_COOLDOWN_STAR if is_star else _SUB_COOLDOWN

    def tick_cooldowns(self) -> None:
        """Decrement all active cooldowns; remove expired ones."""
        for pid in list(self._cooldown):
            self._cooldown[pid] -= 1
            if self._cooldown[pid] <= 0:
                del self._cooldown[pid]

    def on_cooldown(self, player_id: int) -> bool:
        """Return True if *player_id* is on substitution cooldown."""
        return player_id in self._cooldown


# ---------------------------------------------------------------------------
# Substitution decision engine
# ---------------------------------------------------------------------------

def _fatigue_threshold(rank: int) -> float:
    """Return the fatigue threshold for a player at the given importance rank."""
    if rank < 2:
        return _FATIGUE_THRESHOLD_HIGH
    if rank < 5:
        return _FATIGUE_THRESHOLD_MED
    return _FATIGUE_THRESHOLD_LOW


def check_substitutions(
    lineup_state: LineupState,
    tracker: FatigueTracker,
    quarter: int,
    side: Side,
) -> list[SubEvent]:
    """Decide which players on *side* should be subbed out at a dead ball.

    Returns a (possibly empty) list of :class:`SubEvent` objects.
    """
    on_court = lineup_state.on_court(side)
    bench = lineup_state.bench(side)

    if not bench:
        return []

    # Rank on-court players by importance (descending).
    ranked = sorted(on_court, key=player_importance, reverse=True)

    # Build a lookup from player_id -> importance rank (0-indexed).
    rank_of: dict[int, int] = {p.player_id: i for i, p in enumerate(ranked)}

    # Identify players that need subbing, with reason tracking.
    # reason: "fouled_out", "foul_trouble", "fatigue"
    needs_sub: list[tuple[Player, str]] = []
    for p in ranked:
        pid = p.player_id
        rank = rank_of[pid]
        fouls = tracker.fouls(pid)
        fatigue = tracker.fatigue(pid)

        # Fouled out — mandatory sub.
        if fouls >= _FOULED_OUT:
            needs_sub.append((p, "fouled_out"))
            continue

        # Skip recently-subbed-in players for non-mandatory reasons.
        if tracker.on_cooldown(pid) and fouls < _FOULED_OUT:
            continue

        # Foul trouble check (strategic — always pull to protect).
        first_half = quarter <= 2
        if first_half:
            limit = 3 if rank < 2 else _FOUL_TROUBLE_FIRST_HALF
            if fouls >= limit:
                needs_sub.append((p, "foul_trouble"))
                continue
        else:
            if fouls >= _FOUL_TROUBLE_SECOND_HALF:
                needs_sub.append((p, "foul_trouble"))
                continue

        # Fatigue check.
        threshold = _fatigue_threshold(rank)
        if fatigue >= threshold:
            needs_sub.append((p, "fatigue"))
            continue

    # Sort bench by importance (best available first), excluding cooldown players.
    available_bench = sorted(
        [bp for bp in bench if not tracker.on_cooldown(bp.player_id)],
        key=player_importance, reverse=True,
    )

    subs: list[SubEvent] = []
    used_bench_ids: set[int] = set()

    for p, reason in needs_sub:
        # Find the best available bench player not yet assigned.
        replacement: Player | None = None
        for bp in available_bench:
            if bp.player_id not in used_bench_ids:
                replacement = bp
                break

        if replacement is None:
            break  # No more bench players available.

        pid = p.player_id
        rank = rank_of[pid]

        # For fatigue-triggered star subs, compare contributions.
        # Stars (rank < 2) get a 1.20x bonus making them harder to pull;
        # if even after the bonus the star's importance exceeds the bench
        # replacement, skip the sub. Non-stars who hit their fatigue
        # threshold are always subbed.
        if reason == "fatigue" and rank < 2:
            tired_contrib = player_importance(p) * _STAR_BONUS
            bench_contrib = player_importance(replacement)
            if tired_contrib > bench_contrib:
                continue  # Star still more valuable, skip sub.

        subs.append(SubEvent(side=side, off_player_id=pid, on_player_id=replacement.player_id))
        used_bench_ids.add(replacement.player_id)

    return subs
