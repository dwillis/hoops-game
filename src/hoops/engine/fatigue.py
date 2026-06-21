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
_THRESHOLD_FRACTION = 0.95       # sub at 95% of expected playing-time fatigue
_DEFAULT_MIN_SHARE = 0.10        # fallback for players with no min_share data
_MIN_THRESHOLD = 0.30            # floor: everyone can play ~14 min before fatigue sub
_RECOVERY_MULTIPLIER = 2.0       # bench recovery is 2× accumulation rate
_FOUL_TROUBLE_FIRST_HALF = 2    # Q1/Q2: 2+ fouls for non-stars
_FOUL_TROUBLE_SECOND_HALF = 4   # Q3/Q4: 4+ fouls
_FOULED_OUT = 5                  # WBB disqualification
_SUB_COOLDOWN = 10               # possessions before a subbed player can change status again
_SUB_COOLDOWN_STAR = 8           # stars can re-enter faster


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
        self._foul_hold: dict[int, tuple[int, int]] = {}
        self._total_seconds: dict[int, float] = {}
        self._players: dict[int, Player] = {}
        for roster in (home_roster, away_roster):
            for p in roster.players:
                self._fatigue[p.player_id] = 0.0
                self._fouls[p.player_id] = 0
                self._total_seconds[p.player_id] = 0.0
                self._players[p.player_id] = p

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
            self._total_seconds[pid] += duration_seconds

    def rest(self, bench_ids: list[int], duration_seconds: float) -> None:
        """Recover fatigue for benched players (2x recovery rate)."""
        decrement = (duration_seconds / MAX_STAMINA) * _RECOVERY_MULTIPLIER
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

    def set_foul_hold(self, player_id: int, until_quarter: int, until_seconds: int) -> None:
        """Bench *player_id* for foul trouble until a specific game clock."""
        self._foul_hold[player_id] = (until_quarter, until_seconds)

    def on_foul_hold(self, player_id: int, quarter: int, seconds_left: int) -> bool:
        """Return True if *player_id* is benched for foul trouble at this game clock."""
        if player_id not in self._foul_hold:
            return False
        hold_q, hold_s = self._foul_hold[player_id]
        if quarter < hold_q:
            return True
        if quarter == hold_q and seconds_left > hold_s:
            return True
        del self._foul_hold[player_id]
        return False

    def clear_foul_hold(self, player_id: int) -> None:
        """Remove foul hold for *player_id*."""
        self._foul_hold.pop(player_id, None)

    def total_minutes(self, player_id: int) -> float:
        """Return total on-court minutes for *player_id*."""
        return self._total_seconds.get(player_id, 0.0) / 60.0

    def over_target_minutes(self, player_id: int) -> bool:
        """Return True if *player_id* has played more than their target minutes."""
        p = self._players.get(player_id)
        if p is None:
            return False
        ms = p.min_share if p.min_share is not None else _DEFAULT_MIN_SHARE
        target = ms * 200.0
        return self.total_minutes(player_id) >= target

    def remaining_target_minutes(self, player_id: int) -> float:
        """Return how many target minutes *player_id* has left."""
        p = self._players.get(player_id)
        if p is None:
            return 0.0
        ms = p.min_share if p.min_share is not None else _DEFAULT_MIN_SHARE
        target = ms * 200.0
        return max(0.0, target - self.total_minutes(player_id))


# ---------------------------------------------------------------------------
# Substitution decision engine
# ---------------------------------------------------------------------------

def foul_trouble_limit(quarter: int, rank: int) -> int:
    """Foul count at which a player should be pulled to protect them.

    First half: 2 fouls triggers a sub for everyone.
    Q3: 4 fouls triggers a sub.
    Q4: 4 fouls triggers a sub (shorter hold for stars).
    """
    if quarter <= 2:
        return _FOUL_TROUBLE_FIRST_HALF
    return _FOUL_TROUBLE_SECOND_HALF


def foul_trouble_hold(quarter: int, seconds_left: int, rank: int) -> tuple[int, int]:
    """Return (hold_quarter, hold_seconds) — when the player can return.

    First half (Q1-Q2): sit until Q3 starts (rest of the half).
    Q3 with 4 fouls: sit until Q4 starts (rest of Q3), stars return
      with 2 min left in Q3.
    Q4 early (>5:00 left): stars sit ~2 min, role players ~3 min.
    Q4 late (<=5:00 left): no hold — player returns at next dead ball.
    """
    if quarter <= 2:
        if rank < 2:
            return (3, 8 * 60)
        return (3, 10 * 60)
    if quarter == 3:
        if rank < 2:
            return (3, 2 * 60)
        return (4, 10 * 60)
    if seconds_left > 5 * 60:
        hold_secs = seconds_left - (120 if rank < 2 else 180)
        return (4, max(hold_secs, 0))
    return (quarter, seconds_left)


def player_fatigue_threshold(p: Player) -> float:
    """Return the fatigue level at which *p* should be subbed out.

    Derived from the player's historical minutes share so high-minutes
    players have higher thresholds and get subbed later.
    """
    ms = p.min_share if p.min_share is not None else _DEFAULT_MIN_SHARE
    target_minutes = ms * 200.0  # 5 slots × 40 min
    target_fatigue = target_minutes * 60 / MAX_STAMINA
    return max(_MIN_THRESHOLD, target_fatigue * _THRESHOLD_FRACTION)


def check_substitutions(
    lineup_state: LineupState,
    tracker: FatigueTracker,
    quarter: int,
    side: Side,
    seconds_left: int = 600,
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
        if fouls >= foul_trouble_limit(quarter, rank):
            needs_sub.append((p, "foul_trouble"))
            continue

        # Fatigue check — per-player threshold from min_share.
        threshold = player_fatigue_threshold(p)
        if fatigue >= threshold:
            needs_sub.append((p, "fatigue"))
            continue

        # Over-target check — pull players who've exceeded their planned minutes.
        if tracker.over_target_minutes(pid):
            needs_sub.append((p, "fatigue"))
            continue

    # Build two bench pools: under-target (preferred) and all eligible (fallback
    # for fouled-out subs only).
    eligible_bench = [
        bp for bp in bench
        if not tracker.on_cooldown(bp.player_id)
        and tracker.fouls(bp.player_id) < _FOULED_OUT
        and not tracker.on_foul_hold(bp.player_id, quarter, seconds_left)
    ]
    under_target_bench = sorted(
        [bp for bp in eligible_bench if not tracker.over_target_minutes(bp.player_id)],
        key=lambda bp: tracker.remaining_target_minutes(bp.player_id),
        reverse=True,
    )
    all_bench = sorted(eligible_bench, key=player_importance, reverse=True)

    subs: list[SubEvent] = []
    used_bench_ids: set[int] = set()

    for p, reason in needs_sub:
        # For fatigue/foul-trouble, only use under-target bench players.
        # For fouled-out, use anyone available.
        pool = all_bench if reason == "fouled_out" else under_target_bench
        replacement: Player | None = None
        for bp in pool:
            if bp.player_id not in used_bench_ids:
                replacement = bp
                break

        if replacement is None:
            if reason == "fouled_out":
                break
            continue

        subs.append(
            SubEvent(
                side=side,
                off_player_id=p.player_id,
                on_player_id=replacement.player_id,
            )
        )
        used_bench_ids.add(replacement.player_id)

    return subs
