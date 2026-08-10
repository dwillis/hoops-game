"""Lineup-aware blended rates for a 5-player on-court unit.

Given a list of 5 ``Player`` objects and a ``TeamPriors`` baseline,
``compute_lineup_rates()`` returns a frozen ``LineupRates`` dataclass
whose fields are usage-weighted averages of the individual player
advanced-rate stats. When a player lacks a particular rate, the
corresponding team prior is used as a fallback for that player's
contribution.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from hoops.data.distributions import TeamPriors
from hoops.data.rosters import Player
from hoops.engine.fatigue import FatigueTracker, apply_fatigue
from hoops.engine.policy import DefensiveScheme
from hoops.engine.scheme_affinity import scheme_affinity

# Shrinkage constant: at K minutes, player's own rate gets 50% weight.
_SHRINKAGE_K = 200.0


def shrink_rate(player_rate: float, team_rate: float, minutes: float) -> float:
    """Blend a player's rate toward the team prior proportional to minutes.

    w = minutes / (minutes + K), K = 200.
    At 200 min: 50% player / 50% team.
    At 600 min: 75% player / 25% team.
    At  30 min: 13% player / 87% team.
    """
    w = minutes / (minutes + _SHRINKAGE_K) if minutes > 0 else 0.0
    return w * player_rate + (1.0 - w) * team_rate


@dataclass(frozen=True)
class LineupRates:
    """Blended rates for a specific on-court lineup of 5 players."""

    tov_pct: float
    orb_pct: float
    drb_pct: float
    foul_rate: float
    stl_rate: float
    blk_rate: float
    ft_pct: float
    shooters: tuple[tuple[Player, float], ...]
    pace_adj: float = 0.0
    efg_adj: float = 0.0
    shot_weights: tuple[float, ...] | None = None
    """Coach-set shot-selection weights parallel to ``shooters``. ``None``
    means "use the usage weights in ``shooters``" (the default). Only shot
    *selection* uses these — team-rate blends always use usage weights."""


_POSITION_PACE_BONUS = {"G": 0.02, "F": 0.0, "C": -0.02}
_PACE_SCALE = 50.0


def _position_bonus(position: str) -> float:
    pos = position.upper().strip()
    if not pos:
        return 0.0
    parts = [p.strip() for p in pos.replace("-", "/").split("/")]
    bonuses = [_POSITION_PACE_BONUS.get(p, 0.0) for p in parts]
    return sum(bonuses) / len(bonuses) if bonuses else 0.0


_SHOT_SHARE_FLOOR = 0.02
_SHOT_SHARE_CEIL = 0.60
_USAGE_TS_PENALTY_SLOPE = 0.6   # TS pp lost per share pp above natural usage
_USAGE_TS_PENALTY_CAP = 0.10    # overuse beyond +10pp share stops adding penalty


def compute_lineup_rates(
    on_court: list[Player],
    team_priors: TeamPriors,
    fatigue_tracker: FatigueTracker | None = None,
    scheme: DefensiveScheme | None = None,
    shot_distribution: dict[int, float] | None = None,
) -> LineupRates:
    """Compute usage-weighted blended rates for *on_court* players.

    Parameters
    ----------
    on_court:
        Exactly 5 ``Player`` objects representing the current lineup.
    team_priors:
        Team-level priors used as fallback when a player's rate is None.

    Returns
    -------
    LineupRates
        Frozen dataclass with blended rates and shooter tuples.
    """
    if scheme is not None:
        affinity_mult = [scheme_affinity(p).get(scheme, 1.0) for p in on_court]
    else:
        affinity_mult = [1.0] * len(on_court)

    # --- Shrinkage pass: pull per-player rates toward team priors ---
    shrinkage_targets = {
        "tov_pct": team_priors.off_tov_pct,
        "orb_pct": team_priors.off_orb_pct,
        "drb_pct": 1.0 - team_priors.off_orb_pct,
        "foul_rate": team_priors.foul_rate_per_100,
        "stl_pct": 2.0,   # league average per 100 poss
        "blk_pct": 1.5,   # league average per 100 poss
        "ft_pct": team_priors.off_ft_pct,
    }
    _team_ts = team_priors.off_efg + 0.04

    shrunk_players: list[Player] = []
    for p in on_court:
        replacements: dict[str, float] = {}
        for attr, target in shrinkage_targets.items():
            raw = getattr(p, attr)
            if raw is None:
                replacements[attr] = target
            else:
                replacements[attr] = shrink_rate(raw, target, p.minutes)
        # Shrink ts_pct
        if p.ts_pct is not None:
            replacements["ts_pct"] = shrink_rate(p.ts_pct, _team_ts, p.minutes)
        shrunk_players.append(dataclasses.replace(p, **replacements))
    on_court = shrunk_players

    # Apply fatigue AFTER shrinkage, not before — otherwise the shrinkage
    # pass blends a fatigued rate back toward the (undegraded) team prior,
    # diluting the fatigue effect most for exactly the low-minutes bench
    # players it should matter most for.
    if fatigue_tracker is not None:
        on_court = [
            apply_fatigue(p, fatigue_tracker.effective_fatigue(p.player_id))
            for p in on_court
        ]

    # 1. Collect raw usage weights (default 0.20 if missing).
    raw_weights = [max(0.0, p.usage_pct if p.usage_pct is not None else 0.20) for p in on_court]
    total = sum(raw_weights)
    if total <= 0:
        # Avoid division by zero — equal weights.
        weights = [1.0 / len(on_court)] * len(on_court)
    else:
        weights = [w / total for w in raw_weights]

    # 2. Fallback mapping (safety net — shrinkage pass already fills most Nones).
    fallbacks = shrinkage_targets

    _DEFENSIVE_ATTRS = {"stl_pct", "blk_pct", "drb_pct"}

    def _weighted_avg(attr: str) -> float:
        total_val = 0.0
        for i, (player, w) in enumerate(zip(on_court, weights, strict=True)):
            val = getattr(player, attr)
            if val is None:
                val = fallbacks[attr]
            if attr in _DEFENSIVE_ATTRS:
                val *= affinity_mult[i]
            total_val += w * val
        return total_val

    # 3. Build shooter tuples. Coach-set ball distribution (if any) overrides
    #    *shot selection* via a separate weight vector and taxes the eFG of
    #    players pushed above their natural usage — but never touches the team
    #    blends above, which keep using usage weights.
    shooter_players = on_court
    shot_weights = _resolve_shot_weights(on_court, weights, shot_distribution)
    if shot_weights is not None:
        shooter_players = [
            _apply_usage_tax(p, shot_w, natural)
            for p, shot_w, natural in zip(on_court, shot_weights, weights, strict=True)
        ]
    shooters = tuple((p, w) for p, w in zip(shooter_players, weights, strict=True))

    # Pace adjustment: compare lineup tempo proxy to team average.
    _TEAM_AVG_TEMPO = 0.20  # mean usage for a balanced team
    lineup_tempos = []
    for p in on_court:
        u = p.usage_pct if p.usage_pct is not None else 0.20
        lineup_tempos.append(u + _position_bonus(p.position))
    lineup_avg_tempo = sum(lineup_tempos) / len(lineup_tempos)
    pace_adj = max(-3.0, min(3.0, (lineup_avg_tempo - _TEAM_AVG_TEMPO) * _PACE_SCALE))

    # eFG adjustment: compare lineup avg shrunk ts_pct to team TS%.
    lineup_ts_vals = []
    for p, w in zip(on_court, weights, strict=True):
        ts = p.ts_pct if p.ts_pct is not None else _team_ts
        lineup_ts_vals.append(w * ts)
    lineup_avg_ts = sum(lineup_ts_vals)
    efg_adj = max(-0.03, min(0.03, lineup_avg_ts - _team_ts))

    return LineupRates(
        tov_pct=_weighted_avg("tov_pct"),
        orb_pct=_weighted_avg("orb_pct"),
        drb_pct=_weighted_avg("drb_pct"),
        foul_rate=_weighted_avg("foul_rate"),
        stl_rate=_weighted_avg("stl_pct"),
        blk_rate=_weighted_avg("blk_pct"),
        ft_pct=_weighted_avg("ft_pct"),
        shooters=shooters,
        pace_adj=pace_adj,
        efg_adj=efg_adj,
        shot_weights=shot_weights,
    )


def _resolve_shot_weights(
    on_court: list[Player],
    natural_weights: list[float],
    shot_distribution: dict[int, float] | None,
) -> tuple[float, ...] | None:
    """Turn a coach's shot-distribution map into a normalized weight vector.

    Returns ``None`` (use usage weights, exact identity) when the map is empty
    or has no entry for any on-court player. Otherwise: on-court players named
    in the map get their requested share; unnamed on-court players split the
    leftover in proportion to natural usage; every share is clamped to
    ``[0.02, 0.60]`` and renormalized to sum 1."""
    if not shot_distribution:
        return None
    ids = [p.player_id for p in on_court]
    set_ids = [pid for pid in ids if pid in shot_distribution]
    if not set_ids:
        return None

    set_sum = sum(shot_distribution[pid] for pid in set_ids)
    leftover = max(0.0, 1.0 - set_sum)
    unset_natural_sum = sum(
        natural_weights[i] for i, pid in enumerate(ids) if pid not in shot_distribution
    )

    raw: list[float] = []
    for i, pid in enumerate(ids):
        if pid in shot_distribution:
            raw.append(shot_distribution[pid])
        elif unset_natural_sum > 0:
            raw.append(natural_weights[i] * leftover / unset_natural_sum)
        else:
            raw.append(0.0)

    clamped = [min(_SHOT_SHARE_CEIL, max(_SHOT_SHARE_FLOOR, r)) for r in raw]
    total = sum(clamped)
    if total <= 0:
        return tuple(1.0 / len(ids) for _ in ids)
    return tuple(c / total for c in clamped)


def _apply_usage_tax(player: Player, shot_weight: float, natural_weight: float) -> Player:
    """Tax a player's eFG for being pushed above their natural usage share.

    -0.6pp TS per +1pp of shot share above natural usage, capped at -6pp.
    FT% is intentionally not taxed (the penalty models shot difficulty, not
    stroke). Returns the player unchanged when no eFG is available or the
    player is not overused."""
    if player.ts_pct is None:
        return player
    over = max(0.0, shot_weight - natural_weight)
    if over <= 0:
        return player
    penalty = _USAGE_TS_PENALTY_SLOPE * min(over, _USAGE_TS_PENALTY_CAP)
    return dataclasses.replace(player, ts_pct=player.ts_pct - penalty)


# ---------------------------------------------------------------------------
# Per-player shot resolution helpers
# ---------------------------------------------------------------------------


def sample_shooter(lr: LineupRates, rng: np.random.Generator) -> Player:
    """Sample a shooter from the lineup.

    Uses the coach's ball-distribution weights (``shot_weights``) when set,
    otherwise the usage weights carried on ``shooters``. If all weights are
    zero, picks uniformly at random.
    """
    players = [p for p, _ in lr.shooters]
    if lr.shot_weights is not None:
        weights = np.array(lr.shot_weights)
    else:
        weights = np.array([w for _, w in lr.shooters])
    if weights.sum() <= 0:
        return players[rng.integers(len(players))]
    probs = weights / weights.sum()
    idx = rng.choice(len(players), p=probs)
    return players[idx]


def player_shot_zone(
    shooter: Player,
    team_priors: TeamPriors,
    rng: np.random.Generator,
) -> str:
    """Return ``"rim"``, ``"mid"``, or ``"three"`` for a shot attempt.

    Uses the shooter's ``fg3a_share`` when available; otherwise falls
    back to the team's ``shot_mix``.
    """
    sm = team_priors.shot_mix
    if shooter.fg3a_share is not None:
        fg3a = max(0.0, min(shooter.fg3a_share, 0.85))
        rim_mid_total = sm.rim + sm.mid
        if rim_mid_total <= 0:
            rim_prob = (1 - fg3a) * 0.5
        else:
            rim_prob = (1 - fg3a) * (sm.rim / rim_mid_total)
        mid_prob = 1 - fg3a - rim_prob
        probs = [rim_prob, mid_prob, fg3a]
    else:
        probs = [sm.rim, sm.mid, sm.three]

    zones = ["rim", "mid", "three"]
    return zones[rng.choice(len(zones), p=probs)]


def player_zone_make_prob(
    shooter: Player,
    zone: str,
    team_priors: TeamPriors,
) -> float:
    """Return the make probability for *shooter* in *zone*.

    Scales the team baseline by the player's true-shooting percentage
    when available.  Result is clamped to ``[0.05, 0.95]``.
    """
    zone_efg = team_priors.zone_efg
    base = getattr(zone_efg, zone)
    if shooter.ts_pct is not None:
        ratio = shooter.ts_pct / (team_priors.off_efg + 0.04)
        prob = base * ratio
    else:
        prob = base
    return max(0.05, min(prob, 0.95))
