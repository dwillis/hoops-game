"""Scheme–archetype affinity: how well a player's profile fits each
defensive scheme.

Each player is classified into an archetype based on their rate profile,
then an affinity table maps (archetype, scheme) → a multiplier that the
engine can use to adjust defensive effectiveness per possession.
"""

from __future__ import annotations

from hoops.data.rosters import Player
from hoops.engine.policy import DefensiveScheme

# ---------------------------------------------------------------------------
# Affinity table: archetype → {scheme: multiplier}
# ---------------------------------------------------------------------------
_AFFINITY: dict[str, dict[DefensiveScheme, float]] = {
    "rim_protector":     {DefensiveScheme.MAN: 1.00, DefensiveScheme.ZONE: 1.15, DefensiveScheme.PRESS: 0.90},
    "perimeter_stopper": {DefensiveScheme.MAN: 1.10, DefensiveScheme.ZONE: 0.95, DefensiveScheme.PRESS: 1.15},
    "ball_handler":      {DefensiveScheme.MAN: 1.00, DefensiveScheme.ZONE: 1.00, DefensiveScheme.PRESS: 1.10},
    "floor_spacer":      {DefensiveScheme.MAN: 1.00, DefensiveScheme.ZONE: 0.95, DefensiveScheme.PRESS: 1.00},
    "versatile_wing":    {DefensiveScheme.MAN: 1.05, DefensiveScheme.ZONE: 1.00, DefensiveScheme.PRESS: 1.05},
    "default":           {DefensiveScheme.MAN: 1.00, DefensiveScheme.ZONE: 1.00, DefensiveScheme.PRESS: 1.00},
}


def detect_archetype(p: Player) -> str:
    """Classify a player into an archetype from their rate profile.

    Thresholds are calibrated for WBB rates.  Any ``None`` rate is
    treated as 0.0 so a bare ``Player`` with no rate fields safely falls
    through to ``"default"``.
    """
    blk_pct = p.blk_pct if p.blk_pct is not None else 0.0
    drb_pct = p.drb_pct if p.drb_pct is not None else 0.0
    stl_pct = p.stl_pct if p.stl_pct is not None else 0.0
    usage_pct = p.usage_pct if p.usage_pct is not None else 0.0
    ast_pct = p.ast_pct if p.ast_pct is not None else 0.0
    fg3a_share = p.fg3a_share if p.fg3a_share is not None else 0.0

    if blk_pct >= 2.5 and drb_pct >= 12.0:
        return "rim_protector"
    if stl_pct >= 4.0 and blk_pct < 2.5:
        return "perimeter_stopper"
    if usage_pct >= 0.25 and ast_pct >= 8.0:
        return "ball_handler"
    if fg3a_share >= 0.45 and stl_pct < 4.0:
        return "floor_spacer"
    return "default"


def scheme_affinity(p: Player) -> dict[DefensiveScheme, float]:
    """Return per-scheme multipliers for *p* based on archetype."""
    archetype = detect_archetype(p)
    return dict(_AFFINITY[archetype])
