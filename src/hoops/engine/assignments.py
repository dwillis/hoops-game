"""Defensive man-to-man assignments (who guards whom).

The base engine has no per-player defense concept — defense enters only as
team ``def_*`` priors, the scheme, and archetype affinity. This module adds a
bounded, opt-in man-matchup layer used only when the defense is in MAN.

Design (see the plan):

- Each defender gets two synthesized, *lineup-relative* ratings — perimeter
  (P) and interior (I) — from steals/blocks/rebounding/fouls plus an archetype
  bonus. Relative z-scores dodge the absolute-unit ambiguity in the raw rates.
- A deterministic greedy algorithm produces the *default* assignment map.
- All in-game effects are computed as **deltas versus that default map**, so
  when the coach hasn't changed anything (``man_assignments is None`` or equal
  to the default) every effect is exactly zero — no calibration exposure.
"""

from __future__ import annotations

from hoops.data.rosters import Player
from hoops.engine.scheme_affinity import detect_archetype

# Make-probability swing per unit of matchup-rating advantage, and its cap.
_MAKE_DELTA_SCALE = 0.03
_MAKE_DELTA_CAP = 0.03
# Extra shooting-foul probability per G<->C mismatch (relative to default).
_MISMATCH_FOUL_BUMP = 0.01

_ARCH_P = {
    "perimeter_stopper": 0.4,
    "versatile_wing": 0.2,
    "rim_protector": -0.2,
}
_ARCH_I = {
    "rim_protector": 0.4,
    "perimeter_stopper": -0.1,
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rate(p: Player, attr: str) -> float:
    v = getattr(p, attr, None)
    return float(v) if v is not None else 0.0


def _rel_z(vals: list[float]) -> list[float]:
    """Lineup-relative deviation, clamped to [-1, 1]."""
    if not vals:
        return []
    mean = sum(vals) / len(vals)
    denom = max(abs(mean), 1e-6)
    return [_clip((v - mean) / denom, -1.0, 1.0) for v in vals]


def defensive_ratings(defenders: list[Player]) -> dict[int, tuple[float, float]]:
    """Return ``{player_id: (perimeter, interior)}`` for the defending five."""
    z_stl = _rel_z([_rate(p, "stl_pct") for p in defenders])
    z_blk = _rel_z([_rate(p, "blk_pct") for p in defenders])
    z_drb = _rel_z([_rate(p, "drb_pct") for p in defenders])
    z_foul = _rel_z([_rate(p, "foul_rate") for p in defenders])

    ratings: dict[int, tuple[float, float]] = {}
    for i, p in enumerate(defenders):
        arch = detect_archetype(p)
        perim = _clip(
            0.5 * z_stl[i] + 0.2 * z_drb[i] - 0.15 * z_foul[i] + _ARCH_P.get(arch, 0.0),
            -1.0, 1.0,
        )
        interior = _clip(
            0.5 * z_blk[i] + 0.3 * z_drb[i] - 0.15 * z_foul[i] + _ARCH_I.get(arch, 0.0),
            -1.0, 1.0,
        )
        ratings[p.player_id] = (perim, interior)
    return ratings


def opponent_type(p: Player) -> str:
    """Classify an offensive player as ``perimeter`` / ``interior`` / ``hybrid``."""
    pos = (p.position or "").upper()
    share = p.fg3a_share if p.fg3a_share is not None else 0.30
    if "G" in pos or share >= 0.35:
        return "perimeter"
    if "C" in pos or share < 0.15:
        return "interior"
    return "hybrid"


def matchup_rating(rating: tuple[float, float], opp: Player) -> float:
    """The rating dimension relevant to guarding ``opp``."""
    perim, interior = rating
    t = opponent_type(opp)
    if t == "perimeter":
        return perim
    if t == "interior":
        return interior
    return (perim + interior) / 2.0


_POS_ORDER = {"G": 0, "F": 1, "C": 2}


def _pos_letters(p: Player) -> list[str]:
    pos = (p.position or "").upper().replace("-", "/")
    return [x.strip() for x in pos.split("/") if x.strip() in _POS_ORDER]


def _pos_match(defender: Player, opp: Player) -> int:
    """2 = shared position letter, 1 = adjacent (G-F/F-C), 0 = G-C only."""
    d = _pos_letters(defender)
    o = _pos_letters(opp)
    if not d or not o:
        return 1
    if set(d) & set(o):
        return 2
    best = 0
    for a in d:
        for b in o:
            dist = abs(_POS_ORDER[a] - _POS_ORDER[b])
            best = max(best, 2 if dist == 0 else (1 if dist == 1 else 0))
    return best


def default_assignments(
    defenders: list[Player], opponents: list[Player]
) -> dict[int, int]:
    """Greedy default map ``{defender_id: opponent_id}``.

    Opponents are matched in descending usage; each takes the unassigned
    defender maximizing ``2 * position_match + matchup_rating``. Deterministic
    (no RNG)."""
    ratings = defensive_ratings(defenders)
    ordered_opps = sorted(
        opponents,
        key=lambda p: (p.usage_pct if p.usage_pct is not None else 0.20),
        reverse=True,
    )
    used: set[int] = set()
    mapping: dict[int, int] = {}
    for opp in ordered_opps:
        best_def = None
        best_fit = None
        for d in defenders:
            if d.player_id in used:
                continue
            fit = 2 * _pos_match(d, opp) + matchup_rating(ratings[d.player_id], opp)
            if best_fit is None or fit > best_fit:
                best_fit = fit
                best_def = d
        if best_def is not None:
            mapping[best_def.player_id] = opp.player_id
            used.add(best_def.player_id)
    return mapping


def resolve_actual_map(
    coach_map: dict[int, int] | None,
    defenders: list[Player],
    opponents: list[Player],
) -> dict[int, int]:
    """Merge a coach's partial assignment map onto the greedy default.

    Coach pairs are honored only when both players are on court; every other
    slot is filled by the default algorithm over the remaining players."""
    default = default_assignments(defenders, opponents)
    if not coach_map:
        return default

    def_ids = {p.player_id for p in defenders}
    opp_ids = {p.player_id for p in opponents}
    actual: dict[int, int] = {}
    used_def: set[int] = set()
    used_opp: set[int] = set()
    for did, oid in coach_map.items():
        if did in def_ids and oid in opp_ids and did not in used_def and oid not in used_opp:
            actual[did] = oid
            used_def.add(did)
            used_opp.add(oid)

    # Keep default pairings where both endpoints are still free.
    for did, oid in default.items():
        if did in used_def or oid in used_opp:
            continue
        actual[did] = oid
        used_def.add(did)
        used_opp.add(oid)

    # Whatever is left (defenders whose default opponent was taken by a coach
    # pair) is matched greedily against the remaining opponents so the map
    # stays a complete 1:1.
    leftover_def = [p for p in defenders if p.player_id not in used_def]
    leftover_opp = [p for p in opponents if p.player_id not in used_opp]
    for did, oid in default_assignments(leftover_def, leftover_opp).items():
        actual[did] = oid
    return actual


def _defender_of(mapping: dict[int, int], opp_id: int) -> int | None:
    for did, oid in mapping.items():
        if oid == opp_id:
            return did
    return None


def shooter_make_delta(
    actual_map: dict[int, int],
    default_map: dict[int, int],
    ratings: dict[int, tuple[float, float]],
    defenders_by_id: dict[int, Player],
    shooter: Player,
) -> float:
    """Make-probability delta for ``shooter`` vs. the default matchup.

    A tougher-than-default defender lowers the make prob; an easier one raises
    it. Zero when the actual map matches the default for this shooter."""
    a_def = _defender_of(actual_map, shooter.player_id)
    d_def = _defender_of(default_map, shooter.player_id)
    if a_def is None or d_def is None or a_def == d_def:
        return 0.0
    m_actual = matchup_rating(ratings[a_def], shooter)
    m_default = matchup_rating(ratings[d_def], shooter)
    return _clip(-_MAKE_DELTA_SCALE * (m_actual - m_default), -_MAKE_DELTA_CAP, _MAKE_DELTA_CAP)


def shooter_foul_delta(
    actual_map: dict[int, int],
    default_map: dict[int, int],
    defenders_by_id: dict[int, Player],
    shooter: Player,
) -> float:
    """Extra shooting-foul probability from a G<->C mismatch vs. the default."""
    a_def = _defender_of(actual_map, shooter.player_id)
    d_def = _defender_of(default_map, shooter.player_id)
    if a_def is None or d_def is None or a_def == d_def:
        return 0.0

    def is_gc_mismatch(defender_id: int) -> int:
        defender = defenders_by_id.get(defender_id)
        if defender is None:
            return 0
        return 1 if _pos_match(defender, shooter) == 0 else 0

    return _MISMATCH_FOUL_BUMP * (is_gc_mismatch(a_def) - is_gc_mismatch(d_def))
