"""Matchup adjustment: mix one team's offense with another's defense.

The fitted ``TeamPriors`` are season averages against an unweighted schedule.
A possession from team A vs team B is *not* well-modeled by team A's raw
offensive priors alone — that ignores team B's defensive strength.

Standard approach (Dean Oliver / KenPom-style, though simpler than KenPom):

    eff_X = off.off_X + (def_.def_X - league.X)

This re-centers each rate by how far above/below league average team B's
defense is. Team A's offense plus the league-relative quality of team B's
defense.

For the per-zone ``zone_efg`` (which we don't have a defensive analogue
for), we rescale uniformly across zones so the *aggregate* adjusted eFG
matches the per-formula adjusted target. This preserves the team's shot
mix and the relative finishing across zones, while still letting a tough
defense drag the offense down.
"""

from __future__ import annotations

from hoops.data.distributions import LeaguePrior, ShotMix, TeamPriors, ZoneEFG
from hoops.engine.policy import DefensiveScheme, OffensiveScheme


def _nominal_efg(p: TeamPriors) -> float:
    return (
        p.shot_mix.rim * p.zone_efg.rim
        + p.shot_mix.mid * p.zone_efg.mid
        + p.shot_mix.three * p.zone_efg.three * 1.5
    )


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def adjust_offense(
    off: TeamPriors,
    opp: TeamPriors,
    league: LeaguePrior,
) -> TeamPriors:
    """Return ``off`` re-centered for the matchup against ``opp``.

    ``off`` keeps its identity (team_id, name, shot_mix); only the rates
    that the engine consumes per possession are shifted.
    """
    eff_tov = _clip(off.off_tov_pct + (opp.def_tov_pct - league.off_tov_pct), 0.05, 0.40)
    eff_orb = _clip(off.off_orb_pct + (opp.def_orb_pct - league.off_orb_pct), 0.10, 0.55)
    eff_ftr = _clip(off.off_fta_rate + (opp.def_fta_rate - league.off_fta_rate), 0.05, 0.60)
    eff_efg_target = _clip(off.off_efg + (opp.def_efg - league.off_efg), 0.25, 0.65)

    nominal = _nominal_efg(off)
    mult = eff_efg_target / nominal if nominal > 0 else 1.0
    eff_zone = ZoneEFG(
        rim=_clip(off.zone_efg.rim * mult, 0.05, 0.95),
        mid=_clip(off.zone_efg.mid * mult, 0.05, 0.95),
        three=_clip(off.zone_efg.three * mult, 0.05, 0.95),
    )

    return off.model_copy(update={
        "off_tov_pct": eff_tov,
        "off_orb_pct": eff_orb,
        "off_fta_rate": eff_ftr,
        "off_efg": eff_efg_target,
        "zone_efg": eff_zone,
    })


def apply_scheme(off: TeamPriors, scheme: DefensiveScheme) -> TeamPriors:
    """Apply the defense's scheme to the offense's priors.

    Per doc §3.4 we use *flat* adjustments because the scheme-tagged
    coaching data is too thin to fit multiplicative effects per team.
    The directions encoded here:

    - **Man**: baseline (no change).
    - **Zone**: chases shooters off the arc into mid-range; reduces 3pt
      attempts (-3pp), bumps mid-range (+3pp); 3pt make rate dips slightly
      (-2pp on the rate).
    - **Press**: forces turnovers (+3pp TOV%) but yields easier rim
      attempts when broken (+3pp on rim eFG). Stylized but directional.

    These shifts are small enough that league means stay roughly intact
    if every team plays the same scheme — they're matchup-relative.
    """
    if scheme is DefensiveScheme.MAN:
        return off
    if scheme is DefensiveScheme.ZONE:
        new_mix = ShotMix(
            rim=off.shot_mix.rim,
            mid=_clip(off.shot_mix.mid + 0.03, 0.05, 0.85),
            three=_clip(off.shot_mix.three - 0.03, 0.05, 0.70),
        )
        new_zone = ZoneEFG(
            rim=off.zone_efg.rim,
            mid=off.zone_efg.mid,
            three=_clip(off.zone_efg.three - 0.02, 0.05, 0.95),
        )
        return off.model_copy(update={
            "shot_mix": new_mix,
            "zone_efg": new_zone,
        })
    if scheme is DefensiveScheme.PRESS:
        return off.model_copy(update={
            "off_tov_pct": _clip(off.off_tov_pct + 0.03, 0.05, 0.45),
            "zone_efg": ZoneEFG(
                rim=_clip(off.zone_efg.rim + 0.03, 0.05, 0.95),
                mid=off.zone_efg.mid,
                three=off.zone_efg.three,
            ),
        })
    return off


def apply_off_scheme(off: TeamPriors, scheme: OffensiveScheme) -> TeamPriors:
    """Apply the offense's own scheme to their priors.

    Flat adjustments mirroring apply_scheme() for defense:
    - NORMAL: baseline (no change)
    - HURRY_UP: +3 pace, +1.5pp TOV%
    - SLOW_DOWN: -3 pace, -1.5pp TOV%, -1pp all zone eFG
    - THREE_POINT: +5pp three share / -5pp mid share, -1pp three eFG
    """
    if scheme is OffensiveScheme.NORMAL:
        return off
    if scheme is OffensiveScheme.HURRY_UP:
        return off.model_copy(update={
            "pace": off.pace + 3.0,
            "off_tov_pct": _clip(off.off_tov_pct + 0.015, 0.05, 0.40),
        })
    if scheme is OffensiveScheme.SLOW_DOWN:
        return off.model_copy(update={
            "pace": off.pace - 3.0,
            "off_tov_pct": _clip(off.off_tov_pct - 0.015, 0.05, 0.40),
            "zone_efg": ZoneEFG(
                rim=_clip(off.zone_efg.rim - 0.01, 0.05, 0.95),
                mid=_clip(off.zone_efg.mid - 0.01, 0.05, 0.95),
                three=_clip(off.zone_efg.three - 0.01, 0.05, 0.95),
            ),
        })
    if scheme is OffensiveScheme.THREE_POINT:
        return off.model_copy(update={
            "shot_mix": ShotMix(
                rim=off.shot_mix.rim,
                mid=_clip(off.shot_mix.mid - 0.05, 0.05, 0.85),
                three=_clip(off.shot_mix.three + 0.05, 0.05, 0.70),
            ),
            "zone_efg": ZoneEFG(
                rim=off.zone_efg.rim,
                mid=off.zone_efg.mid,
                three=_clip(off.zone_efg.three - 0.01, 0.05, 0.95),
            ),
        })
    return off


# ---------------------------------------------------------------------------
# Home-court advantage
# ---------------------------------------------------------------------------

# Realistic WBB HCA: ~3-4 point advantage per game.
_HCA_FT_PENALTY = 0.02     # away FT% reduced by 2pp (crowd noise)
_HCA_TOV_PENALTY = 0.01    # away TOV% increased by 1pp (hostile environment)
_HCA_EFG_PENALTY = 0.01    # away eFG reduced by 1pp (unfamiliar gym)


def apply_hca(away: TeamPriors) -> TeamPriors:
    """Apply home-court-advantage penalties to the *away* team's priors.

    The home team keeps its raw (post-matchup-adjusted) priors unchanged.
    Called once at game start, after :func:`adjust_offense`.
    """
    new_ft = _clip(away.off_ft_pct - _HCA_FT_PENALTY, 0.05, 0.95)
    new_tov = _clip(away.off_tov_pct + _HCA_TOV_PENALTY, 0.05, 0.40)
    new_efg = _clip(away.off_efg - _HCA_EFG_PENALTY, 0.25, 0.65)

    # Scale zone eFG proportionally so the zone breakdown stays consistent.
    ratio = new_efg / away.off_efg if away.off_efg > 0 else 1.0
    new_zone = ZoneEFG(
        rim=_clip(away.zone_efg.rim * ratio, 0.05, 0.95),
        mid=_clip(away.zone_efg.mid * ratio, 0.05, 0.95),
        three=_clip(away.zone_efg.three * ratio, 0.05, 0.95),
    )

    return away.model_copy(update={
        "off_ft_pct": new_ft,
        "off_tov_pct": new_tov,
        "off_efg": new_efg,
        "zone_efg": new_zone,
    })
