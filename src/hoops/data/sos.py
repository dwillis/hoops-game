"""KenPom-style strength-of-schedule adjustment.

The raw season averages from ``team_box`` describe what each team actually
did, against the actual schedule they played. Two teams with identical raw
eFG might be very different in true offensive ability if one played a much
tougher set of defenses.

The standard Sagarin/Massey/KenPom approach: solve the system

    raw_off[T]  =  adj_off[T]  +  (mean_{opp ∈ sched(T)} adj_def[opp]  -  league_mean)
    raw_def[T]  =  adj_def[T]  +  (mean_{opp ∈ sched(T)} adj_off[opp]  -  league_mean)

iteratively. After convergence ``adj_off[T]`` is what team T would do
against a league-average defense, and ``adj_def[T]`` is what they'd allow
against a league-average offense — apples-to-apples comparable across
teams regardless of schedule strength.

Each four-factor stat is adjusted independently (eFG, TOV%, ORB%, FTR).
``compute_schedule`` produces the (team, opp, games) edges; ``adjust``
runs the iteration. Both are league-agnostic; the projections layer
calls them with the right inputs per league.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

# {team_id: {opponent_id: n_games}}
Schedule = Mapping[int, Mapping[int, int]]


def compute_schedule(team_box: pl.DataFrame) -> dict[int, dict[int, int]]:
    """Build a (team -> opponent -> games) edge map from a raw team_box frame."""
    edges = (
        team_box.group_by(["team_id", "opponent_team_id"])
        .agg(pl.len().alias("n"))
    )
    out: dict[int, dict[int, int]] = {}
    for row in edges.iter_rows(named=True):
        out.setdefault(row["team_id"], {})[row["opponent_team_id"]] = row["n"]
    return out


def adjust(
    raw_off: Mapping[int, float],
    raw_def: Mapping[int, float],
    schedule: Schedule,
    league_mean: float,
    *,
    n_iter: int = 80,
    tol: float = 1e-5,
    damping: float = 0.5,
) -> tuple[dict[int, float], dict[int, float]]:
    """Iteratively SoS-adjust offensive and defensive rates.

    Damping keeps the updates stable when a team's schedule includes many
    teams that haven't been adjusted yet (or non-D-I opponents whose
    raw rates are noisy). Without damping, sufficiently extreme schedules
    can produce small oscillations.
    """
    teams = list(raw_off.keys())
    adj_off: dict[int, float] = dict(raw_off)
    adj_def: dict[int, float] = dict(raw_def)

    for _ in range(n_iter):
        new_off: dict[int, float] = {}
        new_def: dict[int, float] = {}
        for t in teams:
            opps = schedule.get(t, {})
            total = sum(opps.values())
            if total == 0:
                new_off[t] = raw_off[t]
                new_def[t] = raw_def[t]
                continue
            avg_opp_def = sum(
                adj_def.get(o, league_mean) * n for o, n in opps.items()
            ) / total
            avg_opp_off = sum(
                adj_off.get(o, league_mean) * n for o, n in opps.items()
            ) / total
            target_off = raw_off[t] - (avg_opp_def - league_mean)
            target_def = raw_def[t] - (avg_opp_off - league_mean)
            new_off[t] = (1 - damping) * adj_off[t] + damping * target_off
            new_def[t] = (1 - damping) * adj_def[t] + damping * target_def

        delta = max(
            max(abs(new_off[t] - adj_off[t]) for t in teams),
            max(abs(new_def[t] - adj_def[t]) for t in teams),
        )
        adj_off = new_off
        adj_def = new_def
        if delta < tol:
            break

    # Re-center: adjusted league means should equal raw league means.
    # Without this, isolated subgraphs (D-II teams that play mostly each
    # other but appear in our raw box) can drift the absolute scale even
    # though relative ordering is preserved.
    raw_off_mean = sum(raw_off.values()) / len(raw_off)
    raw_def_mean = sum(raw_def.values()) / len(raw_def)
    adj_off_mean = sum(adj_off.values()) / len(adj_off)
    adj_def_mean = sum(adj_def.values()) / len(adj_def)
    off_shift = raw_off_mean - adj_off_mean
    def_shift = raw_def_mean - adj_def_mean
    adj_off = {t: v + off_shift for t, v in adj_off.items()}
    adj_def = {t: v + def_shift for t, v in adj_def.items()}

    return adj_off, adj_def
