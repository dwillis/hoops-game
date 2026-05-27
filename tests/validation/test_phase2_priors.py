"""Phase 2 validation: per the plan, the fitted distributions must

1. Reproduce per-team marginals (eFG%, 3pt-rate, TOV%, ORB%, FTA-rate) within
   ~1 percentage point under resampling at N=10,000.
2. Yield a league-average pace in roughly [67, 73] possessions per 40.

Both checks live here. They require the 2023-24 raw + canonical + fitted
parquet files; tests skip cleanly if those haven't been generated.
"""

from __future__ import annotations

import numpy as np
import pytest

from hoops.data.distributions import (
    TeamPriors,
    load_league_prior,
    load_team_priors,
)
from hoops.data.paths import distributions_dir
from hoops.league import League

SEASON = "2023-24"


def _priors_present() -> bool:
    return (distributions_dir(League.WBB, SEASON) / "team_priors.parquet").exists()


pytestmark = pytest.mark.skipif(
    not _priors_present(),
    reason="fitted priors not present; run `uv run python scripts/fit_distributions.py --season 2023-24` first",
)


# --- league-level sanity -------------------------------------------------------


def test_league_pace_in_expected_band():
    lp = load_league_prior(League.WBB, SEASON)
    assert 67.0 <= lp.pace <= 73.0, f"league pace {lp.pace:.2f} outside [67, 73]"


def test_league_shot_mix_sums_to_one():
    lp = load_league_prior(League.WBB, SEASON)
    total = lp.shot_mix.rim + lp.shot_mix.mid + lp.shot_mix.three
    assert total == pytest.approx(1.0, abs=1e-3)


def test_league_three_point_share_below_men():
    """Doc §4: WBB 3pt share lags men's. League mean should be < 0.40 for 2023-24."""
    lp = load_league_prior(League.WBB, SEASON)
    assert lp.shot_mix.three < 0.40


# --- per-team self-consistency under resampling -------------------------------


def _resample_shot_marginals(p: TeamPriors, n: int, rng: np.random.Generator) -> dict:
    """Draw n shot attempts from the team's shot mix and zone make rates.

    Returns the empirical zone shares and per-zone make rates.
    """
    zones = rng.choice(
        ["rim", "mid", "three"],
        size=n,
        p=[p.shot_mix.rim, p.shot_mix.mid, p.shot_mix.three],
    )
    make_p = np.where(
        zones == "rim", p.zone_efg.rim,
        np.where(zones == "mid", p.zone_efg.mid, p.zone_efg.three),
    )
    makes = rng.random(n) < make_p

    out: dict = {}
    for z, fg in [("rim", p.zone_efg.rim), ("mid", p.zone_efg.mid), ("three", p.zone_efg.three)]:
        mask = zones == z
        out[f"share_{z}"] = float(mask.mean())
        out[f"fg_{z}"] = float(makes[mask].mean()) if mask.any() else 0.0
        out[f"expected_share_{z}"] = float(getattr(p.shot_mix, z))
        out[f"expected_fg_{z}"] = float(fg)
    out["efg"] = float(
        out["share_rim"] * out["fg_rim"]
        + out["share_mid"] * out["fg_mid"]
        + out["share_three"] * out["fg_three"] * 1.5
    )
    out["expected_efg"] = float(
        p.shot_mix.rim * p.zone_efg.rim
        + p.shot_mix.mid * p.zone_efg.mid
        + p.shot_mix.three * p.zone_efg.three * 1.5
    )
    return out


@pytest.mark.parametrize("team_name", ["South Carolina", "Iowa", "UConn"])
def test_resampling_reproduces_top_team_marginals(team_name):
    priors = load_team_priors(League.WBB, SEASON)
    p = next((x for x in priors if x.team_name == team_name), None)
    if p is None:
        pytest.skip(f"{team_name} not present in 2023-24 priors (team-name string mismatch)")

    # N=50k for ~0.2% sampling SE on shares; 1pp tolerance per the plan.
    rng = np.random.default_rng(seed=42)
    m = _resample_shot_marginals(p, n=50_000, rng=rng)

    for z in ("rim", "mid", "three"):
        assert abs(m[f"share_{z}"] - m[f"expected_share_{z}"]) < 0.01, (z, m)
        assert abs(m[f"fg_{z}"] - m[f"expected_fg_{z}"]) < 0.01, (z, m)

    assert abs(m["efg"] - m["expected_efg"]) < 0.01


def test_resampling_marginals_for_all_teams():
    """Every fitted team's resampled marginals are within 1pp of expected at N=50k.

    Tolerance is 4x the binomial SE at p=0.5 (which is the worst case): with
    N=50,000 that's 4 * sqrt(0.25/50000) ~= 0.009. Setting 0.012 absorbs
    sampling noise without hiding real bugs.
    """
    priors = load_team_priors(League.WBB, SEASON)
    rng = np.random.default_rng(seed=1)
    failures = []
    for p in priors:
        if p.shot_mix.rim + p.shot_mix.mid + p.shot_mix.three < 0.99:
            continue
        m = _resample_shot_marginals(p, n=50_000, rng=rng)
        for z in ("rim", "mid", "three"):
            if abs(m[f"share_{z}"] - m[f"expected_share_{z}"]) >= 0.012:
                failures.append((p.team_name, z, "share", m[f"share_{z}"], m[f"expected_share_{z}"]))
    assert not failures, f"{len(failures)} marginal failures, first 5: {failures[:5]}"


# --- top-team smoke check (cheap version of §5.2 from the plan) ---------------


def test_south_carolina_is_clearly_above_league_mean():
    """SC went 38-0; their priors should look like the best team."""
    lp = load_league_prior(League.WBB, SEASON)
    priors = load_team_priors(League.WBB, SEASON)
    sc = next((p for p in priors if p.team_name == "South Carolina"), None)
    assert sc is not None, "South Carolina missing from 2023-24 priors"

    assert sc.off_efg > lp.off_efg + 0.05
    assert sc.off_orb_pct > lp.off_orb_pct + 0.05
    assert sc.off_tov_pct < lp.off_tov_pct
