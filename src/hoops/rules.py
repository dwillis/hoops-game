from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from hoops.data.paths import DATA_ROOT
from hoops.league import League

RULES_DIR = DATA_ROOT / "rules"


class UnsupportedSeasonError(KeyError):
    """Raised when no rules are defined for a (league, season)."""


class Rules(BaseModel):
    model_config = ConfigDict(frozen=True)

    league: League
    structure: Literal["quarters", "halves"]
    quarter_minutes: int | None = None
    half_minutes: int | None = None
    shot_clock_seconds: int
    three_point_distance_ft: float
    bonus: str
    timeouts_per_team: int
    ot_minutes: int
    personal_foul_limit: int

    @property
    def regulation_minutes(self) -> int:
        if self.structure == "quarters":
            assert self.quarter_minutes is not None
            return 4 * self.quarter_minutes
        assert self.half_minutes is not None
        return 2 * self.half_minutes


@lru_cache(maxsize=4)
def _load_table(league: League) -> dict[str, dict]:
    path = RULES_DIR / f"{league.value}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("seasons", {}) or {}


def rules_for(league: League, season: str) -> Rules:
    table = _load_table(league)
    if season not in table:
        raise UnsupportedSeasonError(
            f"No rules for league={league.value!r} season={season!r}; "
            f"v1 supports {sorted(table) if table else '(none)'}"
        )
    return Rules.model_validate(table[season])


def available_seasons(league: League) -> list[str]:
    return sorted(_load_table(league))
