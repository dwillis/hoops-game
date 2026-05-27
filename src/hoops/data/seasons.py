"""Season-string utilities.

We display seasons as ``"2023-24"`` (the doc's convention). External datasets
like sportsdataverse use the season-end year as an integer (``2024``). This
module is the single point that converts between them.
"""

from __future__ import annotations

import re

_SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")


class InvalidSeasonError(ValueError):
    pass


def season_end_year(season: str) -> int:
    """``"2023-24"`` -> ``2024``."""
    m = _SEASON_RE.match(season)
    if not m:
        raise InvalidSeasonError(f"expected 'YYYY-YY' season string, got {season!r}")
    start_full = int(m.group(1))
    end_short = int(m.group(2))
    end_century = (start_full + 1) // 100 * 100
    end_full = end_century + end_short
    if end_full != start_full + 1:
        raise InvalidSeasonError(
            f"season {season!r} second half should be the year after the first"
        )
    return end_full


def season_string(end_year: int) -> str:
    """``2024`` -> ``"2023-24"``."""
    if end_year < 1900 or end_year > 2200:
        raise InvalidSeasonError(f"implausible end_year {end_year!r}")
    start = end_year - 1
    return f"{start}-{end_year % 100:02d}"
