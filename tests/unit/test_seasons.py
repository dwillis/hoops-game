import pytest

from hoops.data.seasons import InvalidSeasonError, season_end_year, season_string


def test_season_end_year_basic():
    assert season_end_year("2023-24") == 2024
    assert season_end_year("2015-16") == 2016
    assert season_end_year("2025-26") == 2026


def test_season_end_year_century_rollover():
    assert season_end_year("1999-00") == 2000


def test_season_end_year_rejects_garbage():
    with pytest.raises(InvalidSeasonError):
        season_end_year("2023-2024")
    with pytest.raises(InvalidSeasonError):
        season_end_year("2023")
    with pytest.raises(InvalidSeasonError):
        season_end_year("2023-25")  # not consecutive


def test_season_string_roundtrip():
    for end in (2016, 2024, 2026, 2000):
        s = season_string(end)
        assert season_end_year(s) == end
