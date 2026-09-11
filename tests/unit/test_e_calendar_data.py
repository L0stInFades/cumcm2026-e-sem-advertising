from __future__ import annotations

import math

import pandas as pd

from pipelines.e.calendar import ADJUSTED_WORKDAYS, HOLIDAY_BLOCKS, day_features, holiday_map
from pipelines.e.data import parse_bounce, parse_duration


def test_holiday_calendar_matches_state_council_notice() -> None:
    hm = holiday_map()
    assert len(hm) == 1 + 8 + 3 + 5 + 3 + 8
    assert hm[pd.Timestamp("2025-01-28")] == "春节" and hm[pd.Timestamp("2025-10-08")] == "国庆节中秋节"
    assert pd.Timestamp("2025-02-08") not in hm and "2025-02-08" in ADJUSTED_WORKDAYS
    assert all(pd.Timestamp(s) <= pd.Timestamp(e) for s, e in HOLIDAY_BLOCKS.values())


def test_day_features_flags() -> None:
    f = day_features(
        ["2025-01-27", "2025-01-28", "2025-02-04", "2025-02-05", "2025-02-08", "2025-02-09", "2026-09-11"]
    ).set_index("date")
    assert bool(f.loc["2025-01-27", "is_pre_holiday"]) and not bool(f.loc["2025-01-27", "is_holiday"])
    assert bool(f.loc["2025-01-28", "is_holiday"]) and f.loc["2025-01-28", "day_type"] == "holiday"
    assert bool(f.loc["2025-02-05", "is_post_holiday"]) and f.loc["2025-02-05", "day_type"] == "workday"
    assert f.loc["2025-02-08", "day_type"] == "adjusted_workday" and bool(f.loc["2025-02-08", "is_workday"])
    assert f.loc["2025-02-09", "day_type"] == "weekend"
    assert f.loc["2026-09-11", "weekday"] == 4 and f.loc["2026-09-11", "day_type"] == "workday"


def test_parsers_handle_every_encoding_in_sheet3() -> None:
    assert math.isnan(parse_bounce("/")) and parse_bounce("0.6109") == 0.6109 and parse_bounce("1") == 1.0
    assert parse_bounce(" 45 ") == 0.45
    assert math.isnan(parse_duration("/")) and parse_duration(" 00:03:16") == 196.0 and parse_duration("0") == 0.0
    assert parse_duration(0) == 0.0
