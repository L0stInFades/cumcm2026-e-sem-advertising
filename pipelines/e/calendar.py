"""Statutory holiday calendar (State Council notice for 2025) and per-day calendar features."""

from __future__ import annotations

from typing import Any

import pandas as pd

# (first day, last day) of every statutory holiday block in 2025, inclusive.
HOLIDAY_BLOCKS: dict[str, tuple[str, str]] = {
    "元旦": ("2025-01-01", "2025-01-01"),
    "春节": ("2025-01-28", "2025-02-04"),
    "清明节": ("2025-04-04", "2025-04-06"),
    "劳动节": ("2025-05-01", "2025-05-05"),
    "端午节": ("2025-05-31", "2025-06-02"),
    "国庆节中秋节": ("2025-10-01", "2025-10-08"),
}
# Weekend days declared working days to compensate for the long holidays (调休).
ADJUSTED_WORKDAYS: tuple[str, ...] = ("2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11")
WEEKDAY_NAMES: tuple[str, ...] = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def holiday_map() -> dict[pd.Timestamp, str]:
    """Every statutory holiday date -> holiday name."""
    out: dict[pd.Timestamp, str] = {}
    for name, (start, end) in HOLIDAY_BLOCKS.items():
        for day in pd.date_range(start, end, freq="D"):
            out[day] = name
    return out


def day_features(dates: Any) -> pd.DataFrame:
    """Calendar covariates for a sequence of dates (weekday, month, holiday flags, day type)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    holidays = holiday_map()
    adjusted = {pd.Timestamp(d) for d in ADJUSTED_WORKDAYS}
    starts = {pd.Timestamp(s) for s, _ in HOLIDAY_BLOCKS.values()}
    ends = {pd.Timestamp(e) for _, e in HOLIDAY_BLOCKS.values()}
    one = pd.Timedelta(days=1)
    rows: list[dict[str, Any]] = []
    for day in idx:
        weekday = int(day.dayofweek)
        is_holiday = day in holidays
        is_adjusted = day in adjusted
        is_weekend = weekday >= 5
        if is_holiday:
            day_type = "holiday"
        elif is_adjusted:
            day_type = "adjusted_workday"
        elif is_weekend:
            day_type = "weekend"
        else:
            day_type = "workday"
        rows.append(
            {
                "date": day,
                "weekday": weekday,
                "weekday_name": WEEKDAY_NAMES[weekday],
                "month": int(day.month),
                "day_of_year": int(day.dayofyear),
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "holiday_name": holidays.get(day, ""),
                "is_adjusted_workday": is_adjusted,
                "is_pre_holiday": (day + one) in starts,
                "is_post_holiday": (day - one) in ends,
                "is_workday": day_type in {"workday", "adjusted_workday"},
                "day_type": day_type,
            }
        )
    return pd.DataFrame(rows)
