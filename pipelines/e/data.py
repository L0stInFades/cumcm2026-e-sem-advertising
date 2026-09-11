"""Loading and normalising the three attachment sheets into analysis frames."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.e.calendar import day_features

_HMS = re.compile(r"^(\d{1,3}):(\d{2}):(\d{2})$")
UNIT_COLS = ["方案ID", "推广单元ID"]


def parse_bounce(value: Any) -> float:
    """Bounce rate as a fraction in [0, 1]; '/' (no visits) becomes NaN."""
    text = str(value).strip()
    if text in {"/", "", "nan", "None"}:
        return math.nan
    x = float(text)
    return x / 100.0 if x > 1.0 else x


def parse_duration(value: Any) -> float:
    """Average visit duration in seconds; 'HH:MM:SS', bare numbers (zero) or '/' (no visits -> NaN)."""
    text = str(value).strip()
    if text in {"/", "", "nan", "None"}:
        return math.nan
    match = _HMS.match(text)
    if match:
        h, m, s = (int(g) for g in match.groups())
        return float(h * 3600 + m * 60 + s)
    return float(text)


def load_sheets(ingest_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sheet1 (unit-day), Sheet2 (daily registrations), Sheet3 (keyword annual statistics)."""
    data = ingest_dir / "data"
    s1 = pd.read_parquet(data / "附件1__Sheet1.parquet")
    s2 = pd.read_parquet(data / "附件1__Sheet2.parquet")
    s3 = pd.read_parquet(data / "附件1__Sheet3.parquet")
    for df in (s1, s2, s3):
        df.columns = [str(c).strip() for c in df.columns]
    s1["日期"] = pd.to_datetime(s1["日期"]).dt.normalize()
    s2["日期"] = pd.to_datetime(s2["日期"]).dt.normalize()
    s1 = s1.sort_values(["日期", *UNIT_COLS]).reset_index(drop=True)
    s2 = s2.sort_values("日期").reset_index(drop=True)
    s3 = s3.sort_values("序号").reset_index(drop=True)
    return s1, s2, s3


def unit_daily(s1: pd.DataFrame) -> pd.DataFrame:
    """Unit-day records with derived ratios and calendar covariates."""
    df = s1.copy()
    clicks = df["点击量"].replace(0, np.nan)
    imp = df["展现量"].replace(0, np.nan)
    df["cpc"] = df["消费额"] / clicks
    df["ctr"] = df["点击量"] / imp
    df["cpm"] = 1000.0 * df["消费额"] / imp
    df["top_share"] = df["上方位展现量"] / imp
    df["first_share"] = df["上方首位展现量"] / imp
    df["top_click_share"] = df["上方位点击量"] / clicks
    df["top_cpc"] = df["上方位消费额"] / df["上方位点击量"].replace(0, np.nan)
    other_clicks = (df["点击量"] - df["上方位点击量"]).replace(0, np.nan)
    df["other_cpc"] = (df["消费额"] - df["上方位消费额"]) / other_clicks
    feats = day_features(df["日期"]).drop(columns=["date"])
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def daily_totals(s1: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    """All 365 days of 2025: account-level totals, registrations and calendar covariates."""
    agg = s1.groupby("日期")[
        ["展现量", "点击量", "消费额", "上方位展现量", "上方首位展现量", "上方位点击量", "上方位消费额"]
    ].sum()
    agg.columns = ["imp", "clicks", "spend", "top_imp", "first_imp", "top_clicks", "top_spend"]
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    out = agg.reindex(days).fillna(0.0)
    out["regs"] = s2.set_index("日期")["新注册数"].reindex(days).astype(float)
    out["active_units"] = s1.groupby("日期")["推广单元ID"].nunique().reindex(days).fillna(0).astype(int)
    out["cpc"] = out["spend"] / out["clicks"].replace(0, np.nan)
    out["ctr"] = out["clicks"] / out["imp"].replace(0, np.nan)
    out["top_share"] = out["top_imp"] / out["imp"].replace(0, np.nan)
    out["first_share"] = out["first_imp"] / out["imp"].replace(0, np.nan)
    feats = day_features(days).set_index("date")
    out = out.join(feats)
    out.index.name = "date"
    return out


def unit_click_matrix(s1: pd.DataFrame, column: str = "点击量") -> pd.DataFrame:
    """Dates (all of 2025) x units matrix of ``column``; days without a record are zero."""
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    mat = s1.pivot_table(index="日期", columns="推广单元ID", values=column, aggfunc="sum")
    return mat.reindex(days).fillna(0.0)


def keyword_table(s3: pd.DataFrame) -> pd.DataFrame:
    """Keyword-level annual statistics with parsed engagement fields."""
    kw = pd.DataFrame(
        {
            "序号": s3["序号"].astype(int),
            "关键词": s3["关键词"].astype(int),
            "方案ID": s3["方案ID"].astype(int),
            "推广单元ID": s3["推广单元ID"].astype(int),
            "spend": s3["消费额"].astype(float),
            "clicks": s3["点击量"].astype(int),
            "views": s3["浏览量"].astype(int),
        }
    )
    kw["bounce"] = s3["跳出率"].map(parse_bounce).astype(float)
    kw["duration_s"] = s3["平均访问时长"].map(parse_duration).astype(float)
    clicks = kw["clicks"].replace(0, np.nan)
    kw["cpc"] = kw["spend"] / clicks
    kw["depth"] = kw["views"] / clicks
    # visits that did not bounce, and total attention time (person-seconds) they generated
    kw["engaged_visits"] = (kw["clicks"] * (1.0 - kw["bounce"].fillna(1.0))).clip(lower=0.0)
    kw["engagement"] = (kw["engaged_visits"] * kw["duration_s"].fillna(0.0)).clip(lower=0.0)
    kw["has_cost"] = kw["spend"] > 0
    kw["has_benefit"] = (kw["clicks"] > 0) | (kw["views"] > 0)
    dup_counts = kw.groupby("关键词")["推广单元ID"].transform("nunique")
    kw["shared_units"] = dup_counts.astype(int)
    return kw
