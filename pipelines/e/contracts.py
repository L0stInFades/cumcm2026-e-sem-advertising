"""Data contracts for Problem E (SEM advertising strategy)."""

from __future__ import annotations

import re

import pandas as pd

from forge.contracts import Column, FrameContract
from forge.xlsx import SheetContract, WorkbookContract

_DURATION = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}\s*$")


def _funnel_consistent(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if (df["点击量"] > df["展现量"]).any():
        errors.append("点击量 > 展现量 in some rows")
    if (df["上方位展现量"] > df["展现量"]).any():
        errors.append("上方位展现量 > 展现量 in some rows")
    if (df["上方首位展现量"] > df["上方位展现量"]).any():
        errors.append("上方首位展现量 > 上方位展现量 in some rows")
    if (df["上方位点击量"] > df["点击量"]).any():
        errors.append("上方位点击量 > 点击量 in some rows")
    if (df["上方位消费额"] > df["消费额"] + 1e-6).any():
        errors.append("上方位消费额 > 消费额 in some rows")
    return errors


def _bounce_and_duration(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for value in df["跳出率"].astype(str):
        v = value.strip()
        if v == "/":
            continue
        try:
            x = float(v)
        except ValueError:
            errors.append(f"跳出率 unparsable {value!r}")
            continue
        if not 0 <= x <= 1:
            errors.append(f"跳出率 out of range {value!r}")
    for value in df["平均访问时长"].astype(str):
        v = value.strip()
        if v != "/" and not _DURATION.match(v):
            errors.append(f"平均访问时长 unparsable {value!r}")
    return errors[:20]


INPUT_CONTRACTS: dict[str, FrameContract] = {
    "附件1__Sheet1": FrameContract(
        name="E.附件1.Sheet1.单元日消费",
        columns=(
            Column("日期", "datetime"), Column("方案ID", "int"), Column("推广单元ID", "int"),
            Column("展现量", "int", min=0), Column("点击量", "int", min=0), Column("消费额", "float", min=0),
            Column("上方位展现量", "int", min=0), Column("上方首位展现量", "int", min=0),
            Column("上方位点击量", "int", min=0), Column("上方位消费额", "float", min=0),
        ),
        min_rows=2000,
        checks=(_funnel_consistent,),
    ),
    "附件1__Sheet2": FrameContract(
        name="E.附件1.Sheet2.每日新注册",
        columns=(Column("日期", "datetime", unique=True), Column("新注册数", "int", min=0)),
        min_rows=365,
        max_rows=366,
    ),
    "附件1__Sheet3": FrameContract(
        name="E.附件1.Sheet3.关键词年度统计",
        columns=(
            Column("序号", "int", unique=True, min=1), Column("关键词", "int"), Column("方案ID", "int"),
            Column("推广单元ID", "int"), Column("消费额", "float", min=0), Column("点击量", "int", min=0),
            Column("浏览量", "int", min=0), Column("跳出率", "any"), Column("平均访问时长", "any"),
        ),
        min_rows=2000,
        checks=(_bounce_and_duration,),
    ),
}

RESULT_CONTRACTS: list[WorkbookContract] = [
    WorkbookContract("result2.xlsx", "result2.xlsx", (
        SheetContract("Sheet1", min_rows=2000, header_len=8),
    )),
    WorkbookContract("result3.xlsx", "result3.xlsx", (
        SheetContract("Sheet1", min_rows=16, header_len=9),
    )),
    WorkbookContract("result4.xlsx", "result4.xlsx", (
        SheetContract("Sheet1", min_rows=7, header_len=9),
    )),
]
