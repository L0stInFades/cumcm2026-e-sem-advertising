"""Paper tables for Problem E: booktabs LaTeX fragments (tabular only) plus identical CSV files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from forge.context import StageContext
from forge.tex import tex_escape
from pipelines.e.classify import LABELS


def _fmt(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if spec == "s":
        return tex_escape(str(value))
    if spec == "d":
        return f"{round(float(value)):,}"
    return format(float(value), spec)


def write_table(
    ctx: StageContext, name: str, df: pd.DataFrame, columns: Sequence[tuple[str, str, str]], align: str | None = None
) -> str:
    """``columns`` = (dataframe column, header, format spec). Writes tables/<name>.tex and tables/<name>.csv."""
    keys = [c for c, _, _ in columns]
    headers = [h for _, h, _ in columns]
    specs = [s for _, _, s in columns]
    if align is None:
        align = "".join("l" if s == "s" else "r" for s in specs)
    lines = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[k], s) for k, s in zip(keys, specs)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    text = "\n".join(lines) + "\n"
    ctx.out("tables", f"{name}.tex").write_text(text, encoding="utf-8")
    df[keys].to_csv(ctx.out("tables", f"{name}.csv"), index=False)
    return text


def build_tables(ctx: StageContext) -> dict[str, Any]:
    eda_dir, cls_dir, alloc_dir, fc_dir = ctx.dep("eda"), ctx.dep("classify"), ctx.dep("allocate"), ctx.dep("forecast")
    names: list[str] = []

    kpis = pd.read_parquet(eda_dir / "unit_kpis.parquet").sort_values("spend", ascending=False)
    kpis["spend_share_pct"] = 100 * kpis["spend_share"]
    kpis["ctr_pct"] = 100 * kpis["ctr"]
    kpis["top_pct"] = 100 * kpis["top_share"]
    kpis["first_pct"] = 100 * kpis["first_share"]
    write_table(
        ctx,
        "tab_unit_kpis",
        kpis,
        [
            ("方案ID", "方案ID", "s"),
            ("推广单元ID", "推广单元ID", "s"),
            ("days", "投放天数", "d"),
            ("spend", "消费额/元", ",.0f"),
            ("spend_share_pct", "占比/\\%", ".1f"),
            ("imp", "展现量", "d"),
            ("clicks", "点击量", "d"),
            ("ctr_pct", "CTR/\\%", ".2f"),
            ("cpc", "CPC/元", ".2f"),
            ("top_pct", "上方位/\\%", ".1f"),
            ("first_pct", "首位/\\%", ".1f"),
        ],
    )
    names.append("tab_unit_kpis")

    coefs = pd.read_parquet(eda_dir / "calendar_regression.parquet")
    term_names = {
        "holiday": "法定节假日",
        "adjusted_workday": "调休上班日",
        "pre_holiday": "假前一日",
        "post_holiday": "假后一日",
        "wd_周一": "周一",
        "wd_周二": "周二",
        "wd_周四": "周四",
        "wd_周五": "周五",
        "wd_周六": "周六",
        "wd_周日": "周日",
    }
    rows = []
    for term, tname in term_names.items():
        row: dict[str, Any] = {"term": tname}
        for target, key in (("regs", "reg"), ("spend", "spend"), ("clicks", "clicks")):
            r = coefs[(coefs["target"] == target) & (coefs["term"] == term)].iloc[0]
            row[f"{key}_eff"] = r["effect_pct"]
            row[f"{key}_ci"] = f"[{r['effect_pct_low']:.1f}, {r['effect_pct_high']:.1f}]"
            row[f"{key}_p"] = r["p"]
        rows.append(row)
    write_table(
        ctx,
        "tab_calendar_effects",
        pd.DataFrame(rows),
        [
            ("term", "因素", "s"),
            ("reg_eff", "注册 效应/\\%", ".1f"),
            ("reg_ci", "95\\% CI", "s"),
            ("reg_p", "$p$", ".3f"),
            ("spend_eff", "消费 效应/\\%", ".1f"),
            ("spend_ci", "95\\% CI", "s"),
            ("spend_p", "$p$", ".3f"),
            ("clicks_eff", "点击 效应/\\%", ".1f"),
            ("clicks_p", "$p$", ".3f"),
        ],
    )
    names.append("tab_calendar_effects")

    attribution = json.loads((eda_dir / "attribution.json").read_text(encoding="utf-8"))
    units = pd.read_parquet(eda_dir / "unit_response.parquet")
    att_rows = []
    for _, u in units.sort_values("推广单元ID").iterrows():
        unit = str(int(u["推广单元ID"]))
        spend = float(kpis.loc[kpis["推广单元ID"] == int(unit), "spend"].iloc[0])
        attributed = attribution["attributed_regs"].get(unit, 0.0)
        att_rows.append(
            {
                "unit": unit,
                "rate": 100 * attribution["rates"][unit],
                "source": {"unit": "单元估计", "pooled": "合并估计"}[attribution["rate_source"][unit]],
                "attributed": attributed,
                "cost_per_reg": spend / attributed if attributed > 0 else np.nan,
                "gamma": u["gamma"],
                "gamma_se": u["gamma_se"],
                "r2": u["r2"],
                "gsrc": {"unit": "单元", "pooled": "合并"}[u["gamma_source"]],
            }
        )
    write_table(
        ctx,
        "tab_attribution_response",
        pd.DataFrame(att_rows),
        [
            ("unit", "推广单元ID", "s"),
            ("rate", "注册率/(人/百次点击)", ".3f"),
            ("source", "来源", "s"),
            ("attributed", "归因注册/人", ",.0f"),
            ("cost_per_reg", "注册成本/元", ".1f"),
            ("gamma", "弹性 $\\gamma$", ".3f"),
            ("gamma_se", "SE", ".3f"),
            ("r2", "$R^2$", ".2f"),
            ("gsrc", "$\\gamma$ 来源", "s"),
        ],
    )
    names.append("tab_attribution_response")

    by_label = pd.read_parquet(cls_dir / "classification_by_label.parquet")
    table = pd.read_parquet(cls_dir / "classification.parquet")
    by_label["spend_share_pct"] = 100 * by_label["spend_share"]
    by_label["share_n"] = 100 * by_label["n"] / by_label["n"].sum()
    by_label["mean_benefit"] = by_label["label"].map(table.groupby("label")["benefit"].mean())
    by_label["median_spend"] = by_label["label"].map(table.groupby("label")["spend"].median())
    write_table(
        ctx,
        "tab_class_summary",
        by_label,
        [
            ("label", "类别", "s"),
            ("n", "数量", "d"),
            ("share_n", "占比/\\%", ".1f"),
            ("spend", "消费额/元", ",.0f"),
            ("spend_share_pct", "消费占比/\\%", ".1f"),
            ("median_spend", "消费中位数/元", ".1f"),
            ("clicks", "点击量", "d"),
            ("cpc", "CPC/元", ".2f"),
            ("mean_benefit", "平均效益指数", ".3f"),
            ("reg_attr", "归因注册/人", ",.0f"),
        ],
    )
    names.append("tab_class_summary")

    summary = json.loads((cls_dir / "classification_summary.json").read_text(encoding="utf-8"))
    rob_rows = []
    method_names = {
        "jenks": "自然断点（主）",
        "median": "中位数",
        "gmm": "高斯混合",
        "equal_weights": "等权效益 + 自然断点",
    }
    for method, counts in summary["counts"].items():
        thr = summary["thresholds"].get(method, {})
        rob_rows.append(
            {
                "method": method_names.get(method, method),
                "cost_thr": thr.get("cost_yuan", np.nan),
                "ben_thr": thr.get("benefit", np.nan),
                **{lab: counts.get(lab, 0) for lab in LABELS},
            }
        )
    write_table(
        ctx,
        "tab_class_robustness",
        pd.DataFrame(rob_rows),
        [
            ("method", "阈值方法", "s"),
            ("cost_thr", "成本阈值/元", ".2f"),
            ("ben_thr", "效益阈值", ".3f"),
            *[(lab, lab, "d") for lab in LABELS],
        ],
    )
    names.append("tab_class_robustness")

    per_unit = (
        table.groupby(["推广单元ID", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
    )
    per_unit["total"] = per_unit[list(LABELS)].sum(axis=1)
    per_unit["推广单元ID"] = per_unit["推广单元ID"].astype(str)
    write_table(
        ctx,
        "tab_class_per_unit",
        per_unit,
        [("推广单元ID", "推广单元ID", "s"), *[(lab, lab, "d") for lab in LABELS], ("total", "合计", "d")],
    )
    names.append("tab_class_per_unit")

    asum = pd.read_parquet(alloc_dir / "allocation_summary.parquet")
    calib = json.loads((alloc_dir / "calibration_check.json").read_text(encoding="utf-8"))
    audit = json.loads((alloc_dir / "allocation_audit.json").read_text(encoding="utf-8"))
    boot = pd.read_parquet(alloc_dir / "bootstrap.parquet")
    q3_rows = []
    for window, wname in (("feb", "2025-02-01 至 02-08"), ("aug", "2025-08-01 至 08-08")):
        w = asum[asum["window"] == window]
        for unit, g in w.groupby("推广单元ID"):
            q3_rows.append(
                {
                    "window": wname,
                    "unit": str(int(unit)),
                    "days": len(g),
                    "budget": g["budget"].sum(),
                    "eligible": int(g["eligible"].iloc[0]),
                    "selected": g["selected"].mean(),
                    "opt_regs": g["opt_regs"].sum(),
                    "prop_regs": g["prop_regs"].sum(),
                    "gain": 100 * (g["opt_regs"].sum() / g["prop_regs"].sum() - 1),
                    "opt_cpc": g["budget"].sum() / g["opt_clicks"].sum(),
                    "prop_cpc": g["budget"].sum() / g["prop_clicks"].sum(),
                }
            )
        b = boot[boot["window"] == window]["ratio"]
        q3_rows.append(
            {
                "window": wname,
                "unit": "合计",
                "days": len(w),
                "budget": w["budget"].sum(),
                "eligible": int(w.groupby("推广单元ID")["eligible"].first().sum()),
                "selected": w["selected"].mean(),
                "opt_regs": w["opt_regs"].sum(),
                "prop_regs": w["prop_regs"].sum(),
                "gain": 100 * (w["opt_regs"].sum() / w["prop_regs"].sum() - 1),
                "opt_cpc": w["budget"].sum() / w["opt_clicks"].sum(),
                "prop_cpc": w["budget"].sum() / w["prop_clicks"].sum(),
                "ci": f"[{100 * (b.quantile(0.025) - 1):.1f}, {100 * (b.quantile(0.975) - 1):.1f}]",
            }
        )
    q3 = pd.DataFrame(q3_rows)
    q3["ci"] = q3["ci"].fillna("")
    write_table(
        ctx,
        "tab_q3_summary",
        q3,
        [
            ("window", "窗口", "s"),
            ("unit", "推广单元", "s"),
            ("days", "天数", "d"),
            ("budget", "预算/元", ",.0f"),
            ("eligible", "候选词", "d"),
            ("selected", "平均入选词", ".1f"),
            ("prop_regs", "历史比例注册/人", ".1f"),
            ("opt_regs", "最优注册/人", ".1f"),
            ("gain", "提升/\\%", ".1f"),
            ("ci", "95\\% 自助区间", "s"),
            ("prop_cpc", "历史 CPC/元", ".2f"),
            ("opt_cpc", "最优 CPC/元", ".2f"),
        ],
    )
    names.append("tab_q3_summary")

    daily_rows = (
        asum.groupby(["window", "date"])
        .agg(
            budget=("budget", "sum"),
            units=("推广单元ID", "nunique"),
            opt_clicks=("opt_clicks", "sum"),
            prop_clicks=("prop_clicks", "sum"),
            actual_clicks=("actual_clicks", "sum"),
            opt_regs=("opt_regs", "sum"),
            prop_regs=("prop_regs", "sum"),
            equal_regs=("equal_regs", "sum"),
        )
        .reset_index()
    )
    daily_rows["gain"] = 100 * (daily_rows["opt_regs"] / daily_rows["prop_regs"] - 1)
    write_table(
        ctx,
        "tab_q3_daily",
        daily_rows,
        [
            ("date", "日期", "s"),
            ("units", "单元数", "d"),
            ("budget", "预算/元", ",.1f"),
            ("actual_clicks", "实际点击", "d"),
            ("prop_clicks", "历史比例点击", ",.1f"),
            ("opt_clicks", "最优点击", ",.1f"),
            ("prop_regs", "历史比例注册", ".1f"),
            ("equal_regs", "均匀注册", ".1f"),
            ("opt_regs", "最优注册", ".1f"),
            ("gain", "提升/\\%", ".1f"),
        ],
    )
    names.append("tab_q3_daily")

    alt = pd.read_parquet(alloc_dir / "window_budget_alternative.parquet")
    alt_rows = []
    for window, wname in (("feb", "2 月窗口"), ("aug", "8 月窗口")):
        a = alt[alt["window"] == window]
        opt = asum[asum["window"] == window]["opt_regs"].sum()
        alt_rows.append(
            {
                "window": wname,
                "regs_daily": opt,
                "regs_window": a["regs_window"].sum(),
                "gain": 100 * (a["regs_window"].sum() / opt - 1),
                "moved": 100 * (a["window_budget_day"] - a["daily_budget"]).abs().sum() / 2 / a["daily_budget"].sum(),
            }
        )
    write_table(
        ctx,
        "tab_q3_window_budget",
        pd.DataFrame(alt_rows),
        [
            ("window", "窗口", "s"),
            ("regs_daily", "逐日预算最优注册/人", ".1f"),
            ("regs_window", "窗口预算最优注册/人", ".1f"),
            ("gain", "额外提升/\\%", ".1f"),
            ("moved", "跨日调动预算比例/\\%", ".1f"),
        ],
    )
    names.append("tab_q3_window_budget")

    sens = (
        pd.read_parquet(alloc_dir / "sensitivity.parquet")
        if (alloc_dir / "sensitivity.parquet").exists()
        else pd.DataFrame()
    )
    if len(sens):
        write_table(
            ctx,
            "tab_q3_sensitivity",
            sens,
            [
                ("scenario", "情形", "s"),
                ("window", "窗口", "s"),
                ("opt_regs", "最优注册/人", ".1f"),
                ("prop_regs", "历史比例注册/人", ".1f"),
                ("gain", "提升/\\%", ".1f"),
                ("selected", "平均入选词", ".1f"),
            ],
        )
        names.append("tab_q3_sensitivity")

    ver_rows = [
        {"item": "Q3 单元-日分配问题数", "value": f"{audit['groups']}"},
        {
            "item": "Q3 独立校验（可行性 / KKT / 配对转移 / 通用凸求解器）",
            "value": "全部通过" if audit["ok"] else "存在失败",
        },
        {"item": "Q3 与通用凸求解器目标值最大相对差", "value": f"{audit['max_cvx_gap']:.2e}"},
        {
            "item": "响应模型校准：历史比例分配预测点击 vs 实际（MAPE / 中位 APE）",
            "value": f"{100 * calib['mape']:.1f}\\% / {100 * calib['median_ape']:.1f}\\%",
        },
    ]
    q4_audit = json.loads((fc_dir / "allocation_audit.json").read_text(encoding="utf-8"))
    ver_rows.append({"item": "Q4 分配问题数（跨日 + 单元-日）", "value": f"{q4_audit['groups']}"})
    ver_rows.append({"item": "Q4 独立校验", "value": "全部通过" if q4_audit["ok"] else "存在失败"})
    cls_check = json.loads((cls_dir / "classification_check.json").read_text(encoding="utf-8"))
    ver_rows.append(
        {"item": "Q2 分类独立复核（规则重推 / 无效词定义 / 计数）", "value": "通过" if cls_check["ok"] else "失败"},
    )
    write_table(
        ctx,
        "tab_verification",
        pd.DataFrame(ver_rows),
        [("item", "检验项", "s"), ("value", "结果", "s")],
        align="p{0.66\\linewidth}l",
    )
    names.append("tab_verification")

    bt = pd.read_parquet(fc_dir / "backtest_summary.parquet")
    tnames = {
        "log_eff": "效率乘子 $\\log\\varepsilon$",
        "log_cpc": "$\\log$ CPC",
        "logit_ctr": "logit CTR",
        "logit_top": "logit 上方位占比",
        "logit_first": "logit 首位占比",
        "log_clicks": "$\\log$ 点击量",
    }
    mnames = {"calendar": "日历回归", "seasonal_naive": "季节朴素", "mean28": "28 日均值"}
    bt["target_name"] = bt["target"].map(tnames)
    bt["model_name"] = bt["model"].map(mnames)
    bt["mape_pct"] = 100 * bt["mape_raw"]
    bt["cov_pct"] = 100 * bt["coverage80"]
    bt["pinball"] = bt["pinball10"] + bt["pinball90"]
    bt = bt.sort_values(["target", "model"])
    write_table(
        ctx,
        "tab_backtest",
        bt,
        [
            ("target_name", "目标", "s"),
            ("model_name", "模型", "s"),
            ("mae", "MAE", ".3f"),
            ("rmse", "RMSE", ".3f"),
            ("mape_pct", "原尺度 MAPE/\\%", ".1f"),
            ("pinball", "Pinball(0.1+0.9)", ".3f"),
            ("cov_pct", "80\\% 覆盖率/\\%", ".1f"),
        ],
    )
    names.append("tab_backtest")

    days = pd.read_parquet(fc_dir / "plan_by_unit_day.parquet")
    d = (
        days.groupby("date")
        .agg(
            units=("推广单元ID", "nunique"),
            spend=("spend", "sum"),
            clicks=("clicks_mean", "sum"),
            clicks_lo=("clicks_q10", "sum"),
            clicks_hi=("clicks_q90", "sum"),
            imp=("imp_mean", "sum"),
            imp_lo=("imp_q10", "sum"),
            imp_hi=("imp_q90", "sum"),
            views=("views_mean", "sum"),
            regs=("regs_mean", "sum"),
            regs_lo=("regs_q10", "sum"),
            regs_hi=("regs_q90", "sum"),
        )
        .reset_index()
    )
    d["cpc"] = d["spend"] / d["clicks"]
    d["cpc_range"] = [f"[{s / hi:.2f}, {s / lo:.2f}]" for s, lo, hi in zip(d["spend"], d["clicks_lo"], d["clicks_hi"])]
    d["clicks_range"] = [f"[{lo:,.0f}, {hi:,.0f}]" for lo, hi in zip(d["clicks_lo"], d["clicks_hi"])]
    d["imp_range"] = [f"[{lo:,.0f}, {hi:,.0f}]" for lo, hi in zip(d["imp_lo"], d["imp_hi"])]
    d["regs_range"] = [f"[{lo:.0f}, {hi:.0f}]" for lo, hi in zip(d["regs_lo"], d["regs_hi"])]
    d["position"] = [
        float(np.average(g["position_mean"], weights=np.maximum(g["clicks_mean"], 1e-9)))
        for _, g in days.groupby("date")
    ]
    write_table(
        ctx,
        "tab_q4_daily",
        d,
        [
            ("date", "日期", "s"),
            ("units", "单元数", "d"),
            ("spend", "投入/元", ",.1f"),
            ("cpc", "CPC/元", ".2f"),
            ("cpc_range", "CPC 范围", "s"),
            ("imp", "展现量", "d"),
            ("imp_range", "展现范围", "s"),
            ("position", "展位", ".2f"),
            ("clicks", "点击量", "d"),
            ("clicks_range", "点击范围", "s"),
            ("views", "浏览量", "d"),
            ("regs", "注册量", ".0f"),
            ("regs_range", "注册范围", "s"),
        ],
    )
    names.append("tab_q4_daily")

    per = pd.DataFrame(json.loads((fc_dir / "plan_by_unit.json").read_text(encoding="utf-8")))
    per["unit"] = per["推广单元ID"].astype(str)
    per["regs_range"] = [f"[{lo:.0f}, {hi:.0f}]" for lo, hi in zip(per["regs_q10"], per["regs_q90"])]
    per["clicks_range"] = [f"[{lo:,.0f}, {hi:,.0f}]" for lo, hi in zip(per["clicks_q10"], per["clicks_q90"])]
    per["cpc"] = per["spent"] / per["clicks_mean"]
    per = per.sort_values("budget", ascending=False)
    write_table(
        ctx,
        "tab_q4_units",
        per,
        [
            ("unit", "推广单元ID", "s"),
            ("budget", "7 日预算/元", ",.1f"),
            ("gamma", "$\\gamma$", ".3f"),
            ("eligible", "候选词", "d"),
            ("cpc", "CPC/元", ".2f"),
            ("clicks_mean", "点击量", ",.0f"),
            ("clicks_range", "点击范围", "s"),
            ("regs_mean", "注册量", ".1f"),
            ("regs_range", "注册范围", "s"),
        ],
    )
    names.append("tab_q4_units")

    ctx.write_json("tables_index.json", names)
    return {"tables": len(names)}
