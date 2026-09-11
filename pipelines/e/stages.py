"""Problem E stages: validate, eda (Q1), classify (Q2), allocate (Q3), forecast (Q4), results, figures, tables.

Conventions (see docs/ENGINEERING_STANDARD.md):
  * every stage is ``@stage(name, deps=(...))`` and returns a JSON-serialisable metrics dict;
  * outputs go only to ``ctx.out(...)``; paper numbers via ``ctx.number("Key", value)``;
  * ``results`` writes result*.xlsx with ``forge.xlsx.write_result`` from the organisers' templates;
  * every forecast carries a backtest and every allocation an independent optimality audit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forge import xlsx
from forge.context import StageContext
from forge.runner import stage
from pipelines.common.validation import run_validation
from pipelines.e import allocate as alloc
from pipelines.e import classify as cls
from pipelines.e import eda as eda_mod
from pipelines.e import forecast as fc
from pipelines.e import response as resp
from pipelines.e.calendar import day_features
from pipelines.e.contracts import INPUT_CONTRACTS
from pipelines.e.data import daily_totals, keyword_table, load_sheets, unit_click_matrix, unit_daily

Q3_WINDOWS: dict[str, tuple[str, str]] = {"feb": ("2025-02-01", "2025-02-08"), "aug": ("2025-08-01", "2025-08-08")}
Q4_DATES = ("2026-09-11", "2026-09-17")
Q4_REFERENCE = ("2025-09-11", "2025-09-17")
RESULT_HEADER = [
    "日期",
    "方案ID",
    "推广单元",
    "关键词",
    "投入金额",
    "预期展位",
    "预期点击量",
    "预期浏览量",
    "预期注册量",
]


@stage("validate", deps=("ingest",), description="Validate the three SEM sheets against their contracts")
def validate(ctx: StageContext) -> dict[str, Any]:
    return run_validation(ctx, INPUT_CONTRACTS)


def _save(ctx: StageContext, df: pd.DataFrame, name: str) -> None:
    df.to_parquet(ctx.out(f"{name}.parquet"), index=False)
    df.to_csv(ctx.out(f"{name}.csv"), index=False)


# ----------------------------------------------------------------------------------------------- Q1
@stage(
    "eda",
    deps=("validate",),
    description="Q1: KPIs, keyword concentration, calendar/holiday effects, registration attribution",
)
def eda(ctx: StageContext) -> dict[str, Any]:
    ctx.seed_everything("eda")
    s1, s2, s3 = load_sheets(ctx.dep("ingest"))
    ud = unit_daily(s1)
    daily = daily_totals(s1, s2)
    kw = keyword_table(s3)
    _save(ctx, ud, "unit_daily")
    _save(ctx, daily.reset_index(), "daily_totals")

    kpis = eda_mod.unit_kpis(s1)
    _save(ctx, kpis, "unit_kpis")
    conc = eda_mod.keyword_concentration(kw)
    _save(ctx, conc.pop("lorenz"), "lorenz")
    _save(ctx, conc.pop("per_unit"), "keyword_per_unit")
    ctx.write_json("concentration.json", conc)

    coef_frames, summaries = [], []
    specs: list[tuple[str, bool, tuple[str, ...]]] = [
        ("regs", True, ()),
        ("regs", True, ("spend",)),
        ("spend", True, ()),
        ("clicks", True, ()),
        ("imp", True, ()),
        ("cpc", False, ()),
        ("ctr", False, ()),
        ("top_share", False, ()),
    ]
    for target, log, controls in specs:
        coef, summary = eda_mod.calendar_regression(daily, target, log, controls)
        coef_frames.append(coef)
        summaries.append(summary)
    coefs = pd.concat(coef_frames, ignore_index=True)
    _save(ctx, coefs, "calendar_regression")
    nonparam = [eda_mod.holiday_nonparametric(daily, t) for t in ("regs", "spend", "clicks")]
    ctx.write_json("calendar_regression_summary.json", {"models": summaries, "mann_whitney": nonparam})
    _save(
        ctx,
        eda_mod.day_type_profile(daily, ("spend", "clicks", "imp", "regs", "cpc", "ctr", "top_share")),
        "day_type_profile",
    )

    attribution = eda_mod.attribution_model(unit_click_matrix(s1), daily)
    _save(ctx, attribution.pop("fitted"), "attribution_fit")
    ctx.write_json("attribution.json", attribution)
    units = resp.unit_response_table(ud)
    _save(ctx, units, "unit_response")

    total_spend = float(s1["消费额"].sum())
    total_clicks, total_imp = int(s1["点击量"].sum()), int(s1["展现量"].sum())
    ctx.number("TotalSpend", total_spend, ",.2f")
    ctx.number("TotalSpendWan", total_spend / 1e4, ".2f")
    ctx.number("TotalClicks", total_clicks, ",d")
    ctx.number("TotalImpressions", total_imp, ",d")
    ctx.number("TotalRegs", int(daily["regs"].sum()), ",d")
    ctx.number("OverallCtrPct", 100 * total_clicks / total_imp, ".2f")
    ctx.number("OverallCpc", total_spend / total_clicks, ".4f")
    ctx.number("TopSharePct", 100 * s1["上方位展现量"].sum() / total_imp, ".2f")
    ctx.number("FirstSharePct", 100 * s1["上方首位展现量"].sum() / total_imp, ".2f")
    ctx.number("TopClickSharePct", 100 * s1["上方位点击量"].sum() / total_clicks, ".2f")
    ctx.number("TopSpendSharePct", 100 * s1["上方位消费额"].sum() / total_spend, ".2f")
    ctx.number("TopCpc", s1["上方位消费额"].sum() / s1["上方位点击量"].sum(), ".4f")
    other_clicks = total_clicks - s1["上方位点击量"].sum()
    ctx.number("OtherCpc", (total_spend - s1["上方位消费额"].sum()) / other_clicks, ".4f")
    ctx.number("UnitDays", len(s1))
    ctx.number("KeywordRows", conc["keywords"])
    ctx.number("UniqueKeywordIds", conc["unique_keyword_ids"])
    ctx.number("SharedKeywordIds", conc["shared_keyword_ids"])
    ctx.number("ActiveKeywords", conc["active_keywords"])
    ctx.number("ZeroKeywords", conc["zero_cost_zero_benefit"])
    ctx.number("GiniSpendActive", conc["gini_spend_active"], ".4f")
    ctx.number("TopTenSpendSharePct", 100 * conc["top10_spend_share"], ".2f")
    ctx.number("TopHundredSpendSharePct", 100 * conc["top100_spend_share"], ".2f")
    big = kpis.sort_values("spend", ascending=False).iloc[0]
    ctx.number("BiggestUnitSpendSharePct", 100 * big["spend_share"], ".2f")
    ctx.number("BiggestUnitId", int(big["推广单元ID"]))

    def term(target: str, name: str) -> pd.Series:
        return coefs[(coefs["target"] == target) & (coefs["term"] == name)].iloc[0]

    for target, key in (("regs", "Reg"), ("spend", "Spend"), ("clicks", "Clicks"), ("imp", "Imp")):
        row = term(target, "holiday")
        ctx.number(f"Holiday{key}EffectPct", row["effect_pct"], ".2f")
        ctx.number(f"Holiday{key}CiLow", row["effect_pct_low"], ".2f")
        ctx.number(f"Holiday{key}CiHigh", row["effect_pct_high"], ".2f")
        ctx.number(f"Holiday{key}P", row["p"], ".4f")
    for name, key in (
        ("adjusted_workday", "AdjustedWorkday"),
        ("pre_holiday", "PreHoliday"),
        ("post_holiday", "PostHoliday"),
        ("wd_周一", "Monday"),
        ("wd_周六", "Saturday"),
        ("wd_周日", "Sunday"),
    ):
        row = term("regs", name)
        ctx.number(f"{key}RegEffectPct", row["effect_pct"], ".2f")
        ctx.number(f"{key}RegP", row["p"], ".4f")
    for name, key in (("holiday", "Holiday"), ("wd_周一", "Monday")):
        row = term("cpc", name)
        ctx.number(f"{key}CpcEffect", row["estimate"], ".4f")
        ctx.number(f"{key}CpcP", row["p"], ".4f")
    ctrl = term("regs|spend", "holiday")
    ctx.number("HolidayRegGivenSpendEffectPct", ctrl["effect_pct"], ".2f")
    ctx.number("HolidayRegGivenSpendCiLow", ctrl["effect_pct_low"], ".2f")
    ctx.number("HolidayRegGivenSpendCiHigh", ctrl["effect_pct_high"], ".2f")
    ctx.number("HolidayRegGivenSpendP", ctrl["p"], ".4f")
    ctx.number("MondayRegGivenSpendEffectPct", term("regs|spend", "wd_周一")["effect_pct"], ".2f")
    ctx.number("SundayRegGivenSpendEffectPct", term("regs|spend", "wd_周日")["effect_pct"], ".2f")
    ctx.number("RegSpendElasticity", term("regs|spend", "log_spend")["estimate"], ".4f")
    ctx.number("RegSpendElasticitySe", term("regs|spend", "log_spend")["se"], ".4f")
    reg_summary = next(s for s in summaries if s["target"] == "regs")
    ctx.number("RegCalendarRsq", reg_summary["r2"], ".4f")
    ctx.number("RegWeekdayWaldP", reg_summary["weekday_wald_p"], ".2e")
    mw = next(m for m in nonparam if m["target"] == "regs")
    ctx.number("HolidayRegMedianRatio", mw["ratio_of_medians"], ".4f")
    ctx.number("HolidayRegMannWhitneyP", mw["p"], ".2e")
    ctx.number("AttributionLambda", attribution["lambda"], ".1f")
    ctx.number("AttributionRsq", attribution["r2"], ".4f")
    ctx.number("AttributionHoldoutRsq", attribution["holdout"]["r2"], ".4f")
    ctx.number("AttributionHoldoutMape", 100 * attribution["holdout"]["mape"], ".2f")
    ctx.number("BaselineRegsPerDay", attribution["baseline_per_day"], ".2f")
    ctx.number("PooledRegRatePerHundredClicks", 100 * attribution["pooled_rate"], ".4f")
    attributed = sum(attribution["attributed_regs"].values())
    ctx.number("AttributedRegs", attributed, ",.0f")
    ctx.number("CostPerAttributedReg", total_spend / attributed, ".2f")
    ctx.number("UnitsWithOwnRate", sum(1 for v in attribution["rate_source"].values() if v == "unit"))
    ctx.number("PooledGamma", units["pooled_gamma"].iloc[0], ".4f")
    ctx.number("PooledGammaSe", units["pooled_se"].iloc[0], ".4f")
    ctx.number("MinUnitGamma", units["gamma"].min(), ".4f")
    ctx.number("MaxUnitGamma", units["gamma"].max(), ".4f")
    ctx.number("UnitsWithOwnGamma", int((units["gamma_source"] == "unit").sum()))
    return {
        "unit_days": len(s1),
        "holiday_reg_effect_pct": float(term("regs", "holiday")["effect_pct"]),
        "holiday_reg_p": float(term("regs", "holiday")["p"]),
        "attribution_r2": attribution["r2"],
        "pooled_gamma": float(units["pooled_gamma"].iloc[0]),
    }


def _rates(ctx: StageContext) -> dict[int, float]:
    attribution = ctx.dep_data("eda", "attribution.json")
    return {int(k): float(v) for k, v in attribution["rates"].items()}


# ----------------------------------------------------------------------------------------------- Q2
@stage(
    "classify", deps=("eda",), description="Q2: five-way cost/benefit keyword classification with threshold robustness"
)
def classify(ctx: StageContext) -> dict[str, Any]:
    seed = ctx.seed_everything("classify")
    _, _, s3 = load_sheets(ctx.dep("ingest"))
    kw = keyword_table(s3)
    methods = tuple(ctx.cfg("classify.methods", ["jenks", "median", "gmm"]))
    main = str(ctx.cfg("classify.main_method", "jenks"))
    result = cls.classify_keywords(kw, _rates(ctx), methods, main, seed)
    table = result.pop("table")
    _save(ctx, table, "classification")
    thr = result["thresholds"][main]
    report = cls.verify_classification(table, thr["log10_cost"], thr["benefit"])
    ctx.write_json("classification_check.json", report)
    per_unit = (
        table.groupby(["方案ID", "推广单元ID", "label"])
        .agg(n=("序号", "size"), spend=("spend", "sum"), clicks=("clicks", "sum"))
        .reset_index()
    )
    _save(ctx, per_unit, "classification_per_unit")
    by_label = (
        table.groupby("label")
        .agg(
            n=("序号", "size"),
            spend=("spend", "sum"),
            clicks=("clicks", "sum"),
            views=("views", "sum"),
            reg_attr=("reg_attr", "sum"),
        )
        .reindex(cls.LABELS)
        .fillna(0)
    )
    by_label["spend_share"] = by_label["spend"] / by_label["spend"].sum()
    by_label["cpc"] = by_label["spend"] / by_label["clicks"].replace(0, np.nan)
    _save(ctx, by_label.reset_index(), "classification_by_label")
    ctx.write_json("classification_summary.json", result)
    if not report["ok"]:
        raise RuntimeError(f"classification checker failed: {report}")
    keys = {"黄金词": "Gold", "重点词": "Key", "潜力词": "Potential", "问题词": "Problem"}
    keys["无效词"] = "Invalid"
    for label, key in keys.items():
        ctx.number(f"{key}Count", int(by_label.loc[label, "n"]))
        ctx.number(f"{key}SpendSharePct", 100 * float(by_label.loc[label, "spend_share"]), ".2f")
        ctx.number(f"{key}Cpc", float(by_label.loc[label, "cpc"]) if by_label.loc[label, "clicks"] > 0 else 0.0, ".4f")
    ctx.number("CostThresholdYuan", thr["cost_yuan"], ".2f")
    ctx.number("CostThresholdLog", thr["log10_cost"], ".4f")
    ctx.number("BenefitThreshold", thr["benefit"], ".4f")
    for comp, key in (("clicks", "Clicks"), ("views", "Views"), ("engagement", "Engagement"), ("reg_attr", "Reg")):
        ctx.number(f"Weight{key}", result["weights"][comp], ".4f")
    ctx.number("AgreementJenksMedianPct", 100 * result["agreement"]["jenks_vs_median"], ".2f")
    ctx.number("AgreementGmmJenksPct", 100 * result["agreement"]["gmm_vs_jenks"], ".2f")
    ctx.number("AgreementJenksEqualWeightsPct", 100 * result["agreement"]["jenks_vs_equal_weights"], ".2f")
    ctx.number("EntropyEqualSpearman", result["agreement"]["entropy_vs_equal_spearman"], ".4f")
    ctx.number("MedianCostThresholdYuan", result["thresholds"]["median"]["cost_yuan"], ".2f")
    ctx.number("GmmCostThresholdYuan", result["thresholds"]["gmm"]["cost_yuan"], ".2f")
    return {"counts": result["counts"][main], "checker_ok": report["ok"], "thresholds": thr}


# ----------------------------------------------------------------------------------------------- Q3
def _load_params(
    ctx: StageContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]], dict[int, float]]:
    s1, _, s3 = load_sheets(ctx.dep("ingest"))
    ud = unit_daily(s1)
    units = pd.read_parquet(ctx.dep("eda") / "unit_response.parquet")
    rates = _rates(ctx)
    kw = keyword_table(s3)
    labels = pd.read_parquet(ctx.dep("classify") / "classification.parquet")[["序号", "label"]]
    params = resp.keyword_parameters(kw, units, rates).merge(labels, on="序号", how="left")
    params["efficiency"] = params["rho"] / params["cpc"]
    pos: dict[int, dict[str, Any]] = {}
    for unit, df in ud.groupby("推广单元ID"):
        pos[int(unit)] = {
            "top": resp.fit_position_model(df, "top_share"),
            "first": resp.fit_position_model(df, "first_share"),
        }
    return ud, units, params, pos, rates


def _eligible(params: pd.DataFrame, active_units: set[int], problem_cap_factor: float) -> pd.DataFrame:
    """Eligibility inside a window: invalid words never enter, problem words keep half the normal cap.

    Keyword IDs shared by several active units are flagged (``dup_best`` marks the most efficient copy)
    for reporting; the copies stay eligible because the data cannot show cross-unit cannibalisation."""
    p = params[params["推广单元ID"].isin(active_units)].copy()
    p["cap_factor"] = np.where(p["label"] == "问题词", problem_cap_factor, 1.0)
    p["dup_best"] = True
    for _, grp in p[p["shared_units"] > 1].groupby("关键词"):
        if len(grp) > 1:
            best = grp["efficiency"].idxmax()
            p.loc[grp.index.difference([best]), "dup_best"] = False
    p["eligible"] = p["label"].isin(["黄金词", "重点词", "潜力词", "问题词"])
    return p


def _evaluate(x: np.ndarray, p: pd.DataFrame, gamma: float, eff: float, pos: dict[str, Any]) -> pd.DataFrame:
    c, s = p["c"].to_numpy(dtype=float), p["s"].to_numpy(dtype=float)
    clicks = eff * resp.clicks_response(x, c, s, gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpc = np.where(clicks > 0, x / clicks, np.nan)
    p_top, p_first = resp.position_shares(np.nan_to_num(cpc, nan=1.0), pos["top"], pos["first"])
    return pd.DataFrame(
        {
            "序号": p["序号"].to_numpy(),
            "关键词": p["关键词"].to_numpy(),
            "方案ID": p["方案ID"].to_numpy(),
            "推广单元ID": p["推广单元ID"].to_numpy(),
            "label": p["label"].to_numpy(),
            "spend": x,
            "clicks": clicks,
            "cpc": cpc,
            "p_top": p_top,
            "p_first": p_first,
            "position": resp.expected_position(p_top, p_first),
            "views": clicks * p["depth"].to_numpy(dtype=float),
            "regs": clicks * p["rho"].to_numpy(dtype=float),
        }
    )


def _caps(p: pd.DataFrame, budget: float, cap_mult: float) -> np.ndarray:
    """Per-keyword daily caps: multiplier x mean daily spend x type factor, never below the historical
    proportional share (so the proportional baseline is always feasible) and jointly attainable."""
    s = p["s"].to_numpy(dtype=float)
    caps = np.maximum(cap_mult * s * p["cap_factor"].to_numpy(dtype=float), alloc.proportional_allocation(s, budget))
    if caps.sum() < 1.2 * budget:  # keep the cap structure but make the budget attainable
        caps = caps * (1.2 * budget / caps.sum())
    return caps


def _solve_unit_day(
    p: pd.DataFrame, gamma: float, budget: float, cap_mult: float, min_spend: float
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    w = (p["rho"] * p["c"]).to_numpy(dtype=float)
    s = p["s"].to_numpy(dtype=float)
    caps = _caps(p, budget, cap_mult)
    res = alloc.solve_group(w, s, gamma, caps, budget, min_spend=min_spend)
    return res["x"], res, w * np.power(s, -gamma), caps


def _day_efficiency(ud: pd.DataFrame, units: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[int, dict[str, Any]]:
    """Calendar-model expectation of the day efficiency multiplier E[eff_{u,d}] for each unit and date."""
    feats = day_features(dates)
    out: dict[int, dict[str, Any]] = {}
    for _, row in units.iterrows():
        unit = int(row["推广单元ID"])
        series = fc.unit_series(
            ud[ud["推广单元ID"] == unit],
            float(row["gamma"]),
            float(row["mean_daily_spend"]),
            float(row["mean_daily_clicks"]),
        )
        if len(series) < 40:
            out[unit] = {"eff": np.ones(len(dates)), "sd": 0.0, "n": len(series), "mean_log": np.zeros(len(dates))}
            continue
        f = fc.forecast_unit(series, "log_eff", feats)
        out[unit] = {"eff": np.exp(f["mean"] + 0.5 * f["sd"] ** 2), "sd": f["sd"], "n": f["n"], "mean_log": f["mean"]}
    return out


def _sensitivity(
    all_groups: dict[str, dict[tuple[int, pd.Timestamp], dict[str, Any]]],
    elig_by_window: dict[str, pd.DataFrame],
    unit_index: pd.DataFrame,
    cap_mult: float,
    min_spend: float,
    problem_cap: float,
) -> pd.DataFrame:
    """Gain of the optimal allocation over the proportional baseline under perturbed modelling choices."""
    scenarios: list[tuple[str, dict[str, Any]]] = [
        ("基准", {}),
        ("上限倍数 2", {"cap": 2.0}),
        ("上限倍数 8", {"cap": 8.0}),
        ("弹性 gamma x0.9", {"gamma_scale": 0.9}),
        ("弹性 gamma x1.1", {"gamma_scale": 1.1}),
        ("预算 −20%", {"budget_scale": 0.8}),
        ("预算 +20%", {"budget_scale": 1.2}),
        ("剔除问题词", {"exclude_problem": True}),
        ("问题词不降上限", {"problem_cap": 1.0}),
        ("最小投放 0", {"min_spend": 0.0}),
    ]
    rows = []
    for name, opt in scenarios:
        for window, groups in all_groups.items():
            elig = elig_by_window[window]
            opt_total = prop_total = 0.0
            selected = []
            for (unit, _date), g in groups.items():
                gamma = float(np.clip(g["gamma"] * opt.get("gamma_scale", 1.0), *resp.GAMMA_BOUNDS))
                budget = g["budget"] * opt.get("budget_scale", 1.0)
                p = g["p"]
                if opt.get("exclude_problem"):
                    p = p[p["label"] != "问题词"]
                    if p.empty:
                        p = g["p"]
                p = p.assign(cap_factor=np.where(p["label"] == "问题词", opt.get("problem_cap", problem_cap), 1.0))
                w = (p["rho"] * p["c"]).to_numpy(dtype=float)
                s = p["s"].to_numpy(dtype=float)
                caps = _caps(p, budget, opt.get("cap", cap_mult))
                r = alloc.solve_group(w, s, gamma, caps, budget, min_spend=opt.get("min_spend", min_spend))
                opt_total += g["eff"] * r["objective"]
                selected.append(r["n_selected"])
                p_all = elig[elig["推广单元ID"] == unit]
                xb = alloc.proportional_allocation(p_all["s"].to_numpy(dtype=float), budget)
                prop_total += g["eff"] * float(
                    np.sum((p_all["rho"] * p_all["c"]).to_numpy() * np.power(xb / p_all["s"].to_numpy(), gamma))
                )
            rows.append(
                {
                    "scenario": name,
                    "window": {"feb": "2 月", "aug": "8 月"}[window],
                    "opt_regs": opt_total,
                    "prop_regs": prop_total,
                    "gain": 100 * (opt_total / max(prop_total, 1e-9) - 1),
                    "selected": float(np.mean(selected)) if selected else 0.0,
                }
            )
    return pd.DataFrame(rows)


@stage(
    "allocate",
    deps=("eda", "classify"),
    description="Q3: keyword selection and daily budget allocation for 2025-02-01..08 and 08-01..08 with audit",
)
def allocate(ctx: StageContext) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed_everything("allocate"))
    ud, units, params, pos, rates = _load_params(ctx)
    cap_mult = float(ctx.cfg("allocate.cap_multiplier", 4.0))
    min_spend = float(ctx.cfg("allocate.min_spend", 0.5))
    problem_cap = float(ctx.cfg("allocate.problem_cap_factor", 0.5))
    n_boot = int(ctx.param("bootstrap", ctx.cfg("allocate.bootstrap", 100)))
    unit_index = units.set_index("推广单元ID")
    s1 = ud[["日期", "方案ID", "推广单元ID", "消费额", "点击量", "展现量"]]
    ctx.write_json("position_models.json", pos)

    rows, summaries, audits = [], [], []
    boot_records: list[dict[str, Any]] = []
    window_alt: list[dict[str, Any]] = []
    eligible_tables = []
    all_groups: dict[str, dict[tuple[int, pd.Timestamp], dict[str, Any]]] = {}
    elig_by_window: dict[str, pd.DataFrame] = {}
    for window, (start, end) in Q3_WINDOWS.items():
        dates = pd.date_range(start, end, freq="D")
        day_eff = _day_efficiency(ud, units, dates)
        actual = s1[(s1["日期"] >= start) & (s1["日期"] <= end) & (s1["消费额"] > 0)]
        active_units = {int(u) for u in actual["推广单元ID"].unique()}
        elig = _eligible(params, active_units, problem_cap)
        elig["window"] = window
        eligible_tables.append(elig)
        groups: dict[tuple[int, pd.Timestamp], dict[str, Any]] = {}
        for _, rec in actual.iterrows():
            unit, date, budget = int(rec["推广单元ID"]), pd.Timestamp(rec["日期"]), float(rec["消费额"])
            p = elig[(elig["推广单元ID"] == unit) & elig["eligible"]]
            if p.empty:
                continue
            gamma = float(unit_index.loc[unit, "gamma"])
            eff = float(day_eff[unit]["eff"][dates.get_loc(date)])
            x, res, a, caps = _solve_unit_day(p, gamma, budget, cap_mult, min_spend)
            ev = _evaluate(x, p, gamma, eff, pos[unit])
            ev.insert(0, "date", date.strftime("%Y-%m-%d"))
            ev["window"] = window
            rows.append(ev[ev["spend"] > 0])
            # baselines evaluated under the same response model and day multiplier
            p_all = elig[elig["推广单元ID"] == unit]
            base = _evaluate(
                alloc.proportional_allocation(p_all["s"].to_numpy(dtype=float), budget), p_all, gamma, eff, pos[unit]
            )
            equal = _evaluate(alloc.equal_allocation(len(p), budget), p, gamma, eff, pos[unit])
            sel = x > 0
            audit = alloc.verify_allocation(x, a, gamma, caps, budget, rng, floor=min_spend)
            audits.append(
                {
                    "window": window,
                    "date": date.strftime("%Y-%m-%d"),
                    "推广单元ID": unit,
                    **{k: v for k, v in audit.items() if k != "checks"},
                    "checks": audit["checks"],
                }
            )
            groups[(unit, date)] = {"p": p, "gamma": gamma, "eff": eff, "budget": budget, "x": x, "a": a, "caps": caps}
            all_groups[window] = groups
            elig_by_window[window] = elig
            summaries.append(
                {
                    "window": window,
                    "date": date.strftime("%Y-%m-%d"),
                    "方案ID": int(rec["方案ID"]),
                    "推广单元ID": unit,
                    "budget": budget,
                    "spent": float(x.sum()),
                    "eligible": len(p),
                    "selected": int(sel.sum()),
                    "capped": audit["n_capped"],
                    "floored": audit["n_floored"],
                    "day_eff": eff,
                    "actual_clicks": float(rec["点击量"]),
                    "actual_regs_expected": float(rec["点击量"]) * rates[unit],
                    "opt_clicks": float(ev["clicks"].sum()),
                    "opt_regs": float(ev["regs"].sum()),
                    "opt_views": float(ev["views"].sum()),
                    "opt_position": float(np.average(ev["position"], weights=np.maximum(ev["clicks"], 1e-9))),
                    "opt_cpc": float(x.sum() / max(ev["clicks"].sum(), 1e-9)),
                    "prop_clicks": float(base["clicks"].sum()),
                    "prop_regs": float(base["regs"].sum()),
                    "prop_cpc": float(budget / max(base["clicks"].sum(), 1e-9)),
                    "equal_clicks": float(equal["clicks"].sum()),
                    "equal_regs": float(equal["regs"].sum()),
                    "audit_ok": audit["ok"],
                    "cvx_gap": audit["cvxpy"].get("relative_gap", np.nan),
                }
            )
        # bootstrap over the elasticity uncertainty: improvement ratio of optimal vs proportional
        for b in range(n_boot):
            opt = prop = 0.0
            for (unit, _date), g in groups.items():
                row = unit_index.loc[unit]
                mu = row["gamma_hat"] if row["gamma_source"] == "unit" else row["pooled_gamma"]
                se = row["gamma_se"] if row["gamma_source"] == "unit" else row["pooled_se"]
                gb = float(np.clip(rng.normal(mu, se), *resp.GAMMA_BOUNDS))
                p = g["p"]
                w = (p["rho"] * p["c"]).to_numpy(dtype=float)
                s = p["s"].to_numpy(dtype=float)
                caps = g["caps"]
                r = alloc.solve_group(w, s, gb, caps, g["budget"], min_spend=min_spend)
                opt += g["eff"] * r["objective"]
                p_all = elig[elig["推广单元ID"] == unit]
                xb = alloc.proportional_allocation(p_all["s"].to_numpy(dtype=float), g["budget"])
                prop += g["eff"] * float(
                    np.sum((p_all["rho"] * p_all["c"]).to_numpy() * np.power(xb / p_all["s"].to_numpy(), gb))
                )
            boot_records.append(
                {"window": window, "draw": b, "opt_regs": opt, "prop_regs": prop, "ratio": opt / max(prop, 1e-9)}
            )
        # alternative budget reading: the window total per unit is also allocated across days
        for unit in sorted(active_units):
            unit_days = [(u, d) for (u, d) in groups if u == unit]
            if not unit_days:
                continue
            gamma = float(unit_index.loc[unit, "gamma"])
            B = float(sum(groups[k]["budget"] for k in unit_days))
            eff_days = np.array([day_eff[unit]["eff"][dates.get_loc(d)] for _, d in unit_days])
            day_cap = 1.5 * max(groups[k]["budget"] for k in unit_days)
            X = alloc.water_filling(eff_days, gamma, np.full(len(unit_days), day_cap), B)
            total = 0.0
            for (u, d), Xd in zip(unit_days, X):
                g = groups[(u, d)]
                r = alloc.solve_group(
                    (g["p"]["rho"] * g["p"]["c"]).to_numpy(dtype=float),
                    g["p"]["s"].to_numpy(dtype=float),
                    gamma,
                    g["caps"],
                    float(Xd),
                    min_spend=min_spend,
                )
                total += g["eff"] * r["objective"]
                window_alt.append(
                    {
                        "window": window,
                        "推广单元ID": unit,
                        "date": d.strftime("%Y-%m-%d"),
                        "daily_budget": g["budget"],
                        "window_budget_day": float(Xd),
                        "day_eff": g["eff"],
                        "regs_window": g["eff"] * r["objective"],
                    }
                )
    sens = _sensitivity(all_groups, elig_by_window, unit_index, cap_mult, min_spend, problem_cap)
    _save(ctx, sens, "sensitivity")
    allocation = pd.concat(rows, ignore_index=True)
    _save(ctx, allocation, "allocation")
    summary = pd.DataFrame(summaries)
    _save(ctx, summary, "allocation_summary")
    _save(ctx, pd.concat(eligible_tables, ignore_index=True), "keyword_parameters")
    boot = pd.DataFrame(boot_records)
    _save(ctx, boot, "bootstrap")
    alt = pd.DataFrame(window_alt)
    _save(ctx, alt, "window_budget_alternative")
    ctx.write_json(
        "allocation_audit.json",
        {
            "ok": all(a["ok"] for a in audits),
            "groups": len(audits),
            "failed": [a for a in audits if not a["ok"]],
            "cvx_available": sum(1 for a in audits if a["cvxpy"].get("available")),
            "max_cvx_gap": float(np.nanmax([a["cvxpy"].get("relative_gap", np.nan) for a in audits])),
            "aggregate_gap": float(
                (
                    sum(a["cvxpy"].get("relaxed_objective", a["objective"]) for a in audits)
                    - sum(a["objective"] for a in audits)
                )
                / sum(a["objective"] for a in audits)
            ),
            "audits": audits,
        },
    )
    # response-model calibration check: proportional allocation at the actual budget should reproduce actual clicks
    calib = summary[["window", "推广单元ID", "actual_clicks", "prop_clicks"]].copy()
    calib["ape"] = np.abs(calib["prop_clicks"] - calib["actual_clicks"]) / np.maximum(calib["actual_clicks"], 1.0)
    ctx.write_json(
        "calibration_check.json",
        {
            "mape": float(calib["ape"].mean()),
            "median_ape": float(calib["ape"].median()),
            "n": len(calib),
            "spend_weighted_mape": float(np.average(calib["ape"], weights=summary["budget"])),
        },
    )
    if not all(a["ok"] for a in audits):
        raise RuntimeError("allocation audit failed for some unit-days; see allocation_audit.json")

    win_tot = summary.groupby("window")[
        [
            "budget",
            "opt_clicks",
            "opt_regs",
            "prop_clicks",
            "prop_regs",
            "equal_regs",
            "actual_clicks",
            "actual_regs_expected",
        ]
    ].sum()
    for window, key in (("feb", "Feb"), ("aug", "Aug")):
        w = win_tot.loc[window]
        ctx.number(f"QthreeBudget{key}", w["budget"], ",.2f")
        ctx.number(f"QthreeOptRegs{key}", w["opt_regs"], ".2f")
        ctx.number(f"QthreePropRegs{key}", w["prop_regs"], ".2f")
        ctx.number(f"QthreeOptClicks{key}", w["opt_clicks"], ",.1f")
        ctx.number(f"QthreePropClicks{key}", w["prop_clicks"], ",.1f")
        ctx.number(f"QthreeActualClicks{key}", w["actual_clicks"], ",.0f")
        ctx.number(f"QthreeRegGainPct{key}", 100 * (w["opt_regs"] / w["prop_regs"] - 1), ".2f")
        ctx.number(f"QthreeClickGainPct{key}", 100 * (w["opt_clicks"] / w["prop_clicks"] - 1), ".2f")
        ctx.number(f"QthreeEqualRegGainPct{key}", 100 * (w["opt_regs"] / w["equal_regs"] - 1), ".2f")
        bw = boot[boot["window"] == window]["ratio"]
        ctx.number(f"QthreeRegGainCiLow{key}", 100 * (bw.quantile(0.025) - 1), ".2f")
        ctx.number(f"QthreeRegGainCiHigh{key}", 100 * (bw.quantile(0.975) - 1), ".2f")
        sw = summary[summary["window"] == window]
        ctx.number(f"QthreeUnitDays{key}", len(sw))
        ctx.number(f"QthreeUnits{key}", int(sw["推广单元ID"].nunique()))
        ctx.number(f"QthreeSelectedKeywords{key}", int(allocation[allocation["window"] == window]["序号"].nunique()))
        ctx.number(f"QthreeEligibleKeywords{key}", int(sw.groupby("推广单元ID")["eligible"].first().sum()))
        ctx.number(f"QthreeAvgCpc{key}", w["budget"] / w["opt_clicks"], ".4f")
        ctx.number(f"QthreePropCpc{key}", w["budget"] / w["prop_clicks"], ".4f")
        aw = alt[alt["window"] == window]
        if len(aw):
            ctx.number(f"QthreeWindowRegs{key}", aw["regs_window"].sum(), ".2f")
            ctx.number(f"QthreeWindowGainPct{key}", 100 * (aw["regs_window"].sum() / w["opt_regs"] - 1), ".2f")
    ctx.number("QthreeAuditGroups", len(audits))
    ctx.number("QthreeMaxCvxGap", float(np.nanmax([a["cvxpy"].get("relative_gap", np.nan) for a in audits])), ".2e")
    relaxed = sum(a["cvxpy"].get("relaxed_objective", a["objective"]) for a in audits)
    achieved = sum(a["objective"] for a in audits)
    ctx.number("QthreeAggregateGapPct", 100 * (relaxed - achieved) / achieved, ".3f")
    ctx.number("QthreeCalibrationMape", 100 * float(calib["ape"].mean()), ".2f")
    ctx.number("QthreeCalibrationMedianApe", 100 * float(calib["ape"].median()), ".2f")
    ctx.number("QthreeRows", len(allocation))
    ctx.number("QthreeCapMultiplier", cap_mult, ".1f")
    ctx.number("QthreeMinSpend", min_spend, ".2f")
    return {"rows": len(allocation), "audit_ok": True, "windows": win_tot.to_dict(orient="index")}


# ----------------------------------------------------------------------------------------------- Q4
@stage(
    "forecast",
    deps=("eda", "classify", "allocate"),
    description="Q4: 2026-09-11..17 forecasts, backtests, stochastic allocation and expected ranges",
)
def forecast(ctx: StageContext) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed_everything("forecast"))
    ud, units, params, pos, _ = _load_params(ctx)
    cap_mult = float(ctx.cfg("allocate.cap_multiplier", 4.0))
    min_spend = float(ctx.cfg("allocate.min_spend", 0.5))
    problem_cap = float(ctx.cfg("allocate.problem_cap_factor", 0.5))
    n_scen = int(ctx.param("scenarios", ctx.cfg("forecast.scenarios", 500)))
    unit_index = units.set_index("推广单元ID")
    dates = pd.date_range(*Q4_DATES, freq="D")
    future = day_features(dates)
    origins = list(pd.date_range("2025-10-09", "2025-12-25", freq="7D"))

    series_by_unit: dict[int, pd.DataFrame] = {}
    for _, row in units.iterrows():
        unit = int(row["推广单元ID"])
        series_by_unit[unit] = fc.unit_series(
            ud[ud["推广单元ID"] == unit],
            float(row["gamma"]),
            float(row["mean_daily_spend"]),
            float(row["mean_daily_clicks"]),
        )
    bt_frames = []
    for unit, series in series_by_unit.items():
        for target in fc.TARGETS:
            bt = fc.rolling_backtest(series, target, origins)
            if len(bt):
                bt.insert(0, "推广单元ID", unit)
                bt_frames.append(bt)
    backtest = pd.concat(bt_frames, ignore_index=True)
    _save(ctx, backtest, "backtest")
    bt_summary = (
        backtest.groupby(["target", "model"])[["mae", "rmse", "mape_raw", "pinball10", "pinball90", "coverage80"]]
        .mean()
        .reset_index()
    )
    _save(ctx, bt_summary, "backtest_summary")

    fc_rows = []
    unit_fc: dict[int, dict[str, dict[str, Any]]] = {}
    for unit, series in series_by_unit.items():
        unit_fc[unit] = {}
        for target, info in fc.TARGETS.items():
            if len(series) < 40:
                continue
            f = fc.forecast_unit(series, target, future)
            unit_fc[unit][target] = f
            sd_total = float(np.sqrt(f["sd"] ** 2 + f["month_sd"] ** 2))
            for i, d in enumerate(dates):
                fc_rows.append(
                    {
                        "推广单元ID": unit,
                        "date": d.strftime("%Y-%m-%d"),
                        "target": target,
                        "mean_t": float(f["mean"][i]),
                        "sd_t": sd_total,
                        "point": float(fc.back_transform(np.array([f["mean"][i]]), info["transform"])[0]),
                        "q10": float(
                            fc.back_transform(np.array([f["mean"][i] - fc.Z80 * sd_total]), info["transform"])[0]
                        ),
                        "q90": float(
                            fc.back_transform(np.array([f["mean"][i] + fc.Z80 * sd_total]), info["transform"])[0]
                        ),
                        "month_known": f["month_known"],
                    }
                )
    forecasts = pd.DataFrame(fc_rows)
    _save(ctx, forecasts, "unit_forecasts")

    # budgets: 2025 same-period spend per unit (main) and the annual-average alternative
    s1 = ud[["日期", "方案ID", "推广单元ID", "消费额"]]
    ref = s1[(s1["日期"] >= Q4_REFERENCE[0]) & (s1["日期"] <= Q4_REFERENCE[1])]
    budget_main = ref.groupby("推广单元ID")["消费额"].sum()
    day_cap_ref = ref.groupby("推广单元ID")["消费额"].max()
    annual = s1.groupby("推广单元ID")["消费额"].sum()
    budget_alt = annual / 365.0 * len(dates)
    budgets = (
        pd.DataFrame({"budget_same_period": budget_main, "budget_annual_avg": budget_alt}).fillna(0.0).reset_index()
    )
    _save(ctx, budgets, "budgets")
    active_units = {int(u) for u, b in budget_main.items() if b > 0}
    elig = _eligible(params, active_units, problem_cap)
    elig_all = _eligible(params, {int(u) for u in units["推广单元ID"]}, problem_cap)
    _save(ctx, elig, "keyword_parameters")

    rows, day_rows, audits, unit_rows = [], [], [], []
    for unit in sorted(active_units):
        p = elig[(elig["推广单元ID"] == unit) & elig["eligible"]]
        if p.empty or unit not in unit_fc or "log_eff" not in unit_fc[unit]:
            continue
        gamma = float(unit_index.loc[unit, "gamma"])
        f_eff = unit_fc[unit]["log_eff"]
        sd_eff = float(np.sqrt(f_eff["sd"] ** 2 + f_eff["month_sd"] ** 2))
        eff_mean = np.exp(f_eff["mean"] + 0.5 * sd_eff**2)
        B = float(budget_main[unit])
        day_cap = float(max(1.5 * day_cap_ref[unit], B / len(dates)))
        # day split with the same floor: a day is either skipped or receives at least the minimum amount
        day_caps = np.full(len(dates), day_cap)
        X = alloc.solve_group(eff_mean, np.ones(len(dates)), gamma, day_caps, B, min_spend=min_spend)["x"]
        day_audit = alloc.verify_allocation(X, eff_mean, gamma, day_caps, B, rng, floor=min_spend)
        audits.append({"推广单元ID": unit, "level": "days", **{k: v for k, v in day_audit.items() if k != "cvxpy"}})
        f_ctr = unit_fc[unit]["logit_ctr"]
        f_top = unit_fc[unit]["logit_top"]
        f_cpc = unit_fc[unit]["log_cpc"]
        kw_sd = float(f_cpc["sd"])
        totals = {k: np.zeros(n_scen) for k in ("clicks", "imp", "views", "regs", "spend")}
        for i, d in enumerate(dates):
            if X[i] <= 0:
                continue
            w = (p["rho"] * p["c"]).to_numpy(dtype=float) * float(eff_mean[i])
            s = p["s"].to_numpy(dtype=float)
            caps = _caps(p, float(X[i]), cap_mult)
            res = alloc.solve_group(w, s, gamma, caps, float(X[i]), min_spend=min_spend)
            x = res["x"]
            sel = x > 0
            a = w * np.power(s, -gamma)
            audit = alloc.verify_allocation(x, a, gamma, caps, float(X[i]), rng, floor=min_spend)
            audits.append(
                {
                    "推广单元ID": unit,
                    "level": "keywords",
                    "date": d.strftime("%Y-%m-%d"),
                    **{k: v for k, v in audit.items() if k != "checks"},
                    "checks": audit["checks"],
                }
            )
            sim = fc.simulate_group(
                x[sel],
                p[sel],
                gamma,
                float(f_eff["mean"][i]),
                sd_eff,
                kw_sd,
                float(f_ctr["mean"][i]),
                float(np.sqrt(f_ctr["sd"] ** 2 + f_ctr["month_sd"] ** 2)),
                pos[unit]["top"],
                pos[unit]["first"],
                float(np.sqrt(f_top["sd"] ** 2 + f_top["month_sd"] ** 2)),
                rng,
                n_scen,
            )
            ps = p[sel]
            stats = {k: fc.summarise(v) for k, v in sim.items()}
            for j in range(int(sel.sum())):
                rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "方案ID": int(ps["方案ID"].iloc[j]),
                        "推广单元ID": unit,
                        "关键词": int(ps["关键词"].iloc[j]),
                        "序号": int(ps["序号"].iloc[j]),
                        "label": ps["label"].iloc[j],
                        "spend": float(x[sel][j]),
                        **{
                            f"{k}_{q}": float(stats[k][q][j])
                            for k in ("clicks", "cpc", "imp", "position", "views", "regs", "p_top")
                            for q in ("mean", "q10", "q50", "q90")
                        },
                    }
                )
            for k in ("clicks", "imp", "views", "regs"):
                totals[k] += np.nansum(sim[k], axis=1)
            totals["spend"] += float(x.sum())
            day_tot = {k: np.nansum(sim[k], axis=1) for k in ("clicks", "imp", "views", "regs")}
            day_cpc = float(x.sum()) / np.maximum(day_tot["clicks"], 1e-9)
            pos_w = np.nansum(sim["position"] * sim["clicks"], axis=1) / np.maximum(day_tot["clicks"], 1e-9)
            day_rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "方案ID": int(ps["方案ID"].iloc[0]),
                    "推广单元ID": unit,
                    "budget_day": float(X[i]),
                    "spend": float(x.sum()),
                    "selected": int(sel.sum()),
                    "eff_mean": float(eff_mean[i]),
                    **{
                        f"{k}_{q}": float(v)
                        for k, arr in day_tot.items()
                        for q, v in zip(
                            ("mean", "q10", "q90"), (arr.mean(), np.quantile(arr, 0.1), np.quantile(arr, 0.9))
                        )
                    },
                    **{
                        f"cpc_{q}": float(v)
                        for q, v in zip(
                            ("mean", "q10", "q90"),
                            (day_cpc.mean(), np.quantile(day_cpc, 0.1), np.quantile(day_cpc, 0.9)),
                        )
                    },
                    **{
                        f"position_{q}": float(v)
                        for q, v in zip(
                            ("mean", "q10", "q90"), (pos_w.mean(), np.quantile(pos_w, 0.1), np.quantile(pos_w, 0.9))
                        )
                    },
                }
            )
        unit_rows.append(
            {
                "推广单元ID": unit,
                "budget": B,
                "spent": float(totals["spend"][0]),
                "gamma": gamma,
                "eligible": len(p),
                "day_split": [float(v) for v in X],
                **{
                    f"{k}_{q}": float(v)
                    for k, arr in totals.items()
                    if k != "spend"
                    for q, v in zip(("mean", "q10", "q90"), (arr.mean(), np.quantile(arr, 0.1), np.quantile(arr, 0.9)))
                },
            }
        )
    alt_expected: dict[str, float] = {}
    for name, budget_series in (("same_period", budget_main), ("annual_avg", budget_alt)):
        total_regs = 0.0
        for unit in sorted(int(u) for u, b in budget_series.items() if b > 0):
            p = elig_all[(elig_all["推广单元ID"] == unit) & elig_all["eligible"]]
            if p.empty or unit not in unit_fc or "log_eff" not in unit_fc[unit]:
                continue
            gamma = float(unit_index.loc[unit, "gamma"])
            f_eff = unit_fc[unit]["log_eff"]
            sd_eff = float(np.sqrt(f_eff["sd"] ** 2 + f_eff["month_sd"] ** 2))
            eff_mean = np.exp(f_eff["mean"] + 0.5 * sd_eff**2)
            B = float(budget_series[unit])
            cap_day = float(max(1.5 * day_cap_ref.get(unit, B / len(dates)), B / len(dates)))
            X = alloc.solve_group(
                eff_mean, np.ones(len(dates)), gamma, np.full(len(dates), cap_day), B, min_spend=min_spend
            )["x"]
            for i in range(len(dates)):
                if X[i] <= 0:
                    continue
                w = (p["rho"] * p["c"]).to_numpy(dtype=float) * float(eff_mean[i])
                s = p["s"].to_numpy(dtype=float)
                total_regs += alloc.solve_group(
                    w, s, gamma, _caps(p, float(X[i]), cap_mult), float(X[i]), min_spend=min_spend
                )["objective"]
        alt_expected[name] = total_regs
    ctx.write_json("budget_alternatives.json", alt_expected)
    plan = pd.DataFrame(rows)
    _save(ctx, plan, "plan")
    days = pd.DataFrame(day_rows)
    _save(ctx, days, "plan_by_unit_day")
    per_unit = pd.DataFrame(unit_rows)
    per_unit.to_json(ctx.out("plan_by_unit.json"), orient="records", force_ascii=False, indent=2)
    audit_ok = all(a["ok"] for a in audits)
    ctx.write_json(
        "allocation_audit.json",
        {"ok": audit_ok, "groups": len(audits), "failed": [a for a in audits if not a["ok"]], "audits": audits},
    )
    if not audit_ok:
        raise RuntimeError("Q4 allocation audit failed; see allocation_audit.json")

    cal = bt_summary[bt_summary["model"] == "calendar"].set_index("target")
    sn = bt_summary[bt_summary["model"] == "seasonal_naive"].set_index("target")
    m28 = bt_summary[bt_summary["model"] == "mean28"].set_index("target")
    for target, key in (
        ("log_eff", "Eff"),
        ("log_cpc", "Cpc"),
        ("logit_ctr", "Ctr"),
        ("logit_top", "Top"),
        ("log_clicks", "Clicks"),
    ):
        ctx.number(f"BacktestCalendarMae{key}", cal.loc[target, "mae"], ".4f")
        ctx.number(f"BacktestNaiveMae{key}", sn.loc[target, "mae"], ".4f")
        ctx.number(f"BacktestMeanMae{key}", m28.loc[target, "mae"], ".4f")
        ctx.number(f"BacktestCalendarCoverage{key}", 100 * cal.loc[target, "coverage80"], ".2f")
        ctx.number(f"BacktestNaiveCoverage{key}", 100 * sn.loc[target, "coverage80"], ".2f")
        ctx.number(f"BacktestMeanCoverage{key}", 100 * m28.loc[target, "coverage80"], ".2f")
        ctx.number(f"BacktestCalendarPinball{key}", cal.loc[target, "pinball10"] + cal.loc[target, "pinball90"], ".4f")
        ctx.number(f"BacktestNaivePinball{key}", sn.loc[target, "pinball10"] + sn.loc[target, "pinball90"], ".4f")
    tot = per_unit[
        [
            "budget",
            "spent",
            "clicks_mean",
            "clicks_q10",
            "clicks_q90",
            "imp_mean",
            "imp_q10",
            "imp_q90",
            "views_mean",
            "views_q10",
            "views_q90",
            "regs_mean",
            "regs_q10",
            "regs_q90",
        ]
    ].sum()
    ctx.number("QfourBudget", tot["budget"], ",.2f")
    ctx.number("QfourBudgetAnnualAlt", float(budgets["budget_annual_avg"].sum()), ",.2f")
    ctx.number("QfourRegsExpectedSamePeriod", alt_expected["same_period"], ".1f")
    ctx.number("QfourRegsExpectedAnnualAlt", alt_expected["annual_avg"], ".1f")
    ctx.number("QfourUnits", len(per_unit))
    ctx.number("QfourSkippedUnits", int(len(units) - len(per_unit)))
    ctx.number("QfourRows", len(plan))
    ctx.number("QfourSelectedKeywords", int(plan["序号"].nunique()))
    for k, key in (("clicks", "Clicks"), ("imp", "Imp"), ("views", "Views"), ("regs", "Regs")):
        ctx.number(f"Qfour{key}Mean", tot[f"{k}_mean"], ",.1f")
        ctx.number(f"Qfour{key}Low", tot[f"{k}_q10"], ",.1f")
        ctx.number(f"Qfour{key}High", tot[f"{k}_q90"], ",.1f")
    ctx.number("QfourCpcMean", tot["spent"] / tot["clicks_mean"], ".4f")
    ctx.number("QfourCpcLow", tot["spent"] / tot["clicks_q90"], ".4f")
    ctx.number("QfourCpcHigh", tot["spent"] / tot["clicks_q10"], ".4f")
    ctx.number(
        "QfourPositionMean",
        float(np.average(days["position_mean"], weights=np.maximum(days["clicks_mean"], 1e-9))),
        ".4f",
    )
    ctx.number("QfourScenarios", n_scen)
    ctx.number("QfourAuditGroups", len(audits))
    ref_clicks = float(ud[(ud["日期"] >= Q4_REFERENCE[0]) & (ud["日期"] <= Q4_REFERENCE[1])]["点击量"].sum())
    ctx.number("QfourRefClicks", ref_clicks, ",.0f")
    ctx.number("QfourClickGainVsRefPct", 100 * (tot["clicks_mean"] / ref_clicks - 1), ".2f")
    return {"rows": len(plan), "units": len(per_unit), "regs_mean": float(tot["regs_mean"]), "audit_ok": audit_ok}


# ----------------------------------------------------------------------------------------------- results
@stage(
    "results",
    deps=("classify", "allocate", "forecast"),
    description="Fill result2/3/4.xlsx from the organisers' templates",
)
def results(ctx: StageContext) -> dict[str, Any]:
    table = pd.read_parquet(ctx.dep("classify") / "classification.parquet").sort_values("序号")
    header2 = ["方案ID", "推广单元", "序号", *cls.LABELS]
    rows2 = [
        [int(r["方案ID"]), int(r["推广单元ID"]), int(r["序号"]), *[int(r["label"] == lab) for lab in cls.LABELS]]
        for _, r in table.iterrows()
    ]
    info2 = xlsx.write_result(ctx.templates / "result2.xlsx", ctx.out("result2.xlsx"), {"Sheet1": (header2, rows2)})

    def plan_rows(df: pd.DataFrame, clicks: str, views: str, regs: str, position: str) -> list[list[Any]]:
        df = df.sort_values(["date", "方案ID", "推广单元ID", "spend"], ascending=[True, True, True, False])
        return [
            [
                str(r["date"]),
                int(r["方案ID"]),
                int(r["推广单元ID"]),
                int(r["关键词"]),
                float(r["spend"]),
                float(r[position]),
                float(r[clicks]),
                float(r[views]),
                float(r[regs]),
            ]
            for _, r in df.iterrows()
        ]

    allocation = pd.read_parquet(ctx.dep("allocate") / "allocation.parquet")
    info3 = xlsx.write_result(
        ctx.templates / "result3.xlsx",
        ctx.out("result3.xlsx"),
        {"Sheet1": (RESULT_HEADER, plan_rows(allocation, "clicks", "views", "regs", "position"))},
    )
    plan = pd.read_parquet(ctx.dep("forecast") / "plan.parquet")
    info4 = xlsx.write_result(
        ctx.templates / "result4.xlsx",
        ctx.out("result4.xlsx"),
        {"Sheet1": (RESULT_HEADER, plan_rows(plan, "clicks_mean", "views_mean", "regs_mean", "position_mean"))},
    )
    summary = {"result2": info2["sheets"], "result3": info3["sheets"], "result4": info4["sheets"]}
    ctx.write_json("results_summary.json", summary)
    return {k: v["Sheet1"]["rows"] for k, v in summary.items()}


@stage(
    "figures", deps=("eda", "classify", "allocate", "forecast"), description="Paper figures (vector PDF + PNG preview)"
)
def figures(ctx: StageContext) -> dict[str, Any]:
    from pipelines.e.figures import build_figures

    return build_figures(ctx)


@stage("tables", deps=("eda", "classify", "allocate", "forecast"), description="Paper tables (booktabs .tex + CSV)")
def tables(ctx: StageContext) -> dict[str, Any]:
    from pipelines.e.tables import build_tables

    return build_tables(ctx)
