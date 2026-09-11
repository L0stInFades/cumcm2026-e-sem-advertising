"""Paper figures for Problem E (vector PDF + PNG preview, Okabe-Ito palette, CJK-capable fonts)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from forge import plotting
from forge.context import StageContext
from pipelines.e.calendar import HOLIDAY_BLOCKS, WEEKDAY_NAMES
from pipelines.e.classify import LABELS

# fixed hue per keyword class (identity colour never depends on ordering)
LABEL_COLORS = {"黄金词": "#E69F00", "重点词": "#0072B2", "潜力词": "#009E73", "问题词": "#D55E00"}
LABEL_COLORS["无效词"] = "#999999"
STRATEGY_COLORS = {"最优分配": "#0072B2", "按历史比例": "#D55E00", "均匀分配": "#009E73"}
MODEL_COLORS = {"calendar": "#0072B2", "seasonal_naive": "#D55E00", "mean28": "#009E73"}
MODEL_NAMES = {"calendar": "日历回归", "seasonal_naive": "季节朴素", "mean28": "28 日均值"}


def _shade_holidays(ax: Any) -> None:
    for start, end in HOLIDAY_BLOCKS.values():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), color="#CC79A7", alpha=0.18, lw=0)


def fig_daily_series(ctx: StageContext, daily: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plotting.new_figure(6.3, 5.2, nrows=3, sharex=True)
    labels = ("日消费额 / 元", "日点击量 / 次", "日新注册数 / 人")
    for ax, col, label in zip(axes, ("spend", "clicks", "regs"), labels):
        ax.plot(daily["date"], daily[col], lw=0.8, color="#0072B2", alpha=0.55, label="日值")
        ax.plot(
            daily["date"],
            daily[col].rolling(7, center=True, min_periods=3).mean(),
            lw=1.8,
            color="#0072B2",
            label="7 日滑动均值",
        )
        _shade_holidays(ax)
        for d in daily.loc[daily["is_adjusted_workday"], "date"]:
            ax.axvline(d, color="#D55E00", lw=0.8, ls=":")
        ax.set_ylabel(label)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("日期（2025 年；阴影 = 法定节假日，虚线 = 调休上班日）")
    return plotting.save(fig, ctx.out("figures", "fig_daily_series.pdf"))


def fig_calendar_effects(ctx: StageContext, coefs: pd.DataFrame) -> dict[str, Any]:
    terms = [f"wd_{w}" for w in WEEKDAY_NAMES if w != "周三"] + [
        "holiday",
        "adjusted_workday",
        "pre_holiday",
        "post_holiday",
    ]
    names = {
        **{f"wd_{w}": w for w in WEEKDAY_NAMES},
        "holiday": "法定节假日",
        "adjusted_workday": "调休上班日",
        "pre_holiday": "假前一日",
        "post_holiday": "假后一日",
    }
    targets = [("regs", "新注册数", "#0072B2"), ("spend", "消费额", "#D55E00")]
    targets.append(("clicks", "点击量", "#009E73"))
    fig, ax = plotting.new_figure(6.3, 3.6)
    y = np.arange(len(terms))
    for k, (target, label, color) in enumerate(targets):
        sub = coefs[coefs["target"] == target].set_index("term").reindex(terms)
        off = (k - 1) * 0.25
        ax.errorbar(
            sub["effect_pct"],
            y + off,
            xerr=[sub["effect_pct"] - sub["effect_pct_low"], sub["effect_pct_high"] - sub["effect_pct"]],
            fmt="o",
            ms=4,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2,
            label=label,
        )
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([names[t] for t in terms])
    ax.set_xlabel("相对周三 / 普通日的效应（%，HC3 稳健 95% 置信区间）")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    return plotting.save(fig, ctx.out("figures", "fig_calendar_effects.pdf"))


def fig_unit_kpis(ctx: StageContext, kpis: pd.DataFrame) -> dict[str, Any]:
    k = kpis.sort_values("spend", ascending=False).reset_index(drop=True)
    fig, axes = plotting.new_figure(6.3, 3.2, ncols=2, gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    ax.barh(np.arange(len(k)), 100 * k["spend_share"], color="#0072B2", height=0.62)
    ax.set_yticks(np.arange(len(k)))
    ax.set_yticklabels([str(u) for u in k["推广单元ID"]], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("年度消费额占比 / %")
    for i, v in enumerate(100 * k["spend_share"]):
        if v >= 3:
            ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=7, color="#333333")
    ax = axes[1]
    size = 12 + 300 * np.sqrt(k["spend_share"])
    ax.scatter(k["cpc"], 100 * k["ctr"], s=size, color="#0072B2", alpha=0.65, edgecolor="white", lw=0.8)
    for _, r in k.iterrows():
        ax.annotate(
            str(r["推广单元ID"])[-4:],
            (r["cpc"], 100 * r["ctr"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
            color="#333333",
        )
    ax.set_xlabel("平均点击成本 CPC / 元")
    ax.set_ylabel("点击率 CTR / %")
    ax.set_yscale("log")
    return plotting.save(fig, ctx.out("figures", "fig_unit_kpis.pdf"))


def fig_lorenz(ctx: StageContext, lorenz: pd.DataFrame, conc: dict[str, Any]) -> dict[str, Any]:
    fig, ax = plotting.new_figure(4.2, 3.4)
    ax.plot(100 * lorenz["keyword_share"], 100 * lorenz["spend_share"], color="#0072B2", label="消费额")
    ax.plot(100 * lorenz["keyword_share"], 100 * lorenz["click_share"], color="#D55E00", label="点击量")
    ax.plot([0, 100], [0, 100], color="#888888", lw=0.8, ls="--", label="均匀分布")
    ax.set_xlabel("关键词累计占比 / %（按消费额降序）")
    ax.set_ylabel("累计占比 / %")
    ax.annotate(
        f"前 10 个词占 {100 * conc['top10_spend_share']:.1f}% 消费",
        (100 * 10 / conc["keywords"], 100 * conc["top10_spend_share"]),
        xytext=(25, 60),
        textcoords="data",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
    )
    ax.legend(loc="lower right")
    return plotting.save(fig, ctx.out("figures", "fig_lorenz.pdf"))


def fig_response_curves(ctx: StageContext, ud: pd.DataFrame, units: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plotting.new_figure(6.3, 6.0, nrows=3, ncols=4)
    for ax, (_, row) in zip(axes.ravel(), units.sort_values("推广单元ID").iterrows()):
        unit = int(row["推广单元ID"])
        d = ud[(ud["推广单元ID"] == unit) & (ud["消费额"] > 0) & (ud["点击量"] > 0)]
        ax.scatter(d["消费额"], d["点击量"], s=6, color="#0072B2", alpha=0.45, lw=0)
        xs = np.geomspace(max(d["消费额"].min(), 0.1), d["消费额"].max(), 50)
        ax.plot(xs, row["mean_daily_clicks"] * (xs / row["mean_daily_spend"]) ** row["gamma"], color="#D55E00", lw=1.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{unit}  gamma={row['gamma']:.2f}" + ("" if row["gamma_source"] == "unit" else "*"), fontsize=7.5)
        ax.tick_params(labelsize=6)
    fig.supxlabel("日消费额 / 元（对数）", fontsize=9)
    fig.supylabel("日点击量 / 次（对数）", fontsize=9)
    return plotting.save(fig, ctx.out("figures", "fig_response_curves.pdf"))


def fig_attribution(ctx: StageContext, fit: pd.DataFrame, attribution: dict[str, Any]) -> dict[str, Any]:
    fig, ax = plotting.new_figure(6.3, 2.8)
    fit["date"] = pd.to_datetime(fit["date"])
    ax.plot(fit["date"], fit["regs"], color="#0072B2", lw=1.0, label="实际新注册数")
    ax.plot(fit["date"], fit["fitted"], color="#D55E00", lw=1.2, label="归因模型拟合")
    split = fit["date"].iloc[attribution["holdout"]["train_days"]]
    ax.axvline(split, color="#444444", lw=0.8, ls="--")
    ax.text(split, ax.get_ylim()[1] * 0.95, " 留出期 →", fontsize=7, va="top")
    _shade_holidays(ax)
    ax.set_ylabel("日新注册数 / 人")
    ax.legend(loc="upper left", ncol=2)
    return plotting.save(fig, ctx.out("figures", "fig_attribution.pdf"))


def fig_cost_benefit(ctx: StageContext, table: pd.DataFrame, thr: dict[str, float]) -> dict[str, Any]:
    fig, ax = plotting.new_figure(6.3, 4.0)
    act = table[table["spend"] > 0]
    for label in LABELS[:4]:
        sub = act[act["label"] == label]
        ax.scatter(
            sub["spend"],
            sub["benefit"],
            s=9,
            color=LABEL_COLORS[label],
            alpha=0.65,
            lw=0,
            label=f"{label}（{len(sub)}）",
        )
    ax.axvline(thr["cost_yuan"], color="#444444", lw=0.9, ls="--")
    ax.axhline(thr["benefit"], color="#444444", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("年度消费额 / 元（对数轴；无效词 = 零消费，未画出）")
    ax.set_ylabel("综合效益指数")
    ax.legend(loc="upper left", markerscale=1.8)
    return plotting.save(fig, ctx.out("figures", "fig_cost_benefit.pdf"))


def fig_class_thresholds(
    ctx: StageContext, table: pd.DataFrame, thresholds: dict[str, dict[str, float]]
) -> dict[str, Any]:
    act = table[table["spend"] > 0]
    fig, axes = plotting.new_figure(6.3, 2.8, ncols=2)
    styles = {
        "jenks": ("-", "#0072B2", "自然断点"),
        "median": ("--", "#D55E00", "中位数"),
        "gmm": (":", "#009E73", "高斯混合"),
    }
    axes[0].hist(act["log10_cost"], bins=40, color="#BBBBBB")
    axes[1].hist(act["benefit"], bins=40, color="#BBBBBB")
    for method, (ls, color, name) in styles.items():
        if method in thresholds:
            axes[0].axvline(thresholds[method]["log10_cost"], ls=ls, color=color, label=name)
            axes[1].axvline(thresholds[method]["benefit"], ls=ls, color=color, label=name)
    axes[0].set_xlabel("log10 年度消费额 / 元")
    axes[1].set_xlabel("综合效益指数")
    axes[0].set_ylabel("关键词数")
    axes[1].legend(loc="upper right")
    return plotting.save(fig, ctx.out("figures", "fig_class_thresholds.pdf"))


def fig_q3_gain(ctx: StageContext, summary: pd.DataFrame, boot: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plotting.new_figure(6.3, 3.0, ncols=3, gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    for ax, (window, title) in zip(axes[:2], (("feb", "2025-02-01 至 02-08"), ("aug", "2025-08-01 至 08-08"))):
        d = summary[summary["window"] == window].groupby("date")[["opt_regs", "prop_regs", "equal_regs"]].sum()
        x = np.arange(len(d))
        ax.plot(x, d["opt_regs"], marker="o", ms=3.5, color=STRATEGY_COLORS["最优分配"], label="最优分配")
        prop_color = STRATEGY_COLORS["按历史比例"]
        ax.plot(x, d["prop_regs"], marker="s", ms=3.5, color=prop_color, label="按历史比例")
        ax.plot(x, d["equal_regs"], marker="^", ms=3.5, color=STRATEGY_COLORS["均匀分配"], label="均匀分配")
        ax.set_xticks(x)
        ax.set_xticklabels([s[-2:] for s in d.index], fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("日")
    axes[0].set_ylabel("预期注册量 / 人")
    axes[0].legend(loc="upper left", fontsize=7)
    ax = axes[2]
    for window, color in (("feb", "#0072B2"), ("aug", "#D55E00")):
        b = boot[boot["window"] == window]
        ax.hist(
            100 * (b["ratio"] - 1),
            bins=20,
            color=color,
            alpha=0.6,
            label={"feb": "2 月窗口", "aug": "8 月窗口"}[window],
        )
    ax.set_xlabel("相对历史比例的提升 / %")
    ax.set_ylabel("自助抽样次数")
    ax.legend(loc="upper right", fontsize=7)
    return plotting.save(fig, ctx.out("figures", "fig_q3_gain.pdf"))


def fig_allocation_mix(ctx: StageContext, allocation: pd.DataFrame, params: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plotting.new_figure(6.3, 2.6, ncols=2, sharey=True)
    for ax, window in zip(axes, ("feb", "aug")):
        a = allocation[allocation["window"] == window]
        opt = a.groupby("label")["spend"].sum()
        p = params[params["window"] == window]
        hist = p.groupby("label")["spend"].sum()
        labels = [lab for lab in LABELS[:4]]
        vals_opt = np.array([opt.get(lab, 0.0) for lab in labels])
        vals_hist = np.array([hist.get(lab, 0.0) for lab in labels])
        vals_opt = 100 * vals_opt / vals_opt.sum()
        vals_hist = 100 * vals_hist / vals_hist.sum()
        x = np.arange(len(labels))
        ax.bar(x - 0.2, vals_hist, width=0.38, color="#BBBBBB", label="2025 年实际")
        ax.bar(x + 0.2, vals_opt, width=0.38, color=[LABEL_COLORS[lab] for lab in labels], label="最优分配")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title({"feb": "2 月窗口", "aug": "8 月窗口"}[window], fontsize=8.5)
    axes[0].set_ylabel("消费额占比 / %")
    axes[0].legend(loc="upper right", fontsize=7)
    return plotting.save(fig, ctx.out("figures", "fig_allocation_mix.pdf"))


def fig_backtest(ctx: StageContext, bt: pd.DataFrame) -> dict[str, Any]:
    targets = [
        ("log_eff", "效率乘子"),
        ("log_cpc", "CPC"),
        ("logit_ctr", "CTR"),
        ("logit_top", "上方位占比"),
        ("log_clicks", "点击量"),
    ]
    fig, axes = plotting.new_figure(6.3, 2.7, ncols=2)
    x = np.arange(len(targets))
    for k, model in enumerate(("calendar", "seasonal_naive", "mean28")):
        sub = bt[bt["model"] == model].set_index("target").reindex([t for t, _ in targets])
        axes[0].bar(x + (k - 1) * 0.27, sub["mae"], width=0.26, color=MODEL_COLORS[model], label=MODEL_NAMES[model])
        axes[1].bar(x + (k - 1) * 0.27, 100 * sub["coverage80"], width=0.26, color=MODEL_COLORS[model])
    axes[1].axhline(80, color="#444444", lw=0.9, ls="--")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in targets], fontsize=7)
    axes[0].set_ylabel("滚动回测 MAE（变换尺度）")
    axes[1].set_ylabel("80% 区间覆盖率 / %")
    axes[0].legend(loc="upper left", fontsize=7)
    return plotting.save(fig, ctx.out("figures", "fig_backtest.pdf"))


def fig_q4_plan(ctx: StageContext, days: pd.DataFrame) -> dict[str, Any]:
    d = days.groupby("date").agg(
        clicks=("clicks_mean", "sum"),
        clicks_lo=("clicks_q10", "sum"),
        clicks_hi=("clicks_q90", "sum"),
        regs=("regs_mean", "sum"),
        regs_lo=("regs_q10", "sum"),
        regs_hi=("regs_q90", "sum"),
        spend=("spend", "sum"),
    )
    d["cpc"] = d["spend"] / d["clicks"]
    d["cpc_lo"] = d["spend"] / d["clicks_hi"]
    d["cpc_hi"] = d["spend"] / d["clicks_lo"]
    fig, axes = plotting.new_figure(6.3, 2.8, ncols=3)
    x = np.arange(len(d))
    for ax, (col, ylabel) in zip(
        axes, (("clicks", "预期点击量 / 次"), ("regs", "预期注册量 / 人"), ("cpc", "预期 CPC / 元"))
    ):
        ax.fill_between(x, d[f"{col}_lo"], d[f"{col}_hi"], color="#0072B2", alpha=0.2, lw=0, label="10%–90% 分位")
        ax.plot(x, d[col], marker="o", ms=3.5, color="#0072B2", label="期望")
        ax.set_xticks(x)
        ax.set_xticklabels([s[-2:] for s in d.index], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("2026 年 9 月")
    axes[0].legend(loc="upper right", fontsize=7)
    return plotting.save(fig, ctx.out("figures", "fig_q4_plan.pdf"))


def fig_q4_units(ctx: StageContext, per_unit: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(per_unit).sort_values("regs_mean", ascending=True)
    fig, ax = plotting.new_figure(5.4, 3.2)
    y = np.arange(len(df))
    ax.errorbar(
        df["regs_mean"],
        y,
        xerr=[df["regs_mean"] - df["regs_q10"], df["regs_q90"] - df["regs_mean"]],
        fmt="o",
        ms=4,
        color="#0072B2",
        ecolor="#0072B2",
        elinewidth=1.2,
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([str(u) for u in df["推广单元ID"]], fontsize=7)
    ax.set_xlabel("2026-09-11 至 09-17 预期注册量 / 人（点 = 期望，横线 = 10%–90% 分位）")
    ax.set_xscale("log")
    return plotting.save(fig, ctx.out("figures", "fig_q4_units.pdf"))


def build_figures(ctx: StageContext) -> dict[str, Any]:
    eda_dir, cls_dir, alloc_dir, fc_dir = ctx.dep("eda"), ctx.dep("classify"), ctx.dep("allocate"), ctx.dep("forecast")
    daily = pd.read_parquet(eda_dir / "daily_totals.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    made = [
        fig_daily_series(ctx, daily),
        fig_calendar_effects(ctx, pd.read_parquet(eda_dir / "calendar_regression.parquet")),
        fig_unit_kpis(ctx, pd.read_parquet(eda_dir / "unit_kpis.parquet")),
        fig_lorenz(
            ctx,
            pd.read_parquet(eda_dir / "lorenz.parquet"),
            json.loads((eda_dir / "concentration.json").read_text(encoding="utf-8")),
        ),
        fig_response_curves(
            ctx, pd.read_parquet(eda_dir / "unit_daily.parquet"), pd.read_parquet(eda_dir / "unit_response.parquet")
        ),
        fig_attribution(
            ctx,
            pd.read_parquet(eda_dir / "attribution_fit.parquet"),
            json.loads((eda_dir / "attribution.json").read_text(encoding="utf-8")),
        ),
    ]
    summary = json.loads((cls_dir / "classification_summary.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(cls_dir / "classification.parquet")
    made += [
        fig_cost_benefit(ctx, table, summary["thresholds"][summary["main"]]),
        fig_class_thresholds(ctx, table, summary["thresholds"]),
    ]
    made += [
        fig_q3_gain(
            ctx,
            pd.read_parquet(alloc_dir / "allocation_summary.parquet"),
            pd.read_parquet(alloc_dir / "bootstrap.parquet"),
        ),
        fig_allocation_mix(
            ctx,
            pd.read_parquet(alloc_dir / "allocation.parquet"),
            pd.read_parquet(alloc_dir / "keyword_parameters.parquet"),
        ),
        fig_backtest(ctx, pd.read_parquet(fc_dir / "backtest_summary.parquet")),
        fig_q4_plan(ctx, pd.read_parquet(fc_dir / "plan_by_unit_day.parquet")),
        fig_q4_units(ctx, json.loads((fc_dir / "plan_by_unit.json").read_text(encoding="utf-8"))),
    ]
    ctx.write_json("figures_index.json", made)
    return {"figures": len(made)}
