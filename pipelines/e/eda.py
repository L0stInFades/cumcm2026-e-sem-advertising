"""Problem 1 analytics: unit KPIs, keyword concentration, calendar/holiday regressions, registration attribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import lsq_linear

from pipelines.e.calendar import WEEKDAY_NAMES

WEEKDAY_BASE = 2  # Wednesday is the reference weekday in every regression
MONTH_BASE = 1  # January is the reference month


def unit_kpis(s1: pd.DataFrame) -> pd.DataFrame:
    """Annual funnel KPIs per (plan, unit): CTR, CPC, position shares, top vs. other CPC."""
    g = s1.groupby(["方案ID", "推广单元ID"])
    k = pd.DataFrame(
        {
            "days": g.size(),
            "first_day": g["日期"].min(),
            "last_day": g["日期"].max(),
            "imp": g["展现量"].sum(),
            "clicks": g["点击量"].sum(),
            "spend": g["消费额"].sum(),
            "top_imp": g["上方位展现量"].sum(),
            "first_imp": g["上方首位展现量"].sum(),
            "top_clicks": g["上方位点击量"].sum(),
            "top_spend": g["上方位消费额"].sum(),
        }
    ).reset_index()
    imp = k["imp"].replace(0, np.nan)
    clicks = k["clicks"].replace(0, np.nan)
    k["ctr"] = k["clicks"] / imp
    k["cpc"] = k["spend"] / clicks
    k["cpm"] = 1000.0 * k["spend"] / imp
    k["top_share"] = k["top_imp"] / imp
    k["first_share"] = k["first_imp"] / imp
    k["top_click_share"] = k["top_clicks"] / clicks
    k["top_cpc"] = k["top_spend"] / k["top_clicks"].replace(0, np.nan)
    k["other_cpc"] = (k["spend"] - k["top_spend"]) / (k["clicks"] - k["top_clicks"]).replace(0, np.nan)
    k["top_ctr"] = k["top_clicks"] / k["top_imp"].replace(0, np.nan)
    k["other_ctr"] = (k["clicks"] - k["top_clicks"]) / (k["imp"] - k["top_imp"]).replace(0, np.nan)
    k["spend_share"] = k["spend"] / k["spend"].sum()
    return k


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = perfectly even)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def keyword_concentration(kw: pd.DataFrame) -> dict[str, Any]:
    """Long-tail statistics of keyword spend and clicks (global and per unit)."""
    spend = kw["spend"].to_numpy(dtype=float)
    order = np.argsort(-spend)
    cum = np.cumsum(spend[order]) / spend.sum()
    n = len(spend)
    lorenz = pd.DataFrame(
        {
            "keyword_share": np.arange(1, n + 1) / n,
            "spend_share": cum,
            "click_share": np.cumsum(kw["clicks"].to_numpy()[order]) / kw["clicks"].sum(),
        }
    )
    per_unit = (
        kw.groupby(["方案ID", "推广单元ID"])
        .apply(
            lambda d: pd.Series(
                {
                    "keywords": len(d),
                    "active": int((d["spend"] > 0).sum()),
                    "zero": int((d["spend"] == 0).sum()),
                    "spend": d["spend"].sum(),
                    "clicks": d["clicks"].sum(),
                    "gini_spend": gini(d["spend"].to_numpy()),
                    "top5_share": float(np.sort(d["spend"].to_numpy())[::-1][:5].sum() / max(d["spend"].sum(), 1e-9)),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return {
        "keywords": int(n),
        "unique_keyword_ids": int(kw["关键词"].nunique()),
        "shared_keyword_ids": int((kw.groupby("关键词")["推广单元ID"].nunique() > 1).sum()),
        "zero_cost_zero_benefit": int(((kw["spend"] == 0) & (kw["clicks"] == 0)).sum()),
        "active_keywords": int((kw["spend"] > 0).sum()),
        "gini_spend_all": gini(spend),
        "gini_spend_active": gini(spend[spend > 0]),
        "top10_spend_share": float(cum[min(9, n - 1)]),
        "top50_spend_share": float(cum[min(49, n - 1)]),
        "top100_spend_share": float(cum[min(99, n - 1)]),
        "top20pct_spend_share": float(cum[max(int(0.2 * n) - 1, 0)]),
        "lorenz": lorenz,
        "per_unit": per_unit,
    }


def _design(daily: pd.DataFrame) -> pd.DataFrame:
    """Regression design: constant, weekday and month dummies, holiday and 调休 indicators."""
    cols: dict[str, Any] = {"const": 1.0}
    for wd in range(7):
        if wd != WEEKDAY_BASE:
            cols[f"wd_{WEEKDAY_NAMES[wd]}"] = (daily["weekday"] == wd).astype(float)
    for m in range(1, 13):
        if m != MONTH_BASE and (daily["month"] == m).any():
            cols[f"month_{m:02d}"] = (daily["month"] == m).astype(float)
    cols["holiday"] = daily["is_holiday"].astype(float)
    cols["adjusted_workday"] = daily["is_adjusted_workday"].astype(float)
    cols["pre_holiday"] = daily["is_pre_holiday"].astype(float)
    cols["post_holiday"] = daily["is_post_holiday"].astype(float)
    return pd.DataFrame(cols, index=daily.index)


def calendar_regression(
    daily: pd.DataFrame, target: str, log: bool, controls: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """OLS of ``target`` (optionally log) on calendar dummies (+ log controls) with HC3 robust standard errors."""
    y = daily[target].astype(float)
    mask = y.notna() & (y > 0 if log else True)
    for ctrl in controls:
        mask &= daily[ctrl].astype(float) > 0
    y = np.log(y[mask]) if log else y[mask]
    X = _design(daily.loc[mask])
    for ctrl in controls:
        X[f"log_{ctrl}"] = np.log(daily.loc[mask, ctrl].astype(float))
    name = f"{target}|{'+'.join(controls)}" if controls else target
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    ci = fit.conf_int(alpha=0.05)
    rows = []
    for term in X.columns:
        est = float(fit.params[term])
        rows.append(
            {
                "target": name,
                "term": term,
                "estimate": est,
                "se": float(fit.bse[term]),
                "t": float(fit.tvalues[term]),
                "p": float(fit.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "effect_pct": 100.0 * (np.exp(est) - 1.0) if log else np.nan,
                "effect_pct_low": 100.0 * (np.exp(ci.loc[term, 0]) - 1.0) if log else np.nan,
                "effect_pct_high": 100.0 * (np.exp(ci.loc[term, 1]) - 1.0) if log else np.nan,
            }
        )
    wd_terms = [c for c in X.columns if c.startswith("wd_")]
    wald = fit.wald_test(np.eye(len(X.columns))[[list(X.columns).index(c) for c in wd_terms]], scalar=True)
    summary = {
        "target": name,
        "log": log,
        "n": int(mask.sum()),
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "weekday_wald_stat": float(wald.statistic),
        "weekday_wald_p": float(wald.pvalue),
        "residual_sd": float(np.sqrt(fit.scale)),
    }
    return pd.DataFrame(rows), summary


def holiday_nonparametric(daily: pd.DataFrame, target: str) -> dict[str, Any]:
    """Mann-Whitney U test of holiday days vs. ordinary workdays for ``target``."""
    hol = daily.loc[daily["is_holiday"], target].dropna()
    work = daily.loc[daily["day_type"] == "workday", target].dropna()
    u = stats.mannwhitneyu(hol, work, alternative="two-sided")
    return {
        "target": target,
        "holiday_n": len(hol),
        "workday_n": len(work),
        "holiday_median": float(hol.median()),
        "workday_median": float(work.median()),
        "ratio_of_medians": float(hol.median() / work.median()) if work.median() else np.nan,
        "u_stat": float(u.statistic),
        "p": float(u.pvalue),
    }


def day_type_profile(daily: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Mean of ``columns`` by weekday, by month and by day type (long format)."""
    frames = []
    for key in ("weekday_name", "month", "day_type"):
        g = daily.groupby(key)[list(columns)].mean()
        g["n_days"] = daily.groupby(key).size()
        g = g.reset_index().rename(columns={key: "level"})
        g.insert(0, "dimension", key)
        g["level"] = g["level"].astype(str)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def attribution_model(
    clicks: pd.DataFrame,
    daily: pd.DataFrame,
    lambdas: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6),
    min_clicks: float = 10_000.0,
    band: tuple[float, ...] = (0.25, 4.0),
) -> dict[str, Any]:
    """Registrations = baseline + sum_u r_u * adstocked clicks_u + calendar effects (r_u >= 0).

    ``clicks``: dates x units matrix.  The adstock decay ``lambda`` is selected by rolling-origin
    validation MAE.  A unit keeps its own rate only when it has enough clicks to identify it and the
    estimate lies within ``band`` times the pooled rate; otherwise the pooled rate is used.
    """
    y = daily["regs"].to_numpy(dtype=float)
    cal = _design(daily).drop(columns=["const"])
    units = [int(u) for u in clicks.columns]
    n = len(y)
    origins = [int(n * f) for f in (0.5, 0.6, 0.7, 0.8)]

    def adstock(mat: np.ndarray, lam: float) -> np.ndarray:
        out = mat.copy()
        for t in range(1, len(out)):
            out[t] += lam * out[t - 1]
        return out

    def fit(Xu: np.ndarray, Xc: np.ndarray, yy: np.ndarray) -> np.ndarray:
        X = np.column_stack([np.ones(len(yy)), Xu, Xc])
        lo = np.concatenate([[0.0], np.zeros(Xu.shape[1]), np.full(Xc.shape[1], -np.inf)])
        hi = np.full(X.shape[1], np.inf)
        return lsq_linear(X, yy, bounds=(lo, hi), lsmr_tol="auto").x

    def predict(beta: np.ndarray, Xu: np.ndarray, Xc: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(Xu)), Xu, Xc]) @ beta

    cv_rows = []
    best_lambda, best_mae = lambdas[0], np.inf
    raw = clicks.to_numpy(dtype=float)
    calm = cal.to_numpy(dtype=float)
    for lam in lambdas:
        Xu = adstock(raw, lam)
        errs = []
        for o in origins:
            beta = fit(Xu[:o], calm[:o], y[:o])
            pred = predict(beta, Xu[o:], calm[o:])
            errs.append(np.abs(pred - y[o:]).mean())
        mae = float(np.mean(errs))
        cv_rows.append({"lambda": lam, "cv_mae": mae})
        if mae < best_mae:
            best_lambda, best_mae = lam, mae
    Xu = adstock(raw, best_lambda)
    beta = fit(Xu, calm, y)
    pred = predict(beta, Xu, calm)
    resid = y - pred
    r_units = {u: float(beta[1 + i]) for i, u in enumerate(units)}
    # pooled rate: same regression with a single aggregate clicks regressor
    pooled_beta = fit(Xu.sum(axis=1, keepdims=True), calm, y)
    pooled_rate = float(pooled_beta[1])
    totals = clicks.sum(axis=0)
    rates: dict[int, float] = {}
    source: dict[int, str] = {}
    for u in units:
        if totals[u] >= min_clicks and band[0] * pooled_rate <= r_units[u] <= band[1] * pooled_rate:
            rates[u], source[u] = r_units[u], "unit"
        else:
            rates[u], source[u] = pooled_rate, "pooled"
    # holdout diagnostics for the selected lambda (train first 70 %, test last 30 %)
    o = int(n * 0.7)
    beta_h = fit(Xu[:o], calm[:o], y[:o])
    pred_h = predict(beta_h, Xu[o:], calm[o:])
    holdout = {
        "train_days": o,
        "test_days": n - o,
        "mae": float(np.abs(pred_h - y[o:]).mean()),
        "mape": float(np.mean(np.abs(pred_h - y[o:]) / np.maximum(y[o:], 1.0))),
        "r2": float(1 - np.sum((y[o:] - pred_h) ** 2) / np.sum((y[o:] - y[o:].mean()) ** 2)),
    }
    return {
        "lambda": float(best_lambda),
        "cv": cv_rows,
        "baseline_per_day": float(beta[0]),
        "unit_rates": r_units,
        "pooled_rate": pooled_rate,
        "rates": rates,
        "rate_source": source,
        "calendar_terms": {c: float(beta[1 + len(units) + j]) for j, c in enumerate(cal.columns)},
        "r2": float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)),
        "mae": float(np.abs(resid).mean()),
        "holdout": holdout,
        "fitted": pd.DataFrame({"date": daily.index, "regs": y, "fitted": pred}),
        # attributed with the rates actually used downstream (unit rate or pooled fallback); the raw
        # NNLS rates are kept for transparency (they are zero for units clipped at the bound)
        "attributed_regs": {u: float((Xu[:, i] * rates[u]).sum()) for i, u in enumerate(units)},
        "attributed_regs_raw": {u: float((Xu[:, i] * r_units[u]).sum()) for i, u in enumerate(units)},
        "baseline_regs": float(beta[0] * n),
    }


def holiday_optimal_cut(
    daily: pd.DataFrame,
    terms: tuple[str, ...] = ("holiday", "wd_周一", "wd_周日"),
    alt_elasticities: tuple[float, ...] = (),
    n_draws: int = 400_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Sampling uncertainty of the marginal-return-equalising spend ratio exp(delta'/(1-theta)).

    The optimal spend ratio on a day type with demand-side effect ``delta'`` relative to an ordinary
    day follows from equalising the marginal registration return of the log-log response
    R = K S^theta.  Both delta' and theta come from the same regression, so their joint sampling
    distribution is propagated by drawing from the HC3 covariance of the two coefficients; the
    critical elasticity ``theta_star`` at which the optimal cut equals the observed cut is reported
    so that the reader can see whether the data can separate "over-cut" from "under-cut".
    """
    y = daily["regs"].astype(float)
    mask = y.notna() & (y > 0) & (daily["spend"].astype(float) > 0)
    X = _design(daily.loc[mask])
    X["log_spend"] = np.log(daily.loc[mask, "spend"].astype(float))
    fit = sm.OLS(np.log(y[mask]), X).fit(cov_type="HC3")
    cols = list(X.columns)
    j_theta = cols.index("log_spend")
    cov = np.asarray(fit.cov_params())
    rng = np.random.default_rng(seed)
    theta = float(fit.params["log_spend"])
    theta_se = float(fit.bse["log_spend"])
    theta_ci = [float(v) for v in fit.conf_int(alpha=0.05).loc["log_spend"]]
    # observed spend cut per day type (same design, spend as the dependent variable)
    spend_fit = sm.OLS(np.log(daily.loc[mask, "spend"].astype(float)), _design(daily.loc[mask])).fit(cov_type="HC3")
    out: dict[str, Any] = {
        "theta": theta,
        "theta_se": theta_se,
        "theta_ci": theta_ci,
        "n": int(mask.sum()),
        "terms": {},
    }

    def cut(delta: np.ndarray | float, th: np.ndarray | float) -> np.ndarray:
        return 1.0 - np.exp(np.asarray(delta, dtype=float) / np.maximum(1.0 - np.asarray(th, dtype=float), 1e-6))

    for term in terms:
        if term not in cols:
            continue
        j = cols.index(term)
        delta = float(fit.params[term])
        sub = cov[np.ix_([j, j_theta], [j, j_theta])]
        draws = rng.multivariate_normal([delta, theta], sub, size=n_draws)
        keep = draws[:, 1] < 0.999  # the ratio is undefined at unit elasticity
        cuts = cut(draws[keep, 0], draws[keep, 1])
        actual = float(1.0 - np.exp(float(spend_fit.params[term])))
        # elasticity at which the optimal cut equals the observed cut: delta / log(1 - actual) = 1 - theta
        theta_star = float(1.0 - delta / np.log(max(1.0 - actual, 1e-9))) if actual > 0 else np.nan
        out["terms"][term] = {
            "delta": delta,
            "delta_se": float(fit.bse[term]),
            "delta_ci": [float(v) for v in fit.conf_int(alpha=0.05).loc[term]],
            "delta_p": float(fit.pvalues[term]),
            "cut": float(cut(delta, theta)),
            "cut_ci": [float(np.quantile(cuts, 0.025)), float(np.quantile(cuts, 0.975))],
            "cut_at_theta_lo": float(cut(delta, theta_ci[0])),
            "cut_at_theta_hi": float(cut(delta, theta_ci[1])),
            "actual_cut": actual,
            "actual_cut_p": float(spend_fit.pvalues[term]),
            "theta_star": theta_star,
            "theta_star_inside_ci": bool(theta_ci[0] <= theta_star <= theta_ci[1])
            if np.isfinite(theta_star)
            else False,
            "prob_optimal_deeper_than_actual": float(np.mean(cuts > actual)),
            "cut_at_alt": {f"{g:.4f}": float(cut(delta, g)) for g in alt_elasticities},
        }
    return out


def attribution_diagnostics(
    clicks: pd.DataFrame,
    daily: pd.DataFrame,
    attribution: dict[str, Any],
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap intervals, boundary flags, collinearity and an additive decomposition of registrations.

    The unit registration rates enter every downstream objective, so their sampling uncertainty and
    their dependence on the calendar controls must be reported: a moving-block bootstrap of the
    residuals gives intervals, refitting without the month dummies gives a specification swing, and
    the residualised design matrix gives the condition number of the click block.
    """
    y = daily["regs"].to_numpy(dtype=float)
    cal = _design(daily).drop(columns=["const"])
    units = [int(u) for u in clicks.columns]
    lam = float(attribution["lambda"])
    raw = clicks.to_numpy(dtype=float)
    Xu = raw.copy()
    for t in range(1, len(Xu)):
        Xu[t] += lam * Xu[t - 1]
    calm = cal.to_numpy(dtype=float)

    def fit(Xc: np.ndarray, yy: np.ndarray, xu: np.ndarray) -> np.ndarray:
        X = np.column_stack([np.ones(len(yy)), xu, Xc])
        lo = np.concatenate([[0.0], np.zeros(xu.shape[1]), np.full(Xc.shape[1], -np.inf)])
        return lsq_linear(X, yy, bounds=(lo, np.full(X.shape[1], np.inf)), lsmr_tol="auto").x

    beta = fit(calm, y, Xu)
    fitted = np.column_stack([np.ones(len(y)), Xu, calm]) @ beta
    resid = y - fitted
    # moving-block bootstrap (block length 7) preserves the weekly dependence of the residuals
    rng = np.random.default_rng(seed)
    block, n = 7, len(y)
    n_blocks = int(np.ceil(n / block))
    draws = np.zeros((n_boot, len(units)))
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        e = np.concatenate([resid[s : s + block] for s in starts])[:n]
        draws[b] = fit(calm, fitted + e, Xu)[1 : 1 + len(units)]
    # specification robustness: drop the month dummies
    month_cols = [i for i, c in enumerate(cal.columns) if c.startswith("month_")]
    keep = [i for i in range(calm.shape[1]) if i not in month_cols]
    beta_nm = fit(calm[:, keep], y, Xu)
    # collinearity of the click block after residualising on the calendar design
    Z = np.column_stack([np.ones(n), calm])
    resid_clicks = Xu - Z @ np.linalg.lstsq(Z, Xu, rcond=None)[0]
    sv = np.linalg.svd(resid_clicks, compute_uv=False)
    rows = []
    for i, u in enumerate(units):
        r_raw = float(beta[1 + i])
        rows.append(
            {
                "推广单元ID": u,
                "rate_raw": r_raw,
                "rate_used": float(attribution["rates"][u]),
                "rate_source": attribution["rate_source"][u],
                "at_zero_bound": bool(r_raw <= 1e-9),
                "boot_low": float(np.quantile(draws[:, i], 0.025)),
                "boot_high": float(np.quantile(draws[:, i], 0.975)),
                "boot_sd": float(np.std(draws[:, i])),
                "boot_share_at_zero": float(np.mean(draws[:, i] <= 1e-9)),
                "rate_no_month": float(beta_nm[1 + i]),
                "swing_ratio": float(beta_nm[1 + i] / r_raw) if r_raw > 1e-9 else np.nan,
                "total_clicks": float(clicks.iloc[:, i].sum()),
            }
        )
    table = pd.DataFrame(rows)
    cal_terms = beta[1 + len(units) :]
    decomposition = {
        "total_regs": float(y.sum()),
        "intercept": float(beta[0] * n),
        "sem": float(sum(Xu[:, i].sum() * beta[1 + i] for i in range(len(units)))),
        "calendar": float(sum(calm[:, j].sum() * cal_terms[j] for j in range(calm.shape[1]))),
        "residual": float(resid.sum()),
    }
    decomposition["implied_non_sem_per_day"] = (decomposition["intercept"] + decomposition["calendar"]) / n
    return {
        "table": table,
        "n_boot": n_boot,
        "n_at_zero_bound": int(table["at_zero_bound"].sum()),
        "max_swing_ratio": float(np.nanmax(np.abs(np.log(table["swing_ratio"].to_numpy(dtype=float))))),
        "condition_number": float(sv[0] / max(sv[-1], 1e-12)),
        "min_singular_value": float(sv[-1]),
        "decomposition": decomposition,
    }
