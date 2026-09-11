"""Keyword response model (concave power law calibrated per unit), position model and keyword parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

GAMMA_BOUNDS = (0.3, 0.95)
MIN_DAYS_FOR_UNIT_GAMMA = 60
POSITION_NON_TOP = 4.0  # ranking index assigned to impressions outside the top region (positions > 3)
POSITION_TOP_OTHER = 2.5  # mean ranking of top-region impressions that are not in first place


def fit_unit_gamma(df: pd.DataFrame) -> dict[str, Any]:
    """log clicks = const + gamma * log spend + weekday dummies, on days with positive spend and clicks."""
    d = df[(df["消费额"] > 0) & (df["点击量"] > 0)]
    n = len(d)
    if n < 10:
        return {"n": n, "gamma": np.nan, "se": np.nan, "r2": np.nan}
    X = pd.DataFrame({"const": 1.0, "log_spend": np.log(d["消费额"].to_numpy(dtype=float))}, index=d.index)
    for wd in range(1, 7):
        X[f"wd{wd}"] = (d["weekday"] == wd).astype(float)
    fit = sm.OLS(np.log(d["点击量"].to_numpy(dtype=float)), X).fit(cov_type="HC3")
    return {
        "n": int(n),
        "gamma": float(fit.params["log_spend"]),
        "se": float(fit.bse["log_spend"]),
        "r2": float(fit.rsquared),
    }


def pooled_gamma(ud: pd.DataFrame) -> dict[str, Any]:
    """Same regression pooled over units with unit fixed effects."""
    d = ud[(ud["消费额"] > 0) & (ud["点击量"] > 0)]
    X = pd.DataFrame({"const": 1.0, "log_spend": np.log(d["消费额"].to_numpy(dtype=float))}, index=d.index)
    units = sorted(d["推广单元ID"].unique())
    for u in units[1:]:
        X[f"u{u}"] = (d["推广单元ID"] == u).astype(float)
    for wd in range(1, 7):
        X[f"wd{wd}"] = (d["weekday"] == wd).astype(float)
    fit = sm.OLS(np.log(d["点击量"].to_numpy(dtype=float)), X).fit(cov_type="HC3")
    return {
        "n": len(d),
        "gamma": float(fit.params["log_spend"]),
        "se": float(fit.bse["log_spend"]),
        "r2": float(fit.rsquared),
    }


def select_gamma(est: dict[str, Any], pooled: float) -> tuple[float, str]:
    """Unit estimate when well identified and inside the concave range, else the pooled elasticity."""
    g = est.get("gamma", np.nan)
    if est.get("n", 0) >= MIN_DAYS_FOR_UNIT_GAMMA and np.isfinite(g) and GAMMA_BOUNDS[0] <= g <= GAMMA_BOUNDS[1]:
        return float(g), "unit"
    return float(np.clip(pooled, *GAMMA_BOUNDS)), "pooled"


def unit_response_table(ud: pd.DataFrame) -> pd.DataFrame:
    """Per-unit elasticity table (estimate, SE, n, R2, selected gamma and its source)."""
    pooled = pooled_gamma(ud)
    rows = []
    for (plan, unit), df in ud.groupby(["方案ID", "推广单元ID"]):
        est = fit_unit_gamma(df)
        gamma, source = select_gamma(est, pooled["gamma"])
        active = df[df["消费额"] > 0]
        rows.append(
            {
                "方案ID": int(plan),
                "推广单元ID": int(unit),
                "n_days": est["n"],
                "gamma_hat": est["gamma"],
                "gamma_se": est["se"],
                "r2": est["r2"],
                "gamma": gamma,
                "gamma_source": source,
                "active_days": len(active),
                "mean_daily_spend": float(active["消费额"].mean()) if len(active) else 0.0,
                "mean_daily_clicks": float(active["点击量"].mean()) if len(active) else 0.0,
                "p95_spend_ratio": float(active["消费额"].quantile(0.95) / max(active["消费额"].mean(), 1e-9))
                if len(active)
                else 1.0,
                "pooled_gamma": pooled["gamma"],
                "pooled_se": pooled["se"],
            }
        )
    return pd.DataFrame(rows)


def clicks_response(x: np.ndarray, c: np.ndarray, s: np.ndarray, gamma: float) -> np.ndarray:
    """Expected clicks at spend ``x`` for keywords with mean daily spend ``s`` and clicks ``c``: c (x/s)^gamma."""
    x = np.asarray(x, dtype=float)
    return c * np.power(np.maximum(x, 0.0) / s, gamma)


def cpc_response(x: np.ndarray, c: np.ndarray, s: np.ndarray, gamma: float) -> np.ndarray:
    """Average CPC at spend ``x`` (rises with spend when gamma < 1)."""
    clicks = clicks_response(x, c, s, gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(clicks > 0, np.asarray(x, dtype=float) / clicks, np.nan)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def fit_position_model(df: pd.DataFrame, share_col: str, min_imp: int = 30) -> dict[str, Any]:
    """logit(share) = a + b log CPC on unit-days (impression-weighted); slope forced non-negative."""
    d = df[(df["展现量"] >= min_imp) & (df["点击量"] > 0) & (df["消费额"] > 0)]
    mean_share = float((df[share_col] * df["展现量"]).sum() / max(df["展现量"].sum(), 1.0)) if len(df) else 0.0
    if len(d) < 20:
        return {
            "a": float(_logit(np.array([mean_share]))[0]),
            "b": 0.0,
            "n": len(d),
            "r2": np.nan,
            "mean_share": mean_share,
            "source": "constant",
        }
    y = _logit(d[share_col].to_numpy(dtype=float))
    X = sm.add_constant(np.log(d["cpc"].to_numpy(dtype=float)))
    fit = sm.WLS(y, X, weights=np.sqrt(d["展现量"].to_numpy(dtype=float))).fit()
    a, b = float(fit.params[0]), float(fit.params[1])
    if b < 0:  # a negative slope has no causal reading (bid up -> lower position); use the constant model
        return {
            "a": float(_logit(np.array([mean_share]))[0]),
            "b": 0.0,
            "n": len(d),
            "r2": float(fit.rsquared),
            "mean_share": mean_share,
            "source": "constant",
        }
    return {"a": a, "b": b, "n": len(d), "r2": float(fit.rsquared), "mean_share": mean_share, "source": "logit"}


def position_shares(
    cpc: np.ndarray, model_top: dict[str, Any], model_first: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Expected top-region and first-place impression shares at a given CPC."""
    lc = np.log(np.maximum(np.asarray(cpc, dtype=float), 1e-6))
    p_top = 1 / (1 + np.exp(-(model_top["a"] + model_top["b"] * lc)))
    p_first = 1 / (1 + np.exp(-(model_first["a"] + model_first["b"] * lc)))
    p_first = np.minimum(p_first, p_top)
    return p_top, p_first


def expected_position(p_top: np.ndarray, p_first: np.ndarray) -> np.ndarray:
    """Expected ranking index: 1 for first place, 2.5 for other top-region slots, 4 outside the top region."""
    return 1.0 * p_first + POSITION_TOP_OTHER * (p_top - p_first) + POSITION_NON_TOP * (1.0 - p_top)


def keyword_parameters(
    kw: pd.DataFrame,
    units: pd.DataFrame,
    rates: dict[int, float],
    kappa_bounds: tuple[float, float] = (0.25, 2.5),
) -> pd.DataFrame:
    """Per-keyword response parameters for active keywords (spend > 0).

    s_i, c_i: mean daily spend / clicks over the unit's active days; depth_i views per click (unit
    fallback); kappa_i bounce-quality multiplier normalised to the unit's click-weighted mean bounce;
    rho_i = r_u * kappa_i expected registrations per click.
    """
    u = units.set_index("推广单元ID")
    active = kw[kw["spend"] > 0].copy()
    days = active["推广单元ID"].map(u["active_days"]).astype(float)
    active["s"] = active["spend"] / days
    active["c"] = active["clicks"] / days
    active["gamma"] = active["推广单元ID"].map(u["gamma"]).astype(float)
    # keywords with clicks but no recorded page views (bounce reported as 0) are tracking failures:
    # both engagement fields are treated as missing and fall back to the unit's click-weighted means.
    untracked = (active["views"] <= 0) & (active["clicks"] > 0)
    active.loc[untracked, "depth"] = np.nan
    active.loc[untracked, "bounce"] = np.nan
    tracked = active[~untracked]
    unit_depth = tracked.groupby("推广单元ID").apply(
        lambda d: d["views"].sum() / max(d["clicks"].sum(), 1), include_groups=False
    )
    active["depth"] = active["depth"].fillna(active["推广单元ID"].map(unit_depth)).fillna(1.0)
    unit_bounce = tracked.groupby("推广单元ID").apply(
        lambda d: float(np.average(d["bounce"].fillna(1.0), weights=np.maximum(d["clicks"], 1))), include_groups=False
    )
    b_i = active["bounce"].fillna(active["推广单元ID"].map(unit_bounce)).fillna(0.5)
    kappa = (1.0 - b_i) / (1.0 - active["推广单元ID"].map(unit_bounce)).clip(lower=0.02)
    active["kappa"] = kappa.clip(*kappa_bounds).fillna(1.0)
    active["rate"] = active["推广单元ID"].map(rates).astype(float)
    active["rho"] = active["rate"] * active["kappa"]
    active["cpc"] = active["spend"] / active["clicks"]
    return active
