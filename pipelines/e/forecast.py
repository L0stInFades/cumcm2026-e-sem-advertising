"""Problem 4: unit-level daily forecasts with rolling-origin backtests, scenario generation and Monte Carlo ranges."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pipelines.e.calendar import day_features

TARGETS: dict[str, dict[str, Any]] = {
    "log_eff": {"label": "效率乘子 log eff", "transform": "exp"},
    "log_cpc": {"label": "log CPC", "transform": "exp"},
    "logit_ctr": {"label": "logit CTR", "transform": "expit"},
    "logit_top": {"label": "logit 上方位占比", "transform": "expit"},
    "logit_first": {"label": "logit 首位占比", "transform": "expit"},
    "log_clicks": {"label": "log 点击量", "transform": "exp"},
}
Z80 = float(stats.norm.ppf(0.9))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def back_transform(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.exp(values) if transform == "exp" else 1 / (1 + np.exp(-values))


def unit_series(ud_unit: pd.DataFrame, gamma: float, mean_spend: float, mean_clicks: float) -> pd.DataFrame:
    """Daily targets for one unit on days with positive spend and clicks."""
    d = ud_unit[(ud_unit["消费额"] > 0) & (ud_unit["点击量"] > 0) & (ud_unit["展现量"] > 0)].copy()
    d = d.sort_values("日期")
    spend = d["消费额"].to_numpy(dtype=float)
    clicks = d["点击量"].to_numpy(dtype=float)
    out = pd.DataFrame({"date": d["日期"].to_numpy()})
    out["log_eff"] = np.log(clicks) - gamma * np.log(spend / mean_spend) - np.log(mean_clicks)
    out["log_cpc"] = np.log(spend / clicks)
    out["logit_ctr"] = _logit(clicks / d["展现量"].to_numpy(dtype=float))
    out["logit_top"] = _logit(d["top_share"].to_numpy(dtype=float))
    out["logit_first"] = _logit(d["first_share"].to_numpy(dtype=float))
    out["log_clicks"] = np.log(clicks)
    feats = day_features(out["date"]).drop(columns=["date"])
    return pd.concat([out.reset_index(drop=True), feats], axis=1)


def _design(feats: pd.DataFrame, months: list[int]) -> np.ndarray:
    cols = [np.ones(len(feats))]
    for wd in range(7):
        if wd != 2:
            cols.append((feats["weekday"] == wd).to_numpy(dtype=float))
    for m in months:
        cols.append((feats["month"] == m).to_numpy(dtype=float))
    cols.append(feats["is_holiday"].to_numpy(dtype=float))
    cols.append(feats["is_adjusted_workday"].to_numpy(dtype=float))
    return np.column_stack(cols)


class CalendarModel:
    """Ridge-stabilised least squares of a target on weekday, month and holiday indicators."""

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = ridge
        self.months: list[int] = []
        self.beta = np.zeros(0)
        self.sd = float("nan")
        self.month_sd = 0.0

    def fit(self, train: pd.DataFrame, target: str) -> CalendarModel:
        months = sorted(int(m) for m in train["month"].unique())
        self.months = [m for m in months if m != months[0]]
        X = _design(train, self.months)
        y = train[target].to_numpy(dtype=float)
        reg = self.ridge * np.eye(X.shape[1])
        reg[0, 0] = 0.0
        self.beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
        resid = y - X @ self.beta
        dof = max(len(y) - X.shape[1], 1)
        self.sd = float(np.sqrt(np.sum(resid**2) / dof))
        n_wd = 6
        month_effects = self.beta[1 + n_wd : 1 + n_wd + len(self.months)]
        self.month_sd = float(np.std(np.concatenate([[0.0], month_effects]))) if len(month_effects) else 0.0
        return self

    def predict(self, feats: pd.DataFrame, month_known: bool = True) -> np.ndarray:
        f = feats.copy()
        if not month_known:
            f["month"] = -1  # reference-month behaviour when the target month never appeared in training
        return _design(f, self.months) @ self.beta


def seasonal_naive(train: pd.DataFrame, target: str, future: pd.DataFrame) -> np.ndarray:
    """Last observed value on the same weekday (falls back to the last observation)."""
    out = np.empty(len(future))
    last = train[target].to_numpy(dtype=float)[-1]
    by_wd = train.groupby("weekday")[target].last()
    for i, wd in enumerate(future["weekday"].to_numpy()):
        out[i] = float(by_wd.get(int(wd), last))
    return out


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def rolling_backtest(
    series: pd.DataFrame, target: str, origins: list[pd.Timestamp], horizon: int = 7, min_train: int = 40
) -> pd.DataFrame:
    """Rolling-origin evaluation of the calendar model vs. seasonal-naive vs. 28-day mean."""
    rows = []
    info = TARGETS[target]
    for origin in origins:
        train = series[series["date"] < origin]
        test = series[(series["date"] >= origin) & (series["date"] < origin + pd.Timedelta(days=horizon))]
        if len(train) < min_train or len(test) == 0:
            continue
        y = test[target].to_numpy(dtype=float)
        y_raw = back_transform(y, info["transform"])
        preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        model = CalendarModel().fit(train, target)
        known = int(test["month"].iloc[0]) in {*model.months, sorted(train["month"].unique())[0]}
        mu = model.predict(test, month_known=known)
        preds["calendar"] = (mu, mu - Z80 * model.sd, mu + Z80 * model.sd)
        sn = seasonal_naive(train, target, test)
        tr = train[target].to_numpy(dtype=float)
        sn_err = tr[7:] - tr[:-7] if len(tr) > 7 else np.array([0.0])
        preds["seasonal_naive"] = (sn, sn + np.quantile(sn_err, 0.1), sn + np.quantile(sn_err, 0.9))
        m28 = float(tr[-28:].mean())
        dev = tr[-28:] - m28
        preds["mean28"] = (np.full(len(y), m28), m28 + np.quantile(dev, 0.1), m28 + np.quantile(dev, 0.9))
        for name, (p, lo, hi) in preds.items():
            p_raw = back_transform(p, info["transform"])
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "origin": origin,
                    "n": len(y),
                    "mae": float(np.mean(np.abs(y - p))),
                    "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                    "mape_raw": float(np.mean(np.abs(y_raw - p_raw) / np.maximum(np.abs(y_raw), 1e-9))),
                    "pinball10": pinball(y, lo, 0.1),
                    "pinball90": pinball(y, hi, 0.9),
                    "coverage80": float(np.mean((y >= lo) & (y <= hi))),
                }
            )
    return pd.DataFrame(rows)


def forecast_unit(series: pd.DataFrame, target: str, future: pd.DataFrame) -> dict[str, Any]:
    """Point forecast, residual sd and between-month sd for the future days (calendar model on all data)."""
    model = CalendarModel().fit(series, target)
    known = all(int(m) in {*model.months, sorted(series["month"].unique())[0]} for m in future["month"].unique())
    mu = model.predict(future, month_known=known)
    return {"mean": mu, "sd": model.sd, "month_sd": model.month_sd, "month_known": known, "n": len(series)}


def simulate_group(
    x: np.ndarray,
    params: pd.DataFrame,
    gamma: float,
    eff_mean: float,
    eff_sd: float,
    kw_sd: float,
    ctr_mean_logit: float,
    ctr_sd: float,
    pos_top: dict[str, Any],
    pos_first: dict[str, Any],
    top_sd: float,
    rng: np.random.Generator,
    n_scen: int = 500,
) -> dict[str, np.ndarray]:
    """Monte Carlo of clicks, CPC, impressions, position, views and registrations for one unit-day allocation.

    Common day shock (log-normal efficiency), keyword idiosyncratic log-normal noise, CTR and position
    day shocks, Poisson registrations.  Returns arrays of shape (n_scen, n_keywords).
    """
    from pipelines.e.response import expected_position, position_shares

    n = len(x)
    c = params["c"].to_numpy(dtype=float)
    s = params["s"].to_numpy(dtype=float)
    rho = params["rho"].to_numpy(dtype=float)
    depth = params["depth"].to_numpy(dtype=float)
    base = c * np.power(np.maximum(x, 0.0) / s, gamma)
    day = np.exp(eff_mean + eff_sd * rng.standard_normal((n_scen, 1)) - 0.5 * eff_sd**2)
    idio = np.exp(kw_sd * rng.standard_normal((n_scen, n)) - 0.5 * kw_sd**2)
    clicks = base[None, :] * day * idio
    with np.errstate(divide="ignore", invalid="ignore"):
        cpc = np.where(clicks > 0, x[None, :] / clicks, np.nan)
    ctr = 1 / (1 + np.exp(-(ctr_mean_logit + ctr_sd * rng.standard_normal((n_scen, 1)))))
    imp = clicks / ctr
    p_top, p_first = position_shares(np.nan_to_num(cpc, nan=1.0), pos_top, pos_first)
    shock = top_sd * rng.standard_normal((n_scen, 1))
    p_top = 1 / (1 + np.exp(-(np.log(p_top / (1 - p_top)) + shock)))
    p_first = np.minimum(p_first, p_top)
    position = expected_position(p_top, p_first)
    views = clicks * depth[None, :]
    regs = rng.poisson(np.maximum(rho[None, :] * clicks, 0.0)).astype(float)
    return {
        "clicks": clicks,
        "cpc": cpc,
        "imp": imp,
        "position": position,
        "views": views,
        "regs": regs,
        "p_top": p_top,
    }


def summarise(samples: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(samples, axis=axis),
        "q10": np.nanquantile(samples, 0.1, axis=axis),
        "q50": np.nanquantile(samples, 0.5, axis=axis),
        "q90": np.nanquantile(samples, 0.9, axis=axis),
    }
