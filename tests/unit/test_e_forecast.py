from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.e.calendar import day_features
from pipelines.e.forecast import (
    CalendarModel,
    back_transform,
    pinball,
    rolling_backtest,
    seasonal_naive,
    simulate_group,
    summarise,
)


def _series(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-01", periods=n, freq="D")
    f = day_features(dates)
    y = 0.3 * (f["weekday"] >= 5).to_numpy() - 0.5 * f["is_holiday"].to_numpy() + 0.1 * rng.standard_normal(n)
    out = pd.DataFrame({"date": dates, "log_eff": y})
    return pd.concat([out, f.drop(columns=["date"])], axis=1)


def test_calendar_model_recovers_weekend_effect() -> None:
    s = _series()
    m = CalendarModel().fit(s, "log_eff")
    future = day_features(pd.date_range("2026-09-11", "2026-09-17"))
    pred = m.predict(future)
    weekend = future["weekday"].to_numpy() >= 5
    assert pred[weekend].mean() - pred[~weekend].mean() > 0.2
    assert 0.05 < m.sd < 0.2


def test_backtest_prefers_calendar_model_on_calendar_driven_series() -> None:
    s = _series(260)
    origins = list(pd.date_range("2025-08-01", "2025-10-30", freq="7D"))
    bt = rolling_backtest(s, "log_eff", origins)
    agg = bt.groupby("model")["mae"].mean()
    assert agg["calendar"] < agg["mean28"]
    assert 0.5 <= bt[bt["model"] == "calendar"]["coverage80"].mean() <= 1.0


def test_pinball_and_transforms() -> None:
    y = np.array([1.0, 2.0])
    assert pinball(y, np.array([0.0, 3.0]), 0.5) == 0.5 * (1.0 + 1.0) / 2
    assert np.allclose(back_transform(np.log([2.0, 3.0]), "exp"), [2.0, 3.0])
    assert np.allclose(back_transform(np.array([0.0]), "expit"), [0.5])
    s = _series(60)
    assert len(seasonal_naive(s, "log_eff", s.tail(7))) == 7


def test_simulation_shapes_and_monotone_ranges() -> None:
    rng = np.random.default_rng(3)
    params = pd.DataFrame({"c": [10.0, 5.0], "s": [20.0, 8.0], "rho": [0.02, 0.05], "depth": [1.5, 2.0]})
    sim = simulate_group(
        np.array([20.0, 8.0]),
        params,
        0.8,
        0.0,
        0.2,
        0.2,
        0.0,
        0.1,
        {"a": 0.0, "b": 0.3},
        {"a": -1.0, "b": 0.3},
        0.2,
        rng,
        n_scen=400,
    )
    assert sim["clicks"].shape == (400, 2)
    st = summarise(sim["clicks"])
    assert (st["q10"] <= st["q50"]).all() and (st["q50"] <= st["q90"]).all()
    assert abs(st["mean"][0] / 10.0 - 1) < 0.15
    assert ((sim["position"] >= 1.0) & (sim["position"] <= 4.0)).all()
