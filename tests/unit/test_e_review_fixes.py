"""Regression tests for the defects the v1.0.0 review surfaced.

Each test pins a property that was violated (or unverified) before the revision:
relaxations must not lower the optimum, the joint relaxation must bound the two-level
solution, cross-unit water filling must equalise marginal returns and exhaust the budget,
and the holiday optimal-cut interval must bracket its own point estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.e.allocate import solve_group, water_filling
from pipelines.e.stages import _cross_unit_allocation, _weights


def _instance(n: int = 40, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(0.0, 1.6, n))
    w = np.exp(rng.normal(0.0, 1.0, n)) * s
    caps = np.maximum(4.0 * s, 0.0)
    return w, s, caps, float(3.0 * s.sum())


@pytest.mark.parametrize("gamma", [0.5, 0.7, 0.9])
def test_dropping_the_floor_with_fixed_caps_cannot_lower_the_optimum(gamma: float) -> None:
    """The v1.0.0 sensitivity coupled the floor to the caps, so m = 0 shrank the feasible set."""
    w, s, caps, budget = _instance()
    m = 0.5
    caps_eff = np.maximum(caps, m)  # the effective box of the floored problem
    base = solve_group(w, s, gamma, caps, budget, min_spend=m)
    relaxed = solve_group(w, s, gamma, caps_eff, budget, min_spend=0.0)
    coupled = solve_group(w, s, gamma, caps, budget, min_spend=0.0)  # the buggy comparison
    assert relaxed["objective"] >= base["objective"] * (1 - 1e-9)
    assert relaxed["n_selected"] >= base["n_selected"]
    # and the coupled variant is exactly the trap: a nominally weaker problem that scores lower
    assert coupled["objective"] <= relaxed["objective"] * (1 + 1e-9)


def test_joint_relaxation_bounds_the_two_level_solution() -> None:
    """Problem 4's certificate: a 7-day x keyword relaxation with the widest box and no floor."""
    gamma, m, n_days = 0.8, 0.5, 7
    w, s, caps, budget = _instance(seed=11)
    eff = np.array([1.0, 0.9, 1.2, 0.7, 1.1, 0.6, 1.3])
    day_caps = np.full(n_days, budget)
    X = solve_group(eff, np.ones(n_days), gamma, day_caps, budget, min_spend=m)["x"]
    two_level = 0.0
    for i in range(n_days):
        if X[i] <= 0:
            continue
        res = solve_group(w * eff[i], s, gamma, caps, float(X[i]), min_spend=m)
        two_level += res["objective"]
    a_joint = np.concatenate([eff[i] * w * np.power(s, -gamma) for i in range(n_days)])
    caps_joint = np.tile(np.maximum(caps, m), n_days)
    x_joint = water_filling(a_joint, gamma, caps_joint, budget)
    joint_bound = float(np.sum(a_joint * np.power(x_joint, gamma)))
    assert joint_bound >= two_level * (1 - 1e-9)


def test_joint_relaxation_without_the_floor_lift_is_not_a_bound() -> None:
    """Documents why caps must be lifted to max(cap, m): tiny caps make the bound invalid."""
    gamma, m = 0.8, 0.5
    s = np.array([0.02, 0.03])
    w = np.array([1.0, 1.0]) * s
    caps = 4.0 * s  # 0.08 and 0.12, both below the floor
    budget = 1.0
    floored = solve_group(w, s, gamma, caps, budget, min_spend=m)
    a = w * np.power(s, -gamma)
    unlifted = float(np.sum(a * np.power(water_filling(a, gamma, caps, budget), gamma)))
    assert unlifted < floored["objective"]  # the unlifted relaxation is below the achieved value
    lifted = float(np.sum(a * np.power(water_filling(a, gamma, np.maximum(caps, m), budget), gamma)))
    assert lifted >= floored["objective"] * (1 - 1e-9)


def test_cross_unit_allocation_equalises_marginals_and_exhausts_the_budget() -> None:
    values = {1: (3.0, 0.7, 100.0), 2: (1.0, 0.85, 50.0), 3: (7.0, 0.6, 20.0)}
    budget = 120.0
    split = _cross_unit_allocation(values, budget, cap_mult=4.0)
    total = sum(split.values())
    assert total == pytest.approx(budget, rel=1e-6)
    interior = {u: x for u, x in split.items() if 1e-9 < x < 4.0 * values[u][2] - 1e-9}
    marginals = [values[u][1] * values[u][0] * x ** (values[u][1] - 1.0) for u, x in interior.items()]
    assert max(marginals) / min(marginals) - 1.0 < 1e-6


def test_cross_unit_allocation_returns_caps_when_the_budget_is_ample() -> None:
    values = {1: (3.0, 0.7, 10.0), 2: (1.0, 0.85, 5.0)}
    split = _cross_unit_allocation(values, 1e6, cap_mult=4.0)
    assert split[1] == pytest.approx(40.0)
    assert split[2] == pytest.approx(20.0)


def test_cross_unit_allocation_survives_extreme_price_sweeps() -> None:
    """The first implementation raised OverflowError on the 1e-300 end of the bisection."""
    values = {u: (float(10.0**k), 0.65, 1.0) for k, u in enumerate((1, 2, 3, 4))}
    split = _cross_unit_allocation(values, 2.0, cap_mult=4.0)
    assert sum(split.values()) == pytest.approx(2.0, rel=1e-6)
    assert all(np.isfinite(v) for v in split.values())


def test_weights_switch_off_and_narrow_the_bounce_factor() -> None:
    import pandas as pd

    p = pd.DataFrame(
        {
            "c": [10.0, 20.0],
            "rate": [0.08, 0.08],
            "kappa": [0.25, 2.5],
            "rho": [0.08 * 0.25, 0.08 * 2.5],
        }
    )
    assert _weights(p) == pytest.approx([0.08 * 0.25 * 10, 0.08 * 2.5 * 20])
    assert _weights(p, "one") == pytest.approx([0.8, 1.6])
    assert _weights(p, (0.5, 2.0)) == pytest.approx([0.08 * 0.5 * 10, 0.08 * 2.0 * 20])
