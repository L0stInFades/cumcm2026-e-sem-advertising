from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipelines.e.allocate import (
    equal_allocation,
    objective_value,
    proportional_allocation,
    solve_group,
    verify_allocation,
    water_filling,
)
from pipelines.e.response import clicks_response, cpc_response, expected_position


def test_water_filling_matches_grid_search_on_small_instance() -> None:
    a = np.array([3.0, 1.0, 2.0])
    gamma = 0.7
    caps = np.array([4.0, 10.0, 10.0])
    budget = 9.0
    x = water_filling(a, gamma, caps, budget)
    assert abs(x.sum() - budget) < 1e-8 and (x <= caps + 1e-9).all()
    best = -np.inf
    grid = np.linspace(0, 9, 181)
    for x1, x2 in itertools.product(grid, grid):
        x3 = budget - x1 - x2
        if x3 < 0 or x1 > caps[0] or x2 > caps[1] or x3 > caps[2]:
            continue
        best = max(best, objective_value(np.array([x1, x2, x3]), a, gamma))
    assert objective_value(x, a, gamma) >= best - 1e-6


def test_water_filling_closed_form_without_caps() -> None:
    a = np.array([1.0, 2.0, 4.0])
    gamma = 0.5
    x = water_filling(a, gamma, np.full(3, 1e9), 7.0)
    # x_i proportional to a_i^(1/(1-gamma)) = a_i^2
    expected = 7.0 * a**2 / (a**2).sum()
    assert np.allclose(x, expected, rtol=1e-6)


def test_caps_bind_and_budget_not_exceeded() -> None:
    a = np.array([10.0, 1.0])
    x = water_filling(a, 0.8, np.array([1.0, 5.0]), 3.0)
    assert abs(x[0] - 1.0) < 1e-9 and abs(x.sum() - 3.0) < 1e-9
    x = water_filling(a, 0.8, np.array([1.0, 1.0]), 3.0)
    assert np.allclose(x, [1.0, 1.0])


@settings(max_examples=40)
@given(
    st.lists(st.floats(min_value=0.05, max_value=50.0), min_size=2, max_size=25),
    st.floats(min_value=0.35, max_value=0.95),
    st.floats(min_value=1.0, max_value=500.0),
)
def test_property_verifier_accepts_solver_output(weights: list[float], gamma: float, budget: float) -> None:
    rng = np.random.default_rng(0)
    w = np.array(weights)
    s = np.linspace(1.0, 3.0, len(w))
    caps = 4.0 * s
    res = solve_group(w, s, gamma, caps, budget, min_spend=0.0)
    a = w * s ** (-gamma)
    sel = res["x"] > 0
    report = verify_allocation(res["x"][sel], a[sel], gamma, caps[sel], budget, rng, n_transfers=30)
    assert report["ok"], report["checks"]


def test_verifier_rejects_a_suboptimal_allocation() -> None:
    rng = np.random.default_rng(1)
    a = np.array([5.0, 1.0, 1.0])
    caps = np.full(3, 100.0)
    report = verify_allocation(np.array([1.0, 4.0, 4.0]), a, 0.6, caps, 9.0, rng)
    assert not report["ok"]
    names = {c["name"]: c["ok"] for c in report["checks"]}
    assert not names["kkt_equal_marginals_interior"] or not names["no_improving_transfer_or_removal"]


def test_min_spend_drops_tiny_keywords_and_respends_budget() -> None:
    w = np.array([100.0, 100.0, 0.001])
    s = np.array([10.0, 10.0, 10.0])
    res = solve_group(w, s, 0.8, 4 * s, 20.0, min_spend=0.5)
    assert res["x"][2] == 0.0 and abs(res["spent"] - 20.0) < 1e-6 and res["n_selected"] == 2
    assert (res["x"][res["x"] > 0] >= 0.5).all()


def test_min_spend_solution_beats_enumeration_of_selections() -> None:
    rng = np.random.default_rng(5)
    w = rng.uniform(0.5, 5.0, 6)
    s = rng.uniform(0.5, 3.0, 6)
    gamma, budget, m = 0.75, 4.0, 0.6
    caps = 4 * s
    res = solve_group(w, s, gamma, caps, budget, min_spend=m)
    a = w * s ** (-gamma)
    # brute force over selections: on a fixed subset the problem is convex; solve it with an independent
    # general-purpose NLP solver (SLSQP) under the lower bound m
    from scipy.optimize import minimize

    best = -np.inf
    for mask in range(1, 2**6):
        idx = [i for i in range(6) if mask >> i & 1]
        if m * len(idx) > budget + 1e-12:
            continue
        sub_a, sub_caps = a[idx], caps[idx]
        x0 = np.clip(np.full(len(idx), budget / len(idx)), m, sub_caps)
        res_nlp = minimize(
            lambda y, sa=sub_a: -float(np.sum(sa * np.power(np.maximum(y, 1e-12), gamma))),
            x0,
            method="SLSQP",
            bounds=[(m, c) for c in sub_caps],
            constraints=[{"type": "ineq", "fun": lambda y: budget - y.sum()}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if res_nlp.success:
            best = max(best, -float(res_nlp.fun))
    assert res["objective"] >= best * (1 - 1e-6)
    report = verify_allocation(res["x"], a, gamma, caps, budget, rng, floor=m)
    assert report["ok"], report["checks"]


def test_baselines_and_response_helpers() -> None:
    s = np.array([1.0, 3.0])
    assert np.allclose(proportional_allocation(s, 8.0), [2.0, 6.0])
    assert np.allclose(equal_allocation(2, 8.0), [4.0, 4.0])
    c = np.array([10.0, 30.0])
    assert np.allclose(clicks_response(s, c, s, 0.8), c)  # passes through the operating point
    assert cpc_response(np.array([2.0, 6.0]), c, s, 0.8)[0] > cpc_response(np.array([1.0, 3.0]), c, s, 0.8)[0]
    pos = expected_position(np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.25]))
    assert pos[0] == pytest.approx(1.0) and pos[1] == pytest.approx(4.0) and 1.0 < pos[2] < 4.0
