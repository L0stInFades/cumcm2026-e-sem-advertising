"""Problem 3/4 budget allocation: separable concave program, exact water-filling solver, independent verifier."""

from __future__ import annotations

from typing import Any

import numpy as np


def water_filling(a: np.ndarray, gamma: float, caps: np.ndarray, budget: float, tol: float = 1e-12) -> np.ndarray:
    """maximise sum_i a_i x_i^gamma  s.t. sum_i x_i <= budget, 0 <= x_i <= caps_i  (0 < gamma < 1, a_i > 0).

    KKT: gamma a_i x_i^(gamma-1) = lambda on interior coordinates, x_i = cap_i where the marginal at the
    cap still exceeds lambda.  The map lambda -> sum_i min(cap_i, (gamma a_i / lambda)^(1/(1-gamma))) is
    strictly decreasing, so lambda is found by bisection; the solution is the unique global optimum.
    """
    a = np.asarray(a, dtype=float)
    caps = np.asarray(caps, dtype=float)
    if len(a) == 0 or budget <= 0:
        return np.zeros_like(a)
    return _bounded_water_filling(a, gamma, np.zeros_like(a), caps, budget, tol)


def _bounded_water_filling(
    a: np.ndarray, gamma: float, lower: np.ndarray, caps: np.ndarray, budget: float, tol: float = 1e-12
) -> np.ndarray:
    """Exact solution of max sum a_i x_i^gamma s.t. sum x_i = budget, lower_i <= x_i <= caps_i (fixed set)."""
    expo = 1.0 / (1.0 - gamma)
    if caps.sum() <= budget:
        return caps.copy()
    if lower.sum() >= budget:
        return lower.copy()

    def total(lam: float) -> np.ndarray:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            raw = np.power(gamma * a / lam, expo)
        return np.clip(np.nan_to_num(raw, nan=0.0, posinf=np.inf), lower, caps)

    lo, hi = 1e-300, 1.0
    while total(hi).sum() > budget:
        hi *= 4.0
    for _ in range(400):
        mid = np.sqrt(lo * hi)
        if total(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < tol:
            break
    x = total(np.sqrt(lo * hi))
    free = (x > lower + 1e-12) & (x < caps - 1e-12)
    gap = budget - x.sum()
    if abs(gap) > 1e-9 and free.any():  # exact budget balance on the free coordinates
        x[free] += gap * x[free] / max(x[free].sum(), 1e-300)
    return np.clip(x, lower, caps)


def solve_group(
    w: np.ndarray,
    s: np.ndarray,
    gamma: float,
    caps: np.ndarray,
    budget: float,
    min_spend: float = 0.5,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Allocate ``budget`` over keywords with value weights ``w`` (objective sum w_i (x_i/s_i)^gamma).

    With ``min_spend`` = m > 0 every keyword is either not selected (x_i = 0) or receives at least m,
    i.e. x_i in {0} U [m, cap_i].  This non-convex selection problem is solved in two phases:
    (1) Lagrangian relaxation -- for a budget price lambda each keyword takes
    x_i(lambda) = clip((gamma a_i / lambda)^(1/(1-gamma)), m, cap_i) and is selected iff its value
    covers the price, a_i x_i^gamma >= lambda x_i; total spend is non-increasing in lambda, so the
    price is found by bisection and fixes the selected set; (2) the convex problem restricted to the
    selected set with lower bound m is solved exactly (bounded water filling), so the KKT conditions
    hold exactly on that set.  The convex relaxation (m = 0) bounds the optimum from above and is used
    by the verifier as an optimality certificate.
    """
    w = np.asarray(w, dtype=float)
    s = np.asarray(s, dtype=float)
    caps = np.asarray(caps, dtype=float)
    a = w * np.power(s, -gamma)
    n = len(a)
    x = np.zeros(n)
    if n == 0 or budget <= 0:
        return {"x": x, "selected": x > 0, "objective": 0.0, "spent": 0.0, "budget": float(budget), "n_selected": 0}
    positive = a > 0
    if budget < min_spend:  # not even one keyword can receive the minimum practical amount
        return {"x": x, "selected": x > 0, "objective": 0.0, "spent": 0.0, "budget": float(budget), "n_selected": 0}
    if min_spend <= 0:
        x[positive] = water_filling(a[positive], gamma, caps[positive], budget, tol)
    else:
        m = float(min_spend)
        caps_eff = np.maximum(caps, m)
        expo = 1.0 / (1.0 - gamma)

        def allocation(lam: float) -> np.ndarray:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                interior = np.power(gamma * a / lam, expo)
            xx = np.clip(np.nan_to_num(interior, nan=0.0, posinf=np.inf), m, caps_eff)
            include = positive & (a * np.power(xx, gamma) >= lam * xx)
            return np.where(include, xx, 0.0)

        full = np.where(positive, caps_eff, 0.0)
        if full.sum() <= budget:
            x = full
        else:
            lo, hi = 1e-300, 1.0
            while allocation(hi).sum() > budget:
                hi *= 4.0
            for _ in range(400):
                mid = np.sqrt(lo * hi)
                if allocation(mid).sum() > budget:
                    lo = mid
                else:
                    hi = mid
                if hi / lo - 1.0 < tol:
                    break
            chosen = allocation(hi) > 0
            if not chosen.any():
                chosen = allocation(lo) > 0
            k = int(chosen.sum())
            x[chosen] = _bounded_water_filling(a[chosen], gamma, np.full(k, m), caps_eff[chosen], budget, tol)
    objective = float(np.sum(a * np.power(x, gamma)))
    return {
        "x": x,
        "selected": x > 0,
        "objective": objective,
        "spent": float(x.sum()),
        "budget": float(budget),
        "n_selected": int((x > 0).sum()),
    }


def objective_value(x: np.ndarray, a: np.ndarray, gamma: float) -> float:
    return float(np.sum(a * np.power(np.maximum(x, 0.0), gamma)))


def verify_allocation(
    x: np.ndarray,
    a: np.ndarray,
    gamma: float,
    caps: np.ndarray,
    budget: float,
    rng: np.random.Generator,
    floor: float = 0.0,
    kkt_tol: float = 1e-6,
    price_tol: float = 0.05,
    n_transfers: int = 200,
    bound_tol: float = 0.05,
) -> dict[str, Any]:
    """Independent optimality audit of an allocation (does not use the solver).

    Feasibility (x_i = 0 or floor <= x_i <= cap_i, budget); KKT conditions (equal marginal value on
    interior coordinates, marginal at capped coordinates not smaller, average value at floored
    coordinates covering the price up to ``price_tol``); random pairwise transfers, removals and
    insertions never improve the objective; and, when cvxpy is available, the convex relaxation
    (no floor) solved by a generic solver bounds the objective from above -- the relative gap is an
    optimality certificate (not informative for budgets of a few floor units, which are flagged).
    """
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    caps = np.asarray(caps, dtype=float)
    caps_eff = np.maximum(caps, floor)
    checks: list[dict[str, Any]] = []
    sel = x > 0
    checks.append({"name": "nonnegative", "ok": bool((x >= -1e-9).all())})
    checks.append({"name": "within_caps", "ok": bool((x <= caps_eff + 1e-6).all())})
    checks.append({"name": "floor_respected", "ok": bool((x[sel] >= floor - 1e-6).all())})
    spent = float(x.sum())
    checks.append({"name": "budget_respected", "ok": bool(spent <= budget + 1e-6 * max(1.0, budget)), "spent": spent})
    all_capped = np.where(a > 0, caps_eff, 0.0).sum() <= budget + 1e-9
    exhausted = abs(spent - budget) <= 1e-6 * max(1.0, budget)
    # with a floor the budget may be unattainable: every selected keyword sits at its cap and the rest < floor
    stuck = floor > 0 and (budget - spent) < floor + 1e-9 and bool(np.all(x[sel] >= caps_eff[sel] - 1e-6))
    checks.append({"name": "budget_exhausted_or_all_capped", "ok": bool(all_capped or exhausted or stuck)})
    marg = np.full(len(x), np.nan)
    marg[sel] = gamma * a[sel] * np.power(x[sel], gamma - 1.0)
    avg = np.full(len(x), np.nan)
    avg[sel] = a[sel] * np.power(x[sel], gamma - 1.0)
    at_cap = x >= caps_eff - 1e-9
    at_floor = (x <= floor + 1e-9) & (floor > 0)
    both = sel & at_cap & at_floor  # cap below the floor: both bounds active, only the selection rule applies
    capped = sel & at_cap & ~at_floor
    floored = sel & at_floor & ~at_cap
    interior = sel & ~at_cap & ~at_floor
    spread = float(np.nanmax(marg[interior]) / np.nanmin(marg[interior]) - 1.0) if interior.sum() >= 2 else 0.0
    checks.append({"name": "kkt_equal_marginals_interior", "ok": spread <= kkt_tol, "relative_spread": spread})
    # shadow price: with interior coordinates it is their common marginal value; otherwise any price
    # between the largest floored marginal and the smallest capped marginal satisfies the KKT system
    lam_low = float(np.nanmax(marg[floored])) if floored.any() else 0.0
    lam_high = float(np.nanmin(marg[capped])) if capped.any() else float("inf")
    if interior.any():
        lam = float(np.nanmedian(marg[interior]))
        cap_ok = not capped.any() or bool(np.all(marg[capped] >= lam * (1 - kkt_tol)))
        floor_marg_ok = not floored.any() or bool(np.all(marg[floored] <= lam * (1 + kkt_tol)))
        floor_avg_ok = not (floored | both).any() or bool(np.all(avg[floored | both] >= lam * (1 - price_tol)))
    else:
        lam = lam_low if np.isfinite(lam_high) is False else min(lam_high, max(lam_low, 0.0))
        cap_ok = lam_low <= lam_high * (1 + kkt_tol)
        floor_marg_ok = True
        floor_avg_ok = not (floored | both).any() or bool(np.all(avg[floored | both] >= lam_low * (1 - price_tol)))
    checks.append({"name": "kkt_capped_marginal_not_smaller", "ok": bool(cap_ok)})
    checks.append({"name": "kkt_floored_marginal_not_larger", "ok": bool(floor_marg_ok)})
    checks.append({"name": "kkt_floored_average_value_covers_price", "ok": bool(floor_avg_ok)})
    base = objective_value(x, a, gamma)
    idx = np.flatnonzero(sel)
    improved = 0
    if len(idx) >= 2:
        for _ in range(n_transfers):
            i, j = rng.choice(idx, size=2, replace=False)
            y = x.copy()
            if rng.random() < 0.3:  # remove keyword i entirely and hand its money to j
                delta = min(x[i], caps_eff[j] - x[j])
                if delta < x[i] - 1e-12 and x[i] - delta < floor - 1e-12:
                    continue
            else:
                delta = min(x[i] - floor, caps_eff[j] - x[j]) * rng.uniform(0.01, 0.5)
            if delta <= 0:
                continue
            y[i] -= delta
            y[j] += delta
            if objective_value(y, a, gamma) > base * (1 + 1e-9):
                improved += 1
    checks.append({"name": "no_improving_transfer_or_removal", "ok": improved == 0, "improving_moves": improved})
    inserted = 0
    out_idx = np.flatnonzero(~sel & (a > 0))
    if len(out_idx) and floor > 0 and interior.any():  # insert an unselected keyword at the floor
        donors = np.flatnonzero(interior)
        slack = price_tol * floor / max(budget, floor)  # phase-1 price and phase-2 price differ slightly
        for k in rng.choice(out_idx, size=min(len(out_idx), 50), replace=False):
            take = floor * x[donors] / x[donors].sum()
            if np.any(x[donors] - take < floor):
                continue
            y = x.copy()
            y[donors] -= take
            y[k] = floor
            if objective_value(y, a, gamma) > base * (1 + slack):
                inserted += 1
    checks.append({"name": "no_improving_insertion", "ok": inserted == 0, "improving_insertions": inserted})
    cvx: dict[str, Any] = {"available": False}
    try:
        import cvxpy as cp

        n = len(x)
        if n <= 1500:
            var = cp.Variable(n, nonneg=True)
            objective = cp.Maximize(cp.sum(cp.multiply(a, cp.power(var, gamma))))
            prob = cp.Problem(objective, [cp.sum(var) <= budget, var <= caps_eff])
            prob.solve(solver=cp.CLARABEL)
            if var.value is not None:
                relaxed = objective_value(np.asarray(var.value).ravel(), a, gamma)
                gap = float((relaxed - base) / max(abs(base), 1e-12))
                tiny = floor > 0 and budget <= 20 * floor
                cvx = {"available": True, "status": prob.status, "relaxed_objective": relaxed, "relative_gap": gap}
                tol = bound_tol if floor > 0 else 1e-4
                checks.append(
                    {
                        "name": "within_convex_relaxation_bound",
                        "ok": bool(gap <= tol or tiny),
                        "relative_gap": gap,
                        "tiny_budget": bool(tiny),
                    }
                )
    except Exception as exc:  # pragma: no cover - solver availability differs across environments
        cvx = {"available": False, "error": repr(exc)}
    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "objective": base,
        "n_selected": int(sel.sum()),
        "n_capped": int(capped.sum()),
        "n_floored": int(floored.sum()),
        "n_both_bounds": int(both.sum()),
        "cvxpy": cvx,
    }


def proportional_allocation(s: np.ndarray, budget: float) -> np.ndarray:
    """Historical-practice baseline: spend split in proportion to each keyword's mean daily spend."""
    s = np.asarray(s, dtype=float)
    return budget * s / s.sum() if s.sum() > 0 else np.zeros_like(s)


def equal_allocation(n: int, budget: float) -> np.ndarray:
    return np.full(n, budget / n) if n else np.zeros(0)
