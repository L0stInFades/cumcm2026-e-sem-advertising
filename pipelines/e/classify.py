"""Problem 2: cost/benefit keyword classification (composite benefit index, transparent thresholds, checker)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LABELS: tuple[str, ...] = ("黄金词", "重点词", "潜力词", "问题词", "无效词")
BENEFIT_COMPONENTS: tuple[str, ...] = ("clicks", "views", "engagement", "reg_attr")


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def entropy_weights(X: np.ndarray) -> np.ndarray:
    """Entropy weight method on a non-negative matrix (rows = items, columns = criteria)."""
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    weights = np.zeros(m)
    for j in range(m):
        col = minmax(X[:, j])
        total = col.sum()
        if total <= 0:
            continue
        p = col / total
        with np.errstate(divide="ignore", invalid="ignore"):
            e = -np.nansum(np.where(p > 0, p * np.log(p), 0.0)) / np.log(n)
        weights[j] = 1.0 - e
    if weights.sum() <= 0:
        return np.full(m, 1.0 / m)
    return weights / weights.sum()


def benefit_components(kw: pd.DataFrame, rates: dict[int, float]) -> pd.DataFrame:
    """Log-scaled benefit components: clicks, views, engagement (attention seconds), attributed registrations."""
    reg_attr = kw["clicks"].to_numpy(dtype=float) * kw["推广单元ID"].map(rates).fillna(0.0).to_numpy(dtype=float)
    comp = pd.DataFrame(
        {
            "clicks": np.log1p(kw["clicks"].to_numpy(dtype=float)),
            "views": np.log1p(kw["views"].to_numpy(dtype=float)),
            "engagement": np.log1p(kw["engagement"].to_numpy(dtype=float)),
            "reg_attr": np.log1p(reg_attr),
        },
        index=kw.index,
    )
    comp["reg_attr_raw"] = reg_attr
    return comp


def benefit_index(comp: pd.DataFrame, weights: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
    """Weighted sum of min-max normalised components (entropy weights unless given)."""
    X = comp[list(BENEFIT_COMPONENTS)].to_numpy(dtype=float)
    w = entropy_weights(X) if weights is None else np.asarray(weights, dtype=float)
    normed = np.column_stack([minmax(X[:, j]) for j in range(X.shape[1])])
    return pd.Series(normed @ w, index=comp.index, name="benefit"), w


def jenks_break(x: np.ndarray) -> float:
    """Exact two-class Fisher/Jenks natural break (minimum within-class sum of squares)."""
    xs = np.sort(np.asarray(x, dtype=float))
    n = len(xs)
    if n < 2:
        return float(xs[0]) if n else 0.0
    cs = np.cumsum(xs)
    cs2 = np.cumsum(xs**2)
    k = np.arange(1, n)  # left class = xs[:k]
    left = cs2[k - 1] - cs[k - 1] ** 2 / k
    right = (cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (n - k)
    best = int(np.argmin(left + right))
    return float(0.5 * (xs[best] + xs[best + 1]))


def gmm_break(x: np.ndarray, seed: int) -> float:
    """Threshold where the two-component Gaussian mixture posterior crosses 0.5 between the component means."""
    from sklearn.mixture import GaussianMixture

    xs = np.asarray(x, dtype=float).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=seed, n_init=3).fit(xs)
    means = gm.means_.ravel()
    lo, hi = float(means.min()), float(means.max())
    grid = np.linspace(lo, hi, 2001).reshape(-1, 1)
    post = gm.predict_proba(grid)[:, int(np.argmax(means))]
    idx = int(np.argmax(post >= 0.5))
    return float(grid[idx, 0])


def threshold(x: np.ndarray, method: str, seed: int = 0) -> float:
    if method == "median":
        return float(np.median(x))
    if method == "jenks":
        return jenks_break(x)
    if method == "gmm":
        return gmm_break(x, seed)
    raise ValueError(f"unknown threshold method {method!r}")


def assign_labels(kw: pd.DataFrame, benefit: pd.Series, cost_thr: float, benefit_thr: float) -> pd.Series:
    """Five-way labels: invalid (no spend, no clicks), else the quadrant of (log10 cost > cost_thr, benefit >= thr)."""
    invalid = (kw["spend"] <= 0) & (kw["clicks"] <= 0)
    high_cost = np.log10(kw["spend"].clip(lower=1e-9)) > cost_thr
    high_benefit = benefit >= benefit_thr
    labels = np.select(
        [
            invalid,
            ~high_cost & high_benefit,
            high_cost & high_benefit,
            ~high_cost & ~high_benefit,
            high_cost & ~high_benefit,
        ],
        ["无效词", "黄金词", "重点词", "潜力词", "问题词"],
        default="问题词",
    )
    return pd.Series(labels, index=kw.index, name="label")


def classify_keywords(
    kw: pd.DataFrame, rates: dict[int, float], methods: tuple[str, ...], main: str, seed: int
) -> dict[str, Any]:
    """Full classification with every threshold method; ``main`` selects the reported labels."""
    comp = benefit_components(kw, rates)
    benefit, weights = benefit_index(comp)
    benefit_equal, _ = benefit_index(comp, np.full(len(BENEFIT_COMPONENTS), 1.0 / len(BENEFIT_COMPONENTS)))
    active = kw["spend"] > 0
    log_cost = np.log10(kw.loc[active, "spend"].to_numpy(dtype=float))
    thresholds: dict[str, dict[str, float]] = {}
    labels: dict[str, pd.Series] = {}
    for method in methods:
        ct = threshold(log_cost, method, seed)
        bt = threshold(benefit[active].to_numpy(dtype=float), method, seed)
        thresholds[method] = {"log10_cost": ct, "cost_yuan": float(10**ct), "benefit": bt}
        labels[method] = assign_labels(kw, benefit, ct, bt)
    out = kw[
        ["序号", "关键词", "方案ID", "推广单元ID", "spend", "clicks", "views", "bounce", "duration_s", "engagement"]
    ].copy()
    out["reg_attr"] = comp["reg_attr_raw"]
    out["benefit"] = benefit
    out["benefit_equal"] = benefit_equal
    out["log10_cost"] = np.log10(kw["spend"].clip(lower=1e-9))
    for method in methods:
        out[f"label_{method}"] = labels[method]
    out["label"] = labels[main]
    agreement = {f"{a}_vs_{b}": float((labels[a] == labels[b]).mean()) for a in methods for b in methods if a < b}
    agreement["entropy_vs_equal_spearman"] = float(benefit[active].corr(benefit_equal[active], method="spearman"))
    # equal-weight benefit with the main thresholds re-derived: sensitivity of labels to the weighting
    bt_equal = threshold(benefit_equal[active].to_numpy(dtype=float), main, seed)
    labels_equal = assign_labels(kw, benefit_equal, thresholds[main]["log10_cost"], bt_equal)
    agreement[f"{main}_vs_equal_weights"] = float((labels[main] == labels_equal).mean())
    out["label_equal_weights"] = labels_equal
    counts = {m: labels[m].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict() for m in methods}
    counts["equal_weights"] = labels_equal.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    thresholds["equal_weights"] = {
        "log10_cost": thresholds[main]["log10_cost"],
        "cost_yuan": thresholds[main]["cost_yuan"],
        "benefit": bt_equal,
    }
    return {
        "table": out,
        "weights": dict(zip(BENEFIT_COMPONENTS, [float(w) for w in weights])),
        "thresholds": thresholds,
        "agreement": agreement,
        "counts": counts,
        "main": main,
    }


def verify_classification(
    table: pd.DataFrame, cost_thr_log10: float, benefit_thr: float, label_col: str = "label"
) -> dict[str, Any]:
    """Independent re-derivation of every label from raw columns; returns a report with ``ok``."""
    checks: list[dict[str, Any]] = []
    lab = table[label_col]
    checks.append({"name": "labels_in_vocabulary", "ok": bool(lab.isin(LABELS).all())})
    invalid_expected = (table["spend"] <= 0) & (table["clicks"] <= 0)
    invalid_ok = bool(((lab == "无效词") == invalid_expected).all())
    checks.append({"name": "invalid_iff_no_spend_no_clicks", "ok": invalid_ok})
    high_cost = table["spend"] > 10**cost_thr_log10
    high_ben = table["benefit"] >= benefit_thr
    expected = np.where(
        invalid_expected,
        "无效词",
        np.where(high_cost, np.where(high_ben, "重点词", "问题词"), np.where(high_ben, "黄金词", "潜力词")),
    )
    mism = int((lab.to_numpy() != expected).sum())
    checks.append({"name": "quadrant_rule_reproduced", "ok": mism == 0, "mismatches": mism})
    counts = lab.value_counts().reindex(LABELS, fill_value=0)
    checks.append({"name": "counts_sum_to_rows", "ok": int(counts.sum()) == len(table)})
    gold = table[lab == "黄金词"]
    prob = table[lab == "问题词"]
    checks.append(
        {
            "name": "gold_cheaper_and_better_than_problem",
            "ok": bool(
                len(gold) == 0
                or len(prob) == 0
                or (
                    gold["spend"].max() <= prob["spend"].min() + 1e-9
                    and gold["benefit"].min() >= prob["benefit"].max() - 1e-12
                )
            ),
        }
    )
    checks.append(
        {
            "name": "benefit_in_unit_interval",
            "ok": bool(((table["benefit"] >= -1e-12) & (table["benefit"] <= 1 + 1e-12)).all()),
        }
    )
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "counts": counts.astype(int).to_dict()}


def axis_diagnostics(table: pd.DataFrame, main_thresholds: dict[str, float], method: str, seed: int) -> dict[str, Any]:
    """How independent are the two classification axes, and what does an efficiency axis give instead?

    The benefit index aggregates volume quantities (clicks, page views, attention seconds, attributed
    registrations) and is therefore strongly co-monotone with spend: the quadrant scheme is mostly a
    banding along one spend-benefit axis plus two small off-diagonal classes.  Reporting the
    correlation, the economic weight of the off-diagonal classes and a dual classification on an
    efficiency axis (attributed registrations per yuan) lets the reader judge the scheme instead of
    taking the two-dimensional picture at face value.
    """
    active = table["spend"] > 0
    a = table.loc[active]
    lc = a["log10_cost"].to_numpy(dtype=float)
    ben = a["benefit"].to_numpy(dtype=float)
    pear = float(np.corrcoef(lc, ben)[0, 1])
    spear = float(pd.Series(lc).corr(pd.Series(ben), method="spearman"))
    clicks_spear = float(a["clicks"].corr(a["benefit"], method="spearman"))
    # efficiency axis: attributed registrations per yuan (log scale), same threshold machinery
    eff = np.log1p(a["reg_attr"].to_numpy(dtype=float) / np.maximum(a["spend"].to_numpy(dtype=float), 1e-9))
    eff_norm = minmax(eff)
    eff_thr = threshold(eff_norm, method, seed)
    eff_series = pd.Series(0.0, index=table.index)
    eff_series.loc[a.index] = eff_norm
    eff_labels = assign_labels(table, eff_series, main_thresholds["log10_cost"], eff_thr)
    counts = eff_labels.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    total_spend = float(table["spend"].sum())
    total_clicks = float(table["clicks"].sum())
    shares = {}
    for label in LABELS:
        sel = table["label"] == label
        shares[label] = {
            "n": int(sel.sum()),
            "spend_share": float(table.loc[sel, "spend"].sum() / max(total_spend, 1e-9)),
            "click_share": float(table.loc[sel, "clicks"].sum() / max(total_clicks, 1e-9)),
        }
    return {
        "pearson_logcost_benefit": pear,
        "r2_logcost_benefit": pear**2,
        "spearman_logcost_benefit": spear,
        "spearman_clicks_benefit": clicks_spear,
        "spearman_logcost_efficiency": float(pd.Series(lc).corr(pd.Series(eff), method="spearman")),
        "efficiency_threshold": float(eff_thr),
        "efficiency_counts": counts,
        "efficiency_agreement": float((eff_labels == table["label"]).mean()),
        "label_shares": shares,
        "efficiency_labels": eff_labels,
    }
