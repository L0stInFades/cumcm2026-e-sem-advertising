from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from pipelines.e.classify import (
    LABELS,
    assign_labels,
    benefit_components,
    benefit_index,
    entropy_weights,
    jenks_break,
    verify_classification,
)


def _frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    spend = np.where(rng.random(n) < 0.3, 0.0, rng.lognormal(2, 1.5, n))
    clicks = np.where(spend > 0, rng.poisson(spend / 1.2) + 1, 0)
    views = np.where(clicks > 0, rng.poisson(clicks * 1.5), 0)
    bounce = np.where(clicks > 0, rng.random(n), np.nan)
    return pd.DataFrame(
        {
            "序号": np.arange(1, n + 1),
            "关键词": np.arange(1, n + 1),
            "方案ID": 1,
            "推广单元ID": 7,
            "spend": spend,
            "clicks": clicks,
            "views": views,
            "bounce": bounce,
            "duration_s": np.where(clicks > 0, rng.random(n) * 300, np.nan),
            "engagement": np.where(clicks > 0, clicks * (1 - np.nan_to_num(bounce)) * 60, 0.0),
        }
    )


def test_jenks_break_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 40), rng.normal(6, 1, 30)])
    thr = jenks_break(x)
    xs = np.sort(x)
    best = min(range(1, len(xs)), key=lambda k: np.var(xs[:k]) * k + np.var(xs[k:]) * (len(xs) - k))
    assert xs[best - 1] < thr < xs[best]
    assert 1.5 < thr < 4.5


def test_entropy_weights_sum_to_one_and_ignore_constant_columns() -> None:
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.random(50), np.ones(50), rng.random(50) ** 4])
    w = entropy_weights(X)
    assert abs(w.sum() - 1) < 1e-12 and w[1] == 0.0 and (w >= 0).all()


def test_labels_follow_quadrant_rule_and_checker_agrees() -> None:
    rng = np.random.default_rng(3)
    kw = _frame(300, rng)
    comp = benefit_components(kw, {7: 0.02})
    benefit, _ = benefit_index(comp)
    active = kw["spend"] > 0
    ct = float(np.median(np.log10(kw.loc[active, "spend"])))
    bt = float(np.median(benefit[active]))
    labels = assign_labels(kw, benefit, ct, bt)
    table = kw.assign(benefit=benefit, label=labels)
    report = verify_classification(table, ct, bt)
    assert report["ok"], report
    assert set(labels.unique()) <= set(LABELS)
    assert (labels[~active] == "无效词").all()
    gold = table[table["label"] == "黄金词"]
    assert (gold["spend"] <= 10**ct).all() and (gold["benefit"] >= bt).all()


@given(st.integers(min_value=20, max_value=200), st.integers(min_value=0, max_value=1000))
def test_property_every_keyword_gets_exactly_one_label(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    kw = _frame(n, rng)
    comp = benefit_components(kw, {7: 0.01})
    benefit, w = benefit_index(comp)
    assert abs(w.sum() - 1) < 1e-9
    labels = assign_labels(kw, benefit, 1.0, 0.5)
    assert len(labels) == n and labels.isin(LABELS).all()
    counts = labels.value_counts()
    assert counts.sum() == n
