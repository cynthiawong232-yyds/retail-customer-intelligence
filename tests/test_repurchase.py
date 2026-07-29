"""Repurchase model invariants.

Model quality itself is reported in the README, not asserted here: a test that
demands PR-AUC > 0.77 becomes a nuisance the moment anything legitimately
changes. What IS asserted are the properties that, if broken, would make every
reported number meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from rci.config import RANDOM_SEED
from rci.features import FEATURE_COLUMNS, build_xy
from rci.repurchase import (
    baseline_recency,
    calibrate,
    evaluate,
    fit_xgboost,
    format_tree,
    importance_table,
    lift_table,
)
from rci.split import train_test_snapshots


@pytest.fixture(scope="module")
def data():
    from rci.clean import build

    train, test = train_test_snapshots(build())
    X_train, y_train = build_xy(train, target="repurchased")
    X_test, y_test = build_xy(test, target="repurchased")
    return X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def fitted(data):
    X_train, y_train, X_test, y_test = data
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    model = fit_xgboost(X_fit, y_fit, X_val, y_val)
    return model, model.predict_proba(X_test)[:, 1]


def test_predictions_are_probabilities(fitted, data):
    _, p = fitted
    _, _, X_test, _ = data
    assert len(p) == len(X_test)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_model_beats_every_baseline(fitted, data):
    """The point of a baseline. If 400 trees cannot beat 'sort by recency',
    the honest recommendation is to ship the rule instead."""
    _, _, X_test, y_test = data
    _, p = fitted
    model_score = evaluate(y_test, p)["pr_auc"]
    recency_score = evaluate(y_test, baseline_recency(X_test))["pr_auc"]
    assert model_score > recency_score
    # And it must beat a model that knows nothing, whose PR-AUC floor is the
    # base rate itself (NOT 0.5, which is the ROC-AUC floor).
    assert model_score > y_test.mean()


def test_xgboost_handles_missing_values_natively(fitted, data):
    """31% of customers have NaN avg_days_between_orders and are still scored.

    This is a real XGBoost property, not an accident: missing gets its own
    branch direction. Logistic regression cannot do this, which is why the
    baseline needs an imputer in a pipeline.
    """
    _, _, X_test, _ = data
    _, p = fitted
    missing = X_test["avg_days_between_orders"].isna().to_numpy()
    assert missing.sum() > 0
    assert np.isfinite(p[missing]).all()


def test_feature_order_is_the_shared_constant(fitted, data):
    """Column order is matched by POSITION at serving time, so a mismatch
    would produce confident nonsense rather than an error."""
    model, _ = fitted
    X_train, _, _, _ = data
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert model.n_features_in_ == len(FEATURE_COLUMNS)


def test_lift_is_monotonic_enough_to_be_useful(fitted, data):
    """The top decile must convert better than the bottom, or the ranking is
    worthless regardless of what the AUC says."""
    _, _, _, y_test = data
    _, p = fitted
    lift = lift_table(y_test.to_numpy(), p)
    assert lift.loc[1, "actual_rate"] > lift.loc[10, "actual_rate"]
    assert lift.loc[1, "lift"] > 1.5
    # Cumulative capture must end at exactly 100% of repurchasers.
    assert np.isclose(lift["cumulative_capture"].iloc[-1], 1.0)


def test_calibration_preserves_ranking(fitted, data):
    """Isotonic regression is monotonic, so it CANNOT reorder anyone.

    That is the whole reason it is the right tool here: it fixes the numbers
    without touching the ordering the business actually acts on. ROC-AUC is
    purely rank-based, so it must survive essentially unchanged.
    """
    _, _, _, y_test = data
    _, p = fitted
    y = y_test.to_numpy()
    idx = np.arange(len(y))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(idx)
    ref, tgt = idx[: len(idx) // 2], idx[len(idx) // 2:]

    p_cal, _ = calibrate(p[ref], y[ref], p[tgt])
    before = evaluate(y[tgt], p[tgt])
    after = evaluate(y[tgt], p_cal)

    assert abs(after["roc_auc"] - before["roc_auc"]) < 0.01
    # And it must actually fix the thing it exists to fix.
    assert after["brier"] < before["brier"]
    assert abs(after["mean_predicted"] - after["base_rate"]) < abs(
        before["mean_predicted"] - before["base_rate"]
    )


def test_evaluate_refuses_to_fake_calibration_for_raw_scores(data):
    """A raw score can be ranked but has no calibration to measure.

    -recency ranks customers fine but -647 is not a probability. Brier must
    come back NaN rather than being invented by squashing the score into 0-1.
    """
    _, _, X_test, y_test = data
    m = evaluate(y_test, baseline_recency(X_test))
    assert np.isnan(m["brier"])
    assert m["pr_auc"] > 0.5          # ranking still works
    assert not np.isnan(m["roc_auc"])


def test_tree_dump_shows_real_thresholds(fitted):
    """The learned tree must be printable, with numeric split points."""
    model, _ = fitted
    text = format_tree(model, FEATURE_COLUMNS, 0)
    assert "<" in text and "leaf" in text
    assert any(col in text for col in FEATURE_COLUMNS)


def test_gain_and_shap_measure_different_things(fitted, data):
    """Documenting a real result: the two rankings disagree.

    recency is mid-table by GAIN (it is split on constantly but each split
    decides little) and top by SHAP (it moves individual predictions most).
    Neither is wrong; they answer different questions. Pinned so the README
    claim cannot silently go stale.
    """
    model, _ = fitted
    _, _, X_test, _ = data
    imp = importance_table(model, FEATURE_COLUMNS)
    gain_rank = list(imp.index).index("recency")

    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test)
    shap_rank = list(
        pd.Series(np.abs(values).mean(axis=0), index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
        .index
    ).index("recency")

    assert gain_rank != shap_rank, "the two importance measures now agree; update the README"
