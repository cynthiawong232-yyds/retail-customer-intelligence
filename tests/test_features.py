"""Feature engineering invariants.

Features are COMPUTED, not read, which makes them the place errors hide.
A wrong feature does not raise an exception; it just quietly makes every
model worse or, worse still, better in a way that will not survive contact
with production. These tests pin down the properties that must hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rci.features import FEATURE_COLUMNS, RFM_COLUMNS, build_features, build_xy
from rci.split import make_snapshot, train_test_snapshots


@pytest.fixture(scope="module")
def snapshots():
    from rci.clean import build

    return train_test_snapshots(build())


@pytest.fixture(scope="module")
def train_xy(snapshots):
    return build_xy(snapshots[0])


def test_shape_and_columns(train_xy):
    X, y = train_xy
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)


def test_x_and_y_are_row_aligned(snapshots, train_xy):
    """The single most dangerous silent bug in this file.

    X and y travel as separate objects. If row 7 of X is a different customer
    than row 7 of y, the model trains on mismatched pairs, learns nothing,
    and never errors.
    """
    train = snapshots[0]
    X, y = train_xy
    assert (X.index.to_numpy() == y.index.to_numpy()).all()
    assert (X.index.to_numpy() == train.labels["customer_id"].to_numpy()).all()


def test_recency_is_measured_from_the_cutoff(snapshots, train_xy):
    """Not from 'today', not from the end of the dataset.

    If recency were measured from the dataset's final date, it would encode
    how far the cutoff sits from the end of the file, which is information
    the model could never have in production.
    """
    train = snapshots[0]
    X, _ = train_xy
    obs = train.observation
    last = obs.groupby("customer_id")["invoice_date"].max()
    expected = (train.cutoff - last).dt.days.reindex(X.index)
    assert (X["recency"].to_numpy() == expected.to_numpy()).all()


def test_recency_is_always_positive(train_xy):
    """A purchase before the cutoff is at least 0 days old. Negative recency
    would mean a future purchase leaked into the observation window."""
    X, _ = train_xy
    assert (X["recency"] >= 0).all()


def test_frequency_counts_orders_not_rows(snapshots, train_xy):
    """An order with 30 products is ONE order, not 30.

    Using .count() instead of .nunique() here would inflate frequency by
    roughly 20x and quietly break RFM, CLV and the repurchase model at once.
    """
    train = snapshots[0]
    X, _ = train_xy
    expected = train.observation.groupby("customer_id")["invoice"].nunique().reindex(X.index)
    assert (X["frequency"].to_numpy() == expected.to_numpy()).all()
    # Sanity: rows per customer should far exceed orders per customer.
    rows = train.observation.groupby("customer_id").size().reindex(X.index)
    assert rows.sum() > 5 * X["frequency"].sum()


def test_recent_window_is_a_subset_of_lifetime(train_xy):
    """Spend in the last 90 days cannot exceed lifetime spend."""
    X, _ = train_xy
    assert (X["spend_last_90d"] <= X["monetary"] + 1e-6).all()
    assert (X["orders_last_90d"] <= X["frequency"]).all()


def test_no_unexpected_missing_values(train_xy):
    """Only avg_days_between_orders may be NaN, and only for one-off buyers.

    Everything else defaults to a real value (usually 0, meaning 'none'
    rather than 'unknown'). An unexpected NaN elsewhere means a join went
    wrong.
    """
    X, _ = train_xy
    allowed = {"avg_days_between_orders"}
    offenders = {c for c in X.columns if X[c].isna().any()} - allowed
    assert not offenders, f"unexpected NaN in {offenders}"
    # And that NaN must correspond exactly to customers with a single order.
    assert (X.loc[X["avg_days_between_orders"].isna(), "frequency"] == 1).all()


def test_features_use_only_the_observation_window():
    """End-to-end leakage check on invented data with a known answer.

    One customer, three purchases: two before the cutoff and one after. The
    features must describe only the first two. If monetary came out as 60
    instead of 30, the future would have leaked in.
    """
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 1],
            "invoice": ["A", "B", "C"],
            "stock_code": ["11111", "22222", "33333"],
            "quantity": [1, 1, 1],
            "invoice_date": pd.to_datetime(["2011-01-01", "2011-02-01", "2011-07-01"]),
            "line_total": [10.0, 20.0, 30.0],
            #                            ^^^^ after the cutoff: must be invisible
        }
    )
    snap = make_snapshot(df, pd.Timestamp("2011-06-01"), window_days=90)
    X = build_features(snap, returns=pd.DataFrame(
        columns=["customer_id", "invoice", "invoice_date", "line_total"]
    ))

    assert np.isclose(X.loc[1, "monetary"], 30.0)   # 10 + 20, NOT 60
    assert X.loc[1, "frequency"] == 2               # A and B, not C
    # Last visible purchase is 1 Feb; cutoff is 1 June; 120 days apart.
    assert X.loc[1, "recency"] == 120
    # The July purchase is the ANSWER, and it did land in the label.
    assert snap.labels.set_index("customer_id").loc[1, "repurchased"] == 1


def test_rfm_columns_are_a_subset_of_features():
    assert set(RFM_COLUMNS) <= set(FEATURE_COLUMNS)
