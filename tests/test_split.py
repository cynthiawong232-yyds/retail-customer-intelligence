"""Leakage tests.

These are the tests that matter. A model can be wrong and still be useful;
a leaking evaluation is worse than useless, because it reports success.

Each test below corresponds to a specific way this pipeline could quietly
start lying, so if one fails, the number it protects should be treated as
fiction until it passes again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rci.config import PREDICTION_WINDOW_DAYS, derive_cutoffs
from rci.split import make_snapshot, train_test_snapshots


@pytest.fixture(scope="module")
def purchases() -> pd.DataFrame:
    from rci.clean import build

    return build()


@pytest.fixture(scope="module")
def snapshots(purchases):
    return train_test_snapshots(purchases)


def test_observation_is_strictly_before_cutoff(snapshots):
    """THE test. No feature-eligible row may sit on or after the cutoff."""
    for snap in snapshots:
        assert snap.observation["invoice_date"].max() < snap.cutoff


def test_every_labelled_customer_was_seen_before_cutoff(snapshots):
    """We cannot score a customer we have never observed.

    If this fails, the negative class is padded with customers who had no
    history, which makes the model look better than it is.
    """
    for snap in snapshots:
        seen = set(snap.observation["customer_id"].unique())
        labelled = set(snap.labels["customer_id"])
        assert labelled <= seen


def test_label_is_consistent_with_future_spend(snapshots):
    """`repurchased` must be exactly `future_spend > 0`, with no third state."""
    for snap in snapshots:
        expected = (snap.labels["future_spend"] > 0).astype(int)
        assert (snap.labels["repurchased"] == expected).all()


def test_label_window_is_fully_inside_the_data(purchases, snapshots):
    """A window running past the end of the data would fake churn.

    Customers would look lapsed only because observation stopped, not
    because they stopped buying.
    """
    last = purchases["invoice_date"].max()
    for snap in snapshots:
        assert snap.cutoff + pd.Timedelta(days=snap.window_days) <= last + pd.Timedelta(days=1)


def test_snapshots_are_out_of_time(snapshots):
    """Train must sit strictly earlier than test, by one full window."""
    train, test = snapshots
    assert train.cutoff < test.cutoff
    assert (test.cutoff - train.cutoff).days == PREDICTION_WINDOW_DAYS


def test_train_label_window_does_not_reach_into_test_labels(snapshots):
    """The two label windows must not overlap, or the same purchase would
    count as evidence twice."""
    train, test = snapshots
    train_end = train.cutoff + pd.Timedelta(days=train.window_days)
    assert train_end <= test.cutoff


def test_derive_cutoffs_packs_windows_against_the_end():
    """Cutoff arithmetic, checked independently of the real data."""
    last = pd.Timestamp("2011-12-09")
    train_cutoff, test_cutoff = derive_cutoffs(last, window_days=90)
    assert test_cutoff == pd.Timestamp("2011-09-10")
    assert train_cutoff == pd.Timestamp("2011-06-12")


def test_synthetic_leak_is_detected():
    """Sanity-check the harness itself on data with a known answer.

    One customer buys only before the cutoff, one only after, one on both
    sides. The snapshot must label them 0, (excluded), and 1 respectively.
    """
    rows = [
        # customer 1: buys before only  -> eligible, did NOT repurchase
        (1, "2011-01-01", 10.0),
        # customer 2: buys after only   -> NOT eligible (unseen at cutoff)
        (2, "2011-07-01", 20.0),
        # customer 3: buys on both sides -> eligible, DID repurchase
        (3, "2011-01-15", 30.0),
        (3, "2011-07-15", 40.0),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "invoice_date", "line_total"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    snap = make_snapshot(df, pd.Timestamp("2011-06-01"), window_days=90)

    labels = snap.labels.set_index("customer_id")
    assert set(labels.index) == {1, 3}, "customer 2 was unseen and must be excluded"
    assert labels.loc[1, "repurchased"] == 0
    assert labels.loc[3, "repurchased"] == 1
    assert np.isclose(labels.loc[3, "future_spend"], 40.0)
    assert np.isclose(labels.loc[1, "future_spend"], 0.0)
