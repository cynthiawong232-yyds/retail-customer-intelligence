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


def test_train_population_is_a_subset_of_test_population(snapshots):
    """A temporal split photographs the base twice; it does not cut it in two.

    Anyone who existed at the June cutoff also existed at the September one,
    because nobody un-becomes a customer. So train MUST be a strict subset of
    test, and the two sizes must NOT sum to the total customer count.

    If this ever fails, the snapshots are being built from different
    populations and the out-of-time comparison is meaningless.
    """
    train, test = snapshots
    tr = set(train.labels["customer_id"])
    te = set(test.labels["customer_id"])
    assert tr <= te, "a train customer vanished from the later snapshot"
    # The later snapshot must have picked up genuinely new customers, or the
    # two cutoffs are not far enough apart to be measuring anything.
    assert len(te - tr) > 0


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

    Every other test in this file runs on the real 776k-row dataset, which
    means a failure could be the data changing rather than the code breaking.
    This one uses FOUR INVENTED ROWS where the correct answer is obvious by
    inspection, so a failure can only mean make_snapshot() is wrong.

    Three customers, covering the three cases that exist:

        cutoff = 2011-06-01, window = 90 days (so label window is Jun-Aug)

        customer 1   Jan  ●                      -> seen before cutoff, never
                          |                         came back = label 0
        customer 2        |         Jul ●        -> first purchase is AFTER
                          |                         the cutoff, so on 1 June
                          |                         we have never heard of
                          |                         them. Must be EXCLUDED.
        customer 3   Jan  ●         Jul ●        -> seen before, came back
                          |                         = label 1, spend 40
                       cutoff
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

    # set_index so we can look a customer up by id rather than row position.
    labels = snap.labels.set_index("customer_id")

    # The subtlest rule in the whole project. Customer 2 is a real customer
    # with a real purchase, but not on 1 June. Including them would add a row
    # the model cannot possibly get right, and would distort the base rate.
    assert set(labels.index) == {1, 3}, "customer 2 was unseen and must be excluded"

    # Customer 1 was around, and stayed away. Negative example.
    assert labels.loc[1, "repurchased"] == 0

    # Customer 3 was around, and came back. Positive example.
    assert labels.loc[3, "repurchased"] == 1

    # np.isclose rather than == because these are floats, and floating point
    # arithmetic makes exact equality unreliable (0.1 + 0.2 != 0.3).
    # Note 40.0, not 70.0: their January purchase is history, not future spend.
    assert np.isclose(labels.loc[3, "future_spend"], 40.0)

    # Zero, not NaN. This is the .fillna(0.0) in make_snapshot doing its job.
    assert np.isclose(labels.loc[1, "future_spend"], 0.0)
