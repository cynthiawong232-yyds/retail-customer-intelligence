"""Segmentation invariants, mostly about the transform step.

The transform is where segmentation silently fails: forget it, and KMeans
clusters on whichever column happens to have the biggest numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rci.features import RFM_COLUMNS, build_features
from rci.segmentation import fit, prepare, profile, quintile_segments
from rci.split import train_test_snapshots


@pytest.fixture(scope="module")
def X():
    from rci.clean import build

    train, _ = train_test_snapshots(build())
    return build_features(train)


def test_prepare_puts_every_column_on_the_same_ruler(X):
    """After scaling, each column has mean 0 and standard deviation 1.

    That is the whole point: it makes 'a big change in recency' and 'a big
    change in spend' comparable, so Euclidean distance stops being a
    referendum on whichever column had the largest raw numbers.
    """
    scaled, _ = prepare(X)
    assert scaled.shape == (len(X), len(RFM_COLUMNS))
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-9)


def test_scaler_is_reusable_on_new_data(X):
    """Serving must .transform() with the TRAINING mean and std.

    Calling .fit_transform() again at prediction time would rescale using the
    new data's own statistics, so an identical customer would get a different
    segment depending on who else happened to be scored alongside them.
    """
    scaled, scaler = prepare(X)
    half = X.iloc[: len(X) // 2]
    reapplied = scaler.transform(np.log1p(half[RFM_COLUMNS]))
    assert np.allclose(reapplied, scaled[: len(half)])


def test_log_actually_compresses_the_tail(X):
    """Skew must drop, or the whales still dominate."""
    raw_skew = X["monetary"].skew()
    logged_skew = np.log1p(X["monetary"]).skew()
    assert raw_skew > 10          # wildly right-skewed to begin with
    assert abs(logged_skew) < 1   # roughly symmetric afterwards


def test_clusters_are_usable_sizes(X):
    """Guards against the failure mode that raw features produce.

    Unscaled KMeans isolates two outlier whales and puts 98% of customers in
    a single bucket. That scores brilliantly on silhouette and is useless, so
    the real check is that no cluster swallows the population.
    """
    _, _, labels = fit(X, k=4)
    sizes = pd.Series(labels).value_counts()
    assert len(sizes) == 4
    assert sizes.max() / len(X) < 0.60    # no bucket dominates
    assert sizes.min() >= 100             # none is a rounding error


def test_fit_is_deterministic(X):
    """Same data must give the same segments every run.

    KMeans starts from random positions. Without a fixed random_state, a
    customer could land in 'Champions' today and 'At-risk' tomorrow with no
    change in their behaviour, which destroys trust in the output.
    """
    _, _, a = fit(X, k=4)
    _, _, b = fit(X, k=4)
    assert (a == b).all()


def test_profile_covers_every_customer(X):
    _, _, labels = fit(X, k=4)
    prof = profile(X, labels)
    assert prof["customers"].sum() == len(X)
    assert np.isclose(prof["pct_customers"].sum(), 100.0, atol=0.5)


def test_quintile_score_is_in_range(X):
    """Three dimensions scored 1-5 each, so the total spans 3 to 15."""
    scores = quintile_segments(X)
    assert scores.min() >= 3
    assert scores.max() <= 15
    assert len(scores) == len(X)


def test_quintile_and_kmeans_disagree_in_the_middle(X):
    """The finding this phase exists to demonstrate, pinned as a test.

    Mid-range RFM scores must contain multiple KMeans segments. If this ever
    stopped being true, the headline claim in the README would be wrong and
    should be rewritten rather than quietly left in place.
    """
    _, _, labels = fit(X, k=4)
    d = pd.DataFrame({"segment": labels, "score": quintile_segments(X).to_numpy()})
    middle = d[d["score"].between(9, 13)]
    # A large share of the base sits in the ambiguous middle...
    assert len(middle) / len(d) > 0.25
    # ...and every score there mixes at least two behavioural segments.
    assert middle.groupby("score")["segment"].nunique().min() >= 2
