"""Phase 1: customer segmentation with KMeans.

WHY THIS EXISTS (the business question)
---------------------------------------
A retailer cannot write 4,951 individual emails. They can write four. So the
job is to collapse thousands of customers into a handful of groups that
behave differently enough to deserve different treatment.

The test of a good segmentation is not a statistic. It is whether someone in
marketing can look at a cluster and say "oh, those are our lapsed regulars,
we should win them back." A cluster nobody can name is a cluster nobody will
ever use.

WHAT KMEANS IS
--------------
Input   X, a table of numbers (here 4,951 customers x 3 RFM columns)
        k, how many groups you want
Output  one cluster number per customer, plus k "centroids" (group centres)

The algorithm, in full:

    1. scatter k centroids at random positions
    2. assign every customer to the NEAREST centroid
    3. move each centroid to the mean of the customers assigned to it
    4. repeat 2 and 3 until nothing moves

That is genuinely all of it. There is no y, no right answer, and nothing is
being predicted. This is UNSUPERVISED learning: find structure, no answer key.

WHY THE DATA MUST BE TRANSFORMED FIRST
--------------------------------------
"Nearest" means straight-line (Euclidean) distance, and distance is measured
in whatever units the columns happen to use. In our raw table:

    monetary   ranges 1.55 to 413,459
    recency    ranges 1 to 557

Monetary is ~700x larger, so it swamps the distance calculation and recency
effectively disappears. Two fixes, applied in order:

    1. LOG    monetary's median is 746 and its max is 413,459. A handful of
              wholesale whales sit so far out that centroids get dragged
              toward them. log() compresses that tail so the difference
              between 100 and 1,000 counts about the same as 10,000 to
              100,000, which is how spend actually behaves commercially.

    2. SCALE  StandardScaler rewrites each column as "standard deviations
              from that column's mean", so every column arrives on the same
              ruler and contributes equally.

`demonstrate_why_scaling_matters()` below proves this rather than asserting
it, by running KMeans both ways and printing the cluster sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from rci.config import RANDOM_SEED
from rci.features import RFM_COLUMNS


def prepare(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log-then-scale the RFM columns. Returns the matrix and the fitted scaler.

    The scaler is returned, not discarded, because it must be SAVED and reused
    at serving time. A new customer has to be transformed with the training
    set's mean and standard deviation, not their own. Re-fitting a scaler at
    prediction time is a classic and silent production bug.
    """
    # .to_numpy() is deliberate, not incidental. Fitting on a DataFrame makes
    # sklearn memorise the column NAMES, and it then warns (rightly) whenever
    # it is later handed a bare array it cannot check. The serving container
    # has no pandas, so it can only send arrays.
    #
    # Rather than silence the warning, make training match serving: both work
    # in plain positional arrays. Column ORDER is then guaranteed the only
    # way it can be, by importing RFM_COLUMNS in both places and asserting it
    # in tests, instead of by a check that cannot run in production anyway.
    rfm = X[RFM_COLUMNS].to_numpy(dtype=np.float64)

    # log1p is log(1 + x), which is defined at x=0. Plain log(0) is -infinity
    # and would poison the whole column. Our values are all >= 1 today, but
    # log1p costs nothing and removes a whole category of future breakage.
    logged = np.log1p(rfm)

    scaler = StandardScaler()
    # fit_transform = learn each column's mean and std, THEN apply them.
    # At serving time we call .transform() only, never .fit_transform().
    scaled = scaler.fit_transform(logged)
    return scaled, scaler


def demonstrate_why_scaling_matters(X: pd.DataFrame, k: int = 4) -> None:
    """Run KMeans raw vs prepared and print what changes. Teaching only."""
    raw = X[RFM_COLUMNS].to_numpy()
    prepared, _ = prepare(X)

    print("\nWHY THE TRANSFORM MATTERS")
    print("=" * 66)
    for name, matrix in (("RAW (no log, no scaling)", raw), ("LOG + SCALED", prepared)):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(matrix)
        sizes = pd.Series(labels).value_counts().sort_index()
        sil = silhouette_score(matrix, labels)
        print(f"\n{name}")
        print(f"  cluster sizes : {list(sizes.to_numpy())}")
        print(f"  silhouette    : {sil:.3f}")
        # Mean spend per cluster, in real pounds, so the failure is legible.
        means = pd.DataFrame({"monetary": X["monetary"].to_numpy(), "c": labels})
        print("  mean spend    : "
              + ", ".join(f"{v:,.0f}" for v in means.groupby("c")["monetary"].mean()))


def choose_k(X: pd.DataFrame, k_range=range(2, 9)) -> pd.DataFrame:
    """Score several values of k so the choice is evidence-based.

    Two numbers, which disagree on purpose:

    INERTIA     total squared distance from each point to its own centroid.
                Always falls as k rises (with k = n customers it hits zero),
                so you never pick the minimum. You look for the "elbow", the
                point after which extra clusters stop buying much.

    SILHOUETTE  for each customer: how much closer are they to their own
                cluster than to the nearest rival cluster? Ranges -1 to +1.
                Above ~0.5 is strong, ~0.25 to 0.5 is workable, near 0 means
                the clusters overlap and are probably not real.
    """
    scaled, _ = prepare(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(scaled, labels),
                "smallest_cluster": pd.Series(labels).value_counts().min(),
            }
        )
    return pd.DataFrame(rows)


def fit(X: pd.DataFrame, k: int) -> tuple[KMeans, StandardScaler, np.ndarray]:
    """Fit the final model. Returns (model, scaler, labels)."""
    scaled, scaler = prepare(X)
    # n_init=10 runs the whole algorithm 10 times from different random starts
    # and keeps the best. KMeans can land in a bad local optimum from an
    # unlucky start, and 10 restarts makes that very unlikely.
    # random_state fixes the randomness so the same data gives the same
    # clusters every run, which matters because these get saved and served.
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(scaled)
    return km, scaler, labels


def profile(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Describe each cluster in business units, not standard deviations.

    The model works in scaled log-space, which is unreadable. This converts
    back to pounds and days so a human can name the groups.
    """
    d = X[RFM_COLUMNS].copy()
    d["segment"] = labels
    out = d.groupby("segment").agg(
        customers=("recency", "size"),
        median_recency=("recency", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
        total_revenue=("monetary", "sum"),
    )
    out["pct_customers"] = (100 * out["customers"] / len(d)).round(1)
    out["pct_revenue"] = (100 * out["total_revenue"] / d["monetary"].sum()).round(1)
    return out


def name_segments(prof: pd.DataFrame) -> dict[int, str]:
    """Attach a human label to each cluster from its own profile.

    Derived from the numbers rather than hardcoded, so the names stay correct
    if the data or k changes. Ranks within this run, not absolute thresholds.
    """
    names: dict[int, str] = {}
    rec_rank = prof["median_recency"].rank()           # 1 = most recent
    val_rank = prof["median_monetary"].rank(ascending=False)   # 1 = highest spend
    freq_rank = prof["median_frequency"].rank(ascending=False)  # 1 = most orders
    n = len(prof)
    for seg in prof.index:
        recent = rec_rank[seg] <= n / 2
        valuable = val_rank[seg] <= n / 2
        frequent = freq_rank[seg] <= n / 2
        if recent and valuable and frequent:
            names[seg] = "Champions"
        elif recent and not valuable:
            names[seg] = "New / Low-value"
        elif not recent and valuable:
            names[seg] = "At-risk high-value"
        else:
            names[seg] = "Lapsed / One-off"
    return names


def quintile_segments(X: pd.DataFrame) -> pd.Series:
    """The rule-based RFM scoring Cindy already wrote, reimplemented.

    This is the approach in porfolio_projects/client_advisor_intelligence_agent
    (`recency_score`, `frequency_score`, `monetary_score`, `rfm_total_score`):
    cut each dimension into 5 equal-sized buckets, score 1 to 5, add them up.

    It is NOT a model. Nothing is fitted and nothing is learned. The bucket
    edges come from the data's own percentiles and the combination rule (plain
    addition) is chosen by a human. That is not a criticism: it is transparent,
    instantly explainable, and needs no maintenance. It is simply a different
    kind of tool, and knowing which is which is the point of this comparison.
    """
    # qcut splits into equal-COUNT buckets (quintiles), not equal-width ones.
    # Recency is reversed: fewer days since purchase is better, so the most
    # recent quintile earns 5.
    r = pd.qcut(X["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # rank(method="first") before qcut on frequency, because frequency has
    # heavy ties (thousands of customers with exactly 1 order) and qcut cannot
    # cut equal-sized bins when one value spans several of them.
    f = pd.qcut(X["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m = pd.qcut(X["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return (r + f + m).rename("rfm_score")
