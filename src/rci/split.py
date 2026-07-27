"""The temporal split. The most important file in this repo.

WHY THIS FILE EXISTS
--------------------
Every model here answers a question about the FUTURE using facts from the
PAST. To measure whether it actually works, the evaluation has to reproduce
that same asymmetry. If any fact from the future reaches the model during
training, the score is fiction.

The mechanism that guarantees it is small and boring: pick a cutoff date,
hand out only the transactions strictly before it, and compute the answer
only from transactions on or after it.

    cutoff = 2011-09-10, window = 90 days

    ....................|........................|
    all purchases       cutoff              cutoff+90d
    <--- FEATURES ---->|<------ LABEL ------>|
      may be used        may NEVER be used
      to build inputs    to build inputs

WHY A FUNCTION AND NOT A CONVENTION
-----------------------------------
It is easy to promise "I only used past data". It is easy to break that
promise by accident, e.g. computing `total_spend` over the whole table and
then filtering rows. Routing every snapshot through one function makes the
guarantee structural rather than a habit, and `tests/test_split.py` asserts
it holds.

OUT-OF-TIME VALIDATION
----------------------
We take two snapshots at different cutoffs. Train on the earlier one, test
on the later one. That is the closest offline imitation of "fit the model
today, run it on customers tomorrow", which is what production does. A
random 80/20 shuffle instead would mix 2011 rows into training and 2010 rows
into test, letting the model learn the future to predict the past.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rci.config import PREDICTION_WINDOW_DAYS, derive_cutoffs


@dataclass
class Snapshot:
    """One point-in-time view of the customer base.

    Attributes
    ----------
    cutoff:
        The "as of" date. Standing here, we can see `observation` and not
        one row more.
    observation:
        Every purchase strictly BEFORE the cutoff. The only legal input to
        feature engineering.
    labels:
        One row per eligible customer: `customer_id`, `repurchased` (0/1),
        and `future_spend` (money in the label window, for the CLV target).
    window_days:
        Length of the label window.
    """

    cutoff: pd.Timestamp
    observation: pd.DataFrame
    labels: pd.DataFrame
    window_days: int

    @property
    def n_customers(self) -> int:
        return len(self.labels)

    @property
    def repurchase_rate(self) -> float:
        return float(self.labels["repurchased"].mean())

    def __str__(self) -> str:
        return (
            f"Snapshot(as of {self.cutoff.date()}, "
            f"{len(self.observation):,} past purchases, "
            f"{self.n_customers:,} customers, "
            f"{self.repurchase_rate:.1%} repurchased in {self.window_days}d)"
        )


def make_snapshot(
    purchases: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_days: int = PREDICTION_WINDOW_DAYS,
) -> Snapshot:
    """Build one leakage-free snapshot.

    Eligible population is customers with at least one purchase before the
    cutoff. A customer whose first ever purchase falls after the cutoff
    cannot be scored, because standing at the cutoff we have never heard of
    them. Including them would quietly inflate the negative class.
    """
    cutoff = pd.Timestamp(cutoff)
    window_end = cutoff + pd.Timedelta(days=window_days)

    observation = purchases[purchases["invoice_date"] < cutoff].copy()
    future = purchases[
        (purchases["invoice_date"] >= cutoff) & (purchases["invoice_date"] < window_end)
    ]

    eligible = pd.Index(observation["customer_id"].unique(), name="customer_id")

    future_spend = (
        future.groupby("customer_id")["line_total"].sum().reindex(eligible).fillna(0.0)
    )

    labels = pd.DataFrame(
        {
            "customer_id": eligible,
            "repurchased": (future_spend > 0).astype(int).to_numpy(),
            "future_spend": future_spend.to_numpy(),
        }
    )

    return Snapshot(
        cutoff=cutoff,
        observation=observation,
        labels=labels,
        window_days=window_days,
    )


def train_test_snapshots(
    purchases: pd.DataFrame,
    window_days: int = PREDICTION_WINDOW_DAYS,
) -> tuple[Snapshot, Snapshot]:
    """The two out-of-time snapshots used by every model in this project."""
    last_date = purchases["invoice_date"].max()
    train_cutoff, test_cutoff = derive_cutoffs(last_date, window_days)
    return (
        make_snapshot(purchases, train_cutoff, window_days),
        make_snapshot(purchases, test_cutoff, window_days),
    )


if __name__ == "__main__":
    from rci.clean import build

    p = build()
    train, test = train_test_snapshots(p)
    print(f"data ends {p['invoice_date'].max().date()}\n")
    print("TRAIN", train)
    print("TEST ", test)

    # The guarantee, checked out loud rather than assumed.
    for name, snap in (("train", train), ("test", test)):
        assert snap.observation["invoice_date"].max() < snap.cutoff
        print(f"  {name}: last observed purchase "
              f"{snap.observation['invoice_date'].max()} < cutoff {snap.cutoff}  OK")
