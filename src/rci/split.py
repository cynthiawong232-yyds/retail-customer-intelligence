"""The temporal split. The most important file in this repo.

VOCABULARY (read this first)
----------------------------
FEATURE   An input the model learns from. One number per customer, e.g.
          "days since last purchase = 71". Features do NOT exist yet in
          this file; they are built in Phase 1. What this file produces is
          the pool of transactions features are *allowed* to be built from.

LABEL     The answer we are trying to predict. Also called the target.
          This file creates two:
            repurchased  = 1 or 0   did they buy in the next 90 days
            future_spend = pounds   how much they spent in those 90 days
          The repurchase model predicts the first, CLV predicts the second.

GRAIN     What one row means. The raw data is TRANSACTION grain (one row per
          product line on an invoice). Models need CUSTOMER grain (one row
          per person). That conversion is Phase 1's job.

TRAIN     The data the model learns from. It sees both features and labels.

TEST      The data the model is GRADED on. It sees the features; we hide the
          labels and use them to mark the exam. Test is for EVALUATION, not
          for prediction. Test labels exist and we know them.

PRODUCTION  The real job, later. Features exist, labels do not exist yet,
          because the 90 days have not happened. That asymmetry is the whole
          reason train and test must be separated by time and not at random.

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
    # Accept a string like "2011-06-12" or a Timestamp; normalise to one type
    # so every comparison below is timestamp-vs-timestamp.
    cutoff = pd.Timestamp(cutoff)

    # The far edge of the label window. cutoff=2011-06-12 + 90d = 2011-09-10.
    window_end = cutoff + pd.Timedelta(days=window_days)

    # ---- THE PAST -------------------------------------------------------
    # Every purchase STRICTLY before the cutoff ("<", never "<=").
    # This is the ONLY table feature engineering is allowed to read.
    # .copy() because we hand this out to callers who will add columns to it,
    # and modifying a slice of the original would be a pandas trap.
    observation = purchases[purchases["invoice_date"] < cutoff].copy()

    # ---- THE FUTURE -----------------------------------------------------
    # Purchases inside the 90-day label window: on/after the cutoff, before
    # the window closes. This is where the ANSWER comes from, and no feature
    # may ever touch it.
    #   >= cutoff    the cutoff day itself belongs to the future
    #   <  window_end   anything past 90 days is beyond the question we asked
    future = purchases[
        (purchases["invoice_date"] >= cutoff) & (purchases["invoice_date"] < window_end)
    ]

    # ---- WHO WE ARE ALLOWED TO SCORE ------------------------------------
    # Only customers we had actually met by the cutoff. A customer whose very
    # first purchase lands after the cutoff is invisible to us on that day, so
    # predicting for them is meaningless. Note this comes from `observation`,
    # not from `purchases`: that single choice is what enforces the rule.
    eligible = pd.Index(observation["customer_id"].unique(), name="customer_id")

    # ---- LABEL 1: how much money, used by the CLV model ------------------
    # groupby(...).sum()  total spend per customer inside the label window
    # .reindex(eligible)  force the result to be exactly our eligible list,
    #                     which INSERTS missing customers as NaN
    # .fillna(0.0)        a customer who bought nothing spent 0, not "unknown"
    #
    # reindex is the important step. Without it, customers who never came back
    # would simply be absent, and we would silently only score returners,
    # which is the population that makes any model look brilliant.
    future_spend = (
        future.groupby("customer_id")["line_total"].sum().reindex(eligible).fillna(0.0)
    )

    # ---- LABEL 2: yes/no, used by the repurchase model -------------------
    # `repurchased` is just "did future_spend exceed zero", as 1/0 rather than
    # True/False because scikit-learn and xgboost both expect numeric targets.
    # .to_numpy() drops the pandas index so the columns align positionally
    # with `eligible` instead of being re-matched by index label.
    labels = pd.DataFrame(
        {
            "customer_id": eligible,
            "repurchased": (future_spend > 0).astype(int).to_numpy(),
            "future_spend": future_spend.to_numpy(),
        }
    )

    # Return the past, the answers, and the date they are separated by, as one
    # object. Bundling them means no downstream code can accidentally pair the
    # observation window from one cutoff with the labels from another.
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
