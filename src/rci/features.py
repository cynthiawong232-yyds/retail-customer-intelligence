"""Turn transactions into one row per customer. Shared by every model.

WHY THIS FILE EXISTS
--------------------
Models cannot read a ledger. They read a table where one row means one thing
you want to predict about. Here that thing is a customer, so 776,577
transaction rows become 4,951 customer rows.

    TRANSACTION GRAIN                    CUSTOMER GRAIN
    one row per product line     ──►     one row per person
    776,577 rows x 9 columns             4,951 rows x ~13 columns

This is also THE place leakage gets introduced, because it is the first place
numbers are *computed* rather than merely read. `monetary` does not exist in
the source file; we invent it by summing. If we sum the wrong rows, every
model downstream is wrong and none of them will complain.

The defence is structural: this module takes a `Snapshot`, and a Snapshot only
ever hands out `observation`, which is already filtered to strictly before the
cutoff. There is no code path here that can reach a future row, because this
module never touches the full transaction table at all.

THE FEATURES, AND WHY EACH ONE
------------------------------
RFM is the backbone. It is 60 years old, predates machine learning, and still
works because it encodes three genuinely different things:

  recency    when did they last buy      -> are they drifting away?
  frequency  how often do they buy       -> is this a habit or a one-off?
  monetary   how much have they spent    -> do they matter commercially?

Two customers can share a monetary value for opposite reasons: one big order,
or thirty small ones. Frequency separates them. Everything else here is a
refinement of those three.
"""

from __future__ import annotations

import pandas as pd

from rci.clean import return_features
from rci.config import CLEAN_PARQUET
from rci.split import Snapshot

# The exact feature list, in the exact order, defined once.
#
# Order matters more than it looks. At serving time the API will build a numpy
# array by hand and hand it to the model. If the columns arrive in a different
# order than training, xgboost will happily predict nonsense: it does not check
# names, only positions. Defining the order here and importing it everywhere is
# what stops that.
FEATURE_COLUMNS = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure",
    "n_products",
    "n_items",
    "avg_basket_value",
    "avg_days_between_orders",
    "spend_last_90d",
    "orders_last_90d",
    "n_returns",
    "returned_value",
]

# Segmentation deliberately uses only the classic three. More dimensions make
# clusters harder to name, and a cluster you cannot name is a cluster nobody
# in marketing will ever use.
RFM_COLUMNS = ["recency", "frequency", "monetary"]


def load_returns() -> pd.DataFrame:
    """The returns/cancellations table that clean.build() wrote alongside."""
    path = CLEAN_PARQUET.with_name("returns.parquet")
    if not path.exists():
        from rci.clean import build

        build(force=True)
    return pd.read_parquet(path)


def build_features(snap: Snapshot, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per eligible customer, computed only from before the cutoff.

    Returns a DataFrame indexed by customer_id, with rows in exactly the same
    order as `snap.labels`. That alignment is not cosmetic: X and y are handed
    to the model as two separate objects, and if row 7 of X is a different
    customer than row 7 of y, the model learns garbage and nothing errors.
    """
    obs = snap.observation
    cutoff = snap.cutoff

    # ---- the core aggregation ------------------------------------------
    # groupby("customer_id") collapses each customer's many transaction rows
    # into one. Everything in .agg() is a summary of that person's history.
    #
    # Note "invoice": "nunique" and NOT "count". A single order that contained
    # 30 different products appears as 30 rows. Counting rows would call that
    # customer a 30-time buyer. nunique counts distinct invoices, i.e. actual
    # shopping trips. This one choice changes `frequency` by roughly 20x.
    g = obs.groupby("customer_id")
    f = g.agg(
        frequency=("invoice", "nunique"),          # distinct orders, not rows
        monetary=("line_total", "sum"),            # total money spent, ever
        n_products=("stock_code", "nunique"),      # breadth of taste
        n_items=("quantity", "sum"),               # total units carried out
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    )

    # ---- time-based features -------------------------------------------
    # Measured from the CUTOFF, never from "today" and never from the end of
    # the dataset. The model is pretending to stand on the cutoff date, so
    # "days since last purchase" has to be measured from there or the number
    # would be impossible to know at prediction time.
    #
    # .dt.days converts a timedelta ("94 days 07:13:00") into a plain integer.
    f["recency"] = (cutoff - f["last_purchase"]).dt.days
    f["tenure"] = (cutoff - f["first_purchase"]).dt.days

    # ---- ratios ---------------------------------------------------------
    # frequency is always >= 1 here (you cannot be in this table without an
    # order), so these divisions are safe without a guard.
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    f["avg_basket_value"] = f["n_items"] / f["frequency"]

    # Their natural buying rhythm. A customer with 10 orders over 464 days
    # buys roughly every 46 days, so a 95-day silence means something. The
    # same silence from someone who buys twice a year means nothing.
    #
    # frequency - 1 because N orders have N-1 gaps between them. Customers
    # with exactly one order have zero gaps, so this is NaN for them, which
    # is honest: they have no rhythm yet. XGBoost handles NaN natively.
    # KMeans does not, which is one reason segmentation uses RFM only.
    gaps = f["frequency"] - 1
    f["avg_days_between_orders"] = (f["tenure"] / gaps).where(gaps > 0)

    # ---- recent-window features ----------------------------------------
    # RFM describes a whole lifetime, which hides direction of travel. A
    # customer who spent 8k over two years but nothing in the last quarter
    # looks identical to a steady one on `monetary` alone. These two columns
    # are what let a model see the difference.
    recent_start = cutoff - pd.Timedelta(days=90)
    recent = obs[obs["invoice_date"] >= recent_start]
    r = recent.groupby("customer_id").agg(
        spend_last_90d=("line_total", "sum"),
        orders_last_90d=("invoice", "nunique"),
    )
    # Customers with no recent activity are absent from `recent`, so the join
    # leaves NaN. Zero is the correct value: they spent nothing, which is a
    # fact, not missing information.
    f = f.join(r, how="left")
    f[["spend_last_90d", "orders_last_90d"]] = f[
        ["spend_last_90d", "orders_last_90d"]
    ].fillna(0.0)

    # ---- return behaviour ----------------------------------------------
    # Returns were separated out during cleaning rather than deleted, because
    # a customer who sends half their order back is worth less than their
    # gross spend suggests. return_features() applies the same cutoff rule.
    if returns is None:
        returns = load_returns()
    rf = return_features(returns, as_of=cutoff).set_index("customer_id")
    f = f.join(rf, how="left")
    f[["n_returns", "returned_value"]] = f[["n_returns", "returned_value"]].fillna(0.0)

    # ---- align to the labels -------------------------------------------
    # reindex forces the rows into exactly the label order. Any customer in
    # labels but missing here would surface as an all-NaN row rather than
    # silently shifting every subsequent row up by one.
    f = f.reindex(snap.labels["customer_id"].to_numpy())
    f.index.name = "customer_id"

    return f[FEATURE_COLUMNS]


def build_xy(snap: Snapshot, target: str = "repurchased", returns=None):
    """Return (X, y), the two objects every scikit-learn style model wants.

    X : DataFrame, 4,951 rows x 13 columns. The inputs. Capital because it is
        a matrix, which is the convention across all of scikit-learn.
    y : Series, 4,951 values. The answer. Lowercase because it is a vector.

    target="repurchased"   -> y is 0/1, for the classifier  (Phase 3)
    target="future_spend"  -> y is pounds, for the regressor (Phase 2)
    """
    X = build_features(snap, returns=returns)
    y = snap.labels.set_index("customer_id")[target].reindex(X.index)
    return X, y


if __name__ == "__main__":
    from rci.clean import build
    from rci.split import train_test_snapshots

    purchases = build()
    train, test = train_test_snapshots(purchases)

    X_train, y_train = build_xy(train)
    X_test, y_test = build_xy(test)

    print(f"X_train {X_train.shape}   y_train {y_train.shape}")
    print(f"X_test  {X_test.shape}   y_test  {y_test.shape}")
    print(f"\npositive rate: train {y_train.mean():.1%}   test {y_test.mean():.1%}")
    print("\nFEATURE SUMMARY (train)")
    print(X_train.describe().T[["mean", "50%", "min", "max"]].round(2).to_string())
    print("\nMISSING VALUES")
    missing = X_train.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  none")
