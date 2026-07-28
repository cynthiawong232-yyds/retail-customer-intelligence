"""Walk one real customer through the temporal split, on the console.

WHY THIS FILE EXISTS
--------------------
The split is easy to describe and hard to believe until you watch it happen
to somebody real. This prints an actual customer's purchase timeline with the
cutoff drawn through it, so you can see which purchases become FEATURES and
which become the ANSWER.

Run:
    python -m rci.explain              # a customer who came back
    python -m rci.explain --lapsed     # one who did not
    python -m rci.explain --id 13085   # a specific customer
"""

from __future__ import annotations

import argparse

import pandas as pd

from rci.clean import build
from rci.split import train_test_snapshots


def describe_customer(snap, purchases: pd.DataFrame, customer_id: int) -> None:
    cutoff = snap.cutoff
    window_end = cutoff + pd.Timedelta(days=snap.window_days)

    mine = purchases[purchases["customer_id"] == customer_id].sort_values("invoice_date")
    past = mine[mine["invoice_date"] < cutoff]
    future = mine[(mine["invoice_date"] >= cutoff) & (mine["invoice_date"] < window_end)]
    label = snap.labels.set_index("customer_id").loc[customer_id]

    print(f"\nCUSTOMER {customer_id}")
    print(f"cutoff {cutoff.date()}   label window {cutoff.date()} to {window_end.date()}")
    print("=" * 70)

    # One line per order, not per product line, so the timeline is readable.
    orders = (
        mine.groupby([mine["invoice_date"].dt.date, "invoice"])["line_total"]
        .sum()
        .reset_index()
        .rename(columns={"invoice_date": "date", "line_total": "value"})
    )

    print("\nTHE PAST  (features may be computed from these)")
    print("-" * 70)
    shown = orders[pd.to_datetime(orders["date"]) < cutoff]
    for _, r in shown.tail(8).iterrows():
        print(f"   {r['date']}   invoice {r['invoice']:<8} GBP {r['value']:>9,.2f}")
    if len(shown) > 8:
        print(f"   ... and {len(shown) - 8} earlier orders")

    print(f"\n{'':>3}{'~' * 64}")
    print(f"   CUTOFF {cutoff.date()}  <- the model stands here")
    print(f"{'':>3}{'~' * 64}\n")

    print("THE FUTURE  (the answer; no feature may ever touch these)")
    print("-" * 70)
    fut = orders[
        (pd.to_datetime(orders["date"]) >= cutoff)
        & (pd.to_datetime(orders["date"]) < window_end)
    ]
    if fut.empty:
        print("   (nothing - this customer did not come back)")
    else:
        for _, r in fut.iterrows():
            print(f"   {r['date']}   invoice {r['invoice']:<8} GBP {r['value']:>9,.2f}")

    # The features that Phase 1 will build, computed here by hand so you can
    # see they use ONLY the rows above the cutoff line.
    print("\nWHAT THE MODEL WILL SEE  (features, from the past only)")
    print("-" * 70)
    recency = (cutoff - past["invoice_date"].max()).days
    frequency = past["invoice"].nunique()
    monetary = past["line_total"].sum()
    tenure = (cutoff - past["invoice_date"].min()).days
    print(f"   recency          {recency:>10,} days since last purchase")
    print(f"   frequency        {frequency:>10,} orders")
    print(f"   monetary         {monetary:>10,.2f} GBP lifetime spend")
    print(f"   tenure           {tenure:>10,} days since first purchase")
    print(f"   avg_order_value  {monetary / frequency:>10,.2f} GBP")

    print("\nTHE ANSWER  (labels, from the future only)")
    print("-" * 70)
    print(f"   repurchased      {int(label['repurchased']):>10}   <- repurchase model predicts this")
    print(f"   future_spend     {label['future_spend']:>10,.2f}   <- CLV model predicts this")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=None, help="specific customer_id")
    ap.add_argument("--lapsed", action="store_true", help="pick one who did NOT return")
    args = ap.parse_args()

    purchases = build()
    train, _ = train_test_snapshots(purchases)

    if args.id is not None:
        customer_id = args.id
    else:
        # Pick a customer with enough history to be interesting: several past
        # orders, and the requested outcome.
        want = 0 if args.lapsed else 1
        counts = train.observation.groupby("customer_id")["invoice"].nunique()
        busy = counts[(counts >= 4) & (counts <= 12)].index
        pool = train.labels[
            train.labels["customer_id"].isin(busy)
            & (train.labels["repurchased"] == want)
        ]
        customer_id = int(pool.iloc[len(pool) // 2]["customer_id"])

    describe_customer(train, purchases, customer_id)


if __name__ == "__main__":
    main()
