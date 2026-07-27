"""Turn raw transactions into a modelling-ready purchase table.

WHY THIS FILE EXISTS
--------------------
Real transaction data is not a list of purchases. It is a ledger, and a
ledger contains corrections. Online Retail II mixes together:

  - real purchases
  - cancellations (invoice number starts with "C", quantity negative)
  - returns and manual adjustments
  - non-product lines: postage, bank charges, samples, discounts, test rows
  - anonymous rows with no customer attached (~22% of the file)

If you model the raw file, "revenue" silently includes refunds, the product
catalogue includes POSTAGE, and the customer count includes nobody.

TWO DECISIONS WORTH DEFENDING IN AN INTERVIEW
---------------------------------------------
1. Rows with no customer_id are dropped, not imputed. You cannot invent a
   customer. They are still real revenue, so they stay valid for product-level
   analysis, but every model here is customer-level.

2. Returns are removed from the purchase table but NOT thrown away. Return
   behaviour is genuinely predictive: a customer who returns half of what
   they buy is worth less than their gross spend suggests. So returns are
   summarised per customer and handed to feature engineering separately.
   Deleting them outright would discard signal.

Every step records how many rows it removed. Knowing what you threw away is
the difference between cleaning and quietly losing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rci.config import CLEAN_PARQUET, NON_PRODUCT_STOCK_CODES


@dataclass
class CleaningReport:
    """Row counts at every step, so the losses are auditable."""

    steps: list[tuple[str, int, int]] = field(default_factory=list)

    def record(self, label: str, before: int, after: int) -> None:
        self.steps.append((label, before, after))

    def __str__(self) -> str:
        lines = [f"{'step':<34}{'rows after':>12}{'removed':>10}"]
        lines.append("-" * 56)
        for label, before, after in self.steps:
            lines.append(f"{label:<34}{after:>12,}{before - after:>10,}")
        return "\n".join(lines)


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Split the ledger into (purchases, returns, report).

    `purchases` is the modelling table: one row per product line actually
    bought, by a known customer, with a positive quantity and price.
    `returns` is the discarded-but-informative negative side.
    """
    report = CleaningReport()
    df = raw.copy()
    n = len(df)
    report.record("raw", n, n)

    # Exact duplicate lines exist in this dataset (same invoice, product,
    # quantity, timestamp). They are data-entry artefacts, not two purchases.
    before, df = len(df), df.drop_duplicates()
    report.record("drop exact duplicates", before, len(df))

    # Nulls in the fields every model depends on.
    before, df = len(df), df.dropna(subset=["invoice_date", "quantity", "price"])
    report.record("drop null date/qty/price", before, len(df))

    # Non-product lines: postage, fees, samples, test rows.
    before = len(df)
    df = df[~df["stock_code"].str.upper().isin({c.upper() for c in NON_PRODUCT_STOCK_CODES})]
    report.record("drop non-product codes", before, len(df))

    # A real product code is 5 digits, optionally followed by letters (85123A).
    # Anything else left at this point is another admin artefact.
    before = len(df)
    df = df[df["stock_code"].str.match(r"^\d{5}\w*$", na=False)]
    report.record("drop malformed stock codes", before, len(df))

    # Customer-level modelling requires a customer.
    before = len(df)
    df = df[df["customer_id"].notna()]
    report.record("drop rows with no customer", before, len(df))

    # Now separate the two sides of the ledger.
    is_cancelled = df["invoice"].str.upper().str.startswith("C")
    is_negative = df["quantity"] <= 0
    returns = df[is_cancelled | is_negative].copy()

    purchases = df[~(is_cancelled | is_negative)].copy()
    report.record("split off returns/cancellations", len(df), len(purchases))

    # Zero and negative prices are giveaways and pricing errors, not sales.
    before = len(purchases)
    purchases = purchases[purchases["price"] > 0]
    report.record("drop non-positive price", before, len(purchases))

    purchases["line_total"] = purchases["quantity"] * purchases["price"]
    returns["line_total"] = returns["quantity"] * returns["price"]

    purchases = purchases.sort_values("invoice_date").reset_index(drop=True)
    return purchases, returns, report


def return_features(returns: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-customer return behaviour, using only returns before `as_of`.

    The `as_of` filter is not optional. These become model features, so they
    must obey the same temporal cutoff as everything else, or we leak.
    """
    window = returns[returns["invoice_date"] < as_of]
    if window.empty:
        return pd.DataFrame(columns=["customer_id", "n_returns", "returned_value"])
    out = (
        window.groupby("customer_id")
        .agg(n_returns=("invoice", "nunique"),
             returned_value=("line_total", lambda s: -s.sum()))
        .reset_index()
    )
    return out


def build(force: bool = False) -> pd.DataFrame:
    """Run the full clean and cache the purchase table."""
    from rci.data import load_raw

    if CLEAN_PARQUET.exists() and not force:
        return pd.read_parquet(CLEAN_PARQUET)

    raw = load_raw()
    purchases, returns, report = clean_transactions(raw)
    purchases.to_parquet(CLEAN_PARQUET, index=False)
    returns.to_parquet(CLEAN_PARQUET.with_name("returns.parquet"), index=False)
    print(report)
    return purchases


if __name__ == "__main__":
    p = build(force=True)
    print(f"\nclean purchases: {len(p):,} rows")
    print(f"customers:       {p['customer_id'].nunique():,}")
    print(f"products:        {p['stock_code'].nunique():,}")
    print(f"date range:      {p['invoice_date'].min()} to {p['invoice_date'].max()}")
