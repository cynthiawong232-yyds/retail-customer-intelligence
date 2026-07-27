"""Cleaning invariants.

Each assertion here corresponds to a documented cleaning decision. If one
fails, either the upstream data changed or a rule was silently dropped, and
both are worth failing a build over.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rci.clean import clean_transactions, return_features
from rci.config import NON_PRODUCT_STOCK_CODES


@pytest.fixture(scope="module")
def cleaned():
    from rci.data import load_raw

    return clean_transactions(load_raw())


def test_no_missing_customers(cleaned):
    purchases, _, _ = cleaned
    assert purchases["customer_id"].notna().all()


def test_all_quantities_and_prices_positive(cleaned):
    purchases, _, _ = cleaned
    assert (purchases["quantity"] > 0).all()
    assert (purchases["price"] > 0).all()


def test_no_cancellation_invoices_in_purchases(cleaned):
    purchases, _, _ = cleaned
    assert not purchases["invoice"].str.upper().str.startswith("C").any()


def test_no_non_product_stock_codes(cleaned):
    purchases, _, _ = cleaned
    banned = {c.upper() for c in NON_PRODUCT_STOCK_CODES}
    assert not purchases["stock_code"].str.upper().isin(banned).any()


def test_stock_codes_are_well_formed(cleaned):
    """5 digits, optionally followed by letters. 85123A is valid, POST is not."""
    purchases, _, _ = cleaned
    assert purchases["stock_code"].str.match(r"^\d{5}\w*$").all()


def test_line_total_is_consistent(cleaned):
    purchases, _, _ = cleaned
    recomputed = purchases["quantity"] * purchases["price"]
    assert ((purchases["line_total"] - recomputed).abs() < 1e-6).all()


def test_returns_are_kept_not_discarded(cleaned):
    """Return behaviour is signal, so the returns frame must be populated."""
    _, returns, _ = cleaned
    assert len(returns) > 0
    assert (returns["quantity"] <= 0).all() | returns["invoice"].str.upper().str.startswith("C").all()


def test_return_features_respect_the_cutoff():
    """Return features are model inputs, so they obey the same time rule."""
    returns = pd.DataFrame(
        {
            "customer_id": [1, 1],
            "invoice": ["C1", "C2"],
            "invoice_date": pd.to_datetime(["2011-01-01", "2011-12-01"]),
            "line_total": [-10.0, -50.0],
        }
    )
    feats = return_features(returns, as_of=pd.Timestamp("2011-06-01"))
    assert len(feats) == 1
    # Only the January return is visible; the December one is in the future.
    assert feats.loc[0, "n_returns"] == 1
    assert feats.loc[0, "returned_value"] == 10.0
