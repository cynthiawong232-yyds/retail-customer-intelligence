"""API contract for the CLV endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rci.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_the_model_comparison_not_a_single_winner(client):
    """The interesting result is that the classic model won, so /health shows
    every contender's MAE rather than one number that hides it."""
    clv = client.get("/health").json()["models"]["clv"]
    assert clv["window_days"] == 90
    mae = clv["mae"]
    assert mae["BG/NBD + Gamma-Gamma"] < mae["XGBoost (raw GBP)"]
    assert mae["BG/NBD + Gamma-Gamma"] < mae["always the mean"]


def test_returns_pounds_a_survival_probability_and_a_decile(client):
    body = client.get("/predict/clv/13069").json()
    assert body["expected_spend_90d"] >= 0
    assert 0.0 <= body["p_still_alive"] <= 1.0
    assert 1 <= body["value_decile"] <= 10


def test_both_models_are_returned_side_by_side(client):
    """Serving only the winner would throw away the finding. The challenger
    is shipped too, with its parts visible."""
    ch = client.get("/predict/clv/13069").json()["challenger"]
    assert ch["model"] == "XGBoost hurdle"
    assert 0.0 <= ch["p_buys_at_all"] <= 1.0
    assert ch["expected_amount_if_they_buy"] >= 0
    # The hurdle IS the product of its two parts, not a separate fit.
    assert ch["estimate"] == pytest.approx(
        ch["p_buys_at_all"] * ch["expected_amount_if_they_buy"], rel=0.02
    )


def test_a_valuable_regular_outranks_a_lapsed_customer(client):
    """13069 buys every ~14 days. 15369 has been silent for 401 days.

    Same pair as the repurchase test, on purpose: CLV must agree with
    repurchase about direction while disagreeing about magnitude, since one
    answers WHETHER and the other WHAT IT IS WORTH.
    """
    good = client.get("/predict/clv/13069").json()
    lapsed = client.get("/predict/clv/15369").json()
    assert good["expected_spend_90d"] > lapsed["expected_spend_90d"]
    assert good["p_still_alive"] > lapsed["p_still_alive"]
    assert good["value_decile"] < lapsed["value_decile"]   # decile 1 is best


def test_the_precompute_and_coverage_limits_are_stated(client):
    """Two real limitations, in every response.

    The endpoint is a lookup for BG/NBD, and BG/NBD cannot score customers
    with no repeat history. A number handed to a budget process without
    either caveat is how a lookup gets mistaken for a live model.
    """
    caveat = client.get("/predict/clv/13069").json()["caveat"]
    assert "precomputed" in caveat
    assert "7.9%" in caveat
    assert "never with zero" in caveat


def test_the_reading_says_ceiling_not_profit(client):
    """Expected spend is revenue, not margin. Saying so is the difference
    between a usable number and one that overstates the retention budget."""
    reading = client.get("/predict/clv/13069").json()["reading"]
    assert "CEILING" in reading
    assert "not a profit number" in reading


def test_unknown_customer_is_404(client):
    assert client.get("/predict/clv/999999").status_code == 404
