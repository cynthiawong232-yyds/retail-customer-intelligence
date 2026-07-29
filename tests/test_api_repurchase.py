"""API contract for the repurchase endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rci.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_lists_both_models(client):
    body = client.get("/health").json()
    assert body["models"]["segmentation"]["algorithm"] == "KMeans"
    rep = body["models"]["repurchase"]
    assert rep["algorithm"] == "XGBoost"
    assert rep["trees"] > 0
    assert rep["pr_auc"] > rep["base_rate"], "model must beat a useless one"


def test_predict_returns_a_probability_and_decile(client):
    body = client.get("/predict/repurchase/13069").json()
    assert 0.0 <= body["probability"] <= 1.0
    assert 1 <= body["decile"] <= 10


def test_a_frequent_recent_buyer_outranks_a_lapsed_one(client):
    """13069 buys every ~14 days. 15369 has been silent for 401 days.

    The ordering is the product. If this ever flips, something is wired
    wrong regardless of what the aggregate metrics say.
    """
    good = client.get("/predict/repurchase/13069").json()
    lapsed = client.get("/predict/repurchase/15369").json()
    assert good["probability"] > lapsed["probability"]
    assert good["decile"] < lapsed["decile"]   # decile 1 is the best


def test_explanations_are_present_and_signed(client):
    """A score with no explanation is not actionable by whoever receives it."""
    body = client.get("/predict/repurchase/15369").json()
    drivers = body["top_drivers"]
    assert len(drivers) == 5
    assert {"feature", "value", "effect_log_odds", "direction"} <= drivers[0].keys()
    # Sorted by absolute impact, strongest first.
    magnitudes = [abs(d["effect_log_odds"]) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # For a customer silent 401 days, recency must be the thing dragging them down.
    recency = next(d for d in drivers if d["feature"] == "recency")
    assert recency["direction"] == "lowers"


def test_the_calibration_caveat_is_always_stated(client):
    """The model under-predicts on this period and the response must say so.

    Shipping an uncalibrated probability without the warning is how a number
    ends up multiplied by a budget in somebody's spreadsheet.
    """
    body = client.get("/predict/repurchase/13069").json()
    assert "uncalibrated" in body["caveat"]
    assert "32.3%" in body["caveat"] and "43.5%" in body["caveat"]


def test_unknown_customer_is_404(client):
    assert client.get("/predict/repurchase/999999").status_code == 404
