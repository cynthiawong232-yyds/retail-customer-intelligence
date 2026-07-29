"""API contract tests.

These run the real app in-process through FastAPI's TestClient, so no server
needs to be started. They cover the things that break silently in serving:
wrong transform, wrong column order, and responses that cannot be interpreted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rci.api import app


@pytest.fixture(scope="module")
def client():
    # `with` triggers the lifespan hook, which is what loads the artifacts.
    # Without it, STATE stays empty and every request 500s.
    with TestClient(app) as c:
        yield c


def test_health_reports_both_dates(client):
    """Two different dates, and conflating them is a real misreading.

    The model was FITTED on the June snapshot. The customer features being
    served are as of September. That is not a mistake; it is what production
    does (fit on history, score today), but the API has to say so or the
    output cannot be interpreted.
    """
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_trained_on_cutoff"] == "2011-06-12"
    assert body["customer_features_as_of"] == "2011-09-10"
    assert body["customers_loaded"] == 5256


def test_segment_returns_a_named_segment(client):
    r = client.post("/segment", json={"recency": 23, "frequency": 12, "monetary": 4848})
    assert r.status_code == 200
    body = r.json()
    assert body["segment_name"] in {
        "Champions",
        "At-risk high-value",
        "New / Low-value",
        "Lapsed / One-off",
    }
    assert 0 <= body["segment_id"] <= 3
    assert body["interpretation"]


def test_response_echoes_its_inputs(client):
    """A label with no inputs shown cannot be debugged.

    Without this you cannot tell a wrong prediction from wrong input data,
    which is most of what goes wrong in production.
    """
    payload = {"recency": 311, "frequency": 11, "monetary": 6251.26}
    body = client.post("/segment", json=payload).json()
    assert body["inputs"]["recency"] == 311
    assert body["inputs"]["frequency"] == 11
    assert body["inputs"]["monetary"] == 6251.26


def test_a_recent_frequent_big_spender_is_a_champion(client):
    """Sanity anchor. If this ever fails, the scaler and model are mismatched.

    The classic silent serving bug is applying a differently-fitted scaler,
    which produces plausible-looking but wrong segments. A customer this
    obviously excellent must land in the top group.
    """
    body = client.post(
        "/segment", json={"recency": 5, "frequency": 40, "monetary": 50000}
    ).json()
    assert body["segment_name"] == "Champions"


def test_a_one_off_ancient_buyer_is_lapsed(client):
    """The opposite anchor."""
    body = client.post(
        "/segment", json={"recency": 500, "frequency": 1, "monetary": 15}
    ).json()
    assert body["segment_name"] == "Lapsed / One-off"


def test_known_customer_lookup(client):
    r = client.get("/customers/15369")
    assert r.status_code == 200
    assert r.json()["inputs"]["monetary"] == pytest.approx(6251.26, rel=1e-4)


def test_unknown_customer_is_404_not_500(client):
    r = client.get("/customers/999999")
    assert r.status_code == 404
    assert "not in the dataset" in r.json()["detail"]


def test_invalid_input_is_rejected_before_the_model(client):
    """Pydantic must reject nonsense rather than let the model answer it.

    A negative recency is impossible. Without validation the model would
    happily return a confident segment for it, which is worse than an error.
    """
    r = client.post("/segment", json={"recency": -5, "frequency": 1, "monetary": 10})
    assert r.status_code == 422

    r = client.post("/segment", json={"recency": 10, "frequency": 0, "monetary": 10})
    assert r.status_code == 422  # zero orders cannot be a customer

    r = client.post("/segment", json={"recency": "yesterday", "frequency": 1, "monetary": 10})
    assert r.status_code == 422


def test_customer_listing_pages(client):
    body = client.get("/customers", params={"limit": 5, "offset": 10}).json()
    assert body["total"] == 5256
    assert len(body["customers"]) == 5
    assert {"customer_id", "recency", "frequency", "monetary"} <= body["customers"][0].keys()


def test_listing_limit_is_capped(client):
    """An unbounded limit is a cheap way for anyone to exhaust the container."""
    body = client.get("/customers", params={"limit": 100000}).json()
    assert len(body["customers"]) <= 100
