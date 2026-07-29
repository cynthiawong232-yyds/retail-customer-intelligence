"""API contract for the recommender endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rci.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_the_recommender(client):
    rec = client.get("/health").json()["models"]["recommender"]
    assert "item2vec" in rec["algorithm"]
    assert rec["embedding_dim"] == 64
    assert rec["products_with_vectors"] > 3000
    assert rec["top_k"] == 12


def test_returns_twelve_products_with_descriptions(client):
    body = client.get("/recommend/13069").json()
    recs = body["recommendations"]
    assert len(recs) == 12
    assert [r["rank"] for r in recs] == list(range(1, 13))
    assert all(r["description"] for r in recs)
    # No duplicates: a list that repeats a product is an obvious defect that
    # aggregate metrics would never show.
    assert len({r["stock_code"] for r in recs}) == 12


def test_each_item_is_labelled_reorder_or_discovery(client):
    """The distinction Phase 4 exists to make. A list that is entirely
    reorders is a SQL query, and the response has to admit which it is."""
    body = client.get("/recommend/13069").json()
    recs = body["recommendations"]
    assert body["n_new_to_customer"] == sum(1 for r in recs if not r["bought_before"])
    assert "reorders" in body["reading"]


def test_the_honest_comparison_is_in_the_caveat(client):
    caveat = client.get("/recommend/13069").json()["caveat"]
    assert "0.238 vs 0.225" in caveat      # barely beats the repeat rule
    assert "0.048 vs 0.020" in caveat      # but is worth real money on new items


def test_similar_items_are_semantically_related(client):
    """The live endpoint, on a product whose neighbours are checkable by eye.

    22423 is REGENCY CAKESTAND 3 TIER. Its neighbours should be the matching
    Regency tea service, learned purely from basket co-occurrence with no
    product taxonomy anywhere in the pipeline.
    """
    body = client.get("/similar/22423?k=5").json()
    assert len(body) == 5
    assert all(0.0 <= item["similarity"] <= 1.0 for item in body)
    # Sorted best-first.
    sims = [item["similarity"] for item in body]
    assert sims == sorted(sims, reverse=True)
    # An item is never its own neighbour.
    assert all(item["stock_code"] != "22423" for item in body)
    assert any("REGENCY" in item["description"].upper() for item in body)


def test_similar_respects_k_and_clamps_it(client):
    assert len(client.get("/similar/22423?k=3").json()) == 3
    assert len(client.get("/similar/22423?k=999").json()) == 50


def test_unknown_ids_are_404(client):
    assert client.get("/recommend/999999").status_code == 404
    assert client.get("/similar/NOT_A_PRODUCT").status_code == 404
