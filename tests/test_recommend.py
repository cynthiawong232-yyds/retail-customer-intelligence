"""Recommender invariants.

Most checks run on a small hand-built dataset where the right answer is
obvious by inspection, because a metric computed on 1.4M candidate rows tells
you nothing about whether the metric itself is correct.

The two claims the README makes about results are checked against the SAVED
artifact rather than by retraining, so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rci.recommend import (
    TOP_K,
    baseline_popular,
    baseline_repeat,
    basket_sequences,
    build_candidates,
    copurchase_matrix,
    customer_vectors,
    evaluate_recommendations,
    label_candidates,
    retrieve,
    top_k_from_scores,
    train_item2vec,
)


# Two product families that never appear in the same basket. Fifteen items
# each, and the size is deliberate: train_item2vec uses negative=10, and
# negative sampling draws its "wrong answers" from the vocabulary itself. With
# a six-product catalogue every negative sample is a genuine co-occurrence, so
# the model pushes items apart as often as it pulls them together and learns
# nothing. Vocabulary must be comfortably larger than the negative count.
# (Found by this test failing on a six-item fixture.)
FAMILY_A = [f"A{i:02d}" for i in range(15)]
FAMILY_B = [f"B{i:02d}" for i in range(15)]


@pytest.fixture(scope="module")
def toy():
    """Two customers, thirty products, structure that is obvious by eye.

    Customer 1 only ever buys family A, customer 2 only family B, and no
    basket ever mixes them. Any working embedding must separate the families.
    """
    rng = np.random.default_rng(0)
    rows = []
    date = pd.Timestamp("2011-01-01")
    for basket in range(80):
        family, cid = (FAMILY_A, 1) if basket % 2 == 0 else (FAMILY_B, 2)
        for code in rng.choice(family, size=6, replace=False):
            rows.append(
                {
                    "customer_id": cid,
                    "invoice": f"INV{basket}",
                    "stock_code": code,
                    "quantity": 1,
                    "price": 1.0,
                    "line_total": 1.0,
                    "invoice_date": date + pd.Timedelta(days=basket),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stage 1
# ---------------------------------------------------------------------------

def test_baskets_become_sentences(toy):
    seqs = basket_sequences(toy)
    assert len(seqs) == toy["invoice"].nunique()
    assert all(len(s) == 6 for s in seqs)


def test_embedding_separates_products_that_never_co_occur(toy):
    """The core claim of item2vec, on data where the answer is known.

    A, B, C share every basket. D, E, F share every other basket. The two
    groups never appear together, so within-group similarity must exceed
    across-group similarity. If this fails, nothing downstream is meaningful.
    """
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=40)
    V = space.vectors
    a = [space.index[c] for c in FAMILY_A]
    b = [space.index[c] for c in FAMILY_B]

    within = float(np.mean([V[a] @ V[a].T, V[b] @ V[b].T]))
    across = float(np.mean(V[a] @ V[b].T))
    assert within > across, f"within {within:.3f} not above across {across:.3f}"


def test_vectors_are_unit_length(toy):
    """Normalised at build time so cosine similarity is a plain dot product.
    Every similarity call downstream assumes this."""
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=5)
    norms = np.linalg.norm(space.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_training_is_reproducible(toy):
    """workers=1 is set for determinism, not speed. gensim's multi-threaded
    training races on batch order, so two seeded runs would still differ."""
    seqs = basket_sequences(toy)
    a = train_item2vec(seqs, dim=16, min_count=1, epochs=5)
    b = train_item2vec(seqs, dim=16, min_count=1, epochs=5)
    assert list(a.items) == list(b.items)
    assert np.allclose(a.vectors, b.vectors)


def test_min_count_excludes_rare_items_from_the_space(toy):
    """A product with too little evidence gets no vector, and therefore can
    never be retrieved. A real coverage limit, so it must be deliberate."""
    extra = toy.iloc[[0]].copy()
    extra["stock_code"] = "RARE"
    extra["invoice"] = "INV_RARE"
    space = train_item2vec(basket_sequences(pd.concat([toy, extra])),
                           dim=16, min_count=5, epochs=5)
    assert "RARE" not in space.index
    assert FAMILY_A[0] in space.index


def test_customer_vector_weights_recent_purchases_more():
    """0.5 ** (days_ago / half_life) is a half-life, so a purchase 90 days old
    counts half as much as today's. Checked directly on the weight matrix."""
    cutoff = pd.Timestamp("2011-04-01")
    rows = []
    for basket, (code, when) in enumerate(
        [("A", "2011-03-31"), ("B", "2010-10-02")]      # today-ish, ~180d ago
    ):
        for _ in range(10):
            rows.append({"customer_id": 1, "invoice": f"I{basket}_{_}",
                         "stock_code": code, "quantity": 1, "price": 1.0,
                         "line_total": 1.0, "invoice_date": pd.Timestamp(when)})
    # Pad so both codes clear min_count and get vectors.
    df = pd.DataFrame(rows)
    space = train_item2vec([["A", "B"]] * 20, dim=8, min_count=1, epochs=5)
    _, _, weights = customer_vectors(df, space, cutoff, half_life_days=90.0)

    w = weights.toarray()[0]
    recent, old = w[space.index["A"]], w[space.index["B"]]
    assert recent > old
    # 180 days is two half-lives, so roughly a quarter of the weight.
    assert 0.2 < old / recent < 0.3


def test_retrieval_returns_distinct_items_in_score_order(toy):
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=20)
    cust_ids, vecs, _ = customer_vectors(toy, space, pd.Timestamp("2011-03-01"))
    got = retrieve(vecs, space, n=4)
    assert got.shape == (len(cust_ids), 4)
    for row, vec in zip(got, vecs):
        assert len(set(row.tolist())) == len(row), "duplicate candidates"
        sims = space.vectors[row] @ vec
        assert np.all(np.diff(sims) <= 1e-6), "not sorted best-first"


# ---------------------------------------------------------------------------
# candidates and labels
# ---------------------------------------------------------------------------

def test_candidates_always_include_what_the_customer_already_bought(toy):
    """43% of future purchases are repeats. Dropping previously-bought items
    before the ranker sees them would discard the strongest signal in the data."""
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=5)
    cust_ids, vecs, weights = customer_vectors(toy, space, pd.Timestamp("2011-03-01"))
    retrieved = retrieve(vecs, space, n=2)
    cands = build_candidates(cust_ids, retrieved, weights, np.array([0]), n_popular=1)

    for i, cid in enumerate(cust_ids):
        own = set(weights.indices[weights.indptr[i]: weights.indptr[i + 1]].tolist())
        got = set(cands.loc[cands["customer_id"] == cid, "item_idx"].tolist())
        assert own <= got


def test_candidates_are_deduplicated(toy):
    """An item both retrieved AND previously bought must appear once, or the
    ranker sees duplicates and every metric double-counts."""
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=5)
    cust_ids, vecs, weights = customer_vectors(toy, space, pd.Timestamp("2011-03-01"))
    cands = build_candidates(cust_ids, retrieve(vecs, space, n=6), weights,
                             np.arange(len(space)), n_popular=6)
    assert not cands.duplicated(["customer_id", "item_idx"]).any()


def test_labels_come_only_from_the_future_window(toy):
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=5)
    bought, not_bought = FAMILY_A[0], FAMILY_B[0]
    cands = pd.DataFrame(
        {"customer_id": [1, 1],
         "item_idx": [space.index[bought], space.index[not_bought]]}
    )
    future = toy[(toy["customer_id"] == 1) & (toy["stock_code"] == bought)].head(1)
    y = label_candidates(cands, future, space)
    assert y.tolist() == [1, 0]


def test_copurchase_matrix_is_symmetric_with_no_self_recommendation(toy):
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=5)
    C = copurchase_matrix(toy, space)
    assert (C.diagonal() == 0).all(), "an item must never recommend itself"
    assert (C != C.T).nnz == 0, "co-occurrence counts must be symmetric"
    # Two items from the same family co-occur; across families they never do.
    assert C[space.index[FAMILY_A[0]], space.index[FAMILY_A[1]]] > 0
    assert C[space.index[FAMILY_A[0]], space.index[FAMILY_B[0]]] == 0


# ---------------------------------------------------------------------------
# metrics, checked against hand-computed answers
# ---------------------------------------------------------------------------

def test_metrics_match_arithmetic_done_by_hand():
    """One customer, four actual purchases, hits at positions 1 and 3 of 4."""
    recs = {1: [10, 11, 12, 13]}
    truth = {1: {10, 12, 20, 21}}
    m = evaluate_recommendations(recs, truth, k=4)

    assert m["recall@4"] == pytest.approx(2 / 4)        # 2 of their 4 purchases
    assert m["precision@4"] == pytest.approx(2 / 4)     # 2 of the 4 slots

    # Average precision: hits at positions 1 and 3 -> 1/1 and 2/3, over
    # min(len(actual), k) = 4.
    assert m["map@4"] == pytest.approx((1.0 + 2 / 3) / 4)

    # NDCG: gains at positions 1 and 3, discounted by 1/log2(pos+1).
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    ideal = sum(1 / np.log2(i + 2) for i in range(4))
    assert m["ndcg@4"] == pytest.approx(dcg / ideal)


def test_a_perfect_list_scores_one_and_a_wrong_one_scores_zero():
    perfect = evaluate_recommendations({1: [1, 2, 3]}, {1: {1, 2, 3}}, k=3)
    assert perfect["precision@3"] == pytest.approx(1.0)
    assert perfect["ndcg@3"] == pytest.approx(1.0)
    assert perfect["recall@3"] == pytest.approx(1.0)

    wrong = evaluate_recommendations({1: [7, 8, 9]}, {1: {1, 2, 3}}, k=3)
    assert all(wrong[k] == 0.0 for k in
               ("recall@3", "precision@3", "map@3", "ndcg@3"))


def test_customers_who_bought_nothing_are_excluded_not_scored_zero():
    """Including them would divide every metric by a constant and make an
    unmeasurable customer look like a modelling failure."""
    m = evaluate_recommendations({1: [1], 2: [1]}, {1: {1}, 2: set()}, k=1)
    assert m["customers_scored"] == 1
    assert m["precision@1"] == pytest.approx(1.0)


def test_ordering_changes_map_but_not_recall():
    """The reason MAP exists. Same items, different order: recall is blind to
    position, MAP is not."""
    truth = {1: {1, 2}}
    early = evaluate_recommendations({1: [1, 2, 9, 9]}, truth, k=4)
    late = evaluate_recommendations({1: [9, 9, 1, 2]}, truth, k=4)
    assert early["recall@4"] == late["recall@4"]
    assert early["map@4"] > late["map@4"]
    assert early["ndcg@4"] > late["ndcg@4"]


def test_top_k_from_scores_respects_score_order():
    cands = pd.DataFrame({"customer_id": [1, 1, 1], "item_idx": [5, 6, 7]})
    got = top_k_from_scores(cands, np.array([0.1, 0.9, 0.5]), k=2)
    assert got[1] == [6, 7]


def test_repeat_baseline_only_ever_suggests_owned_items(toy):
    """Which is why it scores exactly 0 on the new-items table. Not a bug in
    the baseline: it is the definition of the baseline."""
    space = train_item2vec(basket_sequences(toy), dim=16, min_count=1, epochs=5)
    cust_ids, _, weights = customer_vectors(toy, space, pd.Timestamp("2011-03-01"))
    recs = baseline_repeat(cust_ids, weights, k=TOP_K)
    for i, cid in enumerate(cust_ids):
        own = set(weights.indices[weights.indptr[i]: weights.indptr[i + 1]].tolist())
        assert set(recs[int(cid)]) <= own


def test_popular_baseline_gives_everyone_the_same_list(toy):
    recs = baseline_popular(np.array([1, 2, 3]), np.arange(50), k=5)
    assert len({tuple(v) for v in recs.values()}) == 1


# ---------------------------------------------------------------------------
# the claims the README makes, read from the trained artifact
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    import joblib

    from rci.config import ARTIFACTS

    path = ARTIFACTS / "recommend.joblib"
    if not path.exists():
        pytest.skip("run `python -m rci.train_recommend` first")
    return joblib.load(path)


def test_two_stage_beats_every_baseline_on_discovery(artifact):
    """The finding that justifies the whole phase.

    On ALL items the two-stage model barely beats 'repeat what they bought'.
    On NEW items, where repeat scores 0 by construction, embeddings are worth
    roughly 2.4x most-popular. If this ever inverts, the README is wrong.
    """
    new = artifact["metrics_new"]
    best = new["two-stage (pointwise)"]["precision@12"]
    assert best > new["most popular"]["precision@12"] * 2
    assert best > new["co-purchase"]["precision@12"]
    assert best > new["stage 1 only (embeddings)"]["precision@12"]
    assert new["repeat what they bought"]["precision@12"] == 0.0


def test_the_ranker_barely_beats_the_repeat_rule_on_all_items(artifact):
    """Reported honestly rather than buried. The aggregate table is dominated
    by reorders, so a two-stage model looks only marginally better there."""
    allm = artifact["metrics_all"]
    model = allm["two-stage (pointwise)"]["precision@12"]
    rule = allm["repeat what they bought"]["precision@12"]
    assert model > rule
    assert (model - rule) / rule < 0.15, "the gap is now large; update the README"


def test_the_ranking_objective_did_not_beat_the_simple_one(artifact):
    """rank:ndcg optimises the thing we measure, and still did not win.

    Worth pinning because the intuitive expectation is the opposite, and
    'I measured it instead of assuming' is the point of the comparison.
    """
    allm = artifact["metrics_all"]
    assert allm["two-stage (pointwise)"]["ndcg@12"] >= allm["two-stage (rank:ndcg)"]["ndcg@12"]


def test_retrieval_recall_bounds_the_final_recall(artifact):
    """Stage 2 can reorder but never invent, so the candidate set is a hard
    ceiling on anything the full system can achieve."""
    ceiling = artifact["candidate_ceiling"]
    final = artifact["metrics_all"]["two-stage (pointwise)"]["recall@12"]
    assert final < ceiling
    assert ceiling >= artifact["retrieval_recall"]
