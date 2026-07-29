"""Phase 4: which twelve products go in this customer's email?

WHY THIS EXISTS (the business question)
---------------------------------------
Segmentation says WHO. CLV says WHAT THEY ARE WORTH. Repurchase says WHEN to
act. None of them says what to actually put in front of the person. That last
step is where a campaign either earns money or annoys people.

The catalogue here has 4,260 products and an email holds about 12. So the job
is to pick 12 out of 4,260, per customer, and to be right often enough that
the email beats the one the merchandising team would have sent by hand.

WHY TWO STAGES
--------------
Scoring all 4,260 products with an expensive model, for every customer, is
both slow and wasteful: almost all of them are obviously irrelevant. Real
recommender systems therefore split the work in two.

    STAGE 1  RETRIEVAL   cheap, approximate, high recall
    ┌────────────────────────────────────────────────────────┐
    │ item2vec: products bought in the same basket end up     │
    │ near each other in 64-dimensional space                 │
    │   customer vector ──► cosine similarity ──► top 200     │
    └────────────────────────────────────────────────────────┘
                              │
    STAGE 2  RANKING     expensive, precise, high precision
    ┌────────────────────────────────────────────────────────┐
    │ XGBoost scores those ~200 using embedding similarity,   │
    │ popularity, purchase history, price fit, co-purchase    │
    │                        ──► final 12, in order           │
    └────────────────────────────────────────────────────────┘

Stage 1 is allowed to be sloppy as long as the right answers are SOMEWHERE in
its 200. Stage 2 cannot recover an item stage 1 never proposed, so the two
stages are measured by different things: recall for the first, precision for
the second.

WHAT ITEM2VEC IS, IN ONE PARAGRAPH
----------------------------------
Word2Vec learns word meanings from the company they keep: words appearing in
similar sentences get similar vectors. item2vec is the same algorithm with
one substitution.

    sentence  ->  a basket (one invoice)
    word      ->  a product

Products that keep turning up in the same baskets end up close together. No
one tells the model what a product IS; it infers similarity purely from
co-occurrence. That is the entire idea, and it is why the same technique works
for words, products, songs, and anything else that appears in groups.

WHAT THE DATA SAID BEFORE ANY MODELLING
---------------------------------------
Measured on the train snapshot, and each number changed a design decision:

  4,260 products, 26,077 baskets, 20.4 products per basket on average
      -> baskets are huge (this is a wholesale gift retailer, not a shop),
         so the Word2Vec `window` must span a whole basket, not 5 neighbours.

  92.2% of baskets contain 2+ products
      -> a single-item basket teaches item2vec nothing; most baskets do teach.

  top 100 products = 31% of all units sold
      -> demand is only moderately concentrated, so "most popular" is a real
         baseline but not an unbeatable one.

  43.1% of a customer's future purchases are items they ALREADY bought
      -> THE finding. "Recommend what they bought last time" is a brutal
         baseline. It is also the reason this module evaluates twice: once on
         all items, and once on NEW items only. A recommender that merely
         replays purchase history scores well and adds nothing.

  6.5% of products bought after the cutoff never appeared before it
      -> a hard ceiling. No embedding exists for an item never seen, so those
         can never be retrieved. Reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from rci.config import RANDOM_SEED

# The size of the final list. Twelve products is roughly one email or one
# "recommended for you" row, so every metric below is reported @12.
TOP_K = 12

# How many candidates stage 1 hands to stage 2. Big enough that the right
# answers are usually inside it, small enough that ranking stays cheap.
N_CANDIDATES = 200


# ---------------------------------------------------------------------------
# stage 1: item2vec
# ---------------------------------------------------------------------------

def basket_sequences(observation: pd.DataFrame) -> list[list[str]]:
    """Turn the ledger into "sentences", one per invoice.

    Each basket becomes a list of product codes. Order inside a basket is
    meaningless (nobody scans items in a considered sequence), which matters
    for the `window` choice in train_item2vec.

    Single-item baskets are kept rather than dropped: gensim ignores them for
    context but they still count toward min_count, so a product only ever
    bought alone correctly ends up too rare to get a vector.
    """
    grouped = observation.groupby("invoice")["stock_code"].apply(list)
    return [items for items in grouped if len(items) >= 1]


@dataclass
class ItemSpace:
    """The learned product space, plus the bookkeeping to use it.

    Attributes
    ----------
    items:
        Product codes, in vector-row order. This IS the index: row 7 of
        `vectors` is `items[7]`.
    vectors:
        (n_items, dim) float32, **L2-normalised**. Normalising once at build
        time means cosine similarity is a plain dot product later, which turns
        the whole retrieval step into one matrix multiply.
    index:
        product code -> row number, so lookups are dict hits.
    """

    items: np.ndarray
    vectors: np.ndarray
    index: dict[str, int]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def __len__(self) -> int:
        return len(self.items)


def train_item2vec(
    sequences: list[list[str]],
    dim: int = 64,
    min_count: int = 5,
    epochs: int = 10,
) -> ItemSpace:
    """Fit Word2Vec over baskets and return a normalised item space.

    EVERY PARAMETER, AND WHY IT IS WHAT IT IS
    -----------------------------------------
    dim=64
        Length of each product's vector. Too small and distinct products get
        squashed together; too large and there is not enough data to fill the
        space meaningfully. 4,260 items is small, so 64 is generous already.

    window=1000
        How many neighbours count as context. The DEFAULT IS 5, and it is
        wrong here. Word2Vec assumes a sentence has meaningful word order, so
        it only looks a few positions either side. A basket has no order at
        all: a product's context is the WHOLE basket. Baskets run to 271
        items, so the window is set past the largest one, making every pair in
        a basket a training pair. Leaving the default at 5 would silently
        train on an arbitrary slice of each basket.

    sg=1 (skip-gram, not CBOW)
        Skip-gram predicts context from one item; CBOW predicts one item from
        averaged context. Skip-gram is better on small data and on rare items,
        and 4,260 products with a long tail is exactly that.

    negative=10
        Negative sampling: for each real pair, contrast it against 10 random
        products. This is what makes training fast, and a slightly higher
        count than default helps on small corpora.

    min_count=5
        A product bought fewer than 5 times has too little evidence for a
        meaningful vector. Those items get no vector and can never be
        retrieved, which is a real coverage limit reported in the results.

    workers=1
        NOT a performance choice. gensim's multi-threaded training is
        non-deterministic: threads consume batches in a racing order, so two
        runs with the same seed give different vectors. One worker makes the
        run reproducible, and 26k baskets trains in seconds anyway.
    """
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences=sequences,
        vector_size=dim,
        window=1000,          # see docstring: a basket has no word order
        min_count=min_count,
        sg=1,                 # skip-gram
        negative=10,
        epochs=epochs,
        workers=1,            # determinism, deliberately over speed
        seed=RANDOM_SEED,
    )

    items = np.array(model.wv.index_to_key)
    vectors = np.asarray(model.wv.vectors, dtype=np.float32)

    # L2-normalise so that cosine(a, b) == dot(a, b). Done once here rather
    # than inside every similarity call.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    return ItemSpace(
        items=items,
        vectors=vectors,
        index={code: i for i, code in enumerate(items)},
    )


def customer_vectors(
    observation: pd.DataFrame,
    space: ItemSpace,
    cutoff: pd.Timestamp,
    half_life_days: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Place each customer in the same space as the products.

    A customer is represented as a WEIGHTED AVERAGE of the products they
    bought. That is the standard trick and it works because the space is
    already arranged so that "similar" means "bought together": averaging a
    customer's items lands them in the neighbourhood they shop in.

    WHY RECENCY-WEIGHTED, NOT A PLAIN MEAN
    --------------------------------------
    Tastes move. A plain average lets a purchase from 2009 count as much as
    one from last week. Each purchase is therefore weighted by

        0.5 ** (days_ago / half_life_days)

    which is exactly a half-life: a purchase 90 days old counts half as much
    as today's, one 180 days old a quarter, and so on. Choosing 90 to match
    the prediction window is a deliberate alignment, not a tuned number.

    Returns (customer_ids, customer_vectors, weights) where `weights` is the
    sparse customer x item matrix, kept because the ranking stage reuses it
    for "have they bought this before" features.
    """
    obs = observation[observation["stock_code"].isin(space.index)].copy()

    days_ago = (cutoff - obs["invoice_date"]).dt.days.to_numpy(dtype=np.float64)
    obs["weight"] = 0.5 ** (days_ago / half_life_days)

    # Collapse to one weight per (customer, item) pair.
    pairs = obs.groupby(["customer_id", "stock_code"])["weight"].sum()

    customer_ids = np.array(sorted(obs["customer_id"].unique()))
    cust_index = {c: i for i, c in enumerate(customer_ids)}

    rows = np.array([cust_index[c] for c, _ in pairs.index], dtype=np.int32)
    cols = np.array([space.index[s] for _, s in pairs.index], dtype=np.int32)

    # A sparse matrix, because 4,951 customers x 4,260 products is 21M cells
    # of which only ~370k are non-zero. Storing the zeros would be 20x the
    # memory for no information.
    weights = sparse.csr_matrix(
        (pairs.to_numpy(dtype=np.float32), (rows, cols)),
        shape=(len(customer_ids), len(space)),
    )

    # The weighted sum of item vectors, as one sparse-dense matrix multiply
    # rather than a Python loop over 4,951 customers.
    vecs = np.asarray(weights @ space.vectors, dtype=np.float32)

    # Normalise so cosine is again a dot product. A customer whose every item
    # fell below min_count has a zero vector; guard the division rather than
    # emitting NaN, and let the coverage report say how many that was.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    return customer_ids, vecs, weights


def retrieve(
    cust_vecs: np.ndarray,
    space: ItemSpace,
    n: int = N_CANDIDATES,
) -> np.ndarray:
    """Stage 1. Top-n most similar products per customer, by cosine.

    Both sides are already L2-normalised, so the entire retrieval is one dot
    product: (customers x dim) @ (dim x items) -> (customers x items).

    DELIBERATE NON-DECISION: no vector database.
    With 4,260 items this multiply takes single-digit milliseconds and fits in
    a few MB. Adding pgvector or FAISS would be infrastructure that solves a
    problem this project does not have, on a Railway plan where RAM is billed.
    The scaling answer belongs in the README, not in the code. Correctly
    DECLINING to add infrastructure is a better engineering signal than adding
    it, and it is also cheaper.

    argpartition, not argsort: we need the top n out of 4,260, and do not care
    about the order of the other 4,048. Partitioning is O(items) where a full
    sort is O(items log items). The n we keep are then sorted properly.
    """
    scores = cust_vecs @ space.vectors.T            # (customers, items)
    n = min(n, scores.shape[1])
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    # Re-sort the surviving n by actual score, best first.
    ordered = np.take_along_axis(
        part, np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1), axis=1
    )
    return ordered


# ---------------------------------------------------------------------------
# supporting signals
# ---------------------------------------------------------------------------

def item_stats(observation: pd.DataFrame, space: ItemSpace) -> pd.DataFrame:
    """Per-product facts the ranker needs: popularity, reach, price.

    Indexed by vector row number so the ranker can look these up positionally
    alongside the embedding scores, with no string joins in the hot path.
    """
    obs = observation[observation["stock_code"].isin(space.index)]
    g = obs.groupby("stock_code")
    stats = g.agg(
        units=("quantity", "sum"),
        buyers=("customer_id", "nunique"),
        baskets=("invoice", "nunique"),
        price=("price", "median"),
    )
    stats = stats.reindex(space.items).fillna(0.0)
    stats.index = np.arange(len(space))
    # log1p because popularity is heavily right-skewed and trees split more
    # usefully on a compressed scale. (Trees are invariant to monotonic
    # transforms, so this is for readability of the SHAP/importance output,
    # not for accuracy.)
    stats["log_units"] = np.log1p(stats["units"].clip(lower=0))
    stats["log_buyers"] = np.log1p(stats["buyers"])
    return stats


def copurchase_matrix(observation: pd.DataFrame, space: ItemSpace) -> sparse.csr_matrix:
    """Item x item co-occurrence counts. The pre-ML recommender, kept as a baseline.

    Built as B.T @ B where B is a basket x item incidence matrix, so cell
    (i, j) is "how many baskets contained both i and j". This is literally
    "customers who bought this also bought that", the technique that ran
    e-commerce before embeddings, and it is a fair fight: it uses the same
    co-occurrence signal item2vec does, just without the compression.

    The diagonal is zeroed so an item never recommends itself.
    """
    obs = observation[observation["stock_code"].isin(space.index)]
    basket_ids = obs["invoice"].astype("category")
    rows = basket_ids.cat.codes.to_numpy()
    cols = np.array([space.index[s] for s in obs["stock_code"]], dtype=np.int32)

    B = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(basket_ids.cat.categories), len(space)),
    )
    B.data[:] = 1.0                      # presence, not quantity
    C = (B.T @ B).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    return C


# ---------------------------------------------------------------------------
# candidates and features
# ---------------------------------------------------------------------------

def build_candidates(
    cust_ids: np.ndarray,
    retrieved: np.ndarray,
    weights: sparse.csr_matrix,
    popular: np.ndarray,
    n_popular: int = 50,
) -> pd.DataFrame:
    """Assemble the candidate set: retrieved + already-bought + popular.

    THREE SOURCES, EACH FOR A REASON
    --------------------------------
    retrieved   what item2vec thinks they will like. The discovery path.
    bought      what they have already bought. 43% of future purchases are
                repeats, so omitting these would throw away the single
                strongest signal in the data before the ranker ever sees it.
    popular     a safety net. If a customer's embedding is unreliable (few
                purchases, rare items), the global bestsellers are still a
                reasonable fallback, and the ranker can learn when to trust
                them.

    Union, not concatenation: an item retrieved AND previously bought must
    appear once, with both facts attached, or the ranker would see duplicates
    and the metrics would double-count.
    """
    frames = []
    for i, cid in enumerate(cust_ids):
        own = weights.indices[weights.indptr[i]: weights.indptr[i + 1]]
        cand = np.union1d(np.union1d(retrieved[i], own), popular[:n_popular])
        frames.append(
            pd.DataFrame({"customer_id": np.full(len(cand), cid, dtype=np.int64),
                          "item_idx": cand.astype(np.int32)})
        )
    return pd.concat(frames, ignore_index=True)


CANDIDATE_FEATURES = [
    "emb_similarity",
    "emb_rank",
    "log_units",
    "log_buyers",
    "price",
    "bought_before",
    "times_bought_before",
    "weight_before",
    "copurchase",
    "copurchase_rank",
    "price_vs_customer_avg",
    "cust_frequency",
    "cust_recency",
]


def candidate_features(
    candidates: pd.DataFrame,
    cust_ids: np.ndarray,
    cust_vecs: np.ndarray,
    space: ItemSpace,
    weights: sparse.csr_matrix,
    stats: pd.DataFrame,
    copurchase: sparse.csr_matrix,
    customer_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach every signal the ranker gets to see, one row per (customer, item).

    The ranker's whole job is to combine signals that individually rank badly.
    Embedding similarity alone ignores that a customer reorders the same box
    of doilies every month. Purchase history alone can never suggest anything
    new. Popularity alone is identical for everybody. Handing all three to a
    tree model lets it learn *when* each one matters.
    """
    cust_row = {c: i for i, c in enumerate(cust_ids)}
    rows = candidates["customer_id"].map(cust_row).to_numpy()
    cols = candidates["item_idx"].to_numpy()

    out = pd.DataFrame(index=candidates.index)

    # --- embedding signal -------------------------------------------------
    # Row-wise dot product of each customer vector with each candidate's
    # vector. einsum does it without materialising the full 4,951 x 4,260.
    out["emb_similarity"] = np.einsum(
        "ij,ij->i", cust_vecs[rows], space.vectors[cols]
    ).astype(np.float32)
    # Rank within the customer, because an absolute cosine of 0.42 means
    # different things for a focused buyer and a scattered one.
    out["emb_rank"] = (
        out.groupby(candidates["customer_id"])["emb_similarity"]
        .rank(ascending=False, method="first")
        .astype(np.float32)
    )

    # --- item popularity --------------------------------------------------
    for col in ("log_units", "log_buyers", "price"):
        out[col] = stats[col].to_numpy()[cols].astype(np.float32)

    # --- this customer's history with this exact item ---------------------
    # weights[row, col] is the recency-decayed purchase weight from
    # customer_vectors(). Non-zero means "they have bought this before".
    own_weight = np.asarray(weights[rows, cols]).ravel().astype(np.float32)
    out["weight_before"] = own_weight
    out["bought_before"] = (own_weight > 0).astype(np.float32)

    counts = weights.copy()
    counts.data[:] = 1.0
    out["times_bought_before"] = np.asarray(
        counts[rows, cols]
    ).ravel().astype(np.float32)

    # --- co-purchase ------------------------------------------------------
    # "People who bought what this customer bought also bought X." Computed
    # per customer against their own basket history.
    cop = np.zeros(len(candidates), dtype=np.float32)
    start = 0
    for cid, group in candidates.groupby("customer_id", sort=False):
        i = cust_row[cid]
        own = weights.indices[weights.indptr[i]: weights.indptr[i + 1]]
        n = len(group)
        if len(own):
            scores = np.asarray(copurchase[own].sum(axis=0)).ravel()
            cop[start: start + n] = scores[group["item_idx"].to_numpy()]
        start += n
    out["copurchase"] = np.log1p(cop)
    out["copurchase_rank"] = (
        out.groupby(candidates["customer_id"])["copurchase"]
        .rank(ascending=False, method="first")
        .astype(np.float32)
    )

    # --- price fit --------------------------------------------------------
    # A customer who buys 50p stocking fillers is a different proposition from
    # one buying GBP 40 lamps, and the embedding does not encode price.
    avg_price = pd.Series(
        np.asarray((weights @ stats["price"].to_numpy()) /
                   np.maximum(np.asarray(weights.sum(axis=1)).ravel(), 1e-9)),
        index=cust_ids,
    )
    out["price_vs_customer_avg"] = (
        out["price"].to_numpy()
        / np.maximum(avg_price.to_numpy()[rows], 1e-9)
    ).astype(np.float32)

    # --- customer context -------------------------------------------------
    # So the ranker can behave differently for a heavy repeat buyer and a
    # near-lapsed one, rather than applying one policy to everybody.
    cf = customer_features.reindex(candidates["customer_id"].to_numpy())
    out["cust_frequency"] = cf["frequency"].to_numpy(dtype=np.float32)
    out["cust_recency"] = cf["recency"].to_numpy(dtype=np.float32)

    return out[CANDIDATE_FEATURES]


def label_candidates(
    candidates: pd.DataFrame,
    future: pd.DataFrame,
    space: ItemSpace,
) -> np.ndarray:
    """1 if the customer actually bought that product in the label window.

    Built from `future`, which comes from the snapshot's label window and is
    therefore the only place in this module allowed to touch post-cutoff rows.
    """
    fut = future[future["stock_code"].isin(space.index)]
    truth = {
        (int(c), space.index[s])
        for c, s in zip(fut["customer_id"], fut["stock_code"])
    }
    return np.fromiter(
        ((int(c), int(i)) in truth
         for c, i in zip(candidates["customer_id"], candidates["item_idx"])),
        dtype=np.int8,
        count=len(candidates),
    )


# ---------------------------------------------------------------------------
# stage 2: the ranker
# ---------------------------------------------------------------------------

def fit_ranker(X, y, groups, X_val=None, y_val=None, groups_val=None):
    """XGBoost learning-to-rank, one group per customer.

    WHY A RANKER AND NOT A CLASSIFIER
    ---------------------------------
    A classifier asks "will this customer buy this product", each row
    independently. But the product is a LIST: what matters is whether the
    right items are above the wrong ones FOR THAT CUSTOMER. Getting every
    probability slightly too low is harmless if the order survives.

    `rank:ndcg` optimises exactly that. It compares pairs of candidates
    WITHIN a group and never across groups, which is why the group sizes have
    to be passed: xgboost needs to know where one customer's candidates end
    and the next customer's begin. Rows must already be sorted by customer,
    and if they are not, the model trains on nonsense without erroring.

    A pointwise classifier is fitted alongside in train_recommend.py, because
    "the fancier objective is better" is a claim worth measuring rather than
    assuming.
    """
    from xgboost import XGBRanker

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@12",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        early_stopping_rounds=30 if X_val is not None else None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    fit_kwargs = {}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [groups_val]
    model.fit(X, y, group=groups, verbose=False, **fit_kwargs)
    return model


def fit_pointwise(X, y, X_val, y_val):
    """The simpler alternative: score each (customer, item) row on its own."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    return model


def top_k_from_scores(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    k: int = TOP_K,
) -> dict[int, list[int]]:
    """Collapse per-row scores into one ordered list of k items per customer."""
    df = candidates[["customer_id", "item_idx"]].copy()
    df["score"] = scores
    df = df.sort_values(["customer_id", "score"], ascending=[True, False])
    return {
        int(cid): g["item_idx"].to_numpy()[:k].tolist()
        for cid, g in df.groupby("customer_id", sort=False)
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _dcg(hits: np.ndarray) -> float:
    """Discounted cumulative gain: a hit at position 1 is worth more than at 12.

    The discount is 1/log2(position + 1), so positions 1, 2, 3 are worth
    1.00, 0.63, 0.50. That decay is the point: an email's first slot gets
    looked at and its twelfth often does not.
    """
    return float(np.sum(hits / np.log2(np.arange(2, len(hits) + 2))))


def evaluate_recommendations(
    recommended: dict[int, list[int]],
    truth: dict[int, set[int]],
    k: int = TOP_K,
) -> dict:
    """Recall, precision, MAP and NDCG at k, averaged over customers.

    HOW TO READ EACH ONE
    --------------------
    recall@k     of everything they went on to buy, what fraction did we put
                 in the list? Capped harshly here: the median customer buys
                 24 distinct products in 90 days and the list holds 12, so
                 even a perfect model cannot exceed 12/24 for them.
    precision@k  of the 12 we showed, how many were bought? This is the one a
                 merchandiser feels.
    map@k        precision averaged over each hit's position, so being right
                 EARLY counts more. Rewards good ordering, not just good
                 membership.
    ndcg@k       same instinct as MAP, with a smoother log discount, and
                 normalised so 1.0 is the best achievable given how many
                 items the customer actually bought.

    Only customers with at least one purchase in the window are scored.
    Including the rest would divide every metric by a constant and make a
    useless model look like a bad one instead of an unmeasurable one.
    """
    rec, prec, ap, ndcg = [], [], [], []

    for cid, actual in truth.items():
        if not actual:
            continue
        preds = recommended.get(cid, [])[:k]
        hits = np.array([1.0 if p in actual else 0.0 for p in preds])

        rec.append(hits.sum() / len(actual))
        prec.append(hits.sum() / k)

        # Average precision: precision measured at each position that hit.
        if hits.sum():
            positions = np.where(hits == 1)[0] + 1
            precisions = np.cumsum(hits)[positions - 1] / positions
            ap.append(precisions.sum() / min(len(actual), k))
        else:
            ap.append(0.0)

        # Ideal DCG: what the score would be if every top slot were a hit,
        # limited by how many items the customer actually bought.
        ideal = _dcg(np.ones(min(len(actual), k)))
        ndcg.append(_dcg(hits) / ideal if ideal > 0 else 0.0)

    return {
        f"recall@{k}": float(np.mean(rec)),
        f"precision@{k}": float(np.mean(prec)),
        f"map@{k}": float(np.mean(ap)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        "customers_scored": len(rec),
    }


def retrieval_recall(
    retrieved: np.ndarray,
    cust_ids: np.ndarray,
    truth: dict[int, set[int]],
) -> float:
    """Stage 1's own metric, and it is NOT the same as the final one.

    Retrieval is judged on whether the right answers are anywhere in its 200,
    because stage 2 can reorder but can never invent. If this number is low,
    no amount of ranking work will help, and that is the diagnosis you want
    before spending a week tuning the ranker.
    """
    scores = []
    row = {c: i for i, c in enumerate(cust_ids)}
    for cid, actual in truth.items():
        if not actual or cid not in row:
            continue
        got = set(retrieved[row[cid]].tolist())
        scores.append(len(actual & got) / len(actual))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def baseline_popular(cust_ids: np.ndarray, popular: np.ndarray, k: int = TOP_K) -> dict:
    """Everyone gets the same bestsellers. Harder to beat than it sounds."""
    top = popular[:k].tolist()
    return {int(c): list(top) for c in cust_ids}


def baseline_repeat(
    cust_ids: np.ndarray,
    weights: sparse.csr_matrix,
    k: int = TOP_K,
) -> dict:
    """Recommend what they already buy, most recent and frequent first.

    THE baseline in retail. 43% of future purchases are repeats, so this is a
    genuinely strong recommender that requires no model at all. Any embedding
    work has to beat it to be worth deploying.
    """
    out = {}
    for i, cid in enumerate(cust_ids):
        lo, hi = weights.indptr[i], weights.indptr[i + 1]
        items, vals = weights.indices[lo:hi], weights.data[lo:hi]
        order = np.argsort(-vals)[:k]
        out[int(cid)] = items[order].tolist()
    return out


def baseline_copurchase(
    cust_ids: np.ndarray,
    weights: sparse.csr_matrix,
    copurchase: sparse.csr_matrix,
    k: int = TOP_K,
    exclude_own: bool = False,
) -> dict:
    """"Customers who bought this also bought that", pre-embedding style."""
    out = {}
    for i, cid in enumerate(cust_ids):
        lo, hi = weights.indptr[i], weights.indptr[i + 1]
        own = weights.indices[lo:hi]
        if not len(own):
            out[int(cid)] = []
            continue
        scores = np.asarray(copurchase[own].sum(axis=0)).ravel()
        if exclude_own:
            scores[own] = -1.0
        out[int(cid)] = np.argsort(-scores)[:k].tolist()
    return out


def popularity_order(observation: pd.DataFrame, space: ItemSpace) -> np.ndarray:
    """Item row numbers, most units sold first."""
    stats = item_stats(observation, space)
    return stats["units"].to_numpy().argsort()[::-1].astype(np.int32)
