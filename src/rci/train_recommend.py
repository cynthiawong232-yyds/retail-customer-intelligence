"""Train and evaluate the two-stage recommender.

    python -m rci.train_recommend

Evaluated TWICE on purpose, and the second table is the one that matters:

  ALL ITEMS   includes products the customer has bought before. 43% of future
              purchases are repeats, so a system that only replays purchase
              history scores well here while adding nothing a SQL query could
              not do.

  NEW ITEMS   products the customer has never bought, with already-bought
              items removed from every method's candidate list. This is
              DISCOVERY, and it is the only part where an embedding model can
              justify its existence.

Reporting only the first table is the standard way recommender results get
oversold. Reporting both is the honest version and the better interview
answer.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from rci.clean import build
from rci.config import ARTIFACTS, PREDICTION_WINDOW_DAYS, RANDOM_SEED
from rci.features import build_features
from rci.recommend import (
    CANDIDATE_FEATURES,
    N_CANDIDATES,
    TOP_K,
    baseline_copurchase,
    baseline_popular,
    baseline_repeat,
    basket_sequences,
    build_candidates,
    candidate_features,
    copurchase_matrix,
    customer_vectors,
    evaluate_recommendations,
    fit_pointwise,
    fit_ranker,
    item_stats,
    label_candidates,
    popularity_order,
    retrieval_recall,
    retrieve,
    top_k_from_scores,
    train_item2vec,
)
from rci.split import Snapshot, train_test_snapshots


def future_rows(purchases: pd.DataFrame, snap: Snapshot) -> pd.DataFrame:
    """The label window's transactions. The ONLY post-cutoff data used here."""
    end = snap.cutoff + pd.Timedelta(days=snap.window_days)
    return purchases[
        (purchases["invoice_date"] >= snap.cutoff) & (purchases["invoice_date"] < end)
    ]


def prepare(snap: Snapshot, space, copurch, purchases: pd.DataFrame):
    """Everything one snapshot needs: candidates, features, labels, truth."""
    cust_ids, cust_vecs, weights = customer_vectors(snap.observation, space, snap.cutoff)
    retrieved = retrieve(cust_vecs, space, N_CANDIDATES)
    popular = popularity_order(snap.observation, space)
    stats = item_stats(snap.observation, space)

    candidates = build_candidates(cust_ids, retrieved, weights, popular)
    feats = candidate_features(
        candidates, cust_ids, cust_vecs, space, weights, stats, copurch,
        build_features(snap),
    )
    future = future_rows(purchases, snap)
    y = label_candidates(candidates, future, space)

    # Ground truth per customer, in item-row space so everything downstream
    # compares integers rather than strings.
    fut = future[future["stock_code"].isin(space.index)]
    truth = {}
    for cid, g in fut.groupby("customer_id"):
        truth[int(cid)] = {space.index[s] for s in g["stock_code"].unique()}

    return dict(
        cust_ids=cust_ids, cust_vecs=cust_vecs, weights=weights,
        retrieved=retrieved, popular=popular, stats=stats,
        candidates=candidates, X=feats, y=y, truth=truth,
    )


def owned_sets(cust_ids, weights) -> dict[int, set[int]]:
    """What each customer had already bought before the cutoff."""
    return {
        int(c): set(weights.indices[weights.indptr[i]: weights.indptr[i + 1]].tolist())
        for i, c in enumerate(cust_ids)
    }


def main() -> None:
    pd.set_option("display.width", 200)
    purchases = build()
    train, test = train_test_snapshots(purchases)

    # --- stage 1: learn the product space -------------------------------
    # Fitted ONLY on the train snapshot's observation window, and reused for
    # the test snapshot rather than refitted. Two reasons, both deliberate:
    # Word2Vec runs are not aligned across fits (the space rotates), so a
    # ranker trained on one geometry cannot be applied to another; and in
    # production nobody retrains embeddings daily, so a slightly stale space
    # is the realistic case, not a shortcut.
    sequences = basket_sequences(train.observation)
    print("LEARNING THE PRODUCT SPACE")
    print("=" * 78)
    print(f"  baskets (sentences)        {len(sequences):,}")
    print(f"  products in the catalogue  {train.observation['stock_code'].nunique():,}")

    space = train_item2vec(sequences)
    print(f"  products with a vector     {len(space):,}   "
          f"({100*len(space)/train.observation['stock_code'].nunique():.1f}% of catalogue)")
    print(f"  vector size                {space.dim}")
    print("\n  Items below min_count=5 get no vector and can never be retrieved.")

    # --- what did it actually learn? ------------------------------------
    # The sanity check that costs nothing and catches everything. If the
    # neighbours are nonsense, no metric downstream will be trustworthy.
    print("\n\nNEAREST NEIGHBOURS: does the space mean anything?")
    print("=" * 78)
    desc = (
        train.observation.groupby("stock_code")["description"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    )
    for probe in ("22423", "85123A", "23843"):
        if probe not in space.index:
            continue
        i = space.index[probe]
        sims = space.vectors @ space.vectors[i]
        near = np.argsort(-sims)[1:5]
        print(f"\n  {probe}  {desc.get(probe, '')[:45]}")
        for j in near:
            print(f"      {sims[j]:.3f}  {space.items[j]:<8} "
                  f"{desc.get(space.items[j], '')[:45]}")
    print("\n  Nobody told the model what these products ARE. It only saw which")
    print("  ones land in the same basket.")

    copurch = copurchase_matrix(train.observation, space)

    # --- build both snapshots -------------------------------------------
    print("\n\nBUILDING CANDIDATES")
    print("=" * 78)
    tr = prepare(train, space, copurch, purchases)
    te = prepare(test, space, copurch, purchases)
    for name, d in (("train", tr), ("test", te)):
        n_pos = int(d["y"].sum())
        print(f"  {name}: {len(d['candidates']):,} candidate rows over "
              f"{len(d['cust_ids']):,} customers, {n_pos:,} positives "
              f"({100*n_pos/len(d['y']):.2f}%)")

    # --- stage 1's own score ---------------------------------------------
    print("\n\nSTAGE 1 ON ITS OWN (can the ranker possibly succeed?)")
    print("=" * 78)
    r_at_n = retrieval_recall(te["retrieved"], te["cust_ids"], te["truth"])
    print(f"  embeddings alone, recall@{N_CANDIDATES}:  {r_at_n:.3f}")

    # The real ceiling is the WHOLE candidate set, not just the embedding
    # slice, because build_candidates also unions in previously-bought and
    # popular items. Quoting only the retrieval number would understate what
    # the ranker is allowed to find.
    cand_sets: dict[int, set[int]] = {
        int(cid): set(g["item_idx"].to_numpy().tolist())
        for cid, g in te["candidates"].groupby("customer_id", sort=False)
    }
    covered = [
        len(actual & cand_sets.get(cid, set())) / len(actual)
        for cid, actual in te["truth"].items() if actual
    ]
    ceiling = float(np.mean(covered))
    print(f"  full candidate set, recall:      {ceiling:.3f}   <- the real ceiling")
    print("\n  Stage 2 can reorder but never invent, so no ranker can recover an")
    print("  item that never entered the candidate set. If the final recall sits")
    print("  far below this line the ranker is the problem; if it sits close to")
    print("  it, more candidates is the only way forward.")

    # --- stage 2: train the ranker ---------------------------------------
    # Split by CUSTOMER, never by row. Two candidate rows from the same
    # customer are not independent, so a row-wise split would put a customer's
    # own items on both sides and inflate the validation score.
    rng = np.random.default_rng(RANDOM_SEED)
    cust = tr["candidates"]["customer_id"].to_numpy()
    uniq = np.unique(cust)
    val_ids = set(rng.choice(uniq, size=int(0.2 * len(uniq)), replace=False).tolist())
    is_val = np.fromiter((c in val_ids for c in cust), dtype=bool, count=len(cust))

    def groups_of(mask):
        """Group sizes in row order, which is what XGBRanker needs."""
        s = pd.Series(cust[mask])
        return s.groupby(s, sort=False).size().to_numpy()

    X_fit, y_fit = tr["X"][~is_val], tr["y"][~is_val]
    X_val, y_val = tr["X"][is_val], tr["y"][is_val]

    print("\n\nTRAINING THE RANKER")
    print("=" * 78)
    print(f"  fit {len(X_fit):,} rows / {len(np.unique(cust[~is_val])):,} customers")
    print(f"  val {len(X_val):,} rows / {len(val_ids):,} customers")

    ranker = fit_ranker(X_fit, y_fit, groups_of(~is_val), X_val, y_val, groups_of(is_val))
    pointwise = fit_pointwise(X_fit, y_fit, X_val, y_val)
    print(f"  rank:ndcg    trees used {ranker.best_iteration + 1}")
    print(f"  pointwise    trees used {pointwise.best_iteration + 1}")

    # --- evaluate ---------------------------------------------------------
    owned = owned_sets(te["cust_ids"], te["weights"])
    scores_rank = ranker.predict(te["X"])
    scores_point = pointwise.predict_proba(te["X"])[:, 1]

    results_all, results_new = {}, {}

    # ---- table 1: all items ---------------------------------------------
    methods_all = {
        "most popular": baseline_popular(te["cust_ids"], te["popular"]),
        "co-purchase": baseline_copurchase(te["cust_ids"], te["weights"], copurch),
        "repeat what they bought": baseline_repeat(te["cust_ids"], te["weights"]),
        "stage 1 only (embeddings)": {
            int(c): te["retrieved"][i][:TOP_K].tolist()
            for i, c in enumerate(te["cust_ids"])
        },
        "two-stage (pointwise)": top_k_from_scores(te["candidates"], scores_point),
        "two-stage (rank:ndcg)": top_k_from_scores(te["candidates"], scores_rank),
    }
    for name, recs in methods_all.items():
        results_all[name] = evaluate_recommendations(recs, te["truth"])

    # ---- table 2: new items only ----------------------------------------
    # Every method has already-bought items stripped, and the truth is
    # restricted to genuinely new purchases. Same rules for everyone.
    truth_new = {
        cid: items - owned.get(cid, set()) for cid, items in te["truth"].items()
    }
    truth_new = {c: v for c, v in truth_new.items() if v}

    fresh = te["X"]["bought_before"].to_numpy() == 0
    cand_new = te["candidates"][fresh]

    def strip_owned(recs, k=TOP_K):
        return {c: [i for i in items if i not in owned.get(c, set())][:k]
                for c, items in recs.items()}

    methods_new = {
        "most popular": strip_owned(
            {int(c): te["popular"][:200].tolist() for c in te["cust_ids"]}
        ),
        "co-purchase": baseline_copurchase(
            te["cust_ids"], te["weights"], copurch, exclude_own=True
        ),
        "repeat what they bought": {int(c): [] for c in te["cust_ids"]},
        "stage 1 only (embeddings)": strip_owned(
            {int(c): te["retrieved"][i].tolist() for i, c in enumerate(te["cust_ids"])}
        ),
        "two-stage (pointwise)": top_k_from_scores(cand_new, scores_point[fresh]),
        "two-stage (rank:ndcg)": top_k_from_scores(cand_new, scores_rank[fresh]),
    }
    for name, recs in methods_new.items():
        results_new[name] = evaluate_recommendations(recs, truth_new)

    print(f"\n\nTABLE 1: ALL ITEMS  (top {TOP_K}, test snapshot)")
    print("=" * 78)
    print(pd.DataFrame(results_all).T.round(4).to_string())
    print("\n  'repeat what they bought' is the number to beat. If a model cannot,")
    print("  the honest recommendation is to ship the SQL query instead.")

    print(f"\n\nTABLE 2: NEW ITEMS ONLY  (discovery, top {TOP_K})")
    print("=" * 78)
    print(pd.DataFrame(results_new).T.round(4).to_string())
    print(f"\n  {len(truth_new):,} customers bought at least one genuinely new product.")
    print("  Repeat scores 0 by construction: it can only ever suggest old items.")
    print("  This table is where embeddings either earn their place or do not.")

    # --- what the ranker actually leaned on -------------------------------
    print("\n\nWHAT THE RANKER USED (gain)")
    print("=" * 78)
    imp = pd.Series(
        ranker.get_booster().get_score(importance_type="gain")
    ).sort_values(ascending=False)
    imp.index = [CANDIDATE_FEATURES[int(f[1:])] if f.startswith("f") else f
                 for f in imp.index]
    print(imp.round(2).to_string())

    # --- save --------------------------------------------------------------
    # Vectors ship as a plain .npy-style array inside the npz. No vector
    # database: 4,260 x 64 float32 is 1.1MB and brute-force cosine over it is
    # a sub-millisecond matrix multiply. pgvector is the answer at 10M items,
    # and saying so in the README is worth more than building it here.
    path = ARTIFACTS / "recommend.joblib"
    joblib.dump(
        {
            "ranker": ranker,
            "feature_columns": CANDIDATE_FEATURES,
            "item_codes": space.items,
            "item_vectors": space.vectors,
            "descriptions": desc.reindex(space.items).fillna("").to_numpy(),
            "popular": te["popular"],
            "trained_on_cutoff": str(train.cutoff.date()),
            "scored_as_of": str(test.cutoff.date()),
            "metrics_all": results_all,
            "metrics_new": results_new,
            "retrieval_recall": r_at_n,
            "candidate_ceiling": ceiling,
            "top_k": TOP_K,
            "window_days": PREDICTION_WINDOW_DAYS,
        },
        path,
    )
    print(f"\n\nsaved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

    # --- the serving bundle -------------------------------------------------
    # The API gets TWO capabilities, deliberately built by different means:
    #
    #   /recommend/{customer}  PRECOMPUTED. Scoring live would need the sparse
    #       purchase-weight matrix, the item x item co-purchase matrix, the
    #       popularity table and pandas, all to produce twelve integers for one
    #       of 5,256 known customers. On a Railway plan where RAM costs about
    #       $10 per GB-month that is a bad trade, so the twelve integers ship
    #       directly.
    #
    #   /similar/{product}     LIVE, and genuinely so. One 3,795 x 64 float32
    #       matrix is 971 KB and a brute-force cosine over it is a sub-
    #       millisecond dot product. This is the endpoint that actually
    #       demonstrates the embedding, and it needs no gensim: gensim TRAINS
    #       the vectors, it is not required to USE them.
    best = top_k_from_scores(te["candidates"], scores_point)
    cand_bought = dict(
        zip(
            zip(te["candidates"]["customer_id"], te["candidates"]["item_idx"]),
            te["X"]["bought_before"].to_numpy(),
        )
    )
    ids = np.array([int(c) for c in te["cust_ids"]], dtype=np.int64)
    rec_items = np.full((len(ids), TOP_K), -1, dtype=np.int32)
    rec_seen = np.zeros((len(ids), TOP_K), dtype=np.int8)
    for r, cid in enumerate(ids):
        picks = best.get(int(cid), [])
        for c, item in enumerate(picks[:TOP_K]):
            rec_items[r, c] = item
            rec_seen[r, c] = int(cand_bought.get((int(cid), item), 0))

    npz = ARTIFACTS / "recommend_serve.npz"
    np.savez_compressed(
        npz,
        item_codes=space.items,
        item_vectors=space.vectors,
        descriptions=desc.reindex(space.items).fillna("").to_numpy().astype(str),
        customer_ids=ids,
        rec_items=rec_items,
        rec_bought_before=rec_seen,
        as_of=np.array(str(test.cutoff.date())),
    )
    print(f"saved {npz.name}  ({npz.stat().st_size / 1024:.0f} KB)   "
          f"vectors for live /similar, top-{TOP_K} precomputed for /recommend")


if __name__ == "__main__":
    main()
