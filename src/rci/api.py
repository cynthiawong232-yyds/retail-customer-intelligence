"""The serving API. This is the half of the interview gap most candidates skip.

    uvicorn rci.api:app --reload        (local)
    docker build . && railway up        (deployed)

WHY THIS FILE EXISTS
--------------------
A model in a notebook is a claim. A model behind a URL is a product. Most
data science portfolios stop at the notebook, which is exactly the gap this
project exists to close.

Same stack as DropPost's backend: FastAPI on Railway. So this file should
cost setup time, not learning time.

THE TWO RULES OF SERVING
------------------------
1. LOAD ONCE, AT STARTUP. Not per request. Reading a model off disk takes
   tens of milliseconds; doing it inside the handler multiplies that by
   every visitor and turns a 5ms endpoint into a 50ms one for no reason.

2. TRANSFORM WITH THE TRAINING STATISTICS. The scaler was fitted on the
   training set and saved alongside the model. Serving calls .transform(),
   never .fit_transform(). Re-fitting on live data means an identical
   customer gets a different answer depending on who else is being scored,
   which is a genuinely nasty production bug because nothing errors.

MEMORY, BECAUSE IT IS BILLED
----------------------------
Railway charges about $10 per GB-month of RAM. This process deliberately
imports no pandas, no scikit-learn beyond what joblib needs to rebuild the
estimator, no shap and no gensim. The artifacts total well under 1MB.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

# Module-level store, filled once during startup and read-only afterwards.
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the process boots, before it accepts any traffic.

    FastAPI's lifespan hook is the correct place for this. Loading at import
    time also works but makes the module impossible to import in tests
    without the artifacts present.
    """
    bundle = joblib.load(ARTIFACTS / "segmentation.joblib")
    customers = np.load(ARTIFACTS / "customers.npz", allow_pickle=False)

    STATE["kmeans"] = bundle["kmeans"]
    STATE["scaler"] = bundle["scaler"]
    STATE["names"] = bundle["segment_names"]
    STATE["feature_columns"] = bundle["feature_columns"]
    STATE["trained_on"] = bundle["trained_on_cutoff"]

    # The repurchase model. Loaded once, like everything else.
    rep = joblib.load(ARTIFACTS / "repurchase.joblib")
    STATE["repurchase_model"] = rep["model"]
    STATE["repurchase_columns"] = list(rep["feature_columns"])
    STATE["repurchase_metrics"] = rep["metrics_on_test"]
    STATE["repurchase_n_trees"] = rep["n_trees"]
    # Precomputed SHAP, so the container never imports the shap library.
    shap_file = np.load(ARTIFACTS / "shap_test.npz", allow_pickle=False)
    STATE["shap_values"] = shap_file["shap_values"]
    STATE["shap_base"] = float(shap_file["base_value"])
    STATE["shap_ids"] = {int(c): i for i, c in enumerate(shap_file["customer_ids"])}
    STATE["customer_features"] = customers["features"]

    # The CLV pair, shipped two different ways on purpose.
    # The hurdle boosters run LIVE (xgboost is already imported above, so they
    # are nearly free). BG/NBD arrives PRECOMPUTED, because running it would
    # require `lifetimes` and scipy.special in a container whose RAM is billed.
    clv = joblib.load(ARTIFACTS / "clv_serve.joblib")
    STATE["clv_clf"] = clv["hurdle_clf"]
    STATE["clv_reg"] = clv["hurdle_reg"]
    STATE["clv_metrics"] = clv["metrics_on_test"]
    STATE["clv_window"] = clv["window_days"]

    clv_file = np.load(ARTIFACTS / "clv_test.npz", allow_pickle=False)
    STATE["clv_bgnbd"] = clv_file["bgnbd"]
    STATE["clv_alive"] = clv_file["p_alive"]
    STATE["clv_ids"] = {int(c): i for i, c in enumerate(clv_file["customer_ids"])}
    # Sorted once at startup so a decile lookup is a binary search rather than
    # a full comparison against 5,256 values on every request.
    STATE["clv_sorted"] = np.sort(clv_file["bgnbd"])

    # The recommender. Item vectors are loaded to run LIVE (/similar is a real
    # dot product); the per-customer lists are precomputed, because scoring
    # them live would need the co-purchase matrix and pandas for no gain.
    # gensim TRAINS these vectors. It is not needed to USE them, which is why
    # it stays out of requirements-serve.txt.
    rec = np.load(ARTIFACTS / "recommend_serve.npz", allow_pickle=False)
    STATE["item_codes"] = rec["item_codes"]
    STATE["item_vectors"] = rec["item_vectors"]
    STATE["item_desc"] = rec["descriptions"]
    STATE["item_index"] = {str(c): i for i, c in enumerate(rec["item_codes"])}
    STATE["rec_items"] = rec["rec_items"]
    STATE["rec_bought"] = rec["rec_bought_before"]
    STATE["rec_ids"] = {int(c): i for i, c in enumerate(rec["customer_ids"])}

    ids = customers["customer_ids"]
    STATE["customer_ids"] = ids
    STATE["customer_rfm"] = customers["rfm"]
    # The date the stored customer features were measured. NOT the same as
    # the model's training cutoff: the model was fitted on the June snapshot
    # and is applied to the September one, which is exactly what production
    # does (fit on history, score today).
    STATE["features_as_of"] = str(customers["as_of"])
    # id -> row number, so a lookup is a dict hit rather than a linear scan
    # through 5,256 entries on every request.
    STATE["id_index"] = {int(c): i for i, c in enumerate(ids)}

    yield
    STATE.clear()


app = FastAPI(
    title="Retail Customer Intelligence",
    description=(
        "Customer segmentation on UCI Online Retail II. "
        "Real transactions, temporally split, no synthetic data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# The frontend will be served from a different origin (Vercel) than the API
# (Railway), and browsers block that by default. This is the same CORS setup
# DropPost needs for exactly the same reason.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # a public read-only demo; tighten if it ever writes
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CustomerRFM(BaseModel):
    """The request body for /segment.

    Pydantic validates and coerces before our code runs, so a caller sending
    a negative recency or a string gets a clear 422 instead of reaching the
    model and producing a confident, meaningless answer.
    """

    recency: float = Field(..., ge=0, description="days since last purchase")
    frequency: float = Field(..., gt=0, description="number of distinct orders")
    monetary: float = Field(..., gt=0, description="lifetime spend, GBP")

    model_config = {
        "json_schema_extra": {
            "examples": [{"recency": 311, "frequency": 11, "monetary": 6251.26}]
        }
    }


class SegmentResponse(BaseModel):
    """A segment, plus everything needed to interpret it.

    The response deliberately echoes the INPUTS back. A bare label is close
    to unreadable in practice: you cannot tell a wrong prediction from wrong
    input data, and support tickets become guesswork. Echoing three floats
    costs nothing and makes every response self-documenting.
    """

    segment_id: int
    segment_name: str
    distance_to_centroid: float
    interpretation: str
    inputs: dict[str, float]


def _segment(recency: float, frequency: float, monetary: float) -> SegmentResponse:
    """The actual prediction path. Three numbers in, one segment out."""
    # Column order must match training exactly. xgboost and sklearn identify
    # features by POSITION, not name, so a reordering here would silently
    # produce wrong answers rather than an error. feature_columns is carried
    # in the artifact for this reason.
    raw = np.array([[recency, frequency, monetary]], dtype=np.float64)

    # Same two transforms as training, in the same order: log the skew away,
    # then apply the TRAINING scaler.
    logged = np.log1p(raw)
    scaled = STATE["scaler"].transform(logged)

    segment_id = int(STATE["kmeans"].predict(scaled)[0])

    # Distance to the assigned centroid is a cheap, honest confidence signal.
    # A customer sitting far from every centroid is one the model is not
    # really sure about, and saying so is better than a bare label.
    centroid = STATE["kmeans"].cluster_centers_[segment_id]
    distance = float(np.linalg.norm(scaled[0] - centroid))

    name = STATE["names"][segment_id]
    guidance = {
        "Champions": "Highest value and recently active. Protect: early access, loyalty perks.",
        "At-risk high-value": "Spent real money, now gone quiet. Win back: this is the profitable list.",
        "New / Low-value": "Recent but small. Nurture: grow basket size before they lapse.",
        "Lapsed / One-off": "One purchase, long ago. Cheapest to ignore; reactivate only in bulk.",
    }

    return SegmentResponse(
        segment_id=segment_id,
        segment_name=name,
        distance_to_centroid=round(distance, 4),
        interpretation=guidance.get(name, ""),
        inputs={"recency": recency, "frequency": frequency, "monetary": round(monetary, 2)},
    )


@app.get("/health")
def health() -> dict:
    """Liveness check. Railway and Vercel both poll something like this."""
    metrics = STATE.get("repurchase_metrics", {})
    return {
        "status": "ok",
        "model_trained_on_cutoff": STATE.get("trained_on"),
        "customer_features_as_of": STATE.get("features_as_of"),
        "customers_loaded": len(STATE.get("customer_ids", [])),
        "models": {
            "segmentation": {"algorithm": "KMeans", "k": 4},
            "repurchase": {
                "algorithm": "XGBoost",
                "trees": STATE.get("repurchase_n_trees"),
                "pr_auc": round(metrics.get("pr_auc", 0), 4),
                "base_rate": round(metrics.get("base_rate", 0), 4),
            },
            # Reported as a comparison, not a single winner, because the
            # winner is the unfashionable one and that is the finding.
            "clv": {
                "algorithm": "BG/NBD + Gamma-Gamma (served), XGBoost hurdle (challenger)",
                "window_days": STATE.get("clv_window"),
                "mae": {
                    name: round(m["MAE"], 1)
                    for name, m in STATE.get("clv_metrics", {}).items()
                },
            },
            "recommender": {
                "algorithm": "item2vec retrieval + XGBoost ranker (two-stage)",
                "products_with_vectors": len(STATE.get("item_codes", [])),
                "embedding_dim": int(STATE["item_vectors"].shape[1])
                if "item_vectors" in STATE else None,
                "top_k": int(STATE.get("rec_items", np.zeros((1, 0))).shape[1]),
            },
        },
    }


@app.post("/segment", response_model=SegmentResponse)
def segment(body: CustomerRFM) -> SegmentResponse:
    """Segment an arbitrary customer from three numbers."""
    return _segment(body.recency, body.frequency, body.monetary)


@app.get("/customers/{customer_id}", response_model=SegmentResponse)
def segment_known_customer(customer_id: int) -> SegmentResponse:
    """Segment one of the real customers in the dataset, by id."""
    idx = STATE["id_index"].get(customer_id)
    if idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"customer {customer_id} not in the dataset",
        )
    r, f, m = STATE["customer_rfm"][idx]
    return _segment(float(r), float(f), float(m))


class RepurchaseResponse(BaseModel):
    customer_id: int
    probability: float
    decile: int
    reading: str
    top_drivers: list[dict]
    caveat: str


@app.get("/predict/repurchase/{customer_id}", response_model=RepurchaseResponse)
def predict_repurchase(customer_id: int) -> RepurchaseResponse:
    """Probability this customer buys again in the next 90 days, explained."""
    idx = STATE["id_index"].get(customer_id)
    if idx is None:
        raise HTTPException(404, f"customer {customer_id} not in the dataset")

    # reshape(1, -1) because the model expects a 2-D matrix of rows, and a
    # single customer is a matrix with one row, not a 1-D vector.
    row = STATE["customer_features"][idx].reshape(1, -1).astype(np.float32)
    prob = float(STATE["repurchase_model"].predict_proba(row)[0, 1])

    # Which tenth of the customer base this score falls in. Deciles are what
    # a campaign is actually targeted by ("contact the top two deciles"),
    # so it is more directly usable than the raw probability.
    all_rows = STATE["customer_features"].astype(np.float32)
    if "all_probs" not in STATE:
        STATE["all_probs"] = STATE["repurchase_model"].predict_proba(all_rows)[:, 1]
    decile = int(1 + (STATE["all_probs"] > prob).sum() * 10 // len(STATE["all_probs"]))

    # Precomputed SHAP: per-customer contributions in log-odds, positive
    # meaning "pushed the score up". Shipped as data so the container never
    # imports shap, which would cost RAM that Railway bills for.
    drivers: list[dict] = []
    sidx = STATE["shap_ids"].get(customer_id)
    if sidx is not None:
        values = STATE["shap_values"][sidx]
        order = np.argsort(-np.abs(values))[:5]
        cols = STATE["repurchase_columns"]
        drivers = [
            {
                "feature": cols[i],
                "value": None if np.isnan(row[0, i]) else round(float(row[0, i]), 2),
                "effect_log_odds": round(float(values[i]), 4),
                "direction": "raises" if values[i] > 0 else "lowers",
            }
            for i in order
        ]

    return RepurchaseResponse(
        customer_id=customer_id,
        probability=round(prob, 4),
        decile=min(decile, 10),
        reading=(
            f"{prob:.0%} chance of purchasing within 90 days of "
            f"{STATE['features_as_of']}."
        ),
        top_drivers=drivers,
        # Stated in every response on purpose. The model was fitted on a
        # summer window (32.3% base rate) and is scoring an autumn one
        # (43.5%), so it systematically UNDER-states probabilities. The
        # ordering is sound; the absolute number is not, until recalibrated.
        caveat=(
            "Probabilities are uncalibrated for this period: the model was "
            "trained on a 32.3% base-rate window and applied to a 43.5% one "
            "(Christmas seasonality), so absolute values run low. Ranking and "
            "decile are unaffected."
        ),
    )


class CLVResponse(BaseModel):
    customer_id: int
    expected_spend_90d: float
    p_still_alive: float
    value_decile: int
    challenger: dict
    reading: str
    caveat: str


@app.get("/predict/clv/{customer_id}", response_model=CLVResponse)
def predict_clv(customer_id: int) -> CLVResponse:
    """Expected spend over the next 90 days, from two different models.

    Repurchase answers WHETHER someone returns. This answers WHAT THAT IS
    WORTH, which is the number a retention budget is actually built from: an
    80% chance of GBP 40 and a 30% chance of GBP 4,000 are opposite decisions,
    and only the pound figure separates them.

    Both models are returned side by side rather than one being quietly
    picked, because the honest result is that the 2005 probabilistic model
    beat gradient boosting on this dataset. Hiding that would waste the most
    interesting thing the project found.
    """
    cidx = STATE["clv_ids"].get(customer_id)
    idx = STATE["id_index"].get(customer_id)
    if cidx is None or idx is None:
        raise HTTPException(404, f"customer {customer_id} not in the dataset")

    # --- the served model: BG/NBD x Gamma-Gamma, precomputed -------------
    expected = float(STATE["clv_bgnbd"][cidx])
    alive = float(STATE["clv_alive"][cidx])

    # Value decile across the whole customer base. searchsorted on a
    # pre-sorted array is O(log n); decile 1 is the most valuable tenth.
    n = len(STATE["clv_sorted"])
    rank_from_top = n - int(np.searchsorted(STATE["clv_sorted"], expected, side="left"))
    value_decile = min(10, max(1, 1 + (rank_from_top - 1) * 10 // n))

    # --- the challenger: XGBoost hurdle, run live ------------------------
    # Two boosters, exactly as trained: P(spends anything) x E[spend | spends].
    # expm1 inverts the log target the regressor was fitted on.
    row = STATE["customer_features"][idx].reshape(1, -1).astype(np.float32)
    p_buy = float(STATE["clv_clf"].predict_proba(row)[0, 1])
    amount = float(np.expm1(STATE["clv_reg"].predict(row)[0]))
    hurdle = max(0.0, p_buy * amount)

    return CLVResponse(
        customer_id=customer_id,
        expected_spend_90d=round(expected, 2),
        p_still_alive=round(alive, 4),
        value_decile=value_decile,
        challenger={
            "model": "XGBoost hurdle",
            "p_buys_at_all": round(p_buy, 4),
            "expected_amount_if_they_buy": round(amount, 2),
            "estimate": round(hurdle, 2),
        },
        reading=(
            f"Expected to spend GBP {expected:,.0f} in the 90 days after "
            f"{STATE['features_as_of']}, with a {alive:.1%} chance of still "
            f"being an active customer at all. Value decile {value_decile} of 10 "
            f"(1 = most valuable). This figure is the CEILING on retention "
            f"spend before margin, not a profit number."
        ),
        caveat=(
            "BG/NBD is precomputed, so this endpoint scores only customers "
            "known at export time. It also cannot score customers with no "
            "repeat history (7.9% of the base); those are filled with the "
            "median for their group, never with zero. The XGBoost challenger "
            "has no such gap but was less accurate here (MAE 429 vs 389)."
        ),
    )


class Recommendation(BaseModel):
    stock_code: str
    description: str
    rank: int
    bought_before: bool


class RecommendResponse(BaseModel):
    customer_id: int
    recommendations: list[Recommendation]
    n_new_to_customer: int
    reading: str
    caveat: str


@app.get("/recommend/{customer_id}", response_model=RecommendResponse)
def recommend(customer_id: int) -> RecommendResponse:
    """The twelve products to put in this customer's email.

    The list mixes reorders and genuine discoveries, and the response labels
    which is which. That distinction is the whole story of Phase 4: 43% of
    what a customer buys next they have bought before, so a recommender that
    only replays history scores well on paper and adds nothing. Marking each
    item lets whoever receives the list see how much of it is actually new.
    """
    idx = STATE["rec_ids"].get(customer_id)
    if idx is None:
        raise HTTPException(404, f"customer {customer_id} not in the dataset")

    items = STATE["rec_items"][idx]
    seen = STATE["rec_bought"][idx]
    recs = [
        Recommendation(
            stock_code=str(STATE["item_codes"][i]),
            description=str(STATE["item_desc"][i]),
            rank=position + 1,
            bought_before=bool(seen[position]),
        )
        for position, i in enumerate(items)
        if i >= 0
    ]
    n_new = sum(1 for r in recs if not r.bought_before)

    return RecommendResponse(
        customer_id=customer_id,
        recommendations=recs,
        n_new_to_customer=n_new,
        reading=(
            f"{len(recs)} products for the 90 days after "
            f"{STATE['features_as_of']}. {len(recs) - n_new} are reorders, "
            f"{n_new} are new to this customer."
        ),
        caveat=(
            "Ranked by a two-stage model: item2vec retrieval then an XGBoost "
            "ranker. On all items it beats the 'repeat what they bought' rule "
            "only slightly (precision@12 0.238 vs 0.225). On genuinely NEW "
            "items it is worth far more: 0.048 vs 0.020 for most-popular. "
            "Lists are precomputed, so only customers known at export time "
            "can be served."
        ),
    )


class SimilarItem(BaseModel):
    stock_code: str
    description: str
    similarity: float


@app.get("/similar/{stock_code}", response_model=list[SimilarItem])
def similar(stock_code: str, k: int = 8) -> list[SimilarItem]:
    """Products that live in the same neighbourhood as this one. Computed live.

    This is the endpoint that actually demonstrates the embedding. Vectors are
    L2-normalised at build time, so cosine similarity is a plain dot product
    and the whole search is one 3,795 x 64 matrix multiply: about a
    millisecond, on 971 KB of RAM.

    DELIBERATE NON-DECISION: no vector database. At this catalogue size a
    brute-force scan is faster than a network hop to pgvector or FAISS would
    be, and it costs nothing on a plan that bills RAM. The scaling answer
    belongs in the README. Declining to add infrastructure you do not need is
    a better engineering signal than adding it.
    """
    i = STATE["item_index"].get(stock_code)
    if i is None:
        raise HTTPException(404, f"product {stock_code} has no vector")

    sims = STATE["item_vectors"] @ STATE["item_vectors"][i]
    k = max(1, min(k, 50))
    # k+1 then drop self: an item is always its own nearest neighbour at 1.0.
    top = np.argsort(-sims)[: k + 1]
    return [
        SimilarItem(
            stock_code=str(STATE["item_codes"][j]),
            description=str(STATE["item_desc"][j]),
            similarity=round(float(sims[j]), 4),
        )
        for j in top
        if j != i
    ][:k]


@app.get("/customers")
def list_customers(limit: int = 20, offset: int = 0) -> dict:
    """Page through the real customers, so a frontend has something to show."""
    ids = STATE["customer_ids"]
    rfm = STATE["customer_rfm"]
    window = slice(offset, offset + min(limit, 100))
    return {
        "total": len(ids),
        "customers": [
            {
                "customer_id": int(cid),
                "recency": float(row[0]),
                "frequency": float(row[1]),
                "monetary": round(float(row[2]), 2),
            }
            for cid, row in zip(ids[window], rfm[window])
        ],
    }
