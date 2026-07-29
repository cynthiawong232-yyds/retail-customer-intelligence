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
    return {
        "status": "ok",
        "model_trained_on_cutoff": STATE.get("trained_on"),
        "customer_features_as_of": STATE.get("features_as_of"),
        "customers_loaded": len(STATE.get("customer_ids", [])),
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
