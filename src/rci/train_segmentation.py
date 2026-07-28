"""Fit, inspect, and save the segmentation model.

Run:  python -m rci.train_segmentation

Produces artifacts/segmentation.joblib, holding everything the API needs to
segment a customer it has never seen: the fitted scaler, the centroids, and
the segment names.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from rci.clean import build
from rci.config import ARTIFACTS
from rci.features import RFM_COLUMNS, build_features
from rci.segmentation import (
    choose_k,
    fit,
    name_segments,
    profile,
    quintile_segments,
)
from rci.split import train_test_snapshots

K = 4  # chosen from choose_k(): the inertia elbow and a local silhouette peak


def main() -> None:
    pd.set_option("display.width", 200)

    purchases = build()
    train, _ = train_test_snapshots(purchases)
    X = build_features(train)

    print("CHOOSING k")
    print("=" * 78)
    print(choose_k(X).round(3).to_string(index=False))
    print(f"\nchose k={K}: inertia elbow, local silhouette peak, and four groups")
    print("is what a marketing team can actually run campaigns against.")

    # ---- fit ------------------------------------------------------------
    km, scaler, labels = fit(X, K)

    prof = profile(X, labels)
    names = name_segments(prof)
    prof.insert(0, "name", [names[i] for i in prof.index])

    print("\n\nWHO IS IN EACH SEGMENT")
    print("=" * 78)
    print(prof.to_string())

    # ---- the comparison that matters ------------------------------------
    # Her existing quintile rule, scored on the same customers, so the two
    # approaches can be laid side by side.
    quint = quintile_segments(X)
    comp = pd.DataFrame({"kmeans": [names[i] for i in labels], "rfm_score": quint.to_numpy()})

    print("\n\nKMEANS SEGMENT vs RFM QUINTILE SCORE (3=worst, 15=best)")
    print("=" * 78)
    crosstab = pd.crosstab(comp["kmeans"], comp["rfm_score"])
    print(crosstab.to_string())

    print("\nRFM score distribution within each KMeans segment:")
    summary = comp.groupby("kmeans")["rfm_score"].agg(["count", "mean", "min", "max"])
    print(summary.round(2).to_string())

    # ---- save for serving -----------------------------------------------
    # We save the SCALER as well as the model. A new customer must be
    # transformed using the training set's mean and standard deviation.
    # Re-fitting a scaler on live data is a silent, classic production bug.
    artifact = {
        "kmeans": km,
        "scaler": scaler,
        "segment_names": names,
        "feature_columns": RFM_COLUMNS,
        "k": K,
        "trained_on_cutoff": str(train.cutoff.date()),
        "n_training_customers": int(len(X)),
    }
    path = ARTIFACTS / "segmentation.joblib"
    joblib.dump(artifact, path)
    size_kb = path.stat().st_size / 1024
    print(f"\n\nsaved {path.name}  ({size_kb:.0f} KB)")
    print(f"  centroids shape: {km.cluster_centers_.shape}")
    print("  small enough that serving it costs no measurable memory.")


if __name__ == "__main__":
    main()
