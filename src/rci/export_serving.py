"""Build the bundle the API ships with. Run after training.

    python -m rci.export_serving

WHY A SEPARATE STEP
-------------------
Training and serving are different worlds. Training needs pandas, the raw
45MB download, scikit-learn, and eventually gensim and shap. Serving needs a
few small arrays and a model.

Railway bills RAM at about $10 per GB per month, so anything imported at
serve time that is only needed at train time is money spent on nothing. This
script is the boundary: it takes the fat training world and emits a thin
bundle of plain numpy arrays that the API can read without pandas.

WHAT GOES IN THE BUNDLE
-----------------------
customers.npz   the demo customer list: ids plus their RFM values, so the
                deployed page has real people to click on. 5,256 customers
                x 3 floats is tiny.
segmentation.joblib   already written by train_segmentation.py: the fitted
                KMeans, the fitted scaler, and the segment names.
"""

from __future__ import annotations

import numpy as np

from rci.clean import build
from rci.config import ARTIFACTS
from rci.features import FEATURE_COLUMNS, RFM_COLUMNS, build_features
from rci.split import train_test_snapshots


def main() -> None:
    purchases = build()
    _, test = train_test_snapshots(purchases)

    # The TEST snapshot, not train: it is the later photograph, so it holds
    # every customer (5,256 vs 4,951) and their most current behaviour. The
    # demo should show the freshest view of each person.
    X = build_features(test)

    ids = X.index.to_numpy(dtype=np.int64)
    rfm = X[RFM_COLUMNS].to_numpy(dtype=np.float32)
    # The full 13-column matrix too, because the repurchase model needs all of
    # them. NaN survives the round-trip intact, which matters: XGBoost treats
    # missing as its own branch direction rather than something to fill in.
    full = X[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # float32 rather than float64 halves the memory for no meaningful loss:
    # nobody needs 15 significant figures on "days since last purchase".
    # The as-of date travels WITH the data. A customer's RFM is meaningless
    # without knowing when it was measured: customer 14077 is "bought
    # yesterday" on 2011-06-12 and "silent for 3 months" on 2011-09-10, from
    # the same purchase history. Shipping the date stops the API from
    # reporting a segment that cannot be interpreted.
    path = ARTIFACTS / "customers.npz"
    np.savez_compressed(
        path,
        customer_ids=ids,
        rfm=rfm,
        columns=np.array(RFM_COLUMNS),
        features=full,
        feature_columns=np.array(FEATURE_COLUMNS),
        as_of=np.array(str(test.cutoff.date())),
    )

    # ---- segmentation as PLAIN NUMBERS, not a pickled estimator ----------
    # segmentation.joblib holds a real sklearn KMeans and StandardScaler, so
    # unpickling it REQUIRES scikit-learn to be installed. The serving
    # container does not have scikit-learn, and deliberately so: it is ~100MB
    # of dependency on a plan that bills RAM. Loading that artifact in the
    # container fails with ModuleNotFoundError at startup, which is exactly
    # the kind of break that only shows up when you actually build the image.
    #
    # The fix is to ship what was LEARNED rather than the object that learned
    # it. A fitted KMeans is its centroids; a fitted StandardScaler is a mean
    # and a scale. That is 4x3 + 3 + 3 = 18 floats, and both transforms are
    # one line of numpy each:
    #
    #     scaled  = (log1p(x) - mean) / scale
    #     segment = argmin over centroids of ||scaled - centroid||
    #
    # Same argument as the lifetimes fitters in clv.py: a pickle ties the
    # container to the exact library version that wrote it, while an array of
    # floats will still load in five years.
    import joblib

    bundle = joblib.load(ARTIFACTS / "segmentation.joblib")
    names = bundle["segment_names"]
    seg_path = ARTIFACTS / "segmentation.npz"
    np.savez_compressed(
        seg_path,
        centroids=bundle["kmeans"].cluster_centers_.astype(np.float64),
        scaler_mean=bundle["scaler"].mean_.astype(np.float64),
        scaler_scale=bundle["scaler"].scale_.astype(np.float64),
        # Ordered by segment id, so names[i] belongs to centroid row i.
        segment_names=np.array([names[i] for i in range(len(names))]),
        feature_columns=np.array(bundle["feature_columns"]),
        trained_on_cutoff=np.array(str(bundle["trained_on_cutoff"])),
    )

    # Proof, not hope: the hand-written numpy path must agree with sklearn on
    # every customer, or this "optimisation" is a silent model change.
    logged = np.log1p(X[RFM_COLUMNS].to_numpy(dtype=np.float64))
    scaled = (logged - bundle["scaler"].mean_) / bundle["scaler"].scale_
    mine = np.argmin(
        ((scaled[:, None, :] - bundle["kmeans"].cluster_centers_) ** 2).sum(-1), axis=1
    )
    theirs = bundle["kmeans"].predict(bundle["scaler"].transform(logged))
    assert np.array_equal(mine, theirs), "numpy KMeans disagrees with sklearn"
    print(f"wrote {seg_path.name}: {seg_path.stat().st_size} bytes, "
          f"agrees with sklearn on all {len(mine):,} customers")

    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.name}: {len(ids):,} customers x {rfm.shape[1]} features "
          f"({size_kb:.0f} KB)")
    print(f"  id range   {ids.min()} to {ids.max()}")
    print(f"  dtype      {rfm.dtype}")
    print(f"  as of      {test.cutoff.date()}  (the later snapshot: freshest view)")


if __name__ == "__main__":
    main()
