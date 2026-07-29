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
from rci.features import RFM_COLUMNS, build_features
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
        as_of=np.array(str(test.cutoff.date())),
    )

    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.name}: {len(ids):,} customers x {rfm.shape[1]} features "
          f"({size_kb:.0f} KB)")
    print(f"  id range   {ids.min()} to {ids.max()}")
    print(f"  dtype      {rfm.dtype}")
    print(f"  as of      {test.cutoff.date()}  (the later snapshot: freshest view)")


if __name__ == "__main__":
    main()
