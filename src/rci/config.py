"""Project-wide constants and paths.

Everything time-related lives here, in one place, on purpose. The single
most common way a customer-modelling project goes wrong is that different
scripts disagree about "as of when", so the cutoffs are defined once and
imported everywhere.
"""

from pathlib import Path

# --- paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

for _d in (DATA_RAW, DATA_PROCESSED, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)

# UCI Machine Learning Repository, dataset 502. CC BY 4.0.
UCI_DATASET_ID = 502
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

RAW_PARQUET = DATA_RAW / "online_retail_ii.parquet"
CLEAN_PARQUET = DATA_PROCESSED / "transactions_clean.parquet"


# --- the temporal split ----------------------------------------------------
#
# We predict: "will this customer buy again in the next 90 days?"
#
# To answer that honestly we need two snapshots, taken at different dates.
# Standing at a cutoff date, we may use ONLY what happened before it to build
# features, and we measure the answer ONLY from what happened after it.
#
#     2009-12 ................................................. 2011-12-09
#     |------------------- features ------------|--- label ---|            TRAIN
#                          (everything before)   (90 days after)
#
#     2009-12 .................................................|
#     |------------------------ features -------------------|--- label ---|  TEST
#
# Note the test snapshot's FEATURE window legitimately includes the train
# snapshot's LABEL window. That is not leakage: standing at the later cutoff,
# those purchases genuinely are in the past. What we must never do is let a
# snapshot's own label window bleed into its own features.
#
# Training on the earlier snapshot and testing on the later one is called
# out-of-time validation. It is the closest offline proxy for "train today,
# deploy tomorrow", which is what production actually does.

PREDICTION_WINDOW_DAYS = 90

# Cutoffs are derived from the data's own max date rather than hardcoded, so
# the pipeline stays correct if the dataset is ever revised.
#   test_cutoff  = last_date - 90 days
#   train_cutoff = test_cutoff - 90 days


def derive_cutoffs(last_date, window_days: int = PREDICTION_WINDOW_DAYS):
    """Return (train_cutoff, test_cutoff) given the dataset's final timestamp.

    Two non-overlapping label windows of `window_days` each, packed against
    the end of the data so both windows are complete. A label window that
    runs past the end of the data would be silently wrong: customers would
    look inactive purely because we stopped observing them.
    """
    import pandas as pd

    last_date = pd.Timestamp(last_date).normalize()
    test_cutoff = last_date - pd.Timedelta(days=window_days)
    train_cutoff = test_cutoff - pd.Timedelta(days=window_days)
    return train_cutoff, test_cutoff


# --- cleaning rules --------------------------------------------------------
#
# StockCodes that are not products. Found by inspecting codes that are not
# 5-digit-with-optional-letter. Leaving these in would put "postage" in the
# product catalogue and let the recommender suggest a bank charge.
NON_PRODUCT_STOCK_CODES = {
    "POST",       # postage
    "DOT",        # dotcom postage
    "C2",         # carriage
    "M",          # manual adjustment
    "m",
    "BANK CHARGES",
    "B",          # adjust bad debt
    "S",          # samples
    "D",          # discount
    "CRUK",       # charity donation
    "PADS",       # pads to match all cushions
    "TEST001",
    "TEST002",
    "AMAZONFEE",
    "gift_0001_10", "gift_0001_20", "gift_0001_30",
    "gift_0001_40", "gift_0001_50",
}

RANDOM_SEED = 42
