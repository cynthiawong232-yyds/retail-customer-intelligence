"""Download and load the raw Online Retail II dataset.

WHY THIS FILE EXISTS
--------------------
The dataset is never committed to git (see .gitignore). Anyone cloning this
repo runs one command and gets the same data. That is what "reproducible"
means in practice, and it is a thing interviewers check: a repo with a CSV
committed and no download code cannot be re-run by anyone else.

Two download paths, because external hosts fail in boring ways:
  1. the raw .zip from archive.ics.uci.edu  <- the one that actually works
  2. ucimlrepo, the official fetcher, as a fallback

Note: as of 2026-07, `fetch_ucirepo(id=502)` raises DatasetNotFoundError:
"exists in the repository, but is not available for import". Plenty of
tutorials claim otherwise. The zip is therefore the primary path, and
ucimlrepo is kept only in case UCI enables the API for this dataset later.

Both end at the same normalised DataFrame, cached as parquet. Parquet because
re-parsing a 45MB Excel file takes ~30 seconds and reloading parquet takes
milliseconds, and during Phase 0 we reload constantly.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from rci.config import RAW_PARQUET, UCI_DATASET_ID, UCI_ZIP_URL

# The dataset ships with slightly inconsistent naming across its two sheets
# and across the ucimlrepo/zip paths. Normalise to snake_case once, here, so
# nothing downstream has to care which route the data arrived by.
COLUMN_MAP = {
    "Invoice": "invoice",
    "InvoiceNo": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "UnitPrice": "price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}

EXPECTED = ["invoice", "stock_code", "description", "quantity",
            "invoice_date", "price", "customer_id", "country"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Downloaded data is missing expected columns: {missing}. "
            f"Got: {list(df.columns)}"
        )
    df = df[EXPECTED].copy()
    # These four are conceptually text but arrive as `object` holding a mix of
    # str and int (some product descriptions are stored as bare numbers, and
    # some invoice numbers as ints). Mixed-type object columns cannot be
    # written to parquet, so pin them to the nullable string dtype, which
    # keeps NA as NA rather than turning it into the literal "nan".
    for col in ("invoice", "stock_code", "description", "country"):
        df[col] = df[col].astype("string").str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # customer_id is an integer conceptually but has nulls, so it must stay
    # float or become a nullable Int64. Int64 makes the nulls explicit.
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    return df


def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_DATASET_ID)
    # This dataset has no prediction target, so everything lands in features.
    # `original` is present on some versions and is the safest source.
    raw = getattr(ds.data, "original", None)
    if raw is None:
        raw = ds.data.features
    return _normalise(pd.DataFrame(raw))


def _from_zip() -> pd.DataFrame:
    resp = requests.get(UCI_ZIP_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".xlsx", ".csv")))
        with zf.open(name) as fh:
            payload = io.BytesIO(fh.read())
    if name.lower().endswith(".csv"):
        return _normalise(pd.read_csv(payload))
    # The Excel file has one sheet per year; concatenate them.
    sheets = pd.read_excel(payload, sheet_name=None, engine="openpyxl")
    return _normalise(pd.concat(sheets.values(), ignore_index=True))


def load_raw(force: bool = False) -> pd.DataFrame:
    """Return the raw transaction table, downloading and caching if needed."""
    if RAW_PARQUET.exists() and not force:
        return pd.read_parquet(RAW_PARQUET)

    try:
        print("  downloading zip from archive.ics.uci.edu ...")
        df = _from_zip()
        source = "zip"
    except Exception as exc:  # noqa: BLE001 - any failure should fall through
        print(f"  zip path failed ({type(exc).__name__}: {exc}); trying ucimlrepo")
        df = _from_ucimlrepo()
        source = "ucimlrepo"

    df.to_parquet(RAW_PARQUET, index=False)
    print(f"  downloaded via {source}: {len(df):,} rows -> {RAW_PARQUET.name}")
    return df


if __name__ == "__main__":
    frame = load_raw()
    print(frame.head())
    print(frame.dtypes)
