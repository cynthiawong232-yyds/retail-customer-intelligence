# Retail Customer Intelligence

Four customer models on **real** transaction data, trained with a temporal split and served behind an API.

> Work in progress. Phase 1 of 5 complete.

**Data:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), CC BY 4.0. 1,067,371 real transactions from a UK online gift wholesaler, Dec 2009 to Dec 2011. Downloaded at runtime, never committed. **No synthetic data anywhere in this repo.**

| | status |
|---|---|
| Segmentation (KMeans) | done, served at `/segment` |
| Repurchase prediction (XGBoost) | next |
| CLV (BG/NBD + Gamma-Gamma vs XGBoost) | planned |
| Recommendations (item2vec + XGBoost ranker) | planned |

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate    # Python 3.11+
pip install -r requirements.txt
pip install -e .

python -m rci.clean                 # download + clean  (1,067,371 -> 776,577 rows)
python -m rci.train_segmentation    # fit and save the model
python -m rci.export_serving        # build the serving bundle
uvicorn rci.api:app --reload
```

Then `curl localhost:8000/customers/15369`, or open `localhost:8000/docs`.

```bash
pytest          # 45 tests
python -m rci.explain            # walk one real customer through the split
python -m rci.explain --lapsed
```

---

## How the evaluation works

The target is in the future, so the data is split by **time**, never at random.

```
2009-12 ................ cutoff ........... +90d
        |--- features ---|---- label ----|
          strictly before   the answer
```

Two snapshots, at 2011-06-12 and 2011-09-10. Train on the earlier, test on the later. This is out-of-time validation, the closest offline imitation of "fit on history, score today".

A random split would put Nov-2011 rows in training and Jan-2010 rows in test, letting the model learn the future to predict the past. Worse, features like lifetime spend computed across the whole file would already contain the answer.

**Note the two snapshots do not sum to the customer count.** A temporal split photographs the same base twice rather than cutting it in two: 4,951 customers in both, 305 who joined between the cutoffs, 596 who first bought after the later one. Train is necessarily a subset of test.

`tests/test_split.py` enforces all of this, including a synthetic fixture with a known answer.

---

## Findings so far

**Seasonality is visible in the split.** The two label windows have base rates of 32.3% and 43.5%. That 11-point gap is the Christmas run-up, confirmed across both years (Nov revenue ~2x July in 2010 and 2011). Handled with rank-based metrics and recalibration, **not** by shuffling the problem away.

**A better metric can mean a worse model.** Running KMeans on raw RFM scores **0.925** silhouette against **0.372** for the correctly transformed version. The raw version achieves it by isolating two outlier whales and putting 98.5% of customers in one bucket. Cluster sizes get checked alongside the score.

**Adding RFM quintile scores destroys information.** Two real customers:

| customer | recency | frequency | monetary | R | F | M | score | segment |
|---|---|---|---|---|---|---|---|---|
| 15369 | 311 | 11 | £6,251 | 1 | 5 | 5 | **11** | At-risk high-value |
| 14077 | 1 | 4 | £426 | 5 | 4 | 2 | **11** | New / Low-value |

`1+5+5` and `5+4+2` both make 11, and the total forgets which dimension contributed. Scores 9-13 cover 38% of the base and every one contains 3 distinct segments. The rule is accurate at the extremes and blind in the middle, which is where the winnable money is.

**The segments:**

| | customers | % base | % revenue | median spend | median recency |
|---|---|---|---|---|---|
| Champions | 811 | 16.4% | **67.9%** | £4,848 | 23 days |
| At-risk high-value | 1,482 | 29.9% | 21.2% | £1,272 | 184 days |
| New / Low-value | 691 | 14.0% | 5.7% | £835 | 20 days |
| Lapsed / One-off | 1,967 | 39.7% | 5.2% | £280 | 247 days |

k=4 was chosen from the inertia elbow and a local silhouette peak. Silhouette technically preferred k=2; four segments was chosen because two cannot support four campaigns.

---

## Serving

FastAPI, containerised, ~65 KB of artifacts.

```
GET  /health              both dates: model trained on, features as of
POST /segment             three numbers in, a named segment out
GET  /customers/{id}      one of the 5,256 real customers
GET  /customers           paged listing
```

Deliberate choices:

- **Artifacts load once at startup**, not per request.
- **The fitted scaler ships with the model.** Serving calls `.transform()`, never `.fit_transform()`; re-fitting on live data means an identical customer gets a different answer depending on who else is scored alongside them.
- **The scaler is fitted on positional arrays, not DataFrames**, so training matches serving exactly. Column order is guaranteed by a shared constant and a test, not by a name check that cannot run in a container without pandas.
- **`requirements-serve.txt` excludes pandas, scikit-learn's full stack, shap and gensim.** They are training-only, and Railway bills RAM at about $10/GB-month.
- **Responses echo their inputs**, so a wrong prediction can be told apart from wrong input data.

---

## Repo layout

```
src/rci/
  config.py         paths, cutoff arithmetic, cleaning rules
  data.py           download + normalise   (note: ucimlrepo does NOT serve id=502)
  clean.py          ledger -> purchases, with an auditable removal report
  split.py          the temporal split. the most important file here
  features.py       transactions -> one row per customer
  segmentation.py   KMeans, and the quintile rule reimplemented for comparison
  explain.py        prints a real customer's timeline across the cutoff
  api.py            FastAPI serving layer
tests/              45 tests, leakage checks first
```

## Data notes

Cleaning removes, in order: 34,335 exact duplicates, 5,737 non-product rows (`POST`, `BANK CHARGES`, `TEST001`), 243 malformed stock codes, **232,833 rows with no customer id (22%)**, and 17,586 returns/cancellations. Returns are written to a separate file rather than deleted, because return behaviour is predictive.
