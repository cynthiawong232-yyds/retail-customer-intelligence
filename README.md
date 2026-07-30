# Retail Customer Intelligence

Four customer models on **real** transaction data, trained with a temporal split and served behind one API.

**[Live demo](https://retail-customer-intelligence.vercel.app)** | **[API docs](https://retail-customer-intelligence-production.up.railway.app/docs)**

The demo is a real page calling a real API. Pick a customer and all four models answer at once. The API runs as a container on Railway; the frontend is static on Vercel. Both redeploy on every push to `main`.

**Data:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), CC BY 4.0. 1,067,371 real transactions from a UK online gift wholesaler, Dec 2009 to Dec 2011. Downloaded at runtime, never committed. **No synthetic data anywhere in this repo.**

| Model | Question | Served at |
|---|---|---|
| Segmentation (KMeans) | Who are they? | `POST /segment`, `GET /customers/{id}` |
| CLV (BG/NBD + Gamma-Gamma vs XGBoost) | What are they worth? | `GET /predict/clv/{id}` |
| Repurchase (XGBoost + SHAP) | Will they come back? | `GET /predict/repurchase/{id}` |
| Recommender (item2vec + XGBoost ranker) | What do we show them? | `GET /recommend/{id}`, `GET /similar/{code}` |

117 tests. Every model is reported against a baseline it had to beat, and **the cases where it barely did, or did not, are stated rather than buried.**

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate    # Python 3.11+
pip install -r requirements.txt && pip install -e .

python -m rci.clean                 # download + clean  (1,067,371 -> 776,577 rows)
python -m rci.train_segmentation
python -m rci.train_repurchase
python -m rci.train_clv
python -m rci.train_recommend
python -m rci.export_serving        # build the serving bundle
uvicorn rci.api:app --reload
```

Then open `localhost:8000/docs`, or:

```bash
curl localhost:8000/similar/22423          # the demo worth seeing first
curl localhost:8000/predict/clv/13069
pytest                                     # 117 tests
python -m rci.explain                      # walk one real customer across the cutoff
```

---

## How the evaluation works

The target is in the future, so the data is split by **time**, never at random.

```
2009-12 ................ cutoff ........... +90d
        |--- features ---|---- label ----|
          strictly before   the answer
```

Two snapshots, at **2011-06-12** and **2011-09-10**. Train on the earlier, test on the later. This is out-of-time validation, the closest offline imitation of "fit on history, score today".

A random split would put Nov-2011 rows in training and Jan-2010 rows in test, letting the model learn the future to predict the past. Worse, features like lifetime spend computed across the whole file would already contain the answer.

**The two snapshots do not sum to the customer count.** A temporal split photographs the same base twice rather than cutting it in two: 4,951 customers in both, 305 who joined between the cutoffs, 596 who first bought after the later one. Train is necessarily a subset of test.

`tests/test_split.py` enforces this, including a synthetic fixture with a known answer.

### Why the label window is 90 days and not 12 months

Textbook CLV is annual. The dataset spans 24 months, so a 12-month label window would push the cutoff back to 2010-12, leaving one year of feature history and **no room for a second out-of-time snapshot**. The target is therefore 90-day forward spend. Same technique, honest window, stated rather than quietly relabelled.

---

## Findings

### Seasonality is visible in the split, and was not shuffled away

The two label windows have base rates of **32.3% and 43.5%**. That 11-point gap is the Christmas run-up, confirmed across both years (Nov revenue ~2x July in 2010 and 2011). Handled with rank-based metrics and recalibration. A random split would have averaged summer and Christmas into both sides and hidden it.

### A better metric can mean a worse model

KMeans on raw RFM scores **0.925** silhouette against **0.372** for the correctly log-transformed and scaled version. The raw version achieves it by isolating two outlier whales and putting **98.5% of customers in one bucket**. Cluster sizes are checked alongside the score.

### RFM quintile scores destroy information

Two real customers:

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

k=4 from the inertia elbow and a local silhouette peak. Silhouette technically preferred k=2; four was chosen because two cannot support four campaigns.

### XGBoost beat logistic regression by 2%, and that is the finding

```
                     vs base  pr_auc  roc_auc   brier
recency rule            1.58  0.6855   0.7623     NaN
logistic regression     1.74  0.7580   0.7891  0.2007
xgboost                 1.78  0.7742   0.7986  0.1976

base rate (a useless model's PR-AUC floor): 0.4351
```

The large jump is rule to linear model, not linear model to gradient boosting. **The feature engineering did more work than the algorithm.** PR-AUC is the headline rather than ROC-AUC, because ROC-AUC flatters imbalanced problems and PR-AUC's floor is the base rate rather than 0.5.

Business view: **the top decile repurchases at 93.3% against a 43.5% average, 2.15x lift.** The top three deciles capture 53% of all repurchasers.

**Calibration is a separate problem from ranking.** Mean predicted probability 0.309 against an actual 0.435: the model learned a summer world and scores an autumn one. Isotonic recalibration moves mean prediction to 0.435 and improves Brier from 0.2000 to 0.1816 while leaving ROC-AUC at 0.793. Isotonic is monotonic, so it rescales without reordering.

**Gain and SHAP rank features differently, and both are right.** `recency` is 5th by gain (8.4%) and 1st by SHAP. It was split on 172 times, more than any feature, each split deciding little. Gain answers "what did the model rely on to build itself"; SHAP answers "what moves this customer's prediction". The second is what you show a stakeholder.

### A 2005 probabilistic model beat gradient boosting on CLV

```
                              MAE      RMSE     R2  spearman  mean_pred
always the mean            618.47  3239.31 -0.004       NaN     338.50
repeat last 90d            415.79  2200.95  0.537     0.480     348.09
BG/NBD + Gamma-Gamma       388.90  2083.77  0.585     0.552     345.67  <- wins
XGBoost (raw GBP)          429.10  2450.49  0.426     0.526     331.68
XGBoost (log target)       466.94  3016.48  0.130     0.569      80.25
hurdle P(buy) x E[spend]   406.74  2530.00  0.388     0.554     238.47
                                              actual mean 534.25
```

BG/NBD + Gamma-Gamma wins MAE, RMSE and R². The naive "they will repeat last quarter" rule also beat XGBoost on raw pounds. With 4,951 customers and 13 features, the probabilistic model's built-in assumptions do work that gradient boosting would need far more data to learn.

**The log-target model is the trap.** It ranks best (Spearman 0.569) and predicts a mean of £80 against an actual £534. `expm1` undoes the log but not the bias: `E[exp(z)] > exp(E[z])` by Jensen's inequality. Ranking and magnitude are different jobs.

**The target is zero-inflated** (67.7% exact zeros, median £0, skew 20.0), which is why a hurdle model was tried and why Spearman is reported next to MAE.

**BG/NBD cannot score 7.9% of customers.** Its conditional expectation contains a hypergeometric term that will not converge without repeat history, and every undefined customer is verifiably a never-repeater. Filled with the median for their group, never with zero, which would assert they definitely will not buy. XGBoost has no such gap, which is a real argument in its favour despite losing on accuracy.

### A tutorial default produced negative money, and MAE never noticed

Gamma-Gamma was fitted with `penalizer_coef=0.01`, copied from the standard example. `lifetimes` penalises **raw parameter values**, and Gamma-Gamma's `v` is on the scale of money (544 here). Shrinking it collapsed `q`:

| penalizer | q | population mean |
|---|---|---|
| **0.0** | **3.820** | **£366** (empirical is £376) |
| 0.001 | 0.895 | -£1,205 |
| 0.01 | 0.349 | -£21 |

The mean is `v*p/(q-1)` and **exists only if q > 1**. Below that the library returns a negative number rather than raising, so all 1,575 never-repeated customers (30% of the base) had negative predicted spend. MAE did not care: those customers mostly spend £0, so predicting -21 scored nearly as well as predicting 0.

Found by a test asserting non-negativity. Fixed by fitting unregularised, with a hard error if `q <= 1` at both fit time and load time. **A regularisation strength copied from a tutorial is only meaningful on the scale that tutorial's data had.**

### The recommender's strongest competitor is a SQL query

item2vec learned real structure with no taxonomy, no categories and no descriptions, purely from basket co-occurrence:

```
22423   REGENCY CAKESTAND 3 TIER
  0.893   ROSES REGENCY TEACUP AND SAUCER
  0.880   GREEN REGENCY TEACUP AND SAUCER
  0.865   PINK REGENCY TEACUP AND SAUCER
```

But **43.1% of a customer's future purchases are items they have already bought**, so the baseline to beat is "recommend what they bought last time":

```
ALL ITEMS (top 12)          recall@12  precision@12   map@12  ndcg@12
most popular                   0.0250        0.0542   0.0228   0.0600
co-purchase                    0.0591        0.1167   0.0669   0.1312
repeat what they bought        0.1363        0.2254   0.1830   0.2751
stage 1 only (embeddings)      0.0945        0.1601   0.1146   0.1937
two-stage (pointwise)          0.1449        0.2380   0.1900   0.2834
two-stage (rank:ndcg)          0.1440        0.2377   0.1882   0.2823
```

**A 5.6% gain over a SQL query.** Reporting only this table would oversell the model badly, so everything was scored a second time on **new products only**, with already-bought items stripped from every method:

```
NEW ITEMS ONLY (discovery)  recall@12  precision@12   map@12  ndcg@12
most popular                   0.0133        0.0198   0.0084   0.0236
co-purchase                    0.0300        0.0388   0.0188   0.0466
repeat what they bought        0.0000        0.0000   0.0000   0.0000
stage 1 only (embeddings)      0.0347        0.0431   0.0220   0.0531
two-stage (pointwise)          0.0402        0.0476   0.0261   0.0607
```

**On discovery the model is worth 2.4x the most-popular list.** That is where the embeddings earn their place. The ranker agreed: `bought_before` had gain 83.4, more than every other feature combined.

**`window=1000`, not the default 5.** Word2Vec assumes word order carries meaning, so it looks a few positions either side. A basket has no order: a product's context is the whole basket, and ours average 20.4 items and reach 271. The default would have trained on an arbitrary five-item slice with no error and no warning.

**The learning-to-rank objective did not win.** `rank:ndcg` optimises exactly what is measured and scored 0.2823 NDCG against 0.2834 for a plain pointwise classifier. Measured rather than assumed; both are kept.

**Retrieval bounds everything downstream.** The full candidate set has recall 0.457 and the final model reaches 0.145, so the ranker is the bottleneck, not retrieval. If those numbers were close instead, more candidates would be the only way forward.

---

## Serving

FastAPI, containerised, **1.87 MB of artifacts**, no pandas and no scikit-learn in the image.

```
GET  /health                      all four models, their metrics, and both dates
POST /segment                     three numbers in, a named segment out
GET  /customers/{id}              one of the 5,256 real customers
GET  /customers                   paged listing
GET  /predict/repurchase/{id}     probability, decile, and SHAP drivers
GET  /predict/clv/{id}            expected spend, P(still alive), and the challenger
GET  /recommend/{id}              twelve products, each labelled reorder or discovery
GET  /similar/{stock_code}        live nearest neighbours in embedding space
```

### Live vs precomputed, and why each

| Runs live | Precomputed |
|---|---|
| KMeans (`/segment`) | SHAP explanations |
| XGBoost repurchase | BG/NBD + Gamma-Gamma |
| XGBoost CLV hurdle | Per-customer recommendation lists |
| Cosine similarity (`/similar`) | |

The rule is the same each time: anything needing a **training-only dependency** ships as data instead. Running SHAP live would import `shap`; BG/NBD would import `lifetimes` and `scipy.special`; the recommendation lists would need the co-purchase matrix and pandas. Railway bills RAM at about **$10 per GB-month**, so an import that exists only to produce a lookup is money spent on nothing.

`/similar` is the counter-example that proves the rule: `gensim` **trains** the vectors and is not needed to **use** them, so a 971 KB matrix and one dot product stay live.

Every precomputed endpoint says so in its own response, because a lookup silently presented as a live model is a lie of omission.

### Other deliberate choices

- **Artifacts load once at startup**, not per request.
- **The fitted scaler ships with the model.** Serving calls `.transform()`, never `.fit_transform()`; re-fitting on live data means an identical customer gets a different answer depending on who else is scored alongside them.
- **The scaler is fitted on positional arrays, not DataFrames**, so training matches serving exactly. Column order is guaranteed by a shared constant and a test, not a name check that cannot run in a container without pandas.
- **The Dockerfile names each artifact individually.** `COPY artifacts/` shipped 2.6 MB of training-only records the API never opens.
- **Responses echo their inputs**, so a wrong prediction can be told apart from wrong input data.
- **Every repurchase response carries its calibration caveat.** An uncalibrated probability multiplied by a budget is how a number does damage.

### Frontend

A small React + Vite page in [`frontend/`](frontend/): one customer picker, five panels, no router and no component library. 151 KB of JavaScript, 49 KB gzipped.

```bash
cd frontend && npm install && cp .env.example .env.local && npm run dev
```

Every panel renders its model's caveat next to the number rather than in a footnote, and each panel fetches independently so one failing endpoint does not blank the page. Deployment notes are in [`frontend/README.md`](frontend/README.md).

---

## What I would do differently at scale

**Vector search.** At 4,260 items, brute-force cosine is 971 KB and about a millisecond, so a vector database would add a network hop and RAM cost for no benefit. pgvector or FAISS becomes correct somewhere around 10M items, or when embeddings must be shared across services. Declining to build it here was the decision, not an omission.

**Calibration would be scheduled, not one-off.** The 32.3% to 43.5% base-rate shift across a single quarter is seasonal, so calibration should refit on a rolling window rather than being fitted once and trusted.

**Embeddings would be retrained on a schedule, with alignment.** Word2Vec runs are not aligned across fits (the space rotates), so a ranker trained on one geometry cannot be applied to another. Production needs either orthogonal Procrustes alignment or a joint retrain of both stages. Here both snapshots deliberately share one embedding fit.

**Cold start needs a second path.** 6.5% of products bought after the cutoff had never appeared before it, and 10.9% of the catalogue falls below `min_count=5`. No embedding can retrieve either group. The answer is a hybrid: embeddings where there is history, content or category rules elsewhere.

**The 22% of rows with no customer id** are unusable for customer-level modelling but perfectly good for item-to-item co-occurrence. At scale the recommender should train on all baskets, and only the customer models should filter.

**Monitoring would watch the input distribution, not just the metrics.** Every problem in this project announced itself as a distribution shift before it showed up as a bad score, and one of them (negative money) never showed up in the score at all.

---

## Repo layout

```
src/rci/
  config.py           paths, cutoff arithmetic, cleaning rules
  data.py             download + normalise  (note: ucimlrepo does NOT serve id=502)
  clean.py            ledger -> purchases, with an auditable removal report
  split.py            the temporal split. the most important file here
  features.py         transactions -> one row per customer
  segmentation.py     KMeans, and the quintile rule reimplemented for comparison
  repurchase.py       XGBoost, baselines, metrics, calibration, tree printer
  clv.py              BG/NBD + Gamma-Gamma, XGBoost regressor, hurdle model
  recommend.py        item2vec, retrieval, candidate features, ranker, ranking metrics
  train_*.py          one runnable script per model, each printing its own comparison
  export_serving.py   the train/serve boundary
  explain.py          prints a real customer's timeline across the cutoff
  api.py              FastAPI serving layer
tests/                117 tests, leakage checks first
docs/model-cards.md   one card per model: intended use, metrics, limitations
```

## Data notes

Cleaning removes, in order: 34,335 exact duplicates, 5,737 non-product rows (`POST`, `BANK CHARGES`, `TEST001`), 243 malformed stock codes, **232,833 rows with no customer id (22%)**, and 17,586 returns/cancellations. Returns are written to a separate file rather than deleted, because return behaviour is predictive.

Customer 18102 looks like an error and is not: 145 invoices over two years, 3.4% of all revenue. A real wholesale account, kept.
