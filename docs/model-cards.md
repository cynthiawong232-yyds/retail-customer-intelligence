# Model Cards

One card per model. The format follows [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993), trimmed to what is actually knowable here.

The purpose of a model card is to make **misuse harder**. Every card therefore leads with what the model is for, and says plainly what it must not be used for. Sections marked "Limitations" are not disclaimers: each one is a measured property of this specific fit.

**Shared across all four models**

| | |
|---|---|
| Data | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), CC BY 4.0. 1,067,371 transactions, one UK online gift wholesaler, 2009-12-01 to 2011-12-09 |
| After cleaning | 776,577 purchase rows, 5,852 customers, 4,619 products (4,260 of them appear within the training observation window, which is what the embedding sees) |
| Split | Out-of-time. Train cutoff 2011-06-12, test cutoff 2011-09-10, 90-day label window |
| Owner | Portfolio project. Not in production anywhere |
| Licence | Data CC BY 4.0. Code MIT |

**One population caveat applies to every model below.** 22% of rows have no customer id and were dropped, so every customer-level model is fitted on identified customers only. If identification correlates with behaviour (it plausibly does: account holders are likelier to be repeat wholesale buyers than one-off guests), all four models are biased toward the identified population and will generalise worse to anonymous traffic.

---

## 1. Customer Segmentation

**Model:** KMeans, k=4, on log1p + standardised RFM.

### Intended use
Grouping the customer base into a handful of nameable behavioural clusters so that one campaign can address many people. Designed for **campaign targeting and reporting**.

### Not for
- Individual decisions. A segment is a description of a group, not a prediction about a person.
- Pricing, credit, or anything where a customer suffers from being in the wrong cluster. Cluster assignment is unstable near boundaries and the response exposes `distance_to_centroid` for exactly that reason.

### Training
| | |
|---|---|
| Features | recency, frequency, monetary (3 only) |
| Transform | `log1p` then `StandardScaler`, fitted on the training snapshot |
| Selection | k=4 from the inertia elbow plus a local silhouette peak |
| Fitted on | 4,951 customers, cutoff 2011-06-12 |

### Performance
Silhouette 0.372. Segment sizes 811 / 1,482 / 691 / 1,967, and Champions (16.4% of customers) hold 67.9% of revenue.

### Limitations
- **Silhouette preferred k=2.** Four was chosen because two segments cannot support four campaigns. A business constraint overrode the metric, and that is a judgement call, not a measurement.
- **A better silhouette can mean a worse model.** Unscaled RFM scores 0.925 by isolating two outlier whales and putting 98.5% of customers in one cluster. Never read the score without the sizes.
- **Only 3 features.** More dimensions produce clusters nobody can name, and a cluster nobody can name is one marketing will not use.
- **No stability guarantee across refits.** KMeans is seeded (`random_state=42`); change the seed or the data and segment ids may permute. Names are attached by profile, not by id.

---

## 2. Customer Lifetime Value

**Model served:** BG/NBD + Gamma-Gamma. **Challenger returned alongside:** XGBoost hurdle.

### Intended use
Estimating **expected spend over the next 90 days**, to set a ceiling on acquisition and retention budget. The number that turns "who is at risk" into "how much may we spend".

### Not for
- **Profit.** This is revenue before cost of goods, and the API response says so in words.
- **Annual CLV.** The window is 90 days. See the limitation below.
- Individual guarantees. 7.9% of customers receive a group median rather than a personal estimate.

### Training
| | |
|---|---|
| BG/NBD input | repeat frequency, age-at-last-purchase, total age (converted from our recency, which is its opposite) |
| Gamma-Gamma input | repeat frequency, mean repeat order value (first order excluded) |
| Regularisation | BG/NBD `penalizer=0.01`. Gamma-Gamma **unregularised**, see below |
| Fitted params | BG/NBD r=0.6028, alpha=47.5292, a=0.0799, b=0.7127. Gamma-Gamma p=1.896, q=3.820, v=544.11 |

### Performance (test snapshot, 5,256 customers)
| | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|
| always the mean | 618.47 | 3239.31 | -0.004 | n/a |
| repeat last 90d | 415.79 | 2200.95 | 0.537 | 0.480 |
| **BG/NBD + Gamma-Gamma** | **388.90** | **2083.77** | **0.585** | 0.552 |
| XGBoost (raw GBP) | 429.10 | 2450.49 | 0.426 | 0.526 |
| XGBoost (log target) | 466.94 | 3016.48 | 0.130 | **0.569** |
| hurdle | 406.74 | 2530.00 | 0.388 | 0.554 |

### Limitations
- **The window is 90 days, not 12 months.** Two years of data cannot support a 12-month label window and a second out-of-time snapshot. Anyone reading this as annual CLV will overstate budget by roughly 4x.
- **7.9% of customers cannot be scored.** BG/NBD's conditional expectation contains a hypergeometric term that will not converge without repeat history. Every affected customer is verifiably a never-repeater. They receive the median prediction for other never-repeaters, **not zero**, because zero would assert they definitely will not buy.
- **Gamma-Gamma must be fitted unregularised here.** `lifetimes` penalises raw parameter values, and `v` is on the scale of money (544). A penalizer of 0.01 collapses `q` from 3.820 to 0.349, and the model's mean `v*p/(q-1)` exists only for `q > 1`. Below it the library returns negative money with no error. This shipped once and MAE did not detect it. A hard error now guards both fit and load.
- **RMSE is 5.4x MAE.** Error is heavily concentrated in a few very large accounts. Aggregate accuracy says little about any individual whale.
- **The log-target model must not be used for magnitudes.** It ranks best and under-predicts the mean by a factor of 6.6 (Jensen's inequality). It is reported, not served.
- **Zero-inflated target** (67.7% exact zeros). MAE alone can look healthy on a model that predicts near-zero for everyone; read Spearman alongside it.

---

## 3. Repurchase Prediction

**Model:** XGBoost binary classifier, 35 trees, plus precomputed SHAP.

### Intended use
Ranking customers by probability of purchasing within 90 days, so a **finite retention budget** goes to the customers where it changes an outcome. The output is a **ranking**; the decile is the intended unit of action.

### Not for
- **Using the raw probability as a probability.** It is uncalibrated for the scoring period and runs systematically low. Every response says so.
- Withholding service, or any adverse action against a customer. This predicts commercial behaviour, nothing about the person.

### Training
| | |
|---|---|
| Features | 13, order fixed by a shared constant and enforced by a test |
| Target | `repurchased` (did they buy in the next 90 days) |
| Settings | `max_depth=4`, `learning_rate=0.05`, `min_child_weight=5`, `subsample=0.8`, early stopping patience 50 |
| Trees | 85 built, **35 used**. The booster is trimmed to `best_iteration` and the trim is verified bit-identical |
| Missing values | Native. 31% of customers have NaN `avg_days_between_orders` and are still scored |

### Performance (test snapshot)
| | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| recency rule | 0.6855 | 0.7623 | n/a |
| logistic regression | 0.7580 | 0.7891 | 0.2007 |
| **XGBoost** | **0.7742** | **0.7986** | **0.1976** |

Base rate 0.4351, which is the PR-AUC floor. Top decile repurchases at 93.3% against a 43.5% average, **2.15x lift**; top three deciles capture 53% of all repurchasers.

### Limitations
- **Uncalibrated for the scoring period.** Trained on a 32.3% base-rate window, applied to a 43.5% one. Mean predicted 0.309 against actual 0.435. Isotonic recalibration fixes it (Brier 0.2000 to 0.1816) without changing ROC-AUC, because isotonic is monotonic. **Ranking is sound; absolute values are not.**
- **XGBoost beat logistic regression by only 2%.** Most of the signal is linear in the engineered features. If interpretability mattered more than 2%, the honest recommendation is logistic regression.
- **SHAP values are precomputed** for the 5,256 known customers. An unknown customer gets a score with no explanation.
- **Seasonality is not modelled**, only observed. There are two Christmases in this data, which is not enough to fit a seasonal term.
- **Gain and SHAP disagree** on feature ranking (`recency` is 5th by gain, 1st by SHAP). They answer different questions; neither is the "true" importance.

---

## 4. Product Recommender

**Model:** item2vec (gensim Word2Vec, 64-dim) retrieval, then an XGBoost ranker over ~275 candidates.

### Intended use
Choosing **12 products** for one customer's email or "recommended for you" row. Each recommendation is labelled reorder or discovery, because the two have very different value.

### Not for
- **Claiming lift over doing nothing.** Its real competitor is "recommend what they bought before", which it beats by 5.6%. Any business case must be made against that rule, not against random.
- Cold-start customers or new products. Neither has a vector.
- Ranking anything outside this retailer's catalogue and season.

### Training
| | |
|---|---|
| Corpus | 26,077 baskets from the training observation window |
| `window` | **1000**, not the default 5. A basket has no word order, so a product's context is the whole basket (mean 20.4 items, max 271) |
| `sg=1`, `negative=10`, `min_count=5`, `epochs=10`, `workers=1` | skip-gram for a long tail; one worker for determinism, not speed |
| Coverage | 3,795 of 4,260 products get a vector (89.1%) |
| Customer vector | Recency-weighted mean of purchased item vectors, half-life 90 days |
| Candidates | Union of top-200 retrieved, all previously bought, top-50 popular |
| Ranker | XGBoost, 13 candidate features, ~1.35M training rows, grouped by customer |

### Performance (test snapshot, precision@12)
| | all items | new items only |
|---|---|---|
| most popular | 0.0542 | 0.0198 |
| co-purchase | 0.1167 | 0.0388 |
| repeat what they bought | 0.2254 | 0.0000 |
| stage 1 only | 0.1601 | 0.0431 |
| **two-stage (pointwise)** | **0.2380** | **0.0476** |
| two-stage (rank:ndcg) | 0.2377 | 0.0455 |

Reported twice on purpose. 43.1% of future purchases are repeats, so the all-items table is dominated by reorders and flatters every method that can replay history.

### Limitations
- **Cold start, two ways.** 6.5% of products bought after the cutoff never appeared before it, and 10.9% of the catalogue falls below `min_count=5`. Neither group can ever be recommended. A production system needs a content-based fallback.
- **The embedding is one fit, reused for both snapshots.** Word2Vec spaces are not aligned across runs, so refitting per snapshot would silently invalidate the ranker. This is realistic (nobody retrains embeddings daily) but it means the test snapshot is scored with a slightly stale space.
- **`rank:ndcg` did not beat a pointwise classifier** (0.2823 vs 0.2834 NDCG). The learning-to-rank objective is not automatically better.
- **Retrieval is the hard ceiling.** The full candidate set has recall 0.457; the final model reaches 0.145. Stage 2 can reorder but never invent.
- **Recommendation lists are precomputed**, so only customers known at export time can be served. `/similar` is live; `/recommend` is a lookup, and the response says which.
- **Popularity bias.** Two of three candidate sources (popular, and to a degree co-purchase) favour bestsellers, so the long tail is under-served. Not measured here; catalogue coverage and intra-list diversity would be the metrics to add.
- **No feedback loop is modelled.** In production, showing a product causes purchases of it, so a deployed version of this model would train on data it created. Offline metrics cannot see that.

---

## What is not modelled anywhere

- **Price and promotion.** No discount, campaign or margin data exists in this dataset, so none of these models knows whether a purchase was profitable.
- **Returns as an outcome.** Return behaviour is a *feature* (`n_returns`, `returned_value`) but never a target. A customer who buys and returns everything looks good to all four models.
- **Anything about the person.** There is no demographic, geographic (beyond country, ~90% UK) or behavioural web data. These models see transactions and nothing else, which is a limitation and also a privacy property worth keeping.
