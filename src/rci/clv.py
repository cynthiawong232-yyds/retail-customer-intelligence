"""Phase 2: how much will this customer spend in the next 90 days?

WHY THIS EXISTS (the business question)
---------------------------------------
Repurchase (Phase 3) says WHETHER someone comes back. It does not say whether
they are worth chasing. A customer with an 80% chance of spending GBP 40 and
one with a 30% chance of spending GBP 4,000 are completely different
propositions, and only the second justifies a phone call.

CLV sets the CEILING on what you may spend to acquire or keep someone. It is
the number that turns "who is at risk" into "what is the budget".

A NOTE ON THE WINDOW
--------------------
Textbook CLV is 12 months. This dataset spans two years, and a 12-month label
window would push the cutoff back to 2010-12, leaving one year of feature
history and NO room for a second out-of-time snapshot. So the target here is
90-day forward spend. Same technique, honest window, stated plainly rather
than quietly relabelled.

THE THING THAT MAKES THIS HARD
------------------------------
The target is ZERO-INFLATED:

    67.7% of training customers spend exactly GBP 0
    median target = GBP 0
    skew = 20.0

So the single best constant guess is zero, and any model that mostly predicts
near zero will look decent on MAE while being useless. This shapes everything
below, including which metrics get reported.

THREE APPROACHES, COMPARED
--------------------------
1. BG/NBD + GAMMA-GAMMA   the classic probabilistic pair
      BG/NBD      -> how MANY purchases will they make
      Gamma-Gamma -> how MUCH per purchase
      CLV = multiply

   BG/NBD's genuinely clever idea: you never SEE a customer quit. They just
   stop. So it models two things at once, a purchase rate while alive, and a
   per-purchase chance of silently dying forever. Fitting both gives
   P(still alive), which is not observable in the data.

2. XGBOOST REGRESSOR      features -> pounds, one model, no assumptions

3. HURDLE (two-part)      P(buys at all) x E[spend | they buy]
      Part 1 is a classifier, which is Phase 3's exact problem.
      Part 2 is a regressor trained ONLY on customers who did buy.
      This is the standard answer to a zero-inflated target, and note that
      BG/NBD+Gamma-Gamma is structurally the same idea in probabilistic form.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from rci.config import RANDOM_SEED

# lifetimes is unmaintained and emits deprecation noise on modern numpy.
# The fitters themselves work (verified against a smoke test), so the warnings
# are suppressed rather than the library being avoided.
warnings.filterwarnings("ignore", category=FutureWarning, module="lifetimes")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lifetimes")


# ---------------------------------------------------------------------------
# BG/NBD + Gamma-Gamma
# ---------------------------------------------------------------------------

def to_bgnbd_format(X: pd.DataFrame) -> pd.DataFrame:
    """Translate our features into what lifetimes expects.

    THE CLASSIC TRAP, and it silently produces a plausible-looking wrong
    model rather than an error:

        OUR "recency"        = days SINCE last purchase, counting back
                               from the cutoff. Big number = bad customer.

        LIFETIMES "recency"  = AGE at last purchase, counting forward from
                               their FIRST purchase. Big number = long-lived
                               customer, which is good.

    They are near opposites. The conversion is:

        lifetimes_recency = tenure - our_recency
                          = (last_purchase - first_purchase) in days

    Also note `frequency` here means REPEAT purchases, so it is our order
    count minus one. A customer with a single order has frequency 0, and
    BG/NBD treats them as having no repeat behaviour to learn from yet.
    """
    out = pd.DataFrame(
        {
            "frequency": (X["frequency"] - 1).astype(float),
            "recency": (X["tenure"] - X["recency"]).astype(float),
            "T": X["tenure"].astype(float),
        },
        index=X.index,
    )
    # The model is undefined if a purchase happened after the observation
    # ended. If this ever trips, the feature pipeline has a leak.
    assert (out["recency"] <= out["T"]).all(), "recency > T: impossible, check the split"
    assert (out["frequency"] >= 0).all()
    return out


def repeat_order_value(observation: pd.DataFrame) -> pd.Series:
    """Average value of a customer's REPEAT orders, excluding their first.

    Gamma-Gamma is defined on repeat transactions, so feeding it an average
    across all orders (including the first) is subtly wrong. For a customer
    with two orders that is a 50% error, which is not subtle at all.
    """
    orders = (
        observation.groupby(["customer_id", "invoice"])
        .agg(value=("line_total", "sum"), when=("invoice_date", "min"))
        .reset_index()
        .sort_values(["customer_id", "when"])
    )
    # Drop each customer's first order; what remains is their repeat business.
    # cumcount() numbers each customer's orders 0, 1, 2... in date order, so
    # keeping seq > 0 drops exactly the first one. Done this way rather than
    # with .apply(lambda g: g.iloc[1:]) because that is far slower and, under
    # pandas 3.0, include_groups=False strips customer_id out of the result.
    orders["seq"] = orders.groupby("customer_id").cumcount()
    repeats = orders[orders["seq"] > 0]
    return repeats.groupby("customer_id")["value"].mean()


def fit_bgnbd_gg(
    X: pd.DataFrame,
    observation: pd.DataFrame,
    penalizer: float = 0.01,
    gg_penalizer: float = 0.0,
):
    """Fit both halves. Returns (bgf, ggf, monetary) ready for prediction.

    TWO DIFFERENT PENALIZERS, AND THE REASON IS A REAL BUG THIS CODE ONCE HAD
    ---------------------------------------------------------------------
    `penalizer` is L2 regularisation on the fitted parameters. For BG/NBD it
    does what regularisation is supposed to do: r, alpha, a, b are all small
    dimensionless numbers, and 0.01 keeps sparse-history customers from
    dragging the likelihood somewhere silly.

    For Gamma-Gamma the SAME value silently destroys the model, because
    lifetimes penalises the RAW parameter values and Gamma-Gamma's `v` lives
    on the scale of MONEY. Fitted on this data, v = 544. An L2 term pulls it
    toward zero, and the optimiser compensates by collapsing q:

        penalizer   p        q        v        population mean
        0.0         1.896    3.820    544.11        366     <- healthy
        0.001      11.172    0.895     11.33     -1,205
        0.01        3.788    0.349      3.69        -21     <- what we shipped
        0.1         1.029    0.182      0.94         -1

    Gamma-Gamma's mean is v*p/(q-1), so it EXISTS ONLY IF q > 1. Below that
    the underlying gamma has infinite mean and lifetimes returns a negative
    number rather than raising. The consequence was negative predicted spend
    for all 1,575 never-repeated customers (30% of the base), because they
    get zero individual weight and therefore receive the population mean
    unmodified.

    MAE never noticed: -21 is close to the true value for people who mostly
    spend 0, so the aggregate metric looked fine while the model was
    returning negative money. `tests/test_clv.py` asserts non-negativity for
    exactly this reason, and that assertion is what found it.

    Unregularised is the correct answer here, not a workaround: the healthy
    fit reproduces the empirical mean repeat order value (366 vs 376) almost
    exactly. The general lesson is that a regularisation strength copied from
    a tutorial is only meaningful on the scale the tutorial's data had.
    """
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    bg = to_bgnbd_format(X)
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(bg["frequency"], bg["recency"], bg["T"])

    # Gamma-Gamma may only be fitted on customers who actually repeated, and
    # whose repeat spend is positive. One-order customers contribute nothing
    # about how much a REPEAT order is worth.
    monetary = repeat_order_value(observation).reindex(X.index)
    mask = (bg["frequency"] > 0) & monetary.notna() & (monetary > 0)

    ggf = GammaGammaFitter(penalizer_coef=gg_penalizer)
    ggf.fit(bg.loc[mask, "frequency"], monetary[mask])

    # A guard, not a hope. If a future data revision pushes q back under 1,
    # this must fail loudly instead of quietly emitting negative money again.
    q = float(ggf.params_["q"])
    if q <= 1:
        raise ValueError(
            f"Gamma-Gamma fitted q={q:.4f} <= 1, so its mean does not exist "
            "and expected order value will come back negative. Lower "
            "gg_penalizer or check the repeat-order-value distribution."
        )
    return bgf, ggf, monetary


def predict_bgnbd_gg(bgf, ggf, X: pd.DataFrame, monetary: pd.Series, days: int = 90):
    """CLV = expected purchases in the window x expected value per purchase.

    Deliberately NOT lifetimes' customer_lifetime_value(), which discounts
    over whole months and would obscure what is being computed. This is the
    same product, written out.
    """
    bg = to_bgnbd_format(X)

    # How many purchases in the next `days`, given their history.
    # np.array (not asarray) because we patch NaNs below, and asarray on a
    # pandas Series hands back a read-only view.
    n_purchases = np.array(
        bgf.conditional_expected_number_of_purchases_up_to_time(
            days, bg["frequency"], bg["recency"], bg["T"]
        ),
        dtype=float,
    )

    # BG/NBD returns NaN for some customers with frequency == 0, i.e. those
    # who have never placed a second order. Its conditional expectation
    # contains a Gaussian hypergeometric term that does not converge for
    # certain parameter combinations when there is no repeat history to
    # condition on. This is a documented limitation of the model, not a data
    # fault, and it hits ~8% of our customers.
    #
    # Filling with 0 would be a lie: it would assert these people definitely
    # will not buy. They are simply customers the model cannot speak about.
    # The honest stand-in is what the model predicts for the OTHER
    # never-repeated customers where the maths did work, so they are treated
    # as typical of their group rather than as certain non-buyers.
    bad = np.isnan(n_purchases)
    if bad.any():
        never_repeated = (bg["frequency"].to_numpy() == 0) & ~bad
        fallback = (
            float(np.median(n_purchases[never_repeated]))
            if never_repeated.any()
            else float(np.nanmedian(n_purchases))
        )
        n_purchases[bad] = fallback

    # Expected value per purchase. Gamma-Gamma shrinks a customer's observed
    # average TOWARD THE POPULATION MEAN, weighted by how many orders back it
    # up. Someone with 40 orders is trusted; someone with 2 is pulled toward
    # the crowd. That shrinkage is the whole point of using it over a raw
    # average. Concretely, with p=1.896 and q=3.820:
    #
    #     weight = p*x / (p*x + q - 1)
    #     1 repeat order  -> 0.40 own average, 0.60 population mean
    #     5 repeat orders -> 0.77 own average, 0.23 population mean
    #    40 repeat orders -> 0.96 own average, 0.04 population mean
    #
    # Note the weight formula needs q > 1 to behave; see fit_bgnbd_gg for the
    # bug that happens when it does not.
    m = monetary.fillna(monetary.median())
    value = np.asarray(
        ggf.conditional_expected_average_profit(bg["frequency"], m), dtype=float
    )

    return n_purchases * value


def dump_fitters(bgf, ggf) -> dict:
    """Save the lifetimes models as plain numbers.

    The fitted objects themselves cannot be pickled: BetaGeoFitter.fit builds
    a lambda closure internally and joblib refuses it. That is fine, because
    a fitted BG/NBD *is* just four numbers, and Gamma-Gamma three. Storing the
    parameters rather than the object is both smaller and version-proof: a
    dict of floats will still load in five years when the library will not.
    """
    return {
        "bgnbd": {k: float(v) for k, v in bgf.params_.items()},
        "gamma_gamma": {k: float(v) for k, v in ggf.params_.items()},
    }


def load_fitters(params: dict):
    """Rebuild working fitters from saved parameters."""
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    bgf = BetaGeoFitter()
    bgf.params_ = pd.Series(params["bgnbd"])
    ggf = GammaGammaFitter()
    ggf.params_ = pd.Series(params["gamma_gamma"])

    # Same validity check as at fit time. Loading is the other door into this
    # model, and a stale artifact saved before the penalizer fix would
    # otherwise reintroduce negative predictions with no warning.
    q = float(ggf.params_["q"])
    if q <= 1:
        raise ValueError(
            f"loaded Gamma-Gamma has q={q:.4f} <= 1: its mean does not exist. "
            "This artifact predates the penalizer fix; retrain it."
        )
    return bgf, ggf


def bgnbd_coverage(bgf, X: pd.DataFrame, days: int = 90) -> dict:
    """How many customers BG/NBD can actually speak about. Reported, not hidden."""
    bg = to_bgnbd_format(X)
    raw = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(
            days, bg["frequency"], bg["recency"], bg["T"]
        ),
        dtype=float,
    )
    bad = np.isnan(raw)
    return {
        "n": len(raw),
        "undefined": int(bad.sum()),
        "pct_undefined": round(100 * bad.mean(), 1),
        "all_undefined_are_never_repeated": bool(
            (bg["frequency"].to_numpy()[bad] == 0).all()
        ),
    }


# ---------------------------------------------------------------------------
# XGBoost challengers
# ---------------------------------------------------------------------------

def fit_regressor(X_fit, y_fit, X_val, y_val, log_target: bool = False):
    """Direct regression on pounds, optionally on a log scale.

    WHY log_target MATTERS HERE, and it is the opposite of the Phase 3 story:

    Trees are invariant to monotonic transforms of FEATURES, so logging
    inputs did literally nothing for the classifier (verified: identical
    predictions). The TARGET is a different matter. Squared-error loss is
    dominated by large values: one customer at GBP 62,000 generates a
    gradient that swamps a hundred at GBP 60. Modelling log(1+spend) puts
    every customer on comparable footing.

    The catch: predictions come back in log space and expm1() undoes the log
    but NOT the bias. E[exp(z)] > exp(E[z]) by Jensen's inequality, so the
    back-transform systematically under-predicts the mean. Reported honestly
    below rather than silently corrected.
    """
    model = XGBRegressor(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    yf = np.log1p(y_fit) if log_target else y_fit
    yv = np.log1p(y_val) if log_target else y_val
    model.fit(X_fit, yf, eval_set=[(X_val, yv)], verbose=False)
    return model


def predict_regressor(model, X, log_target: bool = False) -> np.ndarray:
    p = model.predict(X)
    if log_target:
        p = np.expm1(p)
    # Spend cannot be negative. Squared-error regression has no such scruple.
    return np.clip(p, 0, None)


def fit_hurdle(X_fit, y_fit, X_val, y_val):
    """Two-part model: P(spends anything) x E[spend | they spend].

    The standard treatment for a zero-inflated target, and the honest way to
    describe what is going on: 68% of customers produce a structural zero,
    and the interesting variation lives entirely in the other 32%.

    Part 1 is exactly Phase 3's problem, so the same classifier settings are
    used. Part 2 is trained ONLY on spenders, which is what lets it learn
    about size without being drowned in zeros.

    Note this is structurally the same idea as BG/NBD + Gamma-Gamma:
    how-likely times how-much. One is probabilistic, one is learned.
    """
    clf = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="binary:logistic", eval_metric="aucpr",
        early_stopping_rounds=50, random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(X_fit, (y_fit > 0).astype(int),
            eval_set=[(X_val, (y_val > 0).astype(int))], verbose=False)

    spenders = y_fit > 0
    spenders_val = y_val > 0
    reg = XGBRegressor(
        n_estimators=2000, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", eval_metric="rmse",
        early_stopping_rounds=50, random_state=RANDOM_SEED, n_jobs=-1,
    )
    # log target for part 2: conditional on buying, spend is still very
    # right-skewed, and this half is pure regression on positive amounts.
    reg.fit(X_fit[spenders], np.log1p(y_fit[spenders]),
            eval_set=[(X_val[spenders_val], np.log1p(y_val[spenders_val]))],
            verbose=False)
    return clf, reg


def predict_hurdle(clf, reg, X) -> np.ndarray:
    p_buy = clf.predict_proba(X)[:, 1]
    amount = np.expm1(reg.predict(X))
    return np.clip(p_buy * amount, 0, None)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred) -> dict:
    """Regression metrics, plus the two that actually matter here.

    MAE   average miss in pounds. Treats a GBP 10 error and a GBP 10,000
          error as 1000 vs 1 of the same thing.
    RMSE  squares errors first, so one huge miss hurts far more than many
          small ones. ALWAYS >= MAE, and the GAP between them tells you how
          much of your error is concentrated in a few big misses.
    R2    fraction of variance explained. 0 means no better than always
          guessing the mean. NEGATIVE means worse than that, which is very
          possible on a zero-inflated target.

    SPEARMAN is the one to lead with. It measures whether the RANKING is
          right, ignoring the magnitudes. Budget allocation is a ranking
          problem: you want to know who the top 500 are, not their exact
          pound values. A model can rank beautifully and still be badly
          calibrated in absolute terms.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # A constant prediction has no ranking to assess, so Spearman is
    # genuinely undefined rather than zero. NaN says "not applicable"; 0.0
    # would falsely imply the baseline ranks as badly as a random model.
    rank = float("nan") if np.ptp(y_pred) == 0 else float(spearmanr(y_true, y_pred).statistic)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "spearman": rank,
        "mean_pred": float(y_pred.mean()),
        "mean_actual": float(y_true.mean()),
    }


def decile_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Sort by prediction, cut into ten groups, compare predicted vs actual.

    The business-readable view, and the one that exposes miscalibration a
    single metric hides: a model can have a fine RMSE while systematically
    under-predicting the top decile, which is exactly where the money is.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                       "p": np.asarray(y_pred, dtype=float)})
    df["decile"] = pd.qcut(
        df["p"].rank(method="first", ascending=False), n_bins,
        labels=range(1, n_bins + 1),
    ).astype(int)
    out = df.groupby("decile").agg(
        customers=("y", "size"),
        predicted=("p", "mean"),
        actual=("y", "mean"),
        actual_total=("y", "sum"),
    )
    out["share_of_revenue"] = (100 * out["actual_total"] / df["y"].sum()).round(1)
    return out
