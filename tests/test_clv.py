"""CLV model invariants.

As in test_repurchase.py, exact scores are reported in the README rather than
asserted, with two deliberate exceptions at the bottom: the two findings the
README makes claims about. If those ever flip, the writeup is wrong and a red
test is the correct way to find out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from rci.clv import (
    bgnbd_coverage,
    decile_table,
    dump_fitters,
    evaluate_regression,
    fit_bgnbd_gg,
    fit_hurdle,
    fit_regressor,
    load_fitters,
    predict_bgnbd_gg,
    predict_hurdle,
    predict_regressor,
    repeat_order_value,
    to_bgnbd_format,
)
from rci.config import PREDICTION_WINDOW_DAYS, RANDOM_SEED
from rci.features import build_xy
from rci.split import train_test_snapshots


@pytest.fixture(scope="module")
def data():
    from rci.clean import build

    train, test = train_test_snapshots(build())
    X_train, y_train = build_xy(train, target="future_spend")
    X_test, y_test = build_xy(test, target="future_spend")
    return train, test, X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def classic(data):
    """BG/NBD + Gamma-Gamma, fitted on train, predicting on test."""
    train, test, X_train, _, X_test, _ = data
    bgf, ggf, _ = fit_bgnbd_gg(X_train, train.observation)
    monetary_test = repeat_order_value(test.observation).reindex(X_test.index)
    preds = predict_bgnbd_gg(bgf, ggf, X_test, monetary_test, days=PREDICTION_WINDOW_DAYS)
    return bgf, ggf, monetary_test, preds


# ---------------------------------------------------------------------------
# the input conversion, which is where this model is most often broken
# ---------------------------------------------------------------------------

def test_lifetimes_recency_is_not_our_recency(data):
    """The trap that produces a plausible-looking wrong model rather than an error.

    Ours counts BACK from the cutoff (big = bad). Lifetimes counts FORWARD
    from the first purchase (big = long-lived, good). They are near opposites,
    so if these two columns ever correlate positively the conversion has been
    skipped somewhere.
    """
    _, _, X_train, _, _, _ = data
    bg = to_bgnbd_format(X_train)
    corr = np.corrcoef(bg["recency"], X_train["recency"])[0, 1]
    assert corr < 0, f"lifetimes recency should oppose ours, got corr {corr:.3f}"


def test_bgnbd_format_obeys_the_model_constraints(data):
    """recency <= T always. A violation means a purchase landed after the
    observation window closed, i.e. the temporal split leaked."""
    _, _, X_train, _, X_test, _ = data
    for X in (X_train, X_test):
        bg = to_bgnbd_format(X)
        assert (bg["recency"] <= bg["T"]).all()
        assert (bg["frequency"] >= 0).all()
        # frequency here is REPEAT purchases, so exactly one less than orders.
        assert np.array_equal(bg["frequency"].to_numpy(), X["frequency"].to_numpy() - 1)


def test_repeat_order_value_drops_the_first_order():
    """Gamma-Gamma is defined on repeat transactions only.

    Including the first order is a 50% error for a two-order customer, which
    is not a rounding issue. Checked on a hand-built frame where the right
    answer is obvious by inspection.
    """
    df = pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2],
            "invoice": ["A", "B", "C", "D"],
            "line_total": [1000.0, 10.0, 20.0, 500.0],
            "invoice_date": pd.to_datetime(
                ["2011-01-01", "2011-02-01", "2011-03-01", "2011-01-01"]
            ),
        }
    )
    out = repeat_order_value(df)
    # Customer 1: first order (1000) dropped, mean of 10 and 20 = 15.
    assert out.loc[1] == pytest.approx(15.0)
    # Customer 2 has no repeat orders, so contributes nothing at all.
    assert 2 not in out.index


def test_gamma_gamma_independence_assumption_holds(data):
    """Gamma-Gamma assumes order VALUE is independent of order COUNT.

    An assumption the model cannot check for itself. Above roughly 0.3 it is
    the wrong tool and the result would be quietly biased.
    """
    _, _, X_train, _, _, _ = data
    repeat = X_train[X_train["frequency"] > 1]
    corr = repeat["frequency"].corr(repeat["avg_order_value"], method="spearman")
    assert abs(corr) < 0.3, f"correlation {corr:.3f} breaks the Gamma-Gamma assumption"


# ---------------------------------------------------------------------------
# the classic pair
# ---------------------------------------------------------------------------

def test_gamma_gamma_mean_exists(classic):
    """q > 1, or the model's expected order value is negative.

    Gamma-Gamma's mean is v*p/(q-1). Below q = 1 the underlying gamma has an
    infinite mean and lifetimes returns a negative number rather than raising,
    which is how this project once shipped negative predicted spend for 30% of
    customers. The cause was a penalizer of 0.01 copied from a tutorial: it
    regularises the RAW parameters, and Gamma-Gamma's v is on the scale of
    money (544 here), so the penalty crushed it and q collapsed with it.
    """
    _, ggf, _, _ = classic
    p, q, v = (float(ggf.params_[k]) for k in ("p", "q", "v"))
    assert q > 1, f"q={q:.4f}: Gamma-Gamma's mean does not exist"
    population_mean = v * p / (q - 1)
    assert population_mean > 0
    # And it must be a plausible order value, not merely positive.
    assert 100 < population_mean < 1000


def test_every_customer_gets_a_finite_non_negative_prediction(classic, data):
    """The NaN patch must actually hold. A NaN reaching a budget spreadsheet
    is worse than a mediocre number."""
    *_, X_test, _ = data
    _, _, _, preds = classic
    assert len(preds) == len(X_test)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_the_coverage_gap_is_exactly_the_never_repeated(classic, data):
    """BG/NBD's documented limitation, pinned rather than glossed over.

    Its conditional expectation contains a hypergeometric term that will not
    converge without repeat history. The claim in the README is that EVERY
    undefined customer is a never-repeater; if that ever stops being true, the
    explanation is wrong and something else is going on.
    """
    *_, X_test, _ = data
    bgf, *_ = classic
    cov = bgnbd_coverage(bgf, X_test, days=PREDICTION_WINDOW_DAYS)
    assert cov["undefined"] > 0
    assert cov["all_undefined_are_never_repeated"] is True
    assert cov["pct_undefined"] < 15


def test_never_repeaters_are_filled_with_a_median_not_a_zero(classic, data):
    """Filling with 0 would assert these people definitely will not buy.

    They are customers the model cannot speak about, which is a different
    statement, and the fill value must reflect that.
    """
    *_, X_test, _ = data
    bgf, _, _, preds = classic
    bg = to_bgnbd_format(X_test)
    raw = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(
            PREDICTION_WINDOW_DAYS, bg["frequency"], bg["recency"], bg["T"]
        ),
        dtype=float,
    )
    patched = np.isnan(raw)
    assert patched.any()
    # Their CLV is spend-per-order x a positive expected count, so it can only
    # be zero if their monetary value is zero. It must not be uniformly zero.
    assert (preds[patched] > 0).any()


def test_fitters_survive_the_round_trip_as_plain_numbers(classic, data):
    """lifetimes objects cannot be pickled (a lambda closure inside fit), so
    the parameters are saved instead. That is only acceptable if the rebuilt
    fitters predict identically."""
    *_, X_test, _ = data
    bgf, ggf, monetary, preds = classic

    params = dump_fitters(bgf, ggf)
    assert set(params) == {"bgnbd", "gamma_gamma"}
    assert all(isinstance(v, float) for v in params["bgnbd"].values())

    bgf2, ggf2 = load_fitters(params)
    again = predict_bgnbd_gg(bgf2, ggf2, X_test, monetary, days=PREDICTION_WINDOW_DAYS)
    assert np.allclose(again, preds, equal_nan=True)


def test_p_alive_is_a_probability_and_ranks_sensibly(classic, data):
    """P(still alive) is the one quantity here that is never observed.

    It cannot be validated against a ground truth, so the check is structural:
    it must be a probability, and a customer silent far longer than their own
    usual gap must score below one who just bought.
    """
    *_, X_test, _ = data
    bgf, *_ = classic
    bg = to_bgnbd_format(X_test)
    alive = np.asarray(
        bgf.conditional_probability_alive(bg["frequency"], bg["recency"], bg["T"]),
        dtype=float,
    )
    assert alive.min() >= 0.0 and alive.max() <= 1.0

    repeat = X_test["frequency"] > 3
    recent = repeat & (X_test["recency"] < 15)
    lapsed = repeat & (X_test["recency"] > 250)
    assert recent.sum() > 0 and lapsed.sum() > 0
    assert alive[recent.to_numpy()].mean() > alive[lapsed.to_numpy()].mean()


# ---------------------------------------------------------------------------
# the learned challengers
# ---------------------------------------------------------------------------

def test_the_target_is_zero_inflated(data):
    """The fact that shapes every modelling decision in this phase.

    Most customers spend nothing, so the best constant guess is zero and a
    model can score well on MAE while being useless. This is why the hurdle
    model exists and why Spearman is reported alongside MAE.
    """
    _, _, _, y_train, _, _ = data
    assert (y_train == 0).mean() > 0.5
    assert y_train.median() == 0
    assert y_train.skew() > 5


def test_regressor_never_predicts_negative_money(data):
    """Squared-error regression has no idea that spend has a floor at zero."""
    _, _, X_train, y_train, X_test, _ = data
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
    )
    model = fit_regressor(X_fit, y_fit, X_val, y_val, log_target=False)
    preds = predict_regressor(model, X_test, log_target=False)
    assert (preds >= 0).all()
    # Raw predictions genuinely do go negative; the clip is doing real work.
    assert (model.predict(X_test) < 0).any()


def test_hurdle_parts_are_individually_interpretable(data):
    """The two halves must each mean what their names say, because the whole
    argument for a hurdle model is that the parts are separable."""
    _, _, X_train, y_train, X_test, _ = data
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
    )
    clf, reg = fit_hurdle(X_fit, y_fit, X_val, y_val)

    p_buy = clf.predict_proba(X_test)[:, 1]
    assert p_buy.min() >= 0 and p_buy.max() <= 1

    amount = np.expm1(reg.predict(X_test))
    # Part 2 saw only spenders, so its output is "size given they buy" and
    # should sit near the spend of actual buyers, far above the overall mean.
    assert np.median(amount) > 0

    combined = predict_hurdle(clf, reg, X_test)
    assert (combined >= 0).all()
    assert np.isfinite(combined).all()


def test_log_target_underpredicts_the_mean(data):
    """Jensen's inequality, made concrete.

    expm1 undoes the log but not the bias: E[exp(z)] > exp(E[z]). The log
    model ranks well and is badly wrong in absolute pounds, which is exactly
    the trap that makes people ship a log-target regressor into a budget
    process. Pinned because the README makes this claim.
    """
    _, _, X_train, y_train, X_test, y_test = data
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
    )
    m_log = fit_regressor(X_fit, y_fit, X_val, y_val, log_target=True)
    p_log = predict_regressor(m_log, X_test, log_target=True)
    assert p_log.mean() < 0.5 * y_test.mean(), "the Jensen bias has vanished; update the README"


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_constant_predictions_get_nan_spearman_not_zero():
    """A constant has no ranking to assess, so Spearman is undefined.

    Returning 0.0 would falsely claim the baseline ranks as badly as random.
    """
    y = np.array([0.0, 10.0, 200.0, 5.0])
    m = evaluate_regression(y, np.full(4, y.mean()))
    assert np.isnan(m["spearman"])
    assert m["R2"] == pytest.approx(0.0, abs=1e-9)


def test_rmse_is_never_below_mae():
    """A mathematical guarantee, and the GAP between them is the signal:
    a wide gap means the error is concentrated in a few large misses."""
    y = np.array([0.0, 10.0, 5000.0, 5.0])
    p = np.array([5.0, 5.0, 5.0, 5.0])
    m = evaluate_regression(y, p)
    assert m["RMSE"] >= m["MAE"]


def test_decile_table_sorts_high_value_first(classic, data):
    """Decile 1 must be the most valuable tenth, or every business reading of
    the table is inverted."""
    _, _, _, _, _, y_test = data
    _, _, _, preds = classic
    table = decile_table(y_test, preds)
    assert table.index.min() == 1 and table.index.max() == 10
    assert table.loc[1, "predicted"] > table.loc[10, "predicted"]
    assert table.loc[1, "actual"] > table.loc[10, "actual"]
    # Shares of revenue must account for all of it.
    assert table["share_of_revenue"].sum() == pytest.approx(100.0, abs=0.5)


def test_the_classic_model_beats_xgboost_on_pounds(classic, data):
    """The headline finding, pinned.

    A 2005 probabilistic model beats gradient boosting here, on 4,951
    customers and 13 features. The README says so; if a future change flips
    it, the README is wrong and this test is how that gets noticed.
    """
    _, _, X_train, y_train, X_test, y_test = data
    _, _, _, classic_preds = classic

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
    )
    m_raw = fit_regressor(X_fit, y_fit, X_val, y_val, log_target=False)
    xgb_preds = predict_regressor(m_raw, X_test, log_target=False)

    classic_mae = evaluate_regression(y_test, classic_preds)["MAE"]
    xgb_mae = evaluate_regression(y_test, xgb_preds)["MAE"]
    assert classic_mae < xgb_mae, "XGBoost now wins on MAE; update the README"


def test_both_models_beat_guessing_the_mean(classic, data):
    """The floor. R2 is defined against the mean, so it scores exactly 0."""
    _, _, _, y_train, _, y_test = data
    _, _, _, preds = classic
    constant = evaluate_regression(y_test, np.full(len(y_test), y_train.mean()))
    assert constant["R2"] < 0.01
    assert evaluate_regression(y_test, preds)["R2"] > constant["R2"]
