"""Train, compare and save the CLV models.

    python -m rci.train_clv

Four contenders, baselines first: a constant, then the probabilistic pair,
then two learned approaches.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rci.clean import build
from rci.clv import (
    bgnbd_coverage,
    decile_table,
    dump_fitters,
    load_fitters,
    evaluate_regression,
    fit_bgnbd_gg,
    fit_hurdle,
    fit_regressor,
    predict_bgnbd_gg,
    predict_hurdle,
    predict_regressor,
    to_bgnbd_format,
)
from rci.config import ARTIFACTS, PREDICTION_WINDOW_DAYS, RANDOM_SEED
from rci.features import FEATURE_COLUMNS, build_xy
from rci.split import train_test_snapshots


def main() -> None:
    pd.set_option("display.width", 220)

    purchases = build()
    train, test = train_test_snapshots(purchases)
    X_train, y_train = build_xy(train, target="future_spend")
    X_test, y_test = build_xy(test, target="future_spend")

    print("THE TARGET: 90-day forward spend")
    print("=" * 78)
    for name, v in (("train", y_train), ("test", y_test)):
        print(f"  {name}: n={len(v):,}  zeros={100*(v==0).mean():.1f}%  "
              f"mean={v.mean():>8,.0f}  median={v.median():>6,.0f}  max={v.max():>9,.0f}")
    print(f"\n  skew {y_train.skew():.1f}. The median customer spends NOTHING, so a")
    print("  model that always predicts 0 is a genuinely hard baseline to beat.")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
    )

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    # --- baseline 0: predict the training mean for everyone --------------
    # The floor. R2 is defined against this, so it scores exactly 0.0 by
    # construction and any model failing to beat it is worse than useless.
    preds["always the mean"] = np.full(len(y_test), y_train.mean())
    results["always the mean"] = evaluate_regression(y_test, preds["always the mean"])

    # --- baseline 1: last 90 days repeats itself -------------------------
    # The naive business guess, and a surprisingly strong one.
    preds["repeat last 90d"] = X_test["spend_last_90d"].to_numpy()
    results["repeat last 90d"] = evaluate_regression(y_test, preds["repeat last 90d"])

    # --- BG/NBD + Gamma-Gamma --------------------------------------------
    bgf, ggf, monetary_train = fit_bgnbd_gg(X_train, train.observation)
    from rci.clv import repeat_order_value

    monetary_test = repeat_order_value(test.observation).reindex(X_test.index)
    preds["BG/NBD + Gamma-Gamma"] = predict_bgnbd_gg(
        bgf, ggf, X_test, monetary_test, days=PREDICTION_WINDOW_DAYS
    )
    results["BG/NBD + Gamma-Gamma"] = evaluate_regression(
        y_test, preds["BG/NBD + Gamma-Gamma"]
    )

    print("\n\nBG/NBD FITTED PARAMETERS")
    print("=" * 78)
    print("  " + bgf.params_.round(4).to_string().replace("\n", "\n  "))
    print("\n  r, alpha describe the PURCHASE RATE across customers (a gamma).")
    print("  a, b describe the DROPOUT probability after each purchase (a beta).")
    print("  Four numbers standing in for 4,951 customers' behaviour.")

    p_, q_, v_ = (float(ggf.params_[k]) for k in ("p", "q", "v"))
    print("\n  GAMMA-GAMMA")
    print(f"    p={p_:.4f}  q={q_:.4f}  v={v_:.4f}")
    print(f"    population mean order value = v*p/(q-1) = {v_ * p_ / (q_ - 1):,.2f}")
    print(f"    empirical mean repeat order value       = {monetary_train.mean():,.2f}")
    print("    q MUST exceed 1 or this mean does not exist and comes back")
    print("    NEGATIVE. It is fitted unregularised on purpose: lifetimes")
    print("    penalises raw parameters, and v is on the scale of money, so")
    print("    the usual penalizer=0.01 collapsed q to 0.35 and produced")
    print("    negative predicted spend for 30% of customers.")

    bg_test = to_bgnbd_format(X_test)
    alive = bgf.conditional_probability_alive(
        bg_test["frequency"], bg_test["recency"], bg_test["T"]
    )
    print(f"\n  P(still alive): mean {alive.mean():.3f}, "
          f"{(alive < 0.2).sum():,} customers below 0.2")
    print("  This is the quantity you cannot observe directly. Nobody tells")
    print("  you they have churned; they just stop.")

    cov = bgnbd_coverage(bgf, X_test, days=PREDICTION_WINDOW_DAYS)
    print(f"\n  COVERAGE LIMIT: undefined for {cov['undefined']:,} of {cov['n']:,} "
          f"customers ({cov['pct_undefined']}%)")
    print(f"  all of them never placed a second order: {cov['all_undefined_are_never_repeated']}")
    print("  The hypergeometric term does not converge without repeat history.")
    print("  Filled with the median prediction for other never-repeated")
    print("  customers, NOT with 0, which would assert they definitely will")
    print("  not buy. XGBoost has no such gap: it scores every customer.")

    # --- XGBoost, raw pounds ---------------------------------------------
    m_raw = fit_regressor(X_fit, y_fit, X_val, y_val, log_target=False)
    preds["XGBoost (raw GBP)"] = predict_regressor(m_raw, X_test, log_target=False)
    results["XGBoost (raw GBP)"] = evaluate_regression(y_test, preds["XGBoost (raw GBP)"])

    # --- XGBoost, log target ---------------------------------------------
    m_log = fit_regressor(X_fit, y_fit, X_val, y_val, log_target=True)
    preds["XGBoost (log target)"] = predict_regressor(m_log, X_test, log_target=True)
    results["XGBoost (log target)"] = evaluate_regression(
        y_test, preds["XGBoost (log target)"]
    )

    # --- hurdle ------------------------------------------------------------
    clf, reg = fit_hurdle(X_fit, y_fit, X_val, y_val)
    preds["hurdle P(buy) x E[spend]"] = predict_hurdle(clf, reg, X_test)
    results["hurdle P(buy) x E[spend]"] = evaluate_regression(
        y_test, preds["hurdle P(buy) x E[spend]"]
    )

    # --- the comparison ----------------------------------------------------
    print("\n\nRESULTS ON THE TEST SNAPSHOT (baselines first)")
    print("=" * 78)
    table = pd.DataFrame(results).T[["MAE", "RMSE", "R2", "spearman", "mean_pred"]]
    print(table.round(3).to_string())
    print(f"\n  actual mean spend: {y_test.mean():,.2f}")
    print("\n  MAE  = average miss in pounds")
    print("  RMSE = squares errors first, so big misses dominate. Always >= MAE;")
    print("         the GAP shows how concentrated the error is in a few whales.")
    print("  R2   = 0.0 means no better than always guessing the mean.")
    print("  spearman = is the RANKING right. Lead with this one: budget")
    print("         allocation needs to know WHO the top 500 are, not exact GBP.")

    # nan-safe: the constant baseline has no ranking, so it cannot win here.
    best = max(results, key=lambda k: np.nan_to_num(results[k]["spearman"], nan=-1.0))
    print(f"\n  best ranking: {best}  (spearman {results[best]['spearman']:.3f})")

    # --- where the money actually is --------------------------------------
    print(f"\n\nDECILES FOR THE BEST RANKER: {best}")
    print("=" * 78)
    print(decile_table(y_test, preds[best]).round(2).to_string())
    print("\n  'predicted' vs 'actual' in the top rows is the calibration check.")
    print("  Under-predicting decile 1 is the expensive kind of wrong.")

    # --- save ---------------------------------------------------------------
    # Round-trip check: the reconstructed fitters must predict identically,
    # or "we saved the model" is not true.
    saved = dump_fitters(bgf, ggf)
    bgf2, ggf2 = load_fitters(saved)
    reloaded = predict_bgnbd_gg(bgf2, ggf2, X_test, monetary_test,
                                days=PREDICTION_WINDOW_DAYS)
    identical = np.allclose(reloaded, preds["BG/NBD + Gamma-Gamma"])
    print(f"\n\nsave/load round-trip identical: {identical}")
    print("  lifetimes fitters cannot be pickled (a lambda closure inside fit),")
    print("  so the parameters are stored instead. A fitted BG/NBD IS just four")
    print("  numbers, and a dict of floats will still load when the library won't.")

    artifact = {
        "lifetimes_params": saved,
        "xgb_log": m_log,
        "hurdle_clf": clf,
        "hurdle_reg": reg,
        "feature_columns": FEATURE_COLUMNS,
        "window_days": PREDICTION_WINDOW_DAYS,
        "trained_on_cutoff": str(train.cutoff.date()),
        "metrics_on_test": {k: v for k, v in results.items()},
        "best_by_spearman": best,
    }
    path = ARTIFACTS / "clv.joblib"
    joblib.dump(artifact, path)
    print(f"\n\nsaved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

    # --- the serving bundle -------------------------------------------------
    # Two models ship by two DIFFERENT mechanisms, and the split is a real
    # engineering trade-off rather than a preference:
    #
    #   HURDLE  ships as boosters and runs LIVE at request time. xgboost is
    #           already loaded in the serving container for the repurchase
    #           model, so a second pair of boosters costs a few hundred KB and
    #           no new dependency at all.
    #
    #   BG/NBD  ships as PRECOMPUTED numbers. Running it live would drag
    #           `lifetimes` (unmaintained, warns on modern numpy) plus
    #           scipy.special into a container whose RAM Railway bills at
    #           about $10 per GB-month. The scoring surface here is 5,256
    #           known customers, so the entire output fits in one small array.
    #
    # The cost of precomputing is stated in the API response rather than
    # hidden: a lookup cannot score a customer who did not exist at export
    # time. A scheduled re-export fixes that; a portfolio demo does not need
    # one. Choosing NOT to build the live path, and saying why, is the same
    # judgement call as declining a vector database in Phase 4.
    serve = {
        "hurdle_clf": clf,
        "hurdle_reg": reg,
        "feature_columns": FEATURE_COLUMNS,
        "window_days": PREDICTION_WINDOW_DAYS,
        "trained_on_cutoff": str(train.cutoff.date()),
        "scored_as_of": str(test.cutoff.date()),
        "metrics_on_test": {k: v for k, v in results.items()},
        "best_by_mae": min(results, key=lambda k: results[k]["MAE"]),
    }
    serve_path = ARTIFACTS / "clv_serve.joblib"
    joblib.dump(serve, serve_path)

    # p_alive travels with the predictions because it is the one quantity in
    # this project that cannot be observed at all. "72% chance this customer
    # is still active" is the most useful sentence BG/NBD produces, and it
    # would be lost if only the pound figure were exported.
    npz_path = ARTIFACTS / "clv_test.npz"
    np.savez_compressed(
        npz_path,
        customer_ids=X_test.index.to_numpy(dtype=np.int64),
        bgnbd=np.asarray(preds["BG/NBD + Gamma-Gamma"], dtype=np.float32),
        p_alive=np.asarray(alive, dtype=np.float32),
        as_of=np.array(str(test.cutoff.date())),
    )
    print(f"saved {serve_path.name}  ({serve_path.stat().st_size / 1024:.0f} KB)"
          f"   hurdle boosters, run live")
    print(f"saved {npz_path.name}  ({npz_path.stat().st_size / 1024:.0f} KB)"
          f"   BG/NBD precomputed, so `lifetimes` never ships")


if __name__ == "__main__":
    main()
