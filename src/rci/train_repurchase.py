"""Train, evaluate and save the repurchase model.

    python -m rci.train_repurchase

Prints everything needed to judge the model honestly: baselines first, then
the model, then the real learned tree, then where it fails.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rci.clean import build
from rci.config import ARTIFACTS, RANDOM_SEED
from rci.features import FEATURE_COLUMNS, build_xy
from rci.repurchase import (
    baseline_logistic,
    baseline_recency,
    calibrate,
    evaluate,
    fit_xgboost,
    format_tree,
    importance_table,
    lift_table,
    shap_contributions,
)
from rci.split import train_test_snapshots


def main() -> None:
    pd.set_option("display.width", 220)

    purchases = build()
    train, test = train_test_snapshots(purchases)
    X_train, y_train = build_xy(train, target="repurchased")
    X_test, y_test = build_xy(test, target="repurchased")

    print("DATA")
    print("=" * 78)
    print(f"  train  {X_train.shape}  cutoff {train.cutoff.date()}  "
          f"positive rate {y_train.mean():.1%}")
    print(f"  test   {X_test.shape}  cutoff {test.cutoff.date()}  "
          f"positive rate {y_test.mean():.1%}")
    print("\n  Note the base rates differ by 11 points. That is Christmas")
    print("  seasonality, found in Phase 0. It will show up again below.")

    # ---- validation split, for early stopping only ----------------------
    # This split IS random, and that is fine here. Both halves come from the
    # SAME snapshot, so they cover the same time period and no future leaks
    # backwards. It is used solely to decide when to stop adding trees, never
    # to report a score. The honest evaluation is still the later snapshot.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    # ---- baselines first -------------------------------------------------
    # Fit these BEFORE the model, deliberately. Deciding what counts as good
    # after seeing the model's score is how people talk themselves into
    # shipping something that beats nothing.
    print("\n\nBASELINES (what the model must beat)")
    print("=" * 78)
    results = {}
    results["recency rule"] = evaluate(y_test, baseline_recency(X_test))
    results["logistic regression"] = evaluate(
        y_test, baseline_logistic(X_train, y_train, X_test)
    )

    # ---- the model -------------------------------------------------------
    model = fit_xgboost(X_fit, y_fit, X_val, y_val)
    p_test = model.predict_proba(X_test)[:, 1]
    results["xgboost"] = evaluate(y_test, p_test)

    table = pd.DataFrame(results).T[["pr_auc", "roc_auc", "brier", "mean_predicted"]]
    table.insert(0, "vs base", (table["pr_auc"] / y_test.mean()).round(2))
    print(table.round(4).to_string())
    print(f"\n  base rate (a useless model's PR-AUC floor): {y_test.mean():.4f}")
    print(f"  trees actually used: {model.best_iteration + 1} of 600 (early stopping)")

    # ---- the learned tree, as promised -----------------------------------
    print("\n\nTHE FIRST TREE, WITH THE THRESHOLDS XGBOOST CHOSE")
    print("=" * 78)
    print(format_tree(model, FEATURE_COLUMNS, 0))
    print("\n  These numbers were LEARNED, not picked by a human. Compare to")
    print("  the quintile boundaries in rfm_summary.csv, which were chosen.")
    print("  Leaf values are log-odds nudges; the final answer sums ~all trees")
    print("  and passes the total through a sigmoid.")

    # ---- what the model relies on ---------------------------------------
    print("\n\nFEATURE IMPORTANCE")
    print("=" * 78)
    imp = importance_table(model, FEATURE_COLUMNS)
    print(imp[["gain", "gain_pct", "weight"]].round(2).to_string())

    # ---- the business view ----------------------------------------------
    print("\n\nLIFT: sort by predicted probability, cut into ten groups")
    print("=" * 78)
    lift = lift_table(y_test.to_numpy(), p_test)
    print(lift.round(3).to_string())
    top = lift.loc[1]
    print(f"\n  Contacting the top 10% reaches customers who repurchase at "
          f"{top['actual_rate']:.1%},")
    print(f"  {top['lift']:.2f}x the {y_test.mean():.1%} average, capturing "
          f"{top['cumulative_capture']:.1%} of all repurchasers.")

    # ---- calibration, where the seasonality bites ------------------------
    print("\n\nCALIBRATION (the Phase 0 seasonality prediction, tested)")
    print("=" * 78)
    print(f"  mean predicted probability : {p_test.mean():.3f}")
    print(f"  actual repurchase rate     : {y_test.mean():.3f}")
    gap = y_test.mean() - p_test.mean()
    print(f"  gap                        : {gap:+.3f}")
    print("\n  The model learned a 32.3% world and is scoring a 43.5% one, so")
    print("  it under-predicts across the board. RANKING is unaffected, which")
    print("  is why PR-AUC and lift are the headline numbers and why the fix")
    print("  is recalibration, not reshuffling the split.")

    # ---- does recalibration actually fix it? ----------------------------
    # Split the TEST snapshot in half. Fit the calibrator on one half, score
    # the other. This is legitimate: it mimics production, where you observe
    # part of the new season and recalibrate on it before scoring the rest.
    # Calibrating on the same rows we then report would be leakage.
    idx = np.arange(len(y_test))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(idx)
    ref, tgt = idx[: len(idx) // 2], idx[len(idx) // 2:]

    p_cal, _ = calibrate(p_test[ref], y_test.to_numpy()[ref], p_test[tgt])
    before = evaluate(y_test.to_numpy()[tgt], p_test[tgt])
    after = evaluate(y_test.to_numpy()[tgt], p_cal)

    print("\n\nRECALIBRATION (fit on half the new season, scored on the other half)")
    print("=" * 78)
    cal = pd.DataFrame({"before": before, "after": after}).T
    print(cal[["pr_auc", "roc_auc", "brier", "mean_predicted", "base_rate"]].round(4).to_string())
    print("\n  Ranking metrics barely move: isotonic regression is monotonic, so")
    print("  it cannot reorder anyone. Brier and mean_predicted are what improve.")
    print("  This is the point: ranking and calibration are separate problems.")

    # ---- explanations, computed offline ---------------------------------
    print("\n\nSHAP: which features move an INDIVIDUAL prediction")
    print("=" * 78)
    shap_values, base_value = shap_contributions(model, X_test)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS)
    mean_abs = mean_abs.sort_values(ascending=False)
    print("  mean |SHAP| per feature (log-odds):")
    print(mean_abs.round(4).to_string())

    # One real customer, explained.
    who = int(np.argmax(p_test))
    cid = X_test.index[who]
    contrib = pd.Series(shap_values[who], index=FEATURE_COLUMNS).sort_values(
        key=np.abs, ascending=False
    )
    print(f"\n  Highest-scoring customer, {cid} (p={p_test[who]:.3f}):")
    for feat, val in contrib.head(5).items():
        direction = "raises" if val > 0 else "lowers"
        print(f"    {feat:<26} = {X_test.loc[cid, feat]:>10,.1f}   "
              f"{direction} score by {abs(val):.3f} log-odds")

    # ---- save -------------------------------------------------------------
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "trained_on_cutoff": str(train.cutoff.date()),
        "n_trees": int(model.best_iteration + 1),
        "metrics_on_test": results["xgboost"],
        "test_cutoff": str(test.cutoff.date()),
        "shap_base_value": base_value,
        "shap_mean_abs": mean_abs.to_dict(),
        "importance_gain_pct": imp["gain_pct"].to_dict(),
    }
    path = ARTIFACTS / "repurchase.joblib"
    joblib.dump(artifact, path)
    print(f"\n\nsaved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

    # Per-customer SHAP, precomputed as plain arrays so the serving container
    # never imports shap. This is the memory decision from the plan, made real.
    shap_path = ARTIFACTS / "shap_test.npz"
    np.savez_compressed(
        shap_path,
        customer_ids=X_test.index.to_numpy(dtype=np.int64),
        shap_values=shap_values.astype(np.float32),
        base_value=np.array(base_value, dtype=np.float32),
        columns=np.array(FEATURE_COLUMNS),
    )
    print(f"saved {shap_path.name}  ({shap_path.stat().st_size / 1024:.0f} KB)  "
          f"so serving needs no shap import")


if __name__ == "__main__":
    main()
