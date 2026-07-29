"""Phase 3: will this customer buy again in the next 90 days?

WHY THIS EXISTS (the business question)
---------------------------------------
Retention budget is finite, and most customers were coming back anyway.
Spending a GBP 10 win-back voucher on someone who would have returned by
themselves is GBP 10 wasted. So the useful output is not "who will churn"
but a RANKING: given budget for 500 vouchers, which 500 customers?

That framing matters because it decides the metric. We do not care much
whether the model says 0.31 or 0.36. We care whether the people it puts at
the top really do buy more often than the people at the bottom.

WHAT XGBOOST IS
---------------
Start with ONE decision tree. A tree is nested yes/no questions:

              recency < 83.5 ?
               /            \
             yes             no
             /                \
     frequency < 3.5 ?      predict high
       /        \
   predict     predict
    low         medium

The hand-made version of this is the RFM quintile rule, which scores each
dimension 1-5 using thresholds a human picked (see `segmentation.py`, where it
is implemented for comparison). XGBoost differs in two ways. It LEARNS the
thresholds from data, and it does not stop at one tree.

BOOSTING, the actual idea:

    tree 1   makes a rough guess           -> wrong by some amount per customer
    tree 2   is trained to predict THAT ERROR
    tree 3   is trained on what is still wrong
    ...
    tree N

    final score = tree1 + tree2 + ... + treeN

Each tree specialises in fixing what the previous ones got wrong. "Gradient"
refers to using the direction of steepest error reduction to decide what each
new tree should chase.

ONE DETAIL THAT CONFUSES EVERYONE:
Leaf values are NOT probabilities. They are log-odds. All the trees' leaf
values are summed, and the total is squeezed into 0-1 by the sigmoid
function, 1/(1+exp(-x)). So a leaf of -0.5 pushes DOWN and +0.5 pushes UP,
and only the total means anything.

WHY XGBOOST HERE RATHER THAN A NEURAL NETWORK
---------------------------------------------
Tabular data with mixed numeric columns is where boosted trees still beat
neural networks. Nets win on images, audio and text. Trees also handle
missing values natively (avg_days_between_orders is NaN for 31% of our
customers), need no feature scaling, and find non-linear effects
automatically: "recency matters enormously up to ~90 days then stops
mattering" is something a tree discovers and a linear model cannot express.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from rci.config import RANDOM_SEED


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Score a set of predicted probabilities.

    PR_AUC (average precision) is the HEADLINE metric here.
        It summarises the precision/recall tradeoff across every threshold.
        Its floor is the base rate, so on a 43.5% positive test set a useless
        model scores 0.435, not 0.5.

    ROC_AUC is reported for comparability but is NOT the headline.
        On imbalanced problems it flatters models, because the huge negative
        class makes the false-positive rate look small no matter what. Saying
        this unprompted is a genuine interview signal.

    BRIER is mean squared error on the probabilities: it measures CALIBRATION,
        i.e. whether "0.3" really happens 30% of the time. Lower is better.
        Ranking and calibration are different things and a model can be
        excellent at one and poor at the other.

    IMPORTANT DISTINCTION, which the recency baseline forces us to be honest
    about: ranking metrics (PR-AUC, ROC-AUC) work on ANY score, because they
    only read the ORDER. Calibration metrics need genuine probabilities.
    "-recency" ranks customers perfectly well but -647 is not a probability,
    so Brier is undefined for it and is reported as NaN rather than faked by
    squashing the score into 0-1, which would invent a calibration that was
    never claimed.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    is_probability = bool(y_prob.min() >= 0.0 and y_prob.max() <= 1.0)
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob) if is_probability else float("nan"),
        "base_rate": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_prob)) if is_probability else float("nan"),
    }


def lift_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """The business version of the score.

    Sort customers by predicted probability, cut into ten equal groups, and
    ask what fraction in each group actually repurchased. This is the table a
    marketing lead understands: "the top decile converts 2x the average, so
    spend there first."

    LIFT is the ratio of a decile's actual rate to the overall base rate.
    Lift of 2.0 in the top decile means targeting it is twice as efficient as
    contacting people at random.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    # rank first so ties do not collapse deciles; descending so decile 1 is best
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False),
                           n_bins, labels=range(1, n_bins + 1)).astype(int)
    base = df["y"].mean()
    out = df.groupby("decile").agg(
        customers=("y", "size"),
        actual_rate=("y", "mean"),
        mean_predicted=("p", "mean"),
        repurchasers=("y", "sum"),
    )
    out["lift"] = out["actual_rate"] / base
    # What share of ALL repurchasers you capture by contacting down to here.
    out["cumulative_capture"] = out["repurchasers"].cumsum() / df["y"].sum()
    return out


# ---------------------------------------------------------------------------
# baselines: what the model has to beat
# ---------------------------------------------------------------------------

def baseline_recency(X: pd.DataFrame) -> np.ndarray:
    """The dumbest defensible rule: recent buyers come back.

    Turned into a score by negating recency, since a SMALLER recency should
    mean a HIGHER probability. Ranking metrics only care about order, so no
    rescaling into 0-1 is needed for PR-AUC or ROC-AUC.

    This baseline matters. If a 400-tree gradient boosting model cannot beat
    "sort by last purchase date", the model is not earning its complexity and
    the honest recommendation is to ship the rule.
    """
    return -X["recency"].to_numpy(dtype=float)


def baseline_logistic(X_train, y_train, X_test) -> np.ndarray:
    """A linear model, the standard "is non-linearity actually needed" check.

    Wrapped in a pipeline because logistic regression, unlike XGBoost, cannot
    handle NaN and is sensitive to feature scale:
      SimpleImputer  fills the 31% missing avg_days_between_orders
      StandardScaler puts every column on the same ruler
    XGBoost needs neither, which is itself worth noticing.
    """
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def fit_xgboost(X_train, y_train, X_val=None, y_val=None) -> XGBClassifier:
    """Fit the classifier.

    PARAMETERS, and why each value:

    n_estimators=600     upper bound on trees. Early stopping picks the real
                         number, so this only has to be generous.
    max_depth=4          shallow. With 4,951 rows, deep trees memorise
                         individual customers instead of learning patterns.
    learning_rate=0.05   how much of each tree's correction to apply. Small
                         means slower but steadier learning: many timid
                         corrections beat a few overconfident ones.
    subsample=0.8        each tree sees a random 80% of customers
    colsample_bytree=0.8 and a random 80% of features
                         Both inject deliberate randomness so trees make
                         DIFFERENT mistakes, which is what makes summing them
                         useful rather than redundant.
    min_child_weight=5   refuse to create a leaf covering fewer than ~5
                         customers. Stops the model inventing rules from
                         coincidences.
    eval_metric="aucpr"  early stopping watches PR-AUC, the metric we
                         actually care about, not accuracy.

    NOT USED: scale_pos_weight. The plan assumed heavy imbalance, but the
    real positive rate is 32%, which is mild. Reweighting would distort the
    probabilities for no ranking benefit. Checking the number beat assuming.
    """
    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=50 if X_val is not None else None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# looking inside the model
# ---------------------------------------------------------------------------

def format_tree(model: XGBClassifier, feature_names: list[str], tree_index: int = 0) -> str:
    """Print one real tree, with the thresholds XGBoost actually learned.

    The raw dump uses tabs for depth and shows leaf values in log-odds. This
    converts it to something readable and annotates the leaves with the
    probability that leaf alone would imply, via the sigmoid.

    Read a leaf as a NUDGE, not an answer: the final prediction sums this
    tree's leaf with 300-odd others before the sigmoid is applied.
    """
    dump = model.get_booster().get_dump(with_stats=False)
    raw = dump[tree_index]

    lines = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        depth = len(line) - len(line.lstrip("\t"))
        body = line.strip()
        indent = "    " * depth

        leaf = re.match(r"(\d+):leaf=([-\d.]+)", body)
        if leaf:
            value = float(leaf.group(2))
            prob = 1 / (1 + np.exp(-value))
            lines.append(f"{indent}=> leaf {value:+.4f} log-odds  (alone: {prob:.1%})")
            continue

        split = re.match(r"(\d+):\[([^<]+)<([-\d.e+]+)\]", body)
        if split:
            feature, threshold = split.group(2), float(split.group(3))
            lines.append(f"{indent}{feature} < {threshold:,.2f} ?")
            continue

        lines.append(f"{indent}{body}")
    return "\n".join(lines)


def trim_to_best_iteration(model: XGBClassifier) -> XGBClassifier:
    """Drop the trees early stopping proved were not worth keeping.

    THE PROBLEM THIS SOLVES
    With early_stopping_rounds=50, training does not halt the moment the
    score peaks. It keeps going for 50 more rounds to PROVE the peak was
    real, and every one of those losing trees is saved into the booster too.

    So a model whose best iteration was 34 ends up storing 85 trees:
        35 that are used  +  50 that exist only as evidence

    Predictions are unaffected, because predict_proba honours best_iteration
    and silently ignores trees past it. But the artifact carries all 85 to
    production, and Railway bills RAM by the GB-month. 59% of the trees in
    that container would never be consulted.

    THE FIX
    Slicing a Booster returns a new one holding just those trees. The slice
    also drops the best_iteration attribute, which is exactly what we want:
    with no early-stopping metadata, prediction uses every stored tree, and
    every stored tree is now precisely the set that was being used anyway.

    Verified bit-identical: max prediction difference 0.0.
    """
    import copy

    n_used = model.best_iteration + 1
    trimmed = copy.deepcopy(model)
    trimmed._Booster = trimmed.get_booster()[0:n_used]
    return trimmed


def calibrate(p_reference: np.ndarray, y_reference: np.ndarray, p_target: np.ndarray):
    """Rescale probabilities so that "0.3" really means 30%.

    WHY THIS IS NEEDED HERE
    The model learned a 32.3% world (summer) and is scoring a 43.5% one
    (Christmas run-up). It therefore under-predicts everyone. The RANKING is
    unharmed, which is why PR-AUC barely moves, but the NUMBERS are wrong,
    and numbers matter the moment anyone multiplies them by a budget.

    WHY IT CANNOT BE FIXED WITH TRAINING DATA
    The training period does not contain the target season, so no amount of
    reworking it will teach the model what a Christmas base rate looks like.
    Calibration needs observations from the new regime. In production this is
    exactly what happens: you observe some of the new period, then recalibrate
    on it, on a schedule.

    ISOTONIC REGRESSION is the method: fit a monotonic (never-decreasing)
    step function mapping predicted -> observed. Monotonic is the key
    property, because it can rescale the probabilities without ever changing
    the order, so ranking performance is preserved by construction.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_reference, y_reference)
    return iso.predict(p_target), iso


def shap_contributions(model: XGBClassifier, X: pd.DataFrame):
    """Per-customer feature contributions, in log-odds.

    GAIN IMPORTANCE (above) is one number per FEATURE for the whole model:
    "frequency mattered most overall". It cannot tell you why THIS customer
    scored 0.8.

    SHAP is one number per feature PER CUSTOMER: "for customer 15369, a
    recency of 401 pushed the score down by 0.6 log-odds". That is what makes
    a prediction explainable to the person acting on it, which is the
    difference between a model being trusted and being ignored.

    The values are additive: base_value + sum(shap for this customer) = the
    model's raw output for them, before the sigmoid. That additivity is what
    makes SHAP defensible rather than a heuristic.

    Computed OFFLINE and shipped as data. `shap` is a heavy import and the
    serving container deliberately does not have it (Railway bills RAM).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return values, float(explainer.expected_value)


def importance_table(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Feature importance by three different definitions, because they disagree.

    gain    total improvement in the objective from splits on this feature.
            Usually the one to quote: it measures usefulness, not activity.
    weight  how many times the feature was split on. Biased toward
            high-cardinality numeric features, which get more places to cut.
    cover   how many customers pass through those splits.

    A feature can rank high on weight and low on gain, meaning it is used
    constantly but never decides much.
    """
    booster = model.get_booster()
    frames = {}
    for kind in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=kind)
        frames[kind] = pd.Series(scores)
    out = pd.DataFrame(frames).reindex(feature_names).fillna(0.0)
    out["gain_pct"] = (100 * out["gain"] / out["gain"].sum()).round(1)
    return out.sort_values("gain", ascending=False)
