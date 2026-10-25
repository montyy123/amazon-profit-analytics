"""Order-level profitability models.

Classification : will this order lose money?   (target: is_loss_making)

Only information available AT ORDER TIME is used. Outcome columns (returned, net_revenue,
cogs_recognised, return_shipping_cost, profit...) are excluded to prevent data leakage.
Evaluation uses a chronological split: train on the past, test on the most recent 20% of orders.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FESTIVE_MONTHS, FIG, NAVY, ORANGE, REPORTS, RED, SEED, TEAL

CAT = ["category", "region", "fulfilment", "shipping_mode", "customer_segment"]
NUM = ["quantity", "list_price", "discount_pct", "selling_price", "unit_cost", "gross_sales",
       "shipping_cost", "fulfilment_cost", "marketing_cost", "customer_return_rate",
       "pre_return_margin_pct", "order_month", "is_festive"]
FEATURES = CAT + NUM
FORBIDDEN = {"returned", "net_revenue", "cogs", "cogs_recognised", "return_shipping_cost",
             "profit", "profit_margin_pct", "is_loss_making", "order_id", "customer_id", "order_date"}
TARGET_CLF = "is_loss_making"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive order-time features (no outcome information)."""
    d = df.copy()
    d["selling_price"] = d["list_price"] * (1 - d["discount_pct"] / 100)
    d["gross_sales"] = d["selling_price"] * d["quantity"]
    known_cost = d["unit_cost"] * d["quantity"] + d["fulfilment_cost"] + d["shipping_cost"] + d["marketing_cost"]
    d["pre_return_margin_pct"] = (d["gross_sales"] - known_cost) / d["gross_sales"] * 100
    if "order_date" in d:
        d["order_month"] = pd.to_datetime(d["order_date"]).dt.month
    d["is_festive"] = d["order_month"].isin(FESTIVE_MONTHS).astype(int)
    return d


def _preprocess():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    d = df.sort_values("order_date").reset_index(drop=True)
    cut = int(len(d) * (1 - test_frac))
    return d.iloc[:cut], d.iloc[cut:]


def _style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 12})


def run_models(df: pd.DataFrame) -> dict:
    _style()
    assert not (set(FEATURES) & FORBIDDEN), "Leakage: forbidden column used as feature"
    d = add_features(df)
    train, test = chronological_split(d)
    Xtr, ytr = train[FEATURES], train[TARGET_CLF]
    Xte, yte = test[FEATURES], test[TARGET_CLF]
    print(f"  Train: {len(train):,} orders ({train.order_date.min():%b %Y}-{train.order_date.max():%b %Y}) | "
          f"Test: {len(test):,} orders ({test.order_date.min():%b %Y}-{test.order_date.max():%b %Y})")

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                            max_leaf_nodes=31, random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("pre", _preprocess()), ("clf", clf)])
        scores = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["accuracy", "roc_auc"], n_jobs=1)
        cv_acc, cv_auc = scores["test_accuracy"].mean(), scores["test_roc_auc"].mean()
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "cv_accuracy": round(float(cv_acc), 4), "cv_auc": round(float(cv_auc), 4),
            "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
            "test_precision": round(float(precision_score(yte, pred)), 4),
            "test_recall": round(float(recall_score(yte, pred)), 4),
            "test_f1": round(float(f1_score(yte, pred)), 4),
            "test_auc": round(float(roc_auc_score(yte, proba)), 4),
        }
        fitted[name] = pipe
        r = results[name]
        print(f"  {name:20s} CV acc {r['cv_accuracy']:.3f} | TEST acc {r['test_accuracy']:.3f}  "
              f"prec {r['test_precision']:.3f}  rec {r['test_recall']:.3f}  AUC {r['test_auc']:.3f}")

    best = max(results, key=lambda k: results[k]["cv_auc"])      # chosen on CV only, never on test
    model = fitted[best]
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    baseline = 1 - yte.mean()
    print(f"  Best (by CV AUC): {best} | majority-class baseline accuracy: {baseline:.3f}")

    # Why are some losses missed? (mostly returns, which are unknowable at order time)
    fn_mask = (yte.values == 1) & (pred == 0)
    missed = {"count": int(fn_mask.sum()),
              "pct_caused_by_returns": round(float(test["returned"].values[fn_mask].mean() * 100), 1) if fn_mask.any() else 0.0}
    print(f"  Missed losses: {missed['count']} ({missed['pct_caused_by_returns']}% caused by customer returns)")

    # ---------- charts ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    palette = [NAVY, TEAL, ORANGE]
    for (name, p), c in zip(fitted.items(), palette):
        pr = p.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, pr)
        ax[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {results[name]['test_auc']:.3f})")
        pre, rec, _ = precision_recall_curve(yte, pr)
        ax[1].plot(rec, pre, color=c, lw=2, label=f"{name} (AP {average_precision_score(yte, pr):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.4); ax[0].set(title="ROC curves (hold-out test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for a in ax: a.legend(loc="lower right" if a is ax[0] else "lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "08_roc_pr_curves.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5, 4.3))
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["Profitable", "Loss-making"],
                                            cmap="Oranges", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {best}\naccuracy {accuracy_score(yte, pred):.1%}"); ax.grid(False)
    plt.tight_layout(); plt.savefig(FIG / "09_confusion_matrix.png", dpi=150); plt.close()

    sample = test.sample(min(4000, len(test)), random_state=SEED)
    imp = permutation_importance(model, sample[FEATURES], sample[TARGET_CLF], scoring="roc_auc",
                                 n_repeats=5, random_state=SEED, n_jobs=1)
    imp_s = pd.Series(imp.importances_mean, index=FEATURES).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp_s.index, imp_s.values, color=ORANGE)
    ax.set(title="What drives loss-making orders? (permutation importance)", xlabel="Drop in AUC when shuffled")
    plt.tight_layout(); plt.savefig(FIG / "10_feature_importance.png", dpi=150); plt.close()

    # calibration
    bins = pd.cut(proba, np.linspace(0, 1, 11), include_lowest=True)
    cal = pd.DataFrame({"p": proba, "y": yte.values}).groupby(bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.plot([0, 1], [0, 1], "k--", alpha=.4, label="Perfect"); ax.plot(cal.p, cal.y, "o-", color=NAVY, label=best)
    ax.set(title="Calibration: predicted vs actual loss rate", xlabel="Predicted probability", ylabel="Observed frequency"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG / "11_calibration.png", dpi=150); plt.close()

    # ---------- artefacts ----------
    joblib.dump(model, REPORTS / "loss_classifier.joblib", compress=3)
    watch = test.assign(loss_probability=proba.round(4), predicted_loss=pred)
    watch.sort_values("loss_probability", ascending=False).head(200)[
        ["order_id", "order_date", "category", "region", "shipping_mode", "discount_pct",
         "gross_sales", "loss_probability", "profit", "is_loss_making"]].to_csv(REPORTS / "highest_loss_risk_orders.csv", index=False)

    tn, fp, fn, tp = (int(x) for x in pd.crosstab(yte, pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy().ravel())
    return {"best_classifier": best, "classification": results,
            "baseline_accuracy": round(float(baseline), 4),
            "test_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "missed_losses": missed,
            "top_drivers": {k: round(float(v), 4) for k, v in imp_s.sort_values(ascending=False).head(6).items()},
            "split": {"train_orders": len(train), "test_orders": len(test),
                      "test_period": f"{test.order_date.min():%Y-%m-%d} to {test.order_date.max():%Y-%m-%d}"}}
