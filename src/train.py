"""Logistic Regression vs Random Forest for churn, plus picking a decision threshold
that actually makes business sense.

Accuracy is a trap here: only ~27% of customers churn, so a model that says
"nobody leaves" is already 73% accurate. ROC-AUC tells us how well the model
*ranks* customers; the threshold decides who we actually call.

Run:  python -m src.train
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, TARGET, load_data

RANDOM_STATE = 42

# Rough, made-up-but-reasonable economics for the threshold analysis:
# a retention offer (discount / call) costs $50; losing a customer costs
# about 12 months of revenue (~$65/month avg) — call it $780.
# Assume the offer saves 1 in 3 customers who were going to leave.
OFFER_COST = 50
CHURN_COST = 780
OFFER_SUCCESS_RATE = 0.33


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), make_column_selector(dtype_include=object)),
    ])


def business_cost(y_true, proba, threshold):
    """Total cost of a retention campaign that targets everyone above `threshold`."""
    flagged = proba >= threshold
    tp = np.sum(flagged & (y_true == 1))
    fp = np.sum(flagged & (y_true == 0))
    fn = np.sum(~flagged & (y_true == 1))
    saved = tp * OFFER_SUCCESS_RATE
    return (tp + fp) * OFFER_COST + (tp - saved) * CHURN_COST + fn * CHURN_COST


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    searches = {
        "Logistic Regression": GridSearchCV(
            Pipeline([("prep", make_preprocessor()),
                      ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}, cv=cv, scoring="roc_auc", n_jobs=-1),
        "Random Forest": GridSearchCV(
            Pipeline([("prep", make_preprocessor()),
                      ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                       random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__max_depth": [6, 10, 14, None], "model__min_samples_leaf": [1, 5, 10, 20],
             "model__max_features": ["sqrt", 0.3]}, cv=cv, scoring="roc_auc", n_jobs=-1),
    }

    results, fitted, probas = {}, {}, {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        proba = search.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_roc_auc": round(float(search.best_score_), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
            "precision@0.5": round(float(precision_score(y_test, pred)), 4),
            "recall@0.5": round(float(recall_score(y_test, pred)), 4),
            "f1@0.5": round(float(f1_score(y_test, pred)), 4),
        }
        fitted[name], probas[name] = search.best_estimator_, proba
        print(f"{name:<20} CV AUC {search.best_score_:.4f} | test AUC {results[name]['test_roc_auc']:.4f} | "
              f"PR-AUC {results[name]['test_pr_auc']:.4f} | P {results[name]['precision@0.5']:.3f} "
              f"R {results[name]['recall@0.5']:.3f}  {search.best_params_}")

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    best, best_proba = fitted[best_name], probas[best_name]
    print(f"\nSelected: {best_name}")

    # ---- ROC + PR curves ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, proba), color in zip(probas.items(), ["#4C72B0", "#55A868"]):
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['test_roc_auc']:.3f})", color=color)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP {results[name]['test_pr_auc']:.3f})", color=color)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Base rate ({y_test.mean():.2f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curve")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    # ---- Threshold trade-off + business cost -------------------------------
    thresholds = np.linspace(0.05, 0.95, 91)
    prec_t = [precision_score(y_test, best_proba >= t, zero_division=0) for t in thresholds]
    rec_t = [recall_score(y_test, best_proba >= t) for t in thresholds]
    f1_t = [f1_score(y_test, best_proba >= t) for t in thresholds]
    costs = np.array([business_cost(y_test.values, best_proba, t) for t in thresholds])
    no_campaign = business_cost(y_test.values, best_proba, 1.01)
    t_cost = float(thresholds[costs.argmin()])
    t_f1 = float(thresholds[int(np.argmax(f1_t))])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(thresholds, prec_t, label="Precision")
    axes[0].plot(thresholds, rec_t, label="Recall")
    axes[0].plot(thresholds, f1_t, label="F1", ls="--")
    axes[0].axvline(t_f1, color="grey", ls=":", label=f"best F1 @ {t_f1:.2f}")
    axes[0].set(xlabel="Decision threshold", title=f"{best_name}: the precision / recall trade-off")
    axes[0].legend()
    axes[1].plot(thresholds, costs / 1000, color="#C44E52")
    axes[1].axhline(no_campaign / 1000, color="k", ls="--", lw=1, label="Do nothing")
    axes[1].axvline(t_cost, color="grey", ls=":", label=f"cheapest @ {t_cost:.2f}")
    axes[1].set(xlabel="Decision threshold", ylabel="Total cost ($k, test set)",
                title="What does each threshold cost the business?")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_analysis.png", dpi=120)
    plt.close(fig)

    cm = confusion_matrix(y_test, best_proba >= t_cost)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix @ threshold {t_cost:.2f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ---- What drives churn? ------------------------------------------------
    perm = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#8172B2")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Permutation importance — what the model relies on")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    # Logistic regression odds ratios are the most explainable view for stakeholders
    lr = fitted["Logistic Regression"]
    names = [n.split("__", 1)[1] for n in lr.named_steps["prep"].get_feature_names_out()]
    odds = pd.Series(np.exp(lr.named_steps["model"].coef_[0]), index=names).sort_values()
    odds_view = pd.concat([odds.head(6), odds.tail(6)])

    joblib.dump({"model": best, "threshold": t_cost}, MODELS_DIR / "churn_model.joblib")
    tn, fp, fn, tp = cm.ravel()
    summary = {
        "selected_model": best_name,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "models": results,
        "threshold": {
            "best_f1_threshold": round(t_f1, 2),
            "cost_optimal_threshold": round(t_cost, 2),
            "precision_at_cost_threshold": round(float(precision_score(y_test, best_proba >= t_cost)), 4),
            "recall_at_cost_threshold": round(float(recall_score(y_test, best_proba >= t_cost)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "cost_do_nothing": int(no_campaign),
            "cost_at_optimal": int(costs.min()),
            "savings": int(no_campaign - costs.min()),
            "assumptions": {"offer_cost": OFFER_COST, "churn_cost": CHURN_COST, "offer_success_rate": OFFER_SUCCESS_RATE},
        },
        "top_permutation_importance": {k: round(float(v), 4) for k, v in imp.iloc[::-1].head(8).items()},
        "logreg_odds_ratios": {k: round(float(v), 3) for k, v in odds_view.items()},
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["threshold"], indent=2))
    print(f"Model saved to {MODELS_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
