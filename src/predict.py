"""Score customers and produce a call list, riskiest first.

Example:
    python -m src.predict data/Telco-Customer-Churn.csv --top 10
"""
import argparse

import joblib
import pandas as pd

from src.data import MODELS_DIR, add_features, clean


def score(raw: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(MODELS_DIR / "churn_model.joblib")
    model, threshold = bundle["model"], bundle["threshold"]
    ids = raw.get("customerID", pd.Series(range(len(raw))))
    X = add_features(clean(raw.drop(columns=["Churn"], errors="ignore")))
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"customerID": ids.values, "churn_probability": proba.round(3),
                        "contact": proba >= threshold})
    return out.sort_values("churn_probability", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", help="optional path to save the full scored list")
    args = parser.parse_args()
    scored = score(pd.read_csv(args.csv))
    print(f"{scored['contact'].sum()} of {len(scored)} customers flagged for a retention offer\n")
    print(scored.head(args.top).to_string(index=False))
    if args.out:
        scored.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
