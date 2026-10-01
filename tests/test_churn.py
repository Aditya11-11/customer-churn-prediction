import numpy as np
import pandas as pd

from src.data import clean, load_data, load_raw
from src.train import business_cost


def test_total_charges_blank_becomes_zero():
    raw = load_raw()
    assert (raw["TotalCharges"] == " ").sum() > 0  # the gotcha exists in the raw file
    df = clean(raw)
    assert df["TotalCharges"].dtype == float
    assert df.loc[df["tenure"] == 0, "TotalCharges"].eq(0).all()


def test_target_is_binary_and_imbalanced():
    df = load_data()
    assert set(df["Churn"].unique()) == {0, 1}
    assert 0.2 < df["Churn"].mean() < 0.35


def test_business_cost_extremes():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8])
    nobody = business_cost(y, p, threshold=1.01)
    assert nobody == 2 * 780  # do nothing: lose both churners
    assert business_cost(y, p, threshold=0.5) < nobody  # perfect targeting is cheaper


def test_scoring_returns_probabilities():
    from src.predict import score

    out = score(load_raw().head(25))
    assert len(out) == 25
    assert out["churn_probability"].between(0, 1).all()
