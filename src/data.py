"""Download and clean the IBM Telco Customer Churn dataset (7,043 customers)."""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_FILE = DATA_DIR / "Telco-Customer-Churn.csv"
TARGET = "Churn"

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def download() -> Path:
    if not RAW_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {RAW_FILE} ...")
        urlretrieve(URL, RAW_FILE)
    return RAW_FILE


def load_raw() -> pd.DataFrame:
    return pd.read_csv(download())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is stored as text, with " " for brand-new customers (tenure = 0).
    # They haven't been billed yet, so 0 is the honest value, not the mean.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if TARGET in df:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=["customerID"], errors="ignore")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                               labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"]).astype(str)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["PaysElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def load_data() -> pd.DataFrame:
    return add_features(clean(load_raw()))
