import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


# =========================
# 1. DATE FEATURES
# =========================

class DateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        df["transaction_hour"] = df["TransactionStartTime"].dt.hour
        df["transaction_day"] = df["TransactionStartTime"].dt.day
        df["transaction_month"] = df["TransactionStartTime"].dt.month
        df["transaction_year"] = df["TransactionStartTime"].dt.year

        return df


# =========================
# 2. AGGREGATION (CUSTOMER LEVEL)
# =========================

class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        agg = df.groupby("CustomerId").agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            max_transaction_amount=("Amount", "max"),
            min_transaction_amount=("Amount", "min")
        ).reset_index()

        return agg


# =========================
# 3. RFM + PROXY TARGET (TASK 4)
# =========================

class RFMTargetGenerator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.snapshot_date = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        self.snapshot_date = df["TransactionStartTime"].max()

        # RFM features
        rfm = df.groupby("CustomerId").agg(
            Recency=("TransactionStartTime", lambda x: (self.snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum")
        ).reset_index()

        # scaling RFM
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

        # clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

        # identify high-risk cluster (low frequency + low monetary)
        cluster_summary = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()

        high_risk_cluster = cluster_summary["Frequency"].idxmin()

        rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

        return rfm[["CustomerId", "is_high_risk"]]


# =========================
# 4. PREPROCESSING PIPELINE
# =========================

def build_preprocessor():

    numeric_features = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "max_transaction_amount",
        "min_transaction_amount"
    ]

    categorical_features = ["CustomerId"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# =========================
# 5. FULL PIPELINE (FINAL)
# =========================

def build_pipeline():

    pipeline = Pipeline(steps=[
        ("date_features", DateFeatures()),
        ("aggregation", AggregateFeatures()),
        ("rfm_target", RFMTargetGenerator()),
        ("preprocessor", build_preprocessor())
    ])

    return pipeline