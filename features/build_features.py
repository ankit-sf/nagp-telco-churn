import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for the Telco Customer Churn dataset.

    I kept this as a sklearn-compatible transformer so the exact same feature logic
    is used during training and during API prediction. This helps avoid data leakage
    and keeps new/unseen customer data consistent with the training data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Nothing needs to be learned from the data here.
        # The transformer only creates rule-based features.
        return self

    def transform(self, X):
        X = X.copy()

        # customerID is an identifier, not a useful modelling feature.
        # Keeping it could make the model learn customer-specific noise.
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        # TotalCharges sometimes has blank strings in the original IBM dataset.
        # Converting here ensures numeric preprocessing works correctly later.
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        # Feature 1: tenure group
        # Customers at different life-cycle stages usually churn differently.
        if "tenure" in X.columns:
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
            ).astype("object")

        # Feature 2: average charges per tenure month
        # This gives a rough customer value/usage indicator while avoiding division by zero.
        if {"TotalCharges", "tenure"}.issubset(X.columns):
            X["avg_charge_per_tenure_month"] = (
                X["TotalCharges"] / X["tenure"].replace(0, np.nan)
            )
            X["avg_charge_per_tenure_month"] = X["avg_charge_per_tenure_month"].replace(
                [np.inf, -np.inf],
                np.nan
            )

        # Feature 3: whether the customer is on a month-to-month contract
        # Month-to-month customers often have less commitment and higher churn risk.
        if "Contract" in X.columns:
            X["is_month_to_month"] = np.where(
                X["Contract"].astype(str).str.lower().eq("month-to-month"),
                "Yes",
                "No"
            )

        # Feature 4: whether the customer has both support-related services.
        # Security and tech support may indicate stickier service relationships.
        if {"OnlineSecurity", "TechSupport"}.issubset(X.columns):
            X["has_security_and_support"] = np.where(
                (X["OnlineSecurity"] == "Yes") & (X["TechSupport"] == "Yes"),
                "Yes",
                "No"
            )

        # Feature 5: whether the customer uses streaming services.
        # Bundled entertainment services can affect retention behaviour.
        if {"StreamingTV", "StreamingMovies"}.issubset(X.columns):
            X["has_streaming_services"] = np.where(
                (X["StreamingTV"] == "Yes") | (X["StreamingMovies"] == "Yes"),
                "Yes",
                "No"
            )

        return X