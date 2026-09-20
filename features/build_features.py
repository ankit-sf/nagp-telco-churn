import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for the Telco Customer Churn dataset.

    This transformer is sklearn-compatible so the same feature logic is used
    during training and during API prediction. This is important to avoid
    data leakage and to keep the data processing consistent for new/unseen data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No learned parameters are required here. The engineering rules are
        # deterministic and based on the structure of the dataset.
        return self

    def transform(self, X):
        X = X.copy()

        # Remove customerID because it is a unique identifier, not a behavioural signal.
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        # TotalCharges may contain blank strings in the IBM dataset.
        # Convert to numeric so missing values can be handled uniformly later.
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        # Feature 1: tenure_group
        # Customers at different lifecycle stages often show different churn behaviour.
        if "tenure" in X.columns:
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
            ).astype("object")

        # Feature 2: avg_charge_per_tenure_month
        # This captures how much the customer is paying on average over their tenure.
        if {"TotalCharges", "tenure"}.issubset(X.columns):
            X["avg_charge_per_tenure_month"] = (
                X["TotalCharges"] / X["tenure"].replace(0, np.nan)
            )
            X["avg_charge_per_tenure_month"] = X["avg_charge_per_tenure_month"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Feature 3: is_month_to_month
        # Month-to-month contracts often have higher churn risk.
        if "Contract" in X.columns:
            X["is_month_to_month"] = np.where(
                X["Contract"].astype(str).str.lower().eq("month-to-month"),
                "Yes",
                "No",
            )

        # Feature 4: has_security_and_support
        # Customers who use both security and support services may be more engaged.
        if {"OnlineSecurity", "TechSupport"}.issubset(X.columns):
            X["has_security_and_support"] = np.where(
                (X["OnlineSecurity"] == "Yes") & (X["TechSupport"] == "Yes"),
                "Yes",
                "No",
            )

        # Feature 5: has_streaming_services
        # Streaming services may reflect deeper product usage and bundling.
        if {"StreamingTV", "StreamingMovies"}.issubset(X.columns):
            X["has_streaming_services"] = np.where(
                (X["StreamingTV"] == "Yes") | (X["StreamingMovies"] == "Yes"),
                "Yes",
                "No",
            )

        return X