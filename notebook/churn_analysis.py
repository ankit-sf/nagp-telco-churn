# %% [markdown]
# # Customer Churn Prediction
#
# ## Business Problem
#
# A telecommunications company wants to identify customers who are likely to churn.
# The goal is to build an end-to-end machine learning solution that helps the
# retention team proactively contact high-risk customers.

# %%
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# This allows the notebook to import code from the project root.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from features.build_features import ChurnFeatureEngineer


RANDOM_STATE = 42

DATA_PATH = PROJECT_ROOT / "data" / "TelcoCustomerChurn.csv"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Business Problem
#
# A telecommunications company wants to identify customers who are likely to churn.
# If high-risk customers can be identified early, the retention team can contact
# them with offers, support, or better plans before they leave.
#
# This is a binary classification problem:
#
# - Input: customer demographics, services, contract, tenure, and billing data
# - Target: `Churn`
# - Target values: `Yes` or `No`

# %% [markdown]
# ## 2. Data Loading and Understanding

# %%
df = pd.read_csv(DATA_PATH)
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.describe(include="all").T

# %%
print("Columns in dataset:")
print(df.columns.tolist())

# %%
print("Missing values before cleaning:")
df.isna().sum()

# %%
print("Duplicate rows:", df.duplicated().sum())

# %%
print("Target distribution:")
df["Churn"].value_counts()

# %%
print("Target distribution percentage:")
(df["Churn"].value_counts(normalize=True) * 100).round(2)

# %%
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("Numerical columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)

# %% [markdown]
# ### Important Data Observation
#
# `TotalCharges` should be numeric, but in the original IBM Telco dataset it can
# contain blank strings. Therefore, it needs to be converted using
# `pd.to_numeric(..., errors="coerce")`.
#
# `customerID` is only an identifier. It should not be used as a model feature
# because it does not describe customer behaviour and could encourage memorization.

# %%
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Missing values after converting TotalCharges:")
df.isna().sum()

# %%
# Drop customerID before modelling.
# This is important because customerID is an identifier, not a predictive feature.
# Dropping it also helps prevent ID-based memorization.
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

df.head()

# %% [markdown]
# ## 3. Exploratory Data Analysis
#
# The purpose of EDA is to understand customer behaviour and identify patterns
# associated with churn.

# %%
sns.set_theme(style="whitegrid")

# %% [markdown]
# ### Visualization 1: Churn Distribution

# %%
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False,
)
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Business insight:** Most customers do not churn, but the churn group is still
# large enough to represent a meaningful retention opportunity.

# %% [markdown]
# ### Visualization 2: Churn by Contract Type

# %%
plt.figure(figsize=(8, 4))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn",
    palette="Set2",
)
plt.title("Churn by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Business insight:** Month-to-month customers usually show higher churn risk.
# Longer contracts appear to create stronger customer commitment.

# %% [markdown]
# ### Visualization 3: Churn by Internet Service

# %%
plt.figure(figsize=(8, 4))
sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn",
    palette="Set2",
)
plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Business insight:** Churn differs across internet service categories. This
# may indicate differences in price, service quality, or customer expectations.

# %% [markdown]
# ### Visualization 4: Monthly Charges by Churn

# %%
plt.figure(figsize=(7, 4))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False,
)
plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Business insight:** Customers who churn often have higher monthly charges,
# suggesting that pricing or value perception may influence churn.

# %% [markdown]
# ### Visualization 5: Tenure Distribution by Churn

# %%
plt.figure(figsize=(8, 4))
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    kde=True,
    bins=30,
    palette="Set2",
)
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Business insight:** Customers with shorter tenure are more likely to churn.
# This suggests that onboarding and early engagement are important.

# %% [markdown]
# ### Visualization 6: Churn by Payment Method

# %%
plt.figure(figsize=(10, 4))
sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn",
    palette="Set2",
)
plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Business insight:** Some payment methods are associated with higher churn.
# This could reflect billing friction or differences in customer segments.

# %% [markdown]
# ## 4. Feature Engineering
#
# Feature engineering code is kept in:
#
# `features/build_features.py`
#
# The feature engineering is placed inside a sklearn-compatible transformer so
# the same logic is used during model training and API prediction.

# %%
feature_engineer = ChurnFeatureEngineer()
engineered_preview = feature_engineer.fit_transform(df.drop(columns=["Churn"]))
engineered_preview.head()

# %% [markdown]
# ### Engineered Features
#
# 1. `tenure_group`
#    - Created by grouping `tenure` into lifecycle bands.
#    - Useful because new customers and long-term customers often churn differently.
#
# 2. `avg_charge_per_tenure_month`
#    - Created as `TotalCharges / tenure`.
#    - Useful because it captures approximate billing intensity over the customer lifetime.
#
# 3. `is_month_to_month`
#    - Created from the `Contract` column.
#    - Useful because month-to-month customers have less commitment.
#
# 4. `has_security_and_support`
#    - Created from `OnlineSecurity` and `TechSupport`.
#    - Useful because support and security services may indicate stronger engagement.
#
# 5. `has_streaming_services`
#    - Created from `StreamingTV` and `StreamingMovies`.
#    - Useful because bundled services may increase customer stickiness.

# %% [markdown]
# ## 5. Data Preparation for Modelling
#
# The data is split into 70% training and 30% testing.
#
# The test data is not used during preprocessing fitting or hyperparameter tuning.
# This helps avoid data leakage.

# %%
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])

# Safety check: customerID should already be removed, but this ensures it is not
# used even if the previous cell is edited later.
X = X.drop(columns=["customerID"], errors="ignore")

if y.isna().any():
    raise ValueError("The Churn column contains values other than 'Yes' and 'No'.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4))

# %%
def build_pipeline(classifier):
    """
    Build a complete feature engineering, preprocessing, and modelling pipeline.

    Keeping everything inside one pipeline ensures that training data, test data,
    and future API data receive the same transformations.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                make_column_selector(dtype_include=["int64", "float64"]),
            ),
            (
                "cat",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "category", "bool"]),
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics required by the assignment."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

# %% [markdown]
# ## 6. Model Development
#
# The assignment requires a Decision Tree Classifier. Two baseline Decision Tree
# configurations are trained first, then hyperparameter tuning is performed using
# GridSearchCV.

# %% [markdown]
# ### 6.1 Baseline Decision Tree Models

# %%
baseline_configs = {
    "Decision Tree - depth 4": DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=25,
        random_state=RANDOM_STATE,
    ),
    "Decision Tree - depth 6 balanced": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

baseline_results = []
trained_baseline_models = {}

for model_name, classifier in baseline_configs.items():
    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    baseline_results.append(
        {
            "model": model_name,
            **metrics,
        }
    )

    trained_baseline_models[model_name] = pipeline

baseline_results_df = pd.DataFrame(baseline_results)
baseline_results_df

# %% [markdown]
# ### 6.2 Hyperparameter Tuning
#
# Hyperparameter tuning systematically searches for better Decision Tree settings.
#
# Tuned hyperparameters:
#
# - `criterion`
# - `max_depth`
# - `min_samples_split`
# - `min_samples_leaf`
# - `class_weight`
#
# `GridSearchCV` uses 5-fold stratified cross-validation.
#
# The scoring metric is F1 score because it balances precision and recall.
# Recall is especially important for churn because missing a real churner can
# mean losing the customer without any retention attempt.

# %%
tuning_pipeline = build_pipeline(
    DecisionTreeClassifier(random_state=RANDOM_STATE)
)

parameter_grid = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [3, 4, 5, 6, 8, None],
    "classifier__min_samples_split": [2, 10, 20],
    "classifier__min_samples_leaf": [5, 10, 20, 30],
    "classifier__class_weight": [None, "balanced"],
}

cross_validator = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=cross_validator,
    n_jobs=-1,
    verbose=1,
    return_train_score=False,
)

grid_search.fit(X_train, y_train)

# %%
print("Best Decision Tree hyperparameters:")
for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")

print(
    "\nBest mean cross-validation F1 score:",
    round(grid_search.best_score_, 4),
)

# %%
tuning_results_df = pd.DataFrame(grid_search.cv_results_)

tuning_columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "param_classifier__criterion",
    "param_classifier__max_depth",
    "param_classifier__min_samples_split",
    "param_classifier__min_samples_leaf",
    "param_classifier__class_weight",
]

top_tuning_results = (
    tuning_results_df[tuning_columns]
    .sort_values("rank_test_score")
    .head(10)
    .reset_index(drop=True)
)

top_tuning_results

# %%
tuning_results_df[tuning_columns].sort_values("rank_test_score").to_csv(
    OUTPUT_DIR / "hyperparameter_tuning_results.csv",
    index=False,
)

print(
    "Saved hyperparameter tuning results to:",
    OUTPUT_DIR / "hyperparameter_tuning_results.csv",
)

# %% [markdown]
# ## 7. Final Model Evaluation
#
# `GridSearchCV` refits the best pipeline using the full training set.
# The final tuned model is then evaluated once on the untouched 30% test set.

# %%
final_model = grid_search.best_estimator_
final_model_name = "Tuned Decision Tree"

final_predictions = final_model.predict(X_test)
final_probabilities = final_model.predict_proba(X_test)[:, 1]

final_metrics = calculate_metrics(y_test, final_predictions)
final_metrics

# %%
tuned_result = {
    "model": final_model_name,
    **final_metrics,
}

all_results_df = pd.concat(
    [
        baseline_results_df,
        pd.DataFrame([tuned_result]),
    ],
    ignore_index=True,
)

all_results_df

# %%
all_results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

print("Saved model comparison to:", OUTPUT_DIR / "model_comparison.csv")

# %%
comparison_plot = all_results_df.melt(
    id_vars="model",
    value_vars=["accuracy", "precision", "recall", "f1_score"],
    var_name="metric",
    value_name="score",
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=comparison_plot,
    x="model",
    y="score",
    hue="metric",
)
plt.title("Decision Tree Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, 1)
plt.legend(title="Metric")
plt.tight_layout()
plt.show()

# %%
confusion = confusion_matrix(y_test, final_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["No Churn", "Churn"],
)

display.plot(
    cmap="Blues",
    values_format="d",
)

plt.title("Confusion Matrix - Tuned Decision Tree")
plt.tight_layout()
plt.show()

# %%
print("Final model metrics:")
print("Accuracy:", round(final_metrics["accuracy"], 4))
print("Precision:", round(final_metrics["precision"], 4))
print("Recall:", round(final_metrics["recall"], 4))
print("F1 Score:", round(final_metrics["f1_score"], 4))

# %% [markdown]
# ### Business Interpretation
#
# - **Accuracy** shows the overall percentage of correct predictions.
# - **Precision** tells us how many predicted churners actually churned.
# - **Recall** tells us how many actual churners were identified.
# - **F1 score** balances precision and recall.
#
# For telecom churn prediction, recall is usually prioritized because missing a
# true churner means the retention team may not contact that customer before they
# leave. False positives are less costly because the business may simply contact
# an extra customer.

# %% [markdown]
# ## 8. Model Interpretation
#
# Decision Trees can be interpreted using feature importance and tree structure.

# %%
preprocessor = final_model.named_steps["preprocessor"]
classifier = final_model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importance_values = classifier.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importance_values,
    }
).sort_values(
    "importance",
    ascending=False,
)

importance_df.head(15)

# %%
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df.head(10),
    y="feature",
    x="importance",
    hue="feature",
    palette="viridis",
    legend=False,
)
plt.title("Top 10 Feature Importances - Tuned Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(26, 12))
plot_tree(
    classifier,
    max_depth=3,
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Tuned Decision Tree Visualization - First Three Levels")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Key Interpretation
#
# The strongest churn drivers are usually related to contract type, tenure,
# monthly charges, internet service, support services, and payment method.
#
# These features are meaningful from a business perspective because they reflect:
#
# - customer commitment
# - service usage
# - price sensitivity
# - support experience
# - customer lifecycle stage

# %% [markdown]
# ## 9. Save the Final Pipeline
#
# The complete pipeline is saved rather than only the Decision Tree.
#
# The saved pipeline includes:
#
# - feature engineering
# - missing-value handling
# - categorical encoding
# - fitted Decision Tree model
#
# This makes the pipeline reusable for API prediction.

# %%
model_path = MODEL_DIR / "churn_model.pkl"
joblib.dump(final_model, model_path)

serializable_metrics = {
    metric_name: float(metric_value)
    for metric_name, metric_value in final_metrics.items()
}

metadata = {
    "model_name": final_model_name,
    "random_state": RANDOM_STATE,
    "test_size": 0.30,
    "target_mapping": {
        "No": 0,
        "Yes": 1,
    },
    # customerID is intentionally excluded because it was dropped before modelling.
    # The API should not require customerID for prediction.
    "required_columns": [
        column for column in X.columns
        if column != "customerID"
    ],
    "metrics": serializable_metrics,
    "hyperparameter_tuning": {
        "enabled": True,
        "method": "GridSearchCV",
        "scoring": "f1",
        "cross_validation": "5-fold StratifiedKFold",
        "best_cross_validation_f1": float(grid_search.best_score_),
        "best_parameters": grid_search.best_params_,
    },
    "class_imbalance_handling": (
        "GridSearchCV compared class_weight=None with "
        "class_weight='balanced'."
    ),
    "business_priority": (
        "Recall is important because missing a genuine churner may cause the "
        "company to lose the customer without attempting retention."
    ),
}

metadata_path = MODEL_DIR / "model_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as metadata_file:
    json.dump(
        metadata,
        metadata_file,
        indent=4,
        default=str,
    )

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "final_model_metrics.json",
    "w",
    encoding="utf-8",
) as metrics_file:
    json.dump(
        serializable_metrics,
        metrics_file,
        indent=4,
    )

print("Saved model:", model_path)
print("Saved metadata:", metadata_path)
print("Saved feature importance:", OUTPUT_DIR / "feature_importance.csv")
print("Saved final metrics:", OUTPUT_DIR / "final_model_metrics.json")

# %% [markdown]
# ## 10. API Usage
#
# The saved model can be loaded by the FastAPI application in `app.py`.
#
# The API endpoint is:
#
# `POST /predict`
#
# It accepts a customer JSON object, applies the saved preprocessing pipeline,
# and returns:
#
# - churn prediction
# - churn probability

# %% [markdown]
# Example API response:
#
# ```json
# {
#   "prediction": "Yes",
#   "churn_probability": 0.82
# }
# ```

# %% [markdown]
# ## 11. Final Business Conclusion
#
# This project built an end-to-end churn prediction workflow:
#
# Business Problem → Data → Preparation → EDA → Feature Engineering →
# Model → Evaluation → Interpretation → Saved Model → API-ready Pipeline
#
# The tuned Decision Tree model provides a practical churn-risk screening tool.
# For a telecom company, recall is especially important because the business
# wants to identify as many likely churners as possible before they leave.
#
# Although false positives may create extra retention contacts, missing true
# churners can be more costly because those customers may be lost permanently.