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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# This lets the notebook import files from the project root.
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
# ## 1. Data Understanding and Preparation

# %%
df = pd.read_csv(DATA_PATH)
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.describe(include="all").T

# %% [markdown]
# ### Missing-value analysis
#
# In the IBM Telco churn dataset, `TotalCharges` may look numeric but can contain blank
# strings. These blanks become missing values after numeric conversion.

# %%
missing_before = df.isna().sum()
missing_before[missing_before > 0]

# %%
df["TotalCharges_numeric_check"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges_numeric_check"].isna().sum()

# %%
df = df.drop(columns=["TotalCharges_numeric_check"])

# %% [markdown]
# ### Duplicate analysis

# %%
df.duplicated().sum()

# %% [markdown]
# ### Numerical and categorical feature identification

# %%
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

numerical_cols, categorical_cols

# %% [markdown]
# `customerID` is an identifier and should not be used as a predictive feature.
# The target variable is `Churn`.

# %% [markdown]
# ### Target variable analysis

# %%
df["Churn"].value_counts()

# %%
df["Churn"].value_counts(normalize=True)

# %% [markdown]
# **Observation:** The dataset is moderately imbalanced. Most customers do not churn.
# Because the business wants to find customers who may leave, recall will be very important.

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
sns.set_theme(style="whitegrid")

# %% [markdown]
# ### Visualization 1: Churn distribution

# %%
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# %% [markdown]
# **Business insight:** Non-churn customers are the majority, but the churn group is large
# enough to represent a meaningful retention opportunity.

# %% [markdown]
# ### Visualization 2: Churn by contract type

# %%
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
plt.title("Churn by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.show()

# %% [markdown]
# **Business insight:** Month-to-month customers usually show higher churn. Longer contracts
# appear to create stronger customer commitment.

# %% [markdown]
# ### Visualization 3: Churn by internet service

# %%
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="InternetService", hue="Churn", palette="Set2")
plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.show()

# %% [markdown]
# **Business insight:** Churn behaviour differs across internet service categories. This can
# help the business design service-specific retention campaigns.

# %% [markdown]
# ### Visualization 4: Monthly charges by churn

# %%
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette="Set2", legend=False)
plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.show()

# %% [markdown]
# **Business insight:** Customers who churn often have higher monthly charges, suggesting
# price sensitivity may be an important churn driver.

# %% [markdown]
# ### Visualization 5: Tenure distribution by churn

# %%
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="tenure", hue="Churn", kde=True, bins=30, palette="Set2")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.show()

# %% [markdown]
# **Business insight:** Customers with shorter tenure are more likely to churn. Early-life
# customer engagement may be very important.

# %% [markdown]
# ### Visualization 6: Payment method and churn

# %%
plt.figure(figsize=(9, 4))
sns.countplot(data=df, x="PaymentMethod", hue="Churn", palette="Set2")
plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25, ha="right")
plt.show()

# %% [markdown]
# **Business insight:** Some payment methods, especially electronic check in many versions
# of this dataset, are associated with higher churn. This could reflect customer segment
# differences or payment friction.

# %% [markdown]
# ## 3. Feature Engineering
#
# The project creates multiple features inside `features/build_features.py`.
#
# Main engineered features:
#
# 1. `tenure_group`
#    - Created by grouping tenure into customer life-cycle bands.
#    - Useful because new customers and long-term customers often behave differently.
#
# 2. `avg_charge_per_tenure_month`
#    - Created as `TotalCharges / tenure`.
#    - Useful because it approximates customer value and billing intensity over time.
#
# Additional features:
# - `is_month_to_month`
# - `has_security_and_support`
# - `has_streaming_services`
#
# These are created inside the sklearn pipeline to avoid training/serving mismatch.

# %%
feature_engineer = ChurnFeatureEngineer()
sample_features = feature_engineer.fit_transform(df.drop(columns=["Churn"]))
sample_features.head()

# %% [markdown]
# ## 4. Model Development

# %%
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train.shape, X_test.shape

# %%
def build_pipeline(classifier):
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
            ("num", numeric_pipeline, make_column_selector(dtype_include=["int64", "float64"])),
            ("cat", categorical_pipeline, make_column_selector(dtype_include=["object", "category", "bool"])),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline

# %%
model_configs = {
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

results = []
trained_models = {}

for name, classifier in model_configs.items():
    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    results.append(metrics)
    trained_models[name] = pipeline

model_results = pd.DataFrame(results)
model_results

# %% [markdown]
# ### Model selection
#
# For churn prediction, recall is very important because the company does not want to miss
# customers who are likely to leave. However, precision also matters because the retention
# team has limited capacity.
#
# The final model is selected using F1 score first, with recall as the secondary priority.

# %%
final_model_name = model_results.sort_values(
    ["f1_score", "recall"],
    ascending=False
).iloc[0]["model"]

final_model = trained_models[final_model_name]
final_model_name

# %% [markdown]
# ## 5. Model Evaluation

# %%
final_pred = final_model.predict(X_test)

final_metrics = {
    "accuracy": accuracy_score(y_test, final_pred),
    "precision": precision_score(y_test, final_pred),
    "recall": recall_score(y_test, final_pred),
    "f1_score": f1_score(y_test, final_pred),
}

final_metrics

# %%
cm = confusion_matrix(y_test, final_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Final Decision Tree")
plt.show()

# %% [markdown]
# ### Business interpretation
#
# - **Accuracy** shows the overall percentage of correct predictions.
# - **Precision** tells us, out of customers predicted to churn, how many actually churned.
# - **Recall** tells us, out of all true churners, how many the model successfully found.
# - **F1 score** balances precision and recall.
#
# For a telecom churn use case, **recall should usually be prioritized**. Missing a true
# churner means the retention team may never contact that customer. A false positive is
# less costly because the business may simply send an offer to a customer who was not
# going to churn.

# %% [markdown]
# ## 6. Model Interpretation

# %%
preprocessor = final_model.named_steps["preprocessor"]
classifier = final_model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances,
    }
).sort_values("importance", ascending=False)

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
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Key finding:** The most important features usually relate to contract type, tenure,
# internet service, charges, and support services. These are strong indicators of customer
# commitment, satisfaction, and price sensitivity.

# %%
plt.figure(figsize=(22, 10))
plot_tree(
    classifier,
    max_depth=3,
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree Visualization - Top Levels")
plt.show()

# %% [markdown]
# ## 7. Model Saving

# %%
model_path = MODEL_DIR / "churn_model.pkl"
joblib.dump(final_model, model_path)

metadata = {
    "model_name": final_model_name,
    "random_state": RANDOM_STATE,
    "target_mapping": {"No": 0, "Yes": 1},
    "required_columns": list(X.columns),
    "metrics": final_metrics,
}

with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

model_path
