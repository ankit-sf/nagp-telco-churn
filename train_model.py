import json
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
from sklearn.tree import DecisionTreeClassifier

from features.build_features import ChurnFeatureEngineer


# A fixed random seed makes the train/test split and model results reproducible.
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "TelcoCustomerChurn.csv"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load the Telco customer churn dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please place TelcoCustomerChurn.csv inside the data folder."
        )

    return pd.read_csv(DATA_PATH)


def build_pipeline(classifier):
    """
    Create one complete machine learning pipeline.

    Keeping feature engineering, missing-value treatment, encoding, and modelling
    together ensures that training data and future API data receive the same
    transformations.
    """

    numeric_pipeline = Pipeline(
        steps=[
            # Median imputation is less sensitive to extreme values than the mean.
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Fill missing categories before one-hot encoding.
            ("imputer", SimpleImputer(strategy="most_frequent")),

            # Unknown categories from future API requests will not cause errors.
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                make_column_selector(
                    dtype_include=["int64", "float64"]
                ),
            ),
            (
                "cat",
                categorical_pipeline,
                make_column_selector(
                    dtype_include=["object", "category", "bool"]
                ),
            ),
        ]
    )

    # The feature engineer runs before preprocessing so newly created columns
    # are also included in the numeric/categorical transformations.
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(model, X_test, y_test):
    """Calculate the evaluation metrics required by the assignment."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    return metrics, y_pred


def tune_decision_tree(X_train, y_train):
    """
    Tune a Decision Tree using five-fold stratified cross-validation.

    The search includes class_weight='balanced'. This gives more importance
    to the minority churn class during training without creating duplicate
    synthetic records.

    F1 score is used for tuning because it balances precision and recall.
    """

    base_model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    tuning_pipeline = build_pipeline(base_model)

    # These values provide a reasonable search without making training
    # unnecessarily slow for this assignment-sized dataset.
    parameter_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 4, 5, 6, 8, None],
        "classifier__min_samples_leaf": [5, 10, 20, 30],
        "classifier__min_samples_split": [2, 10, 20],
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

    print("\nBest tuned Decision Tree parameters:")
    print(grid_search.best_params_)
    print(
        "Best cross-validation F1 score: "
        f"{grid_search.best_score_:.4f}"
    )

    # Save all tested configurations so the tuning process is visible.
    tuning_results = pd.DataFrame(grid_search.cv_results_)

    useful_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_classifier__criterion",
        "param_classifier__max_depth",
        "param_classifier__min_samples_leaf",
        "param_classifier__min_samples_split",
        "param_classifier__class_weight",
    ]

    available_columns = [
        column for column in useful_columns
        if column in tuning_results.columns
    ]

    tuning_results[available_columns].sort_values(
        "rank_test_score"
    ).to_csv(
        OUTPUT_DIR / "hyperparameter_tuning_results.csv",
        index=False,
    )

    return grid_search


def save_confusion_matrix(y_test, y_pred):
    """Save a visual confusion matrix for the selected final model."""
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Final Decision Tree")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=150,
    )
    plt.close()


def save_feature_importance(model):
    """
    Save feature importance values from the final Decision Tree.

    One-hot encoding creates transformed feature names such as
    Contract_Month-to-month. Keeping those names makes the model easier
    to explain in the assignment.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

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

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    return importance_df


def select_final_model(model_results, trained_models):
    """
    Select the final Decision Tree.

    F1 score is the primary selection metric because it balances precision
    and recall. Recall is the tie-breaker because finding potential churners
    is the main business objective.
    """
    selected_row = model_results.sort_values(
        by=["f1_score", "recall"],
        ascending=False,
    ).iloc[0]

    selected_model_name = selected_row["model"]
    selected_model = trained_models[selected_model_name]

    return selected_model_name, selected_model


def main():
    df = load_data()

    print("\n==============================")
    print("DATA UNDERSTANDING")
    print("==============================")
    print(f"Dataset shape: {df.shape}")

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nTarget distribution:")
    print(df["Churn"].value_counts())

    print("\nTarget distribution as percentages:")
    print(
        (df["Churn"].value_counts(normalize=True) * 100).round(2)
    )

    # Convert the target into binary labels for classification.
    # No = 0 means the customer stayed.
    # Yes = 1 means the customer churned.
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "The Churn column contains values other than 'Yes' and 'No'."
        )

    X = df.drop(columns=["Churn"])

    # The required split is 70% training and 30% testing.
    # Stratification keeps the churn ratio similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # These are the two required baseline Decision Tree configurations.
    # The balanced version gives additional weight to churn customers.
    baseline_configs = {
        "decision_tree_depth_4": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=25,
            random_state=RANDOM_STATE,
        ),
        "decision_tree_depth_6_balanced": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    comparison_rows = []
    trained_models = {}

    print("\n==============================")
    print("BASELINE DECISION TREE MODELS")
    print("==============================")

    for model_name, classifier in baseline_configs.items():
        pipeline = build_pipeline(classifier)

        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train)

        metrics, _ = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        comparison_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        trained_models[model_name] = pipeline

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")

    print("\n==============================")
    print("HYPERPARAMETER TUNING")
    print("==============================")

    grid_search = tune_decision_tree(
        X_train,
        y_train,
    )

    tuned_model = grid_search.best_estimator_

    tuned_metrics, _ = evaluate_model(
        tuned_model,
        X_test,
        y_test,
    )

    comparison_rows.append(
        {
            "model": "tuned_decision_tree",
            **tuned_metrics,
        }
    )

    trained_models["tuned_decision_tree"] = tuned_model

    print("\nTuned Decision Tree test metrics:")
    print(f"Accuracy:  {tuned_metrics['accuracy']:.4f}")
    print(f"Precision: {tuned_metrics['precision']:.4f}")
    print(f"Recall:    {tuned_metrics['recall']:.4f}")
    print(f"F1 Score:  {tuned_metrics['f1_score']:.4f}")

    # Save the comparison of the two baselines and the tuned Decision Tree.
    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")
    print(comparison_df.to_string(index=False))

    # All candidates are Decision Trees, so selecting the best one still follows
    # the assignment requirement to build and use a Decision Tree Classifier.
    final_model_name, final_model = select_final_model(
        comparison_df,
        trained_models,
    )

    final_metrics, final_predictions = evaluate_model(
        final_model,
        X_test,
        y_test,
    )

    print("\n==============================")
    print("FINAL MODEL")
    print("==============================")
    print(f"Selected model: {final_model_name}")

    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    save_confusion_matrix(
        y_test,
        final_predictions,
    )

    importance_df = save_feature_importance(final_model)

    print("\nTop 10 important features:")
    print(importance_df.head(10).to_string(index=False))

    # Save the entire pipeline, not only the Decision Tree.
    # This allows the API to apply exactly the same preprocessing.
    model_path = MODEL_DIR / "churn_model.pkl"
    joblib.dump(final_model, model_path)

    metadata = {
        "model_name": final_model_name,
        "random_state": RANDOM_STATE,
        "test_size": 0.30,
        "target_mapping": {
            "No": 0,
            "Yes": 1,
        },
        "required_columns": list(X.columns),
        "metrics": final_metrics,
        "hyperparameter_tuning": {
            "enabled": True,
            "scoring": "f1",
            "cross_validation_folds": 5,
            "best_parameters": grid_search.best_params_,
            "best_cross_validation_f1": grid_search.best_score_,
        },
        "class_imbalance_handling": (
            "The tuning process compared class_weight=None and "
            "class_weight='balanced'."
        ),
        "business_priority": (
            "Recall is important because missing a genuine churner may cause "
            "the business to lose a customer without attempting retention."
        ),
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
            default=str,
        )

    with open(
        OUTPUT_DIR / "final_model_metrics.json",
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            final_metrics,
            metrics_file,
            indent=4,
        )

    print("\nSaved files:")
    print(f"- {model_path}")
    print(f"- {MODEL_DIR / 'model_metadata.json'}")
    print(f"- {OUTPUT_DIR / 'model_comparison.csv'}")
    print(f"- {OUTPUT_DIR / 'hyperparameter_tuning_results.csv'}")
    print(f"- {OUTPUT_DIR / 'feature_importance.csv'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix.png'}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()