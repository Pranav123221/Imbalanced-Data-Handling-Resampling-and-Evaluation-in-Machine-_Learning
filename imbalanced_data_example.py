"""
Imbalanced Data Handling: Practical Example
============================================

Demonstrates:
- Stratified train/test split
- Baseline Logistic Regression
- Random Oversampling
- Random Undersampling
- SMOTE
- ADASYN
- Class-weighted Logistic Regression
- Evaluation with precision, recall, F1, ROC-AUC, Average Precision
- Confusion matrix for the best model by F1-score

Important:
- Resampling happens only on the training data via imbalanced-learn pipelines.
- The test set remains untouched.
- The data is synthetic, so no external dataset is required.

Install:
    pip install numpy pandas matplotlib scikit-learn imbalanced-learn

Run:
    python imbalanced_data_example.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("default")


def create_dataset():
    """Generate a reproducible binary classification dataset with class imbalance."""
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        class_sep=1.0,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """Return useful binary-classification metrics for a fitted model."""
    y_pred = model.predict(X_test)

    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        result["ROC-AUC"] = roc_auc_score(y_test, y_score)
        # Average Precision summarizes the precision-recall curve.
        result["Average Precision"] = average_precision_score(y_test, y_score)

    return result


def main():
    # 1. Generate data and inspect the class distribution.
    X, y = create_dataset()

    print("=" * 72)
    print("IMBALANCED DATA HANDLING — END-TO-END EXAMPLE")
    print("=" * 72)
    print("\nDataset shape:", X.shape)
    print("\nOriginal target distribution:")
    print(y.value_counts().sort_index())
    print("\nClass percentages:")
    print((y.value_counts(normalize=True).sort_index() * 100).round(2))

    # Optional plot of the original class distribution.
    y.value_counts().sort_index().plot(
        kind="bar", title="Original Class Distribution", rot=0
    )
    plt.xlabel("Target class")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.show()

    # 2. Split first. Stratification preserves class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())
    print("\nTest target distribution (kept untouched):")
    print(y_test.value_counts().sort_index())

    # 3. Baseline model: scaling is fitted on training data only.
    models = {
        "Baseline Logistic Regression": SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )
    }

    # 4. Samplers are applied only during model.fit() on training data.
    samplers = {
        "Random Oversampling": RandomOverSampler(random_state=42),
        "Random Undersampling": RandomUnderSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
    }

    for name, sampler in samplers.items():
        models[name] = ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("sampler", sampler),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )

    # 5. Class weighting changes the learning objective, not dataset size.
    models["Class-Weighted Logistic Regression"] = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    # 6. Fit and evaluate every model on the same untouched test set.
    results = []
    fitted_models = {}

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            fitted_models[name] = model
            results.append(evaluate_model(model, X_test, y_test, name))
            print(f"  Completed: {name}")
        except ValueError as exc:
            # Some samplers can fail for certain dataset sizes or neighborhood settings.
            print(f"  Skipped {name}: {exc}")

    if not results:
        raise RuntimeError("No model completed successfully. Check dependencies and data.")

    results_df = pd.DataFrame(results).sort_values(
        by="F1 Score", ascending=False
    )

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(results_df.round(4).to_string(index=False))

    # 7. Plot the main precision/recall/F1 trade-off.
    plot_df = results_df.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )

    plot_data = plot_df.pivot(index="Model", columns="Metric", values="Score")
    ax = plot_data.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Imbalanced Classification: Model Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

    # 8. Select by F1 as an illustrative rule, not a universal best criterion.
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    y_pred = best_model.predict(X_test)

    print("\n" + "=" * 72)
    print(f"CLASSIFICATION REPORT — {best_name}")
    print("=" * 72)
    print(classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Class 0 (majority)", "Class 1 (minority)"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.show()

    print("\nNotes:")
    print("- The dataset is synthetic; results are for demonstration only.")
    print("- No resampling is applied to the test set.")
    print("- The highest F1-score is only an example selection rule.")
    print("- Choose metrics and thresholds based on the real problem's error costs.")


if __name__ == "__main__":
    main()
