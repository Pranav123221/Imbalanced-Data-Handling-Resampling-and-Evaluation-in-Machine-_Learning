"""
ADASYN: Adaptive Synthetic Sampling for Imbalanced Classification

Install:
    pip install numpy matplotlib scikit-learn imbalanced-learn

Run:
    python adasyn_imbalanced_classification.py

Compares baseline Logistic Regression with an ADASYN + Logistic Regression
pipeline. The train/test split happens BEFORE resampling, and the test set
is never resampled.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, classification_report, ConfusionMatrixDisplay,
)

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42


def show_distribution(y, title):
    classes, counts = np.unique(y, return_counts=True)
    print(f"\\n{title}")
    for label, count in zip(classes, counts):
        print(f"Class {label}: {count:,} ({count / len(y):.2%})")


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    scores = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "Average Precision": average_precision_score(y_test, probabilities),
    }

    print(f"\\n{name}")
    for metric, value in scores.items():
        print(f"{metric:20s}: {value:.4f}")

    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    return scores, predictions


def main():
    # 1. Generate an imbalanced binary classification dataset.
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.92, 0.08],
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )
    show_distribution(y, "Original class distribution")

    # 2. Split BEFORE resampling to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    show_distribution(y_train, "Training distribution before ADASYN")
    show_distribution(y_test, "Untouched test distribution")

    # 3. Baseline pipeline.
    baseline = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    # 4. ADASYN pipeline: scaling and resampling are fitted only on training data.
    adasyn_model = ImbPipeline([
        ("scaler", StandardScaler()),
        ("adasyn", ADASYN(random_state=RANDOM_STATE, n_neighbors=5)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    # 5. Cross-validation on training data only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline_cv = cross_val_score(
        baseline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
    )
    adasyn_cv = cross_val_score(
        adasyn_model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
    )

    print("\\n5-Fold Cross-Validation F1")
    print(f"Baseline: {baseline_cv.mean():.4f} +/- {baseline_cv.std():.4f}")
    print(f"ADASYN:   {adasyn_cv.mean():.4f} +/- {adasyn_cv.std():.4f}")

    # 6. Fit both models on the full training partition.
    baseline.fit(X_train, y_train)
    adasyn_model.fit(X_train, y_train)

    # Show the resampled training distribution.
    X_train_scaled = adasyn_model.named_steps["scaler"].transform(X_train)
    _, y_resampled = adasyn_model.named_steps["adasyn"].fit_resample(
        X_train_scaled, y_train
    )
    show_distribution(y_resampled, "Training distribution after ADASYN")

    # 7. Evaluate against the same untouched test set.
    baseline_scores, _ = evaluate_model(
        "Baseline Logistic Regression", baseline, X_test, y_test
    )
    adasyn_scores, adasyn_predictions = evaluate_model(
        "ADASYN + Logistic Regression", adasyn_model, X_test, y_test
    )

    # 8. Compare F1-score.
    labels = ["Baseline", "ADASYN"]
    values = [baseline_scores["F1-score"], adasyn_scores["F1-score"]]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)
    plt.ylabel("F1-score")
    plt.title("Baseline vs ADASYN")
    plt.ylim(0, 1)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
        )
    plt.tight_layout()
    plt.show()

    # 9. Confusion matrix for the ADASYN model.
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        adasyn_predictions,
        display_labels=["Majority (0)", "Minority (1)"],
    )
    plt.title("ADASYN Model - Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
