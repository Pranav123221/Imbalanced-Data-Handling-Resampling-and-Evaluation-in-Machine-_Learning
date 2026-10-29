# Imbalanced Data Handling, Resampling and Evaluation

An end-to-end machine learning project exploring techniques for handling imbalanced datasets, improving minority-class learning, and evaluating classification models using appropriate metrics.

This repository implements data-level and algorithm-level approaches, including random oversampling, random undersampling, SMOTE, ADASYN, class weighting, and threshold-aware classification evaluation.

## 1. Project Overview

Class imbalance occurs when the target classes in a classification dataset have significantly different numbers of observations.

For example, a fraud detection dataset might contain 99% legitimate transactions and only 1% fraudulent transactions. A model that predicts every transaction as legitimate could achieve 99% accuracy while failing to identify fraud.

Therefore, accuracy alone is often insufficient for evaluating models trained on imbalanced datasets.

This project explores how to handle class imbalance and evaluate whether the resulting models identify minority-class observations effectively.

### Objectives

- Analyze target class distributions.
- Identify the extent of class imbalance.
- Apply appropriate preprocessing techniques.
- Implement multiple resampling strategies.
- Explore class-weighted learning.
- Train and compare classification models.
- Evaluate precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion matrices.
- Prevent data leakage during preprocessing and resampling.
- Compare baseline and imbalance-aware approaches.

## 2. What Is Imbalanced Data?

A dataset is imbalanced when one or more target classes contain substantially fewer observations than other classes.

Consider the following example:

| Class | Number of samples | Distribution |
|---|---:|---:|
| Class 0 — Majority | 9,500 | 95% |
| Class 1 — Minority | 500 | 5% |
| Total | 10,000 | 100% |

The imbalance ratio is:

\[
\text{Imbalance Ratio} =
\frac{\text{Majority Class Count}}
{\text{Minority Class Count}}
\]

For this dataset:

\[
\text{Imbalance Ratio}=\frac{9500}{500}=19
\]

The majority class contains 19 times as many observations as the minority class.

A high imbalance ratio can make it difficult for a classifier to learn patterns associated with the minority class, although the actual impact depends on the dataset, features, and model.

## 3. Why Is Class Imbalance a Problem?

Standard classification algorithms optimize a training objective based on the observations they receive. When one class dominates the training data, the learned decision boundary may favor the majority class.

Potential consequences include:

- Low minority-class recall.
- Poor detection of rare events.
- Misleadingly high accuracy.
- A high number of false negatives.
- Poor probability estimates or unsuitable decision thresholds.
- Unequal error costs across classes.

### Example: Fraud detection

Suppose a dataset contains 10,000 transactions:

- 9,900 legitimate transactions.
- 100 fraudulent transactions.

A classifier predicts every transaction as legitimate.

Its accuracy is:

\[
\text{Accuracy}=\frac{9900}{10000}=99\%
\]

However, its fraud recall is:

\[
\text{Recall}=\frac{TP}{TP+FN}=0
\]

The model fails to detect every fraudulent transaction despite its high accuracy.

This demonstrates why imbalanced classification requires metrics beyond accuracy.

## 4. Types of Imbalance-Handling Techniques

There are several ways to address class imbalance. They can be grouped into data-level and algorithm-level approaches.

### 4.1 Random Oversampling

Random oversampling increases the representation of the minority class by sampling existing minority observations with replacement.

**Advantages**
- Simple to implement.
- Preserves the original minority observations.
- Can improve minority-class learning.

**Limitations**
- Duplicated observations may encourage overfitting.
- Does not generate new feature combinations.

### 4.2 Random Undersampling

Random undersampling reduces the number of majority-class observations.

**Advantages**
- Reduces training dataset size.
- Can reduce training time.
- May help balance the training classes.

**Limitations**
- May discard useful majority-class information.
- Results can vary depending on which observations are removed.

### 4.3 SMOTE

SMOTE stands for **Synthetic Minority Over-sampling Technique**.

Rather than simply duplicating minority observations, SMOTE generates synthetic minority samples by interpolating between a minority observation and one of its minority-class nearest neighbors.

A simplified interpolation formula is:

\[
x_{\text{new}}=x_i+\lambda(x_{nn}-x_i)
\]

Where:

- \(x_i\) is an existing minority-class observation.
- \(x_{nn}\) is one of its minority-class nearest neighbors.
- \(\lambda\) is a random value between 0 and 1.

**Advantages**
- Creates synthetic minority observations.
- Can improve minority-class representation.
- Avoids relying exclusively on exact duplicates.

**Limitations**
- Synthetic observations may be unrealistic.
- Can amplify noise or overlap between classes.
- Standard SMOTE is primarily intended for numerical feature spaces.
- Categorical data requires an appropriate alternative, such as SMOTENC.

### 4.4 ADASYN

ADASYN stands for **Adaptive Synthetic Sampling**.

It generates synthetic minority observations with greater emphasis on minority examples that are harder to learn, based on the local distribution of neighboring classes.

**Advantages**
- Adaptively allocates synthetic samples.
- Can emphasize difficult minority-class regions.

**Limitations**
- May generate additional samples near noisy or ambiguous decision boundaries.
- Can perform poorly when the minority class contains substantial noise.

### 4.5 Class Weighting

Class weighting modifies the loss function so that errors associated with selected classes receive different weights.

For example, scikit-learn classifiers that support `class_weight="balanced"` calculate class weights inversely proportional to observed class frequencies.

A common weighting rule is:

\[
w_c=\frac{N}{K N_c}
\]

Where:

- \(N\) is the total number of training observations.
- \(K\) is the number of classes.
- \(N_c\) is the number of observations in class \(c\).

Class weighting does not modify the dataset itself. It changes how the learning algorithm penalizes errors.

**Advantages**
- Does not require synthetic samples.
- Preserves the original training observations.
- Supported by several classification algorithms.

**Limitations**
- Not supported by every estimator.
- Can increase false positives if weights are poorly chosen.
- Does not guarantee better minority-class performance.

## 5. Project Workflow

The recommended workflow is:

1. Load and inspect the dataset.
2. Analyze the target distribution.
3. Separate features and target.
4. Split into training and test sets using stratification.
5. Fit preprocessing transformations on training data only.
6. Train a baseline classifier without resampling.
7. Apply imbalance-handling techniques to training data only.
8. Train models using the transformed training data.
9. Evaluate all models on the same untouched test set.
10. Compare classification metrics and confusion matrices.
11. Select a model based on the application's error costs and requirements.

### Critical rule: Avoid data leakage

**Split the dataset before resampling.**

Never apply SMOTE, random oversampling, or random undersampling to the entire dataset before splitting it into training and test sets.

Doing so can cause information from the test set to influence the training process and produce overly optimistic evaluation results.

For cross-validation, put resampling inside an imbalanced-learn pipeline so that each training fold is resampled independently.

## 6. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn

## 7. Installation

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

A corresponding `requirements.txt` could contain:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 8. Practical Implementation

The following example builds a binary classification workflow using a synthetic dataset. It compares an ordinary baseline classifier against multiple imbalance-handling strategies.

The synthetic dataset makes the example reproducible without downloading a separate file.

### Step 1: Import libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN
)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
```

### Step 2: Generate an imbalanced dataset

```python
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    flip_y=0.01,
    class_sep=1.0,
    random_state=42
)

X = pd.DataFrame(
    X,
    columns=[f"feature_{i}" for i in range(X.shape[1])]
)

y = pd.Series(y, name="target")

print("Feature shape:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())
```

The generated dataset has a deliberately imbalanced target distribution. The actual class counts can vary slightly from the nominal proportions because of the dataset-generation configuration.

### Step 3: Inspect class distribution

```python
class_counts = y.value_counts().sort_index()

print(class_counts)

print(
    "\nClass percentages:\n",
    (y.value_counts(normalize=True) * 100).round(2)
)
```

Visualize the distribution:

```python
sns.countplot(x=y)

plt.title("Original Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()
```

This step helps determine the severity of class imbalance before selecting an approach.

### Step 4: Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())
```

Stratification preserves approximately the original class proportions in both partitions.

The test set remains untouched by resampling.

### Step 5: Define evaluation metrics

```python
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        )
    }

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]

        results["ROC-AUC"] = roc_auc_score(
            y_test, y_score
        )

        results["PR-AUC (AP)"] = average_precision_score(
            y_test, y_score
        )

    return results
```

The `average_precision_score` output is Average Precision (AP), a summary of the precision-recall curve. It is commonly reported as a PR-AUC-style metric, but it is not identical to every numerical integration definition of area under the precision-recall curve.

### Step 6: Train a baseline model

```python
baseline_model = SklearnPipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

baseline_model.fit(X_train, y_train)

baseline_results = evaluate_model(
    baseline_model,
    X_test,
    y_test,
    "Baseline Logistic Regression"
)

print(baseline_results)
```

The baseline provides a reference point for evaluating the impact of imbalance-handling techniques.

Do not assume the baseline will always have poor recall; the result depends on the generated dataset and model configuration.

### Step 7: Create resampling pipelines

The resampling process must occur after the training split and during model fitting.

Using an `imblearn` pipeline ensures that the sampler is applied to training data when `fit()` is called. At prediction time, the sampler is not applied to the test data.

```python
samplers = {
    "Random Oversampling": RandomOverSampler(
        random_state=42
    ),

    "Random Undersampling": RandomUnderSampler(
        random_state=42
    ),

    "SMOTE": SMOTE(
        random_state=42
    ),

    "ADASYN": ADASYN(
        random_state=42
    )
}
```

Create the models:

```python
models = {
    "Baseline": baseline_model
}

for name, sampler in samplers.items():
    models[name] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", sampler),
        ("classifier", LogisticRegression(max_iter=2000))
    ])
```

Here, scaling is fitted using the training fold before resampling. The synthetic dataset contains continuous numerical features, making it suitable for this example.

### Step 8: Train and evaluate the models

```python
results = []

for name, model in models.items():
    try:
        model.fit(X_train, y_train)

        result = evaluate_model(
            model,
            X_test,
            y_test,
            name
        )

        results.append(result)

        print(f"\n{name}")
        print("-" * len(name))
        print(classification_report(
            y_test,
            model.predict(X_test),
            zero_division=0
        ))

    except ValueError as error:
        print(f"\n{name} could not be fitted: {error}")
```

Some samplers may fail on particular datasets or very small training sets. For example, SMOTE and ADASYN require enough minority-class neighbors for their configured neighborhood size.

Inspect errors rather than silently treating failed runs as successful results.

### Step 9: Compare the results

```python
results_df = pd.DataFrame(results)

print(
    results_df.sort_values(
        by="Recall",
        ascending=False
    ).to_string(index=False)
)
```

Visualize selected metrics:

```python
metrics_to_plot = [
    "Precision",
    "Recall",
    "F1 Score"
]

plot_df = results_df.melt(
    id_vars="Model",
    value_vars=metrics_to_plot,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Comparison of Imbalanced Classification Techniques")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()
```

Interpret the results rather than assuming one resampling method will always perform best.

A technique may improve recall but reduce precision, creating more false positives. The preferred trade-off depends on the application.

### Step 10: Analyze the confusion matrix

```python
best_model_name = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)

plt.title(f"Confusion Matrix: {best_model_name}")
plt.tight_layout()
plt.show()
```

For binary classification:

- **True Positive (TP):** Minority-class observation correctly predicted as positive.
- **True Negative (TN):** Majority-class observation correctly predicted as negative.
- **False Positive (FP):** Majority-class observation incorrectly predicted as positive.
- **False Negative (FN):** Minority-class observation missed by the model.

Selecting the model with the highest F1-score is only an example selection rule. In a real project, choose the selection metric according to the relative cost of false positives and false negatives.

## 9. Class Weighting: An Alternative to Resampling

Class weighting is another approach for imbalanced classification.

```python
weighted_model = SklearnPipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        )
    )
])

weighted_model.fit(X_train, y_train)

weighted_results = evaluate_model(
    weighted_model,
    X_test,
    y_test,
    "Class-Weighted Logistic Regression"
)

print(weighted_results)
```

Unlike oversampling and undersampling, class weighting does not change the number of training observations. It changes the penalty associated with mistakes on different classes.

Compare this approach with resampling methods on the same test set.

## 10. Classification Metrics Explained

### Accuracy

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

Measures the proportion of correct predictions.

**Limitation:** Can be misleading when one class dominates.

### Precision

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

Measures how many predicted positive observations are actually positive.

High precision is particularly useful when false positives are expensive.

### Recall (Sensitivity)

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

Measures how many actual positive observations the model identifies.

High recall is particularly important when missing a positive observation is costly.

### F1-score

\[
F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

Balances precision and recall using their harmonic mean.

### ROC-AUC

Measures ranking performance across classification thresholds using the receiver operating characteristic curve.

ROC-AUC is useful but can appear strong even when precision is poor for extremely rare positive classes.

### Precision-Recall curve and Average Precision

The precision-recall curve examines the trade-off between precision and recall across thresholds. Average Precision summarizes performance over this curve.

For highly imbalanced binary classification, precision-recall analysis is often particularly informative.

### Confusion matrix

Provides a direct count of correct and incorrect predictions for each class. It helps reveal which types of errors the model makes.

## 11. Decision Threshold Optimization

Most binary classifiers use a default probability threshold of 0.5 when converting predicted probabilities into class labels. This is not necessarily the best threshold for an imbalanced classification problem.

A lower threshold may increase recall while decreasing precision.

Threshold selection should be performed on a validation set or through appropriate cross-validation, not by repeatedly tuning against the final test set.

Example threshold analysis:

```python
from sklearn.metrics import precision_recall_curve

model = weighted_model

probabilities = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    precision[:-1],
    label="Precision"
)

plt.plot(
    thresholds,
    recall[:-1],
    label="Recall"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall Across Thresholds")
plt.legend()
plt.tight_layout()
plt.show()
```

This code visualizes the relationship between threshold, precision, and recall. For a valid final evaluation, perform threshold selection on separate validation data and evaluate the selected threshold once on the held-out test set.

## 12. Handling Imbalanced Data During Cross-Validation

For robust evaluation, resampling should occur separately inside each training fold.

An imbalanced-learn pipeline is useful because it applies the sampler during fitting rather than before the cross-validation split.

Example:

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv_pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("sampler", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=2000))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision"
}

cv_results = cross_validate(
    cv_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

for metric, values in cv_results.items():
    if metric.startswith("test_"):
        print(
            metric,
            round(values.mean(), 4),
            "+/-",
            round(values.std(), 4)
        )
```

This estimates cross-validation performance on the training partition. The final test set should remain separate until model and threshold decisions are complete.

## 13. Common Mistakes to Avoid

1. **Resampling before splitting:** Can leak information from the test set into training.
2. **Using accuracy alone:** May hide poor minority-class recall.
3. **Assuming SMOTE is always superior:** Different datasets require different approaches.
4. **Ignoring class overlap:** Synthetic samples can amplify ambiguity.
5. **Tuning against the test set:** Makes final evaluation overly optimistic.
6. **Ignoring precision:** Higher recall can come with many false positives.
7. **Applying numerical scaling blindly:** The transformation should match the features and model.
8. **Using SMOTE on arbitrary encoded categories:** Standard SMOTE may create invalid intermediate category values.
9. **Ignoring class support:** Precision and recall can be unstable when few positive observations exist.
10. **Treating resampled validation data as representative:** Evaluation should reflect the expected real-world class distribution whenever possible.

## 14. Suggested Repository Structure

```text
Imbalanced-Data-Handling-Resampling-and-Evaluation/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── imbalanced_data_analysis.ipynb
│
├── src/
│   ├── data_generation.py
│   ├── data_analysis.py
│   ├── resampling.py
│   ├── train_models.py
│   ├── evaluation.py
│   └── threshold_analysis.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

The `data/` folder can be omitted if the project generates synthetic data. Keep only files that actually exist in your repository.

## 15. Future Improvements

- Compare SMOTE, Borderline-SMOTE, SMOTENC, and other appropriate sampling strategies.
- Evaluate ensemble approaches such as Balanced Random Forest.
- Compare class weighting across multiple classifiers.
- Add stratified cross-validation with multiple random seeds.
- Implement threshold selection using a validation set.
- Analyze calibration and probability reliability.
- Compare performance on real-world fraud or medical classification datasets.
- Add automated tests and reproducible experiment configurations.
- Create an interactive Streamlit interface for comparing resampling techniques.

## 16. Learning Outcomes

This project provides practical experience with:

- Class-distribution analysis.
- Data preprocessing and stratified splitting.
- Random oversampling and undersampling.
- Synthetic minority sampling with SMOTE and ADASYN.
- Class-weighted classification.
- Precision, recall, F1-score, ROC-AUC, and Average Precision.
- Confusion-matrix analysis.
- Threshold selection.
- Cross-validation and leakage prevention.
- Model comparison on imbalanced datasets.

## 17. References

- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [imbalanced-learn: Over-sampling](https://imbalanced-learn.org/stable/over_sampling.html)
- [imbalanced-learn: Under-sampling](https://imbalanced-learn.org/stable/under_sampling.html)
- [Scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation
