# Module 4: Evaluation Metrics & Performance Reporting

## Overview

The evaluation engine in Raikura is responsible for calculating performance metrics across classification, regression, and probabilistic models. It is integrated into `AutoMLPipeline.evaluate()` and supports both scalar and graphical outputs.

---

## Capabilities

* Unified interface for classification and regression metrics
* Visualizations: confusion matrix, ROC curve, PR curve
* Multi-metric evaluation
* Output aggregation into structured dictionary

---

## Supported Metrics

### Classification:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* Log Loss
* Matthews Correlation Coefficient
* Cohen’s Kappa

### Regression:

* R² (coefficient of determination)
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

---

## Entry Point

**File**: `core/evaluation.py`

```python
def compute_metrics(y_true, y_pred, metrics):
    results = {}
    for metric in metrics:
        if metric == "accuracy":
            results["accuracy"] = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            results["f1"] = f1_score(y_true, y_pred)
        elif metric == "roc_auc":
            results["roc_auc"] = roc_auc_score(y_true, y_pred)
        # ... additional metrics
    return results
```

---

## Visual Evaluation

```python
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True)


def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr)
```

Included automatically when `AutoMLPipeline.evaluate()` is called with config:

```python
evaluation={
  "metrics": ["accuracy", "f1", "roc_auc"],
  "visuals": True
}
```

---

## Example

```python
from raikura import AutoMLPipeline
pipe = AutoMLPipeline("random_forest")
pipe.configure(...)
pipe.train(X_train, y_train)
results = pipe.evaluate(X_test, y_test)
print(results)
```

---

## Advanced Features

* Works on multi-class classification (via macro/micro averaging)
* Regression metrics auto-detected from model type
* SHAP-compatible predictions for explainability alignment
* Probabilistic evaluation support (log loss, ROC AUC, etc.)

---

## How to Add Custom Metrics

**Add to `evaluation.py`:**

```python
elif metric == "custom_metric":
    results["custom_metric"] = my_custom_func(y_true, y_pred)
```

Then include it in config:

```yaml
metrics: ["custom_metric"]
```

---

## Summary

Raikura's evaluation layer enables flexible, multi-faceted model validation, with the ability to report multiple scores and diagnostic plots in one call. It’s built to integrate with explainability, fairness, and AutoML modules.

Next Module: **Explainability & SHAP Integration**
