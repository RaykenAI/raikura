# Module 10: Fairness, Bias, and Ethical ML Audits

## Overview

Raikura integrates fairness and bias detection to promote ethical and responsible machine learning. This module enables demographic parity checks, disparate impact measurement, subgroup performance comparison, and SHAP-based bias localization.

---

## Capabilities

* Group-level evaluation (e.g. gender, race, income bracket)
* Disparate impact ratio
* Equal opportunity difference
* SHAP-based bias localization
* Fairness-aware evaluation summary

---

## Entry Point

**File**: `fairness/metrics.py`

```python
def compute_disparate_impact(y_true, y_pred, sensitive_attr):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive_attr})
    p_y_given_group = df.groupby("group")["y_pred"].mean()
    ratio = p_y_given_group.min() / p_y_given_group.max()
    return ratio
```

---

## Example Usage

```python
from raikura.fairness import compute_disparate_impact

ratio = compute_disparate_impact(y_true, y_pred, sensitive_attr=df["gender"])
print("Disparate Impact Ratio:", ratio)
```

---

## Visuals and Reporting

```python
def plot_group_accuracy(y_true, y_pred, sensitive_attr):
    accs = {}
    for g in set(sensitive_attr):
        idx = sensitive_attr == g
        accs[g] = accuracy_score(y_true[idx], y_pred[idx])
    plt.bar(accs.keys(), accs.values())
    plt.title("Accuracy by Group")
```

---

## Configuration Example

```yaml
fairness:
  metrics: ["disparate_impact", "equal_opportunity"]
  sensitive_column: "gender"
  visuals: true
```

---

## CLI Usage

```bash
raikura audit --config fairness.yaml
```

---

## Advanced Features

* Works with any binary classifier
* Integrates with `AutoMLPipeline.evaluate()` if fairness config is passed
* Will integrate with `Fairlearn` in future versions

---

## Summary

Raikura enables ethical ML pipelines by exposing hidden disparities and subgroup imbalances, combining statistical fairness metrics with explainability. It’s a built-in toolkit to build trustable AI at scale.

---

Next Steps:

* Would you like **Module 11: REST API & Deployment Layer** next?
