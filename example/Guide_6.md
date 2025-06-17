# Module 5: Explainability & SHAP Integration

## Overview

Raikura integrates SHAP (SHapley Additive exPlanations) to provide interpretable insights into model predictions. It supports both global and local explainability using SHAP’s TreeExplainer, DeepExplainer, and KernelExplainer depending on the model type.

---

## Entry Point

**File**: `explain/explainer.py`

```python
class Explainability:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.explainer = self._select_explainer()

    def _select_explainer(self):
        if hasattr(self.model, 'predict_proba'):
            return shap.Explainer(self.model.predict_proba, self.data)
        else:
            return shap.Explainer(self.model.predict, self.data)

    def shap_summary_plot(self, X):
        shap_values = self.explainer(X)
        shap.summary_plot(shap_values, X)

    def shap_force_plot(self, X, index=0):
        shap_values = self.explainer(X)
        return shap.plots.force(shap_values[index])

    def shap_waterfall_plot(self, X, index=0):
        shap_values = self.explainer(X)
        return shap.plots.waterfall(shap_values[index])
```

---

## Basic Usage

```python
from raikura import Explainability
explainer = Explainability(model=pipe.model, data=X_train)
explainer.shap_summary_plot(X_train)
```

---

## Supported Visuals

| Plot Type  | Purpose                          |
| ---------- | -------------------------------- |
| Summary    | Global feature importance        |
| Force Plot | Local explanation per sample     |
| Waterfall  | Local contribution decomposition |

All SHAP visuals are interactive when run inside Jupyter Notebooks.

---

## Example Integration

```python
pipe = AutoMLPipeline("xgboost")
pipe.configure(...)
pipe.train(X_train, y_train)

explainer = Explainability(pipe.model, X_train)
explainer.shap_summary_plot(X_train)
explainer.shap_force_plot(X_test, index=0)
```

---

## Advanced Support

* Multi-class classification support using SHAP’s `maskers`
* Automatically selects `TreeExplainer`, `KernelExplainer`, or `DeepExplainer`
* Pluggable into REST APIs via static image generation

---

## Limitations

* Torch models must implement `.predict` compatible with numpy
* Large datasets should be downsampled for plotting
* Force plots require JavaScript-enabled environments for full interactivity

---

## Summary

Raikura’s SHAP integration makes model behavior transparent. You get both global feature importance and local decision reasoning out-of-the-box using a unified class interface, enabling trust and auditability in ML pipelines.

Next Module: **AutoML Search & Hyperparameter Optimization**
