# Module 6: AutoML Search & Hyperparameter Optimization

## Overview

Raikura supports powerful hyperparameter optimization out-of-the-box using:

* Grid Search
* Random Search
* Bayesian Optimization (via Optuna)

This module allows you to automatically find the best configuration for any supported model using performance metrics as guidance.

---

## Entry Point

**File**: `automl/search.py`

```python
class AutoMLSearch:
    def __init__(self, X, y, model_type="xgboost", search_type="bayesian", scoring="accuracy", max_trials=50):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.search_type = search_type
        self.scoring = scoring
        self.max_trials = max_trials

    def run(self):
        if self.search_type == "grid":
            return self._grid_search()
        elif self.search_type == "random":
            return self._random_search()
        elif self.search_type == "bayesian":
            return self._bayesian_optimization()
        else:
            raise ValueError("Invalid search type")
```

---

## Supported Algorithms

| Search Type | Library Used | Notes                            |
| ----------- | ------------ | -------------------------------- |
| Grid        | scikit-learn | Exhaustive across all params     |
| Random      | scikit-learn | Random combinations              |
| Bayesian    | Optuna       | Efficient + probabilistic search |

---

## Example Usage

```python
from raikura.automl import AutoMLSearch

search = AutoMLSearch(X_train, y_train,
                      model_type="lightgbm",
                      search_type="bayesian",
                      scoring="f1",
                      max_trials=50)

best_model = search.run()
```

The result is a trained and fitted model with the best hyperparameter configuration.

---

## Configuration Example (YAML)

```yaml
automl:
  model_type: xgboost
  search_type: bayesian
  scoring: f1
  max_trials: 40
```

---

## Model Compatibility

Supported model types:

* xgboost
* lightgbm
* random\_forest
* torch\_ann (search space defined manually)
* custom models (must implement `.fit()` and accept hyperparams)

---

## Advanced Features

* Objective-based early stopping (Optuna)
* Logging trials and performance
* Visual plots for performance vs. hyperparams

---

## Extendability

To add a new optimizer:

* Create `*.py` in `automl/`
* Register in `AutoMLSearch.run()`

---

## Summary

Raikura’s AutoML engine abstracts away the complexity of model tuning. By using declarative config or direct API calls, you can automate the process of selecting optimal parameters for powerful models in a scalable, repeatable way.

Next Module: **Time Series Forecasting**
