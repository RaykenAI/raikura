# Module 1: AutoMLPipeline – Core Engine

## Overview

`AutoMLPipeline` is the heart of Raikura. It orchestrates the entire ML workflow, allowing users to plug in any model, preprocessing strategy, and evaluation metric. The design is modular, allowing flexibility, extensibility, and config-based reproducibility.

---

## Responsibilities

* Pipeline configuration and validation
* Data preprocessing delegation
* Model instantiation and training
* Prediction
* Metric evaluation
* Saving and loading models

---

## Internal Implementation

**File**: `core/pipeline.py`

```python
class AutoMLPipeline:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None
        self.config = None

    def configure(self, preprocessing=None, model_params=None, evaluation=None):
        self.config = Config(preprocessing, model_params, evaluation)

    def train(self, X, y):
        X_processed = preprocess_data(X, self.config.preprocessing)
        self.model = get_model(self.model_type, self.config.model_params)
        self.model.fit(X_processed, y)

    def predict(self, X):
        X_processed = preprocess_data(X, self.config.preprocessing)
        return self.model.predict(X_processed)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return compute_metrics(y, preds, self.config.evaluation["metrics"])

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
```

---

## Dependencies

* `preprocess_data`: calls Raikura's modular preprocessing engine
* `get_model`: model selector from `models/`
* `Config`: class for schema-driven config management
* `compute_metrics`: evaluates predictions using multiple metric backends

---

## Key Methods

### `configure()`

Accepts preprocessing rules, model hyperparameters, and evaluation specs:

```python
pipe.configure(
  preprocessing={"scaling": True, "expand_polynomial": True},
  model_params={"n_estimators": 100},
  evaluation={"metrics": ["accuracy", "f1"]}
)
```

### `train(X, y)`

1. Preprocesses `X`
2. Instantiates the model
3. Fits it on `(X_processed, y)`

### `predict(X)`

Returns predictions using the same config-driven preprocessing pipeline.

### `evaluate(X, y)`

Computes all metrics on test predictions using unified interface.

### `save_model(path)` / `load_model(path)`

Persists or reloads fitted models using `joblib`

---

## Best Practices

* Always run `configure()` before training
* Use `config.yaml` for reproducibility (see config module)
* Pipe can be used in CLI and REST API via JSON/YAML configs

---

## Example

```python
pipe = AutoMLPipeline("random_forest")
pipe.configure(
  preprocessing={"scaling": True},
  model_params={"n_estimators": 200},
  evaluation={"metrics": ["accuracy"]}
)
pipe.train(X_train, y_train)
pipe.evaluate(X_test, y_test)
```

---

## Extension Points

* Add new models in `models/`
* Add new metrics in `evaluation.py`
* Add preprocessing techniques to `preprocess/`

---

## Future Directions

* Add support for pipeline serialization with config + model together
* Integrate MLFlow tracking or wandb
* Support model versioning and registries

---

## Summary

The `AutoMLPipeline` encapsulates Raikura's philosophy: composable, reproducible, intelligent ML pipelines without the boilerplate. It enables declarative config-driven workflows, lowering the barrier to complex AI pipelines.

Next Module: **Preprocessing Engine**
