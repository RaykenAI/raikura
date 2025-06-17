# Module 2: Preprocessing Engine

## Overview

The preprocessing engine in Raikura is responsible for preparing raw data into model-consumable form. It supports tabular, categorical, text, and time-series inputs and dynamically adapts based on pipeline configuration.

---

## Supported Techniques

### 1. Scaling

* `StandardScaler`
* `MinMaxScaler`
* `RobustScaler`

### 2. Encoding

* `OneHotEncoder`
* `OrdinalEncoder`

### 3. Feature Engineering

* `PolynomialFeatures` (configurable degree, interactions)
* `FeatureExpander` for combining polynomial and interaction terms

### 4. Text Preprocessing

* HuggingFace tokenization (for transformer compatibility)
* Padding, truncation, batching

### 5. Time Series

* Lag features
* Rolling statistics
* Temporal windowing

---

## Entry Point

```python
from raikura.preprocess import preprocess_data
```

This function auto-selects steps based on the pipeline’s config.

---

## Example Config

```python
preprocessing={
    "scaling": "standard",
    "expand_polynomial": True,
    "poly_degree": 2,
    "poly_interaction_only": False,
    "text_column": "description",
    "text_model": "bert-base-uncased"
}
```

---

## Internal Architecture

**File**: `preprocess/__init__.py`

```python
def preprocess_data(X, config):
    if config.get("scaling"):
        X = apply_scaling(X, config["scaling"])
    if config.get("expand_polynomial"):
        X = FeatureExpander(degree=config.get("poly_degree", 2),
                            interaction_only=config.get("poly_interaction_only", False)).fit_transform(X)
    if config.get("text_column"):
        text_embeddings = tokenize_text(X[config["text_column"]], model_name=config.get("text_model", "bert-base-uncased"))
        X = pd.concat([X.drop(columns=[config["text_column"]]), text_embeddings], axis=1)
    return X
```

---

## FeatureExpander Class

Combines polynomial expansion with custom control over interaction terms.

```python
class FeatureExpander:
    def __init__(self, degree=2, interaction_only=False):
        self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    def fit_transform(self, X):
        return pd.DataFrame(self.poly.fit_transform(X), columns=self.poly.get_feature_names_out())
```

---

## How It Connects to the Pipeline

Used internally by `AutoMLPipeline.train()` and `predict()`:

```python
X_processed = preprocess_data(X, self.config.preprocessing)
```

---

## Extension Points

* Add `custom_scaler.py` under `preprocess/`
* Extend `FeatureExpander` with domain-specific features
* Plug in `text_cleaner.py` for classical NLP pipelines

---

## Summary

Raikura’s preprocessing engine makes pipelines modular and reproducible. Its tight integration with config files enables fast, dynamic customization without changing code.

Next Module: **Model Zoo and Custom Model API**
