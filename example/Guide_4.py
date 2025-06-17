# Module 3: Model Zoo and Custom Model API

## Overview

Raikura includes a diverse set of models spanning traditional machine learning, deep learning, and transformer-based architectures. The library is designed to be easily extensible with your own custom models using a consistent API.

---

## Core Entry Point

**File**: `models/__init__.py`

```python
def get_model(model_type, params=None):
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "xgboost":
        return xgb.XGBClassifier(**params)
    elif model_type == "lightgbm":
        return lgb.LGBMClassifier(**params)
    elif model_type == "torch_ann":
        return TorchTabularModel(**params)
    elif model_type == "bert_classifier":
        return TransformerClassifier(model_name=params.get("model_name", "bert-base-uncased"))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

---

## Categories

### 1. Traditional ML

* `RandomForestClassifier`
* `XGBClassifier`
* `LGBMClassifier`
* `LogisticRegression`
* `SVC`

### 2. Deep Learning

**File**: `models/torch_models.py`

```python
class TorchTabularModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
```

Wrapped using `skorch` or custom training loop inside pipeline.

### 3. Transformers

**File**: `models/transformer.py`

```python
class TransformerClassifier:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def fit(self, X, y):
        # Custom tokenization + training loop using Trainer API
        pass

    def predict(self, X):
        # Tokenize + run model forward pass
        pass
```

---

## Training Interface

All models must implement:

* `.fit(X, y)`
* `.predict(X)`
* Optionally `.predict_proba(X)` for probabilistic outputs

---

## Adding a Custom Model

1. Create your model class in `models/custom_model.py`

```python
class MyModel:
    def __init__(self, **params):
        self.model = SomeSklearnCompatibleModel(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
```

2. Register in `get_model()`:

```python
elif model_type == "my_model":
    return MyModel(**params)
```

3. Use in pipeline:

```python
AutoMLPipeline("my_model")
```

---

## Internals

* Uses `joblib` to serialize models (works for sklearn, XGBoost, LGBM)
* Torch models use `.state_dict()` internally (soon to support TorchScript)
* Transformers use `transformers.Trainer` for training abstraction

---

## Advanced Features

* Hyperparameter passthrough via YAML or `configure()`
* Torch + transformer models are fully compatible with `AutoMLPipeline`
* Auto-type inference for classification vs regression coming soon

---

## Summary

Raikura’s model architecture provides plug-and-play extensibility without sacrificing usability. You can integrate any new architecture that follows the `.fit()`/`.predict()` interface. Transformers and Torch models are first-class citizens in the pipeline.

Next Module: **Evaluation Metrics & Performance Reporting**
