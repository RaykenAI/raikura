# Raikura Complete Tutorial: Internals, Code, and Usage

This tutorial takes you from basic usage to in-depth understanding of how Raikura works under the hood — covering all major components, architectures, module structure, and extensibility.
This will be in a bunch of different parts, so follow the file numbers.
---

## 🔧 1. What is Raikura?

Raikura is a full-stack machine learning library designed to replace and improve upon scikit-learn. It includes modular components for:

* Data preprocessing
* Model training (classical + deep learning + transformers)
* Evaluation
* Explainability (SHAP)
* AutoML search
* Time series support
* Fairness auditing
* Config-based pipelines
* REST deployment via FastAPI

---

## 🏗 2. Core Architecture

At the center of Raikura is the `AutoMLPipeline` class, which unifies preprocessing, training, evaluation, and deployment.

**File: `core/pipeline.py`**

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

## 🧹 3. Preprocessing Layer

**File: `preprocess/scalers.py`, `encoders.py`, `feature_expander.py`**

* StandardScaler, MinMaxScaler, RobustScaler wrappers
* OneHotEncoder, OrdinalEncoder
* `FeatureExpander` handles polynomial, interaction, lags, rolling
* NLP tokenizer: uses HuggingFace
* TimeSeriesPreprocessor: creates temporal windows

---

## 📦 4. Models

**File: `models/base.py`, `models/torch_models.py`, `models/transformer.py`**

* Classical models: RandomForest, XGBoost, LightGBM, SVM, LogisticRegression
* Torch ANN: MLP for tabular with dropout, batchnorm
* Transformers: fine-tunes `bert-base-uncased` for text classification

```python
class TorchANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
```

---

## 📊 5. Evaluation & Metrics

**File: `core/evaluation.py`**

* Unified metric engine supporting classification and regression
* Metrics include: `accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `r2`, `mae`, `rmse`
* Integrated confusion matrix and ROC plotting

---

## 🧠 6. Explainability

**File: `explain/explainer.py`**

* Wrapper for SHAP
* Auto-selects TreeExplainer or KernelExplainer based on model
* Generates: summary\_plot, force\_plot, waterfall

```python
explainer = Explainability(pipe.model, X_train)
explainer.shap_summary_plot(X_train)
```

---

## ⚙️ 7. Config-Driven Pipelines

**File: `config/config_loader.py`**

* YAML files define full pipelines for reproducibility
* Compatible with CLI: `raikura train --config config.yaml`

```yaml
model_type: random_forest
preprocessing:
  scaling: true
  expand_polynomial: true
  poly_degree: 2
evaluation:
  metrics: [accuracy, f1]
```

---

## 🧠 8. AutoML

**File: `automl/search.py`**

* Grid Search, Random Search, and Bayesian Optimization via Optuna
* Search space defined via config or API

```python
from raikura.automl import AutoMLSearch
search = AutoMLSearch(X, y, model_type="xgboost", search_type="bayesian")
best_model = search.run()
```

---

## 📈 9. Time Series Support

**File: `preprocess/timeseries.py`**

* `TimeSeriesPreprocessor` adds lag/rolling features
* Supports forecasting via XGBoost or LightGBM
* Future-ready for LSTM/Seq2Seq models

---

## ⚖️ 10. Fairness Auditing

**File: `fairness/metrics.py`**

* Computes disparate impact, statistical parity, equalized odds
* Group disaggregated metrics
* Plots bias distributions

---

## 🔀 11. Multi-Modal Fusion

**File: `fusion/fusion_builder.py`**

* Auto-merges tabular, categorical, and textual columns
* Combines vectorized text with dense features
* Trains unified model across modalities

---

## 🌐 12. REST API Deployment

**File: `core/api.py`**

* FastAPI app for serving prediction endpoint

```bash
uvicorn raikura.core.api:app --reload
```

Send POST request to `/predict` with JSON:

```json
{
  "columns": ["feature1", "feature2"],
  "data": [[0.1, 0.2]]
}
```

---

## ✅ Summary of Capabilities

| Module     | Features                                     |
| ---------- | -------------------------------------------- |
| Preprocess | Scaling, encoding, time series, fusion, text |
| Models     | Classical ML, Torch MLP, Transformers        |
| AutoML     | Grid, Random, Bayesian + config              |
| Evaluation | Unified classification + regression metrics  |
| Explain    | SHAP, feature importances, force plots       |
| Deployment | REST API (FastAPI)                           |
| Fusion     | Multi-modal (text + tabular) pipelines       |
| Fairness   | Bias metrics and reports                     |

---

## 📘 Getting Started

```bash
pip install raikura
```

```python
from raikura import AutoMLPipeline
pipe = AutoMLPipeline("random_forest")
pipe.configure(...)
pipe.train(X, y)
pipe.evaluate(X_test, y_test)
pipe.save_model("model.pkl")
```

Let me know if you’d like an in-depth breakdown of any specific module.
