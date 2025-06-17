# Module 7: Time Series Forecasting

## Overview

Raikura provides native support for time series forecasting using both classical and ML-based methods. The pipeline enables lag-based feature engineering, rolling statistics, and future prediction using XGBoost, LGBM, or Prophet.

---

## Capabilities

* Lag features
* Rolling window statistics (mean, std, min, max)
* Timestamp-based train/test splits
* Autoregressive modeling
* Native support for Prophet, XGBoost, and LGBM for forecasting

---

## Entry Point

**File**: `timeseries/features.py`

```python
def generate_lag_features(df, target_col, lags=[1, 2, 3]):
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df

def generate_rolling_features(df, target_col, windows=[3, 7]):
    for w in windows:
        df[f"{target_col}_rollmean{w}"] = df[target_col].rolling(window=w).mean()
        df[f"{target_col}_rollstd{w}"] = df[target_col].rolling(window=w).std()
    return df
```

---

## Model Usage Example

```python
from raikura.timeseries import generate_lag_features, generate_rolling_features

data = generate_lag_features(data, target_col="sales", lags=[1, 2, 3])
data = generate_rolling_features(data, target_col="sales", windows=[7])

# Drop NaNs due to shifting
data = data.dropna()

X = data.drop(columns=["sales"])
y = data["sales"]

pipe = AutoMLPipeline("xgboost")
pipe.configure(...)
pipe.train(X, y)
pipe.evaluate(X_test, y_test)
```

---

## Prophet Integration

```python
from raikura.models.prophet_model import ProphetWrapper
model = ProphetWrapper()
model.fit(df)
future = model.predict(periods=30)
```

---

## CLI Usage

```bash
raikura forecast --config ts_config.yaml
```

Example YAML:

```yaml
pipeline:
  model_type: xgboost
  preprocessing:
    lag_features: [1, 2, 3]
    rolling_windows: [7, 14]
```

---

## Advanced Features

* Seasonality extraction (monthly, quarterly)
* Time-aware splits
* Integration with SHAP for feature explanation

---

## Summary

Raikura’s time series module enables ML-based forecasting pipelines through lag/rolling feature engineering and flexible model integration. Whether you're working on financial, retail, or sensor data, Raikura adapts to your time-based tasks easily.

Next Module: **NLP & Transformers**
