# Module 9: Multimodal Fusion (Text + Tabular)

## Overview

Raikura supports multimodal learning by fusing transformer-based textual representations with structured tabular data. This allows for richer context in domains like healthcare, finance, and customer support.

---

## Use Case Example

**Scenario**: Predict loan approval using both customer profile (tabular) and application description (text).

---

## Core Pipeline

1. Tokenize text column into embeddings using a pretrained transformer
2. Concatenate embeddings with tabular features
3. Feed into a hybrid model (e.g., MLP, XGBoost, LGBM)

---

## Preprocessing Stage

```python
from raikura.text.utils import tokenize_text

text_embeddings = tokenize_text(X["application_text"], model_name="bert-base-uncased")
X_fused = pd.concat([X.drop(columns=["application_text"]), text_embeddings], axis=1)
```

This returns a fused feature matrix with both numerical and semantic data.

---

## Model Training

```python
pipe = AutoMLPipeline("lightgbm")
pipe.configure(
  preprocessing={"scaling": True},
  model_params={"num_leaves": 50},
  evaluation={"metrics": ["accuracy", "roc_auc"]}
)
pipe.train(X_fused, y)
```

---

## Configuration (YAML)

```yaml
pipeline:
  model_type: lightgbm
  preprocessing:
    text_column: "application_text"
    text_model: "bert-base-uncased"
    scaling: true
  evaluation:
    metrics: ["accuracy", "f1"]
```

---

## CLI

```bash
raikura multimodal --config config_multi.yaml
```

---

## Architecture

```
          [ Tabular Columns ]
                   ↓
       [ Numerical Feature Vector ]
                   ↓
       ─────────────────────────────
                   ↑
    [ Transformer Embeddings from Text Column ]
                   ↓
         [ Concatenated Feature Vector ]
                   ↓
           [ Classifier (e.g. XGB) ]
```

---

## Supported Models

* All tabular models: RandomForest, LGBM, XGBoost, TorchANN
* Future: Cross-modal attention fusion (for Torch models)

---

## Summary

Raikura allows flexible fusion of semantic and structured knowledge, enabling modern multimodal ML workflows. This design enhances performance on datasets where textual context adds meaning to tabular patterns.

Next Module: **Fairness, Bias, and Ethical ML Audits**
