# Module 12: CLI Utilities & YAML Configuration System

## Overview

Raikura provides a flexible CLI interface and YAML-based configuration system that enables full ML pipeline automation without writing code. This supports training, prediction, evaluation, deployment, and AutoML tasks.

---

## CLI Architecture

**File**: `cli/main.py`

```python
import typer
from raikura.core.pipeline import AutoMLPipeline

app = typer.Typer()

@app.command()
def train(config: str):
    from raikura.utils.config_loader import load_config
    cfg = load_config(config)
    pipe = AutoMLPipeline(cfg.pipeline.model_type)
    pipe.configure(cfg.pipeline)
    pipe.train(cfg.data.X_train, cfg.data.y_train)
    pipe.evaluate(cfg.data.X_test, cfg.data.y_test)
```

Run via:

```bash
raikura train --config configs/pipeline_config.yaml
```

---

## CLI Commands

| Command    | Description                  |
| ---------- | ---------------------------- |
| `train`    | Train model via config file  |
| `evaluate` | Run evaluation pipeline      |
| `automl`   | Launch AutoML search         |
| `serve`    | Launch API service           |
| `forecast` | Run time series forecasting  |
| `nlp`      | Fine-tune transformer model  |
| `audit`    | Run fairness & bias audit    |
| `version`  | Show current Raikura version |

---

## YAML Config Example

```yaml
pipeline:
  model_type: lightgbm
  preprocessing:
    scaling: standard
    text_column: "summary"
    text_model: "distilbert-base-uncased"
  model_params:
    num_leaves: 50
  evaluation:
    metrics: ["f1", "roc_auc"]

data:
  X_train: path/to/X_train.csv
  y_train: path/to/y_train.csv
  X_test: path/to/X_test.csv
  y_test: path/to/y_test.csv
```

---

## Loading Configs Programmatically

```python
from raikura.utils.config_loader import load_config
cfg = load_config("configs/your_config.yaml")
```

---

## CLI Packaging

**setup.py** includes:

```python
entry_points={
  'console_scripts': [
    'raikura=cli.main:app',
  ],
}
```

After installing Raikura, commands like `raikura train` become globally available.

---

## Auto-Discovery

* YAML schema validated against required sections
* CLI auto-suggests valid commands with `--help`

---

## Summary

The Raikura CLI + YAML system abstracts ML operations into declarative workflows. It enables repeatable, configurable machine learning pipelines ideal for teams, production, and no-code execution.

This concludes the 12-part comprehensive Raikura module documentation series.
Would you like a zip export of all markdown modules?
