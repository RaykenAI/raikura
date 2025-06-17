# Module 11: REST API & Deployment Layer

## Overview

Raikura supports serving trained models as production-ready APIs using FastAPI. This enables real-time predictions, model introspection, and interactive documentation via OpenAPI. Models can be deployed locally, in Docker, or on cloud platforms.

---

## Capabilities

* FastAPI-based REST endpoint
* Auto input validation with Pydantic
* Model loading and prediction
* Interactive Swagger docs (localhost/docs)
* Support for tabular, text, and multimodal models

---

## API Structure

**File**: `api/main.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: str

@app.post("/predict")
def predict(input: InputData):
    df = pd.DataFrame([input.dict()])
    preds = model.predict(df)
    return {"prediction": preds.tolist()}
```

---

## Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"feature1": 1.5, "feature2": 0.8, "feature3": "male"}'
```

---

## Running the Server

```bash
uvicorn api.main:app --reload
```

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger UI.

---

## Dockerization

**Dockerfile**

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run**

```bash
docker build -t raikura-api .
docker run -p 8000:8000 raikura-api
```

---

## Cloud Deployment Options

* **Render**: One-click deploy via `render.yaml`
* **Railway**: GitHub integration + Docker auto build
* **AWS Lambda**: via `Mangum` adapter for serverless APIs

---

## CLI Support

```bash
raikura serve --model models/best_model.pkl --port 8000
```

---

## Advanced Features (Roadmap)

* Model metadata inspection (`/meta`)
* Batch prediction endpoint (`/batch_predict`)
* Secure token-based access
* Logging + monitoring hooks (Prometheus compatible)

---

## Summary

Raikura’s deployment layer simplifies turning ML pipelines into production-grade services. With FastAPI, Docker, and CLI interfaces, your models are one step away from live, scalable endpoints with modern devops tooling.

Next Module: **CLI Utilities & YAML Configuration System**
