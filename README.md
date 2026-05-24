# Cancer Gene Expression Classifier - REST API

## Overview
A production-style REST API that serves the cancer type classifier from the [Cancer Gene Expression project](https://github.com/shlokdhama/cancer-type-classification).
Given 20,531 RNA-Seq gene expression values, returns the predicted cancer type and confidence scores for all five cancer subtypes.

**Live API**: https://cancer-api-xxxx.onrender.com  
**Interactive docs**: https://cancer-api-xxxx.onrender.com/docs

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API info and status |
| GET | /health | Health check |
| POST | /predict | Predict cancer type from expression values |

## Example Request

```bash
curl -X POST "https://cancer-api-xxxx.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"expression_values": [0.0, 2.01, 3.26, ...]}'
```

## Example Response

```json
{
  "predicted_cancer_type": "PRAD",
  "confidence": 1.0,
  "all_probabilities": {
    "BRCA": 0.0,
    "COAD": 0.0,
    "KIRC": 0.0,
    "LUAD": 0.0,
    "PRAD": 1.0
  }
}
```

## Why an API and not just a script

Jupyter notebooks and Streamlit apps are useful for exploration and demonstration. A REST API is how ML models get used in production: other systems such as clinical dashboards, hospital software, and research pipelines can call the endpoint programmatically without knowing anything about the underlying model. This project demonstrates that deployment pattern.

## Tech Stack

FastAPI, uvicorn, scikit-learn, deployed on Render

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Model

PCA (200 components) + Logistic Regression trained on TCGA RNA-Seq data. 99.38% accuracy, 99.47% Macro F1 across 5 cancer types. Full methodology in the [parent repository](https://github.com/shlokdhama/cancer-type-classification).
