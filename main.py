from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import numpy as np
from model import load_model

app = FastAPI(
    title="Cancer Gene Expression Classifier API",
    description="Classifies cancer type from RNA-Seq gene expression data using PCA + Logistic Regression trained on TCGA data.",
    version="1.0.0"
)

# load model once at startup
scaler, pca, model = load_model()

CANCER_TYPES = ["BRCA", "KIRC", "COAD", "LUAD", "PRAD"]

class GeneExpressionInput(BaseModel):
    expression_values: list[float]
    
    @field_validator('expression_values')
    @classmethod
    def check_length(cls, v):
        if len(v) != 20531:
            raise ValueError(f"Expected 20531 gene expression values, got {len(v)}")
        return v

class PredictionOutput(BaseModel):
    predicted_cancer_type: str
    confidence: float
    all_probabilities: dict

@app.get("/")
def root():
    return {
        "message": "Cancer Gene Expression Classifier API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: GeneExpressionInput):
    try:
        X = np.array(input_data.expression_values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        
        prediction = model.predict(X_pca)[0]
        probabilities = model.predict_proba(X_pca)[0]
        
        prob_dict = {
            cancer: round(float(prob), 4)
            for cancer, prob in zip(model.classes_, probabilities)
        }
        
        return PredictionOutput(
            predicted_cancer_type=prediction,
            confidence=round(float(max(probabilities)), 4),
            all_probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))