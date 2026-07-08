import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import uvicorn

rf = joblib.load(r'C:\Python\Projects\credit_card_fraud\fraud_model.pkl')
explainer = shap.TreeExplainer(rf)

feature_names = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                 'V11','V12','V13','V14','V15','V16','V17','V18','V19',
                 'V20','V21','V22','V23','V24','V25','V26','V27','V28',
                 'Amount_scaled','Time_scaled']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return open(r'C:\Python\Projects\credit_card_fraud\index.html', 
                encoding='utf-8').read()

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount_scaled: float; Time_scaled: float

class Transaction(BaseModel):
    class Config:
        extra = 'allow'

@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()
    
    # Handle raw CSV upload (has Amount/Time instead of scaled)
    if 'Amount' in data and 'Amount_scaled' not in data:
        from sklearn.preprocessing import StandardScaler
        data['Amount_scaled'] = (data['Amount'] - 88.35) / 250.12
        data['Time_scaled'] = (data.get('Time', 0) - 94813) / 47488
    
    # Remove non-feature columns
    features = np.array([[data.get(f, 0.0) for f in feature_names]])
    
    prob = rf.predict_proba(features)[0][1]
    shap_vals = explainer.shap_values(features)
    top = pd.Series(
        shap_vals[0,:,1],
        index=feature_names
    ).abs().sort_values(ascending=False).head(5)
    
    return {
        "probability": round(float(prob), 4),
        "top_features": {k: round(float(v), 4) for k, v in top.items()}
    }

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)