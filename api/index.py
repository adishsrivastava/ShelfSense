"""
ShelfSense AI — FastAPI Serverless Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.mlp import ShelfMLP
from src.core.tensor_ops import run_tensor_diagnostics, svd_feature_compression, verify_gradients
from src.forecasting.sequence_models import DemandForecastLSTM, match_sku_attention

app = FastAPI(title="ShelfSense AI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TrainRequest(BaseModel):
    learning_rate: float = 0.05
    optimizer: str = "nesterov"
    l2_lambda: float = 0.0001
    epochs: int = 40

class ForecastRequest(BaseModel):
    history: List[float] = [48, 44, 39, 35, 29, 24, 18, 14]
    horizon: int = 6

@app.get("/api/health")
def health():
    return {"status": "online", "service": "ShelfSense AI", "seed": 42}

@app.post("/api/train-mlp")
def train_mlp_endpoint(req: TrainRequest):
    np.random.seed(42)
    X = np.random.randn(240, 8)
    w_true = np.array([[1.4, -1.8, 0.9, 0.5, -0.7, 1.1, -0.4, 1.3]]).T
    y = ((X @ w_true + np.random.normal(0, 0.3, (240, 1))) > 0.0).astype(float)
    mlp = ShelfMLP(lr=req.learning_rate, l2_reg=req.l2_lambda, seed=42)
    loss_curve = []
    for _ in range(min(req.epochs, 100)):
        mlp.step(X[:180], y[:180], optimizer=req.optimizer)
        yhat_val, _ = mlp.forward(X[180:])
        loss_curve.append(round(mlp.compute_loss(y[180:], yhat_val), 4))
    yhat_val, _ = mlp.forward(X[180:])
    acc = float(np.mean((yhat_val >= 0.5) == y[180:]))
    return {
        "optimizer": req.optimizer,
        "final_val_loss": loss_curve[-1],
        "val_accuracy_pct": round(acc * 100, 2),
        "loss_curve": loss_curve,
        "gradient_check": verify_gradients(mlp, X[:8], y[:8])
    }

@app.get("/api/diagnostics")
def diagnostics_endpoint():
    np.random.seed(42)
    X_demo = np.random.randn(100, 8)
    mlp = ShelfMLP(seed=42)
    return {
        "tensor_ops": run_tensor_diagnostics(),
        "svd_diagnostics": svd_feature_compression(X_demo, k_components=5),
        "grad_verification": verify_gradients(mlp, X_demo[:6], (X_demo[:6, :1] > 0).astype(float))
    }

@app.post("/api/forecast")
def forecast_endpoint(req: ForecastRequest):
    lstm = DemandForecastLSTM(seed=42)
    q_emb = np.array([0.85, 0.12, 0.44, 0.91])
    cat_embs = np.array([[0.83, 0.15, 0.42, 0.89], [0.71, 0.28, 0.39, 0.75], [0.12, 0.88, 0.65, 0.20], [-0.3, 0.45, 0.81, 0.10]])
    cat_names = ["SKU-104: Coca-Cola Zero 330ml", "SKU-108: Coca-Cola Classic 330ml", "SKU-209: Pepsi Max 330ml", "SKU-412: Monster Ultra 500ml"]
    return {"lstm_forecast": lstm.forecast(req.history, horizon=req.horizon), "sku_matcher": match_sku_attention(q_emb, cat_embs, cat_names)}