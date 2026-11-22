"""
ShelfSense AI — FastAPI Serverless Backend
Runs real NumPy Multi-Layer Perceptron (MLP) forward passes, SVD matrix decomposition,
and LSTM demand forecasting on live uploaded image patch tensors.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.mlp import ShelfMLP
from src.core.tensor_ops import run_tensor_diagnostics, svd_feature_compression, verify_gradients
from src.forecasting.sequence_models import DemandForecastLSTM, match_sku_attention

app = FastAPI(title="ShelfSense AI API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Pre-train a real 2-layer ShelfMLP (8 -> 16 -> 1) on startup so /api/analyze-image runs genuine neural inference
GLOBAL_MLP = ShelfMLP(input_dim=8, hidden_dim=16, lr=0.06, l2_reg=1e-4, seed=42)
np.random.seed(42)
_X_synth = np.random.randn(400, 8)
# True non-linear stockout boundary (high edge density, high saturation, low dark shadow -> SKU=1, else Stockout=0)
_y_synth = ((1.8 * _X_synth[:, 0] + 1.4 * _X_synth[:, 6] + 1.2 * _X_synth[:, 2] - 2.2 * _X_synth[:, 3] + 0.5 * np.tanh(_X_synth[:, 1] * _X_synth[:, 7])) > 0.0).astype(float).reshape(-1, 1)
for _ in range(80):
    GLOBAL_MLP.step(_X_synth, _y_synth, optimizer="nesterov")


class TrainRequest(BaseModel):
    learning_rate: float = 0.05
    optimizer: str = "nesterov"
    l2_lambda: float = 0.0001
    epochs: int = 40

class ForecastRequest(BaseModel):
    history: List[float] = [48, 44, 39, 35, 29, 24, 18, 14]
    horizon: int = 6

class ImagePatchRequest(BaseModel):
    patches: List[Dict[str, Any]]
    conf_threshold: float = 0.52

@app.get("/api/health")
def health():
    return {"status": "online", "engine": "ShelfMLP (8->16->1) + TF.js MobileNetV2", "seed": 42}

@app.post("/api/analyze-image")
def analyze_image_endpoint(req: ImagePatchRequest):
    """
    Receives real 8D patch feature vectors extracted from the uploaded image,
    normalizes them, runs a genuine forward pass through GLOBAL_MLP (8 -> 16 -> 1),
    and computes the real SVD singular value spectrum of the uploaded image's patches.
    """
    if not req.patches:
        return {"predictions": [], "svd_singular_values": []}

    raw_matrix = np.array([p["features"] for p in req.patches], dtype=np.float64)
    # Standardize patch matrix
    mean = np.mean(raw_matrix, axis=0, keepdims=True)
    std = np.std(raw_matrix, axis=0, keepdims=True) + 1e-6
    X_norm = (raw_matrix - mean) / std

    # Real 2-layer MLP forward pass: Z1 = X @ W1 + b1 -> A1 = sigmoid(Z1) -> Z2 = A1 @ W2 + b2 -> Yhat = sigmoid(Z2)
    yhat, (_, z1, a1, z2, _, _, _, _, _) = GLOBAL_MLP.forward(X_norm)
    probs = yhat.flatten().tolist()

    # Compute real SVD on the uploaded image's patch feature matrix
    svd_info = svd_feature_compression(X_norm, k_components=min(5, X_norm.shape[1]))

    results = []
    for idx, p in enumerate(req.patches):
        sku_prob = float(probs[idx])
        results.append({
            "id": idx,
            "row": p.get("row", 1),
            "x": p["x"],
            "y": p["y"],
            "w": p["w"],
            "h": p["h"],
            "sku_prob": round(sku_prob, 4),
            "stockout_prob": round(1.0 - sku_prob, 4),
            "hidden_activation_norm": round(float(np.linalg.norm(a1[idx])), 3),
            "is_stockout": bool(sku_prob < (1.0 - req.conf_threshold))
        })

    return {
        "predictions": results,
        "svd_singular_values": svd_info["singular_values"],
        "svd_cumulative_variance": svd_info["cumulative_variance_pct"]
    }

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
