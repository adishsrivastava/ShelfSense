"""Automated Verification Suite for ShelfSense AI"""
import numpy as np
from src.core.mlp import ThresholdNeuron, ShelfMLP
from src.core.tensor_ops import verify_gradients, svd_feature_compression
from src.core.learning_rules import oja_principal_axis, energy_state_recovery
from src.vision.detector import compute_iou, detect_shelf_stockouts
from src.forecasting.sequence_models import DemandForecastLSTM, match_sku_attention

def test_pipeline():
    tn = ThresholdNeuron([1.0, 1.0], 1.5)
    assert tn.forward(np.array([[1, 1], [1, 0]])).tolist() == [1, 0]
    mlp = ShelfMLP(seed=42)
    X = np.random.randn(12, 8)
    y = (X[:, :1] > 0).astype(float)
    chk = verify_gradients(mlp, X, y)
    assert chk["passed_verification"] is True
    assert round(compute_iou([0, 0, 100, 100], [50, 0, 150, 100]), 3) == 0.333
    lstm = DemandForecastLSTM(seed=42)
    assert len(lstm.forecast([50, 40, 30, 20], horizon=4)["forecast_units"]) == 4