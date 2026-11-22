"""
From-Scratch Neural Network & Optimizer Engine (Pure NumPy)
Includes Threshold Logic Neuron, Two-Layer MLP, Analytical Backpropagation,
and Batch GD, Mini-Batch SGD, Momentum, and Nesterov Accelerated Gradient (NAG).
"""
import numpy as np

class ThresholdNeuron:
    """Binary threshold logic gate."""
    def __init__(self, weights: np.ndarray, threshold: float):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.threshold = float(threshold)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (np.dot(x, self.weights) >= self.threshold).astype(int)


class ShelfMLP:
    """
    Two-Layer Feed-Forward Network (8 -> 16 -> 1) for Shelf Patch Stockout Classification.
    """
    def __init__(self, input_dim=8, hidden_dim=16, lr=0.05, gamma=0.9, l2_reg=1e-4, seed=42):
        self.config = {
            "input_dim": input_dim, "hidden_dim": hidden_dim,
            "learning_rate": lr, "momentum_gamma": gamma,
            "l2_lambda": l2_reg, "seed": seed
        }
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / (hidden_dim + 1))
        self.b2 = np.zeros((1, 1))
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

    def forward(self, X: np.ndarray, params=None):
        W1, b1, W2, b2 = params if params is not None else (self.W1, self.b1, self.W2, self.b2)
        z1 = X @ W1 + b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ W2 + b2
        yhat = self.sigmoid(z2)
        return yhat, (X, z1, a1, z2, yhat, W1, b1, W2, b2)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, W1=None, W2=None) -> float:
        W1 = self.W1 if W1 is None else W1
        W2 = self.W2 if W2 is None else W2
        eps = 1e-8
        bce = -np.mean(y_true * np.log(y_pred + eps) + (1.0 - y_true) * np.log(1.0 - y_pred + eps))
        l2_term = 0.5 * self.config["l2_lambda"] * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        return float(bce + l2_term)

    def backward(self, cache, y_true: np.ndarray):
        X, z1, a1, z2, yhat, W1, b1, W2, b2 = cache
        m = X.shape[0]
        l2 = self.config["l2_lambda"]
        dz2 = (yhat - y_true) / m
        dW2 = a1.T @ dz2 + l2 * W2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (a1 * (1.0 - a1))
        dW1 = X.T @ dz1 + l2 * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, X_batch: np.ndarray, y_batch: np.ndarray, optimizer="nesterov"):
        lr = self.config["learning_rate"]
        gamma = self.config["momentum_gamma"]
        if optimizer == "nesterov":
            lookahead = (self.W1 - gamma * self.vW1, self.b1 - gamma * self.vb1,
                         self.W2 - gamma * self.vW2, self.b2 - gamma * self.vb2)
            _, cache = self.forward(X_batch, params=lookahead)
        else:
            _, cache = self.forward(X_batch)
        dW1, db1, dW2, db2 = self.backward(cache, y_batch)
        if optimizer in ["momentum", "nesterov"]:
            self.vW1 = gamma * self.vW1 + lr * dW1
            self.vb1 = gamma * self.vb1 + lr * db1
            self.vW2 = gamma * self.vW2 + lr * dW2
            self.vb2 = gamma * self.vb2 + lr * db2
            self.W1 -= self.vW1; self.b1 -= self.vb1
            self.W2 -= self.vW2; self.b2 -= self.vb2
        else:
            self.W1 -= lr * dW1; self.b1 -= lr * db1
            self.W2 -= lr * dW2; self.b2 -= lr * db2