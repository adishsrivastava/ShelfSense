"""
Unsupervised Principal Axis Extraction (Oja's Rule) & Stochastic Energy Recovery.
"""
import numpy as np

def oja_principal_axis(X: np.ndarray, lr: float = 0.01, epochs: int = 40):
    np.random.seed(42)
    w = np.random.randn(X.shape[1])
    w /= np.linalg.norm(w)
    for _ in range(epochs):
        for x in X:
            y = np.dot(w, x)
            w = w + lr * y * (x - y * w)
    return w / np.linalg.norm(w)

def energy_state_recovery(patterns: np.ndarray, noisy_query: np.ndarray, T: float = 0.25, steps: int = 15):
    n_features = patterns.shape[1]
    W = (patterns.T @ patterns) / len(patterns)
    np.fill_diagonal(W, 0.0)
    state = noisy_query.copy().astype(float)
    energy_trace = []
    np.random.seed(42)
    for _ in range(steps):
        for i in range(n_features):
            delta_e = 2.0 * state[i] * np.dot(W[i], state)
            prob_flip = 1.0 / (1.0 + np.exp(np.clip(delta_e / T, -30, 30)))
            if np.random.rand() < prob_flip:
                state[i] *= -1.0
        energy = -0.5 * float(state.T @ W @ state)
        energy_trace.append(round(energy, 4))
    return {"recovered_state": state.tolist(), "energy_trace": energy_trace}
