"""
Tensor Operations, SVD Feature Compression & Centered Finite-Difference Gradient Verification.
"""
import numpy as np

def run_tensor_diagnostics(batch_size=4, channels=3, height=8, width=8):
    np.random.seed(42)
    shelf_tensor = np.random.rand(batch_size, channels, height, width)
    channel_bias = np.array([0.1, -0.05, 0.2]).reshape(1, 3, 1, 1)
    broadcasted = shelf_tensor + channel_bias
    flattened = broadcasted.reshape(batch_size, -1)
    gram_matrix = np.inner(flattened, flattened)
    return {
        "original_shape": list(shelf_tensor.shape),
        "broadcast_bias_shape": list(channel_bias.shape),
        "flattened_shape": list(flattened.shape),
        "gram_matrix_shape": list(gram_matrix.shape),
        "gram_trace": round(float(np.trace(gram_matrix)), 4)
    }

def svd_feature_compression(X: np.ndarray, k_components: int = 5):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = (S ** 2) / np.sum(S ** 2)
    cum_var = np.cumsum(var_explained)
    X_rec = (U[:, :k_components] * S[:k_components]) @ Vt[:k_components, :]
    frob_error = np.linalg.norm(X - X_rec, ord="fro") / np.linalg.norm(X, ord="fro")
    return {
        "singular_values": [round(float(s), 4) for s in S],
        "cumulative_variance_pct": [round(float(c * 100), 2) for c in cum_var],
        "rank_k": k_components,
        "relative_frobenius_error": round(float(frob_error), 4)
    }

def verify_gradients(mlp, X_sample: np.ndarray, y_sample: np.ndarray, eps: float = 1e-5) -> dict:
    _, cache = mlp.forward(X_sample)
    dW1_ana, _, _, _ = mlp.backward(cache, y_sample)
    dW1_num = np.zeros_like(mlp.W1)
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            orig = mlp.W1[i, j]
            mlp.W1[i, j] = orig + eps
            lp = mlp.compute_loss(y_sample, mlp.forward(X_sample)[0])
            mlp.W1[i, j] = orig - eps
            lm = mlp.compute_loss(y_sample, mlp.forward(X_sample)[0])
            dW1_num[i, j] = (lp - lm) / (2.0 * eps)
            mlp.W1[i, j] = orig
    rel_error = np.linalg.norm(dW1_ana - dW1_num) / (np.linalg.norm(dW1_ana) + np.linalg.norm(dW1_num) + 1e-12)
    return {"epsilon": eps, "relative_l2_error": float(rel_error), "passed_verification": bool(rel_error < 1e-7)}