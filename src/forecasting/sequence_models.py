"""
LSTM Shelf Depletion Forecaster & Self-Attention SKU Matcher.
"""
import numpy as np

class DemandForecastLSTM:
    def __init__(self, input_dim=1, hidden_dim=8, seed=42):
        np.random.seed(seed)
        concat_dim = input_dim + hidden_dim
        self.Wf = np.random.randn(hidden_dim, concat_dim) * 0.25
        self.bf = np.ones((hidden_dim, 1)) * 0.5
        self.Wi = np.random.randn(hidden_dim, concat_dim) * 0.25
        self.bi = np.zeros((hidden_dim, 1))
        self.Wc = np.random.randn(hidden_dim, concat_dim) * 0.25
        self.bc = np.zeros((hidden_dim, 1))
        self.Wo = np.random.randn(hidden_dim, concat_dim) * 0.25
        self.bo = np.zeros((hidden_dim, 1))
        self.Wy = np.random.randn(1, hidden_dim) * 0.4
        self.by = np.zeros((1, 1))
        self.hidden_dim = hidden_dim

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def forecast(self, history_seq: list, horizon: int = 6) -> dict:
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        arr = np.array(history_seq, dtype=float)
        mean, std = float(np.mean(arr)), float(np.std(arr) + 1e-6)
        norm_seq = (arr - mean) / std
        for val in norm_seq:
            z = np.vstack([h, [[val]]])
            f_t = self.sigmoid(self.Wf @ z + self.bf)
            i_t = self.sigmoid(self.Wi @ z + self.bi)
            c = f_t * c + i_t * np.tanh(self.Wc @ z + self.bc)
            o_t = self.sigmoid(self.Wo @ z + self.bo)
            h = o_t * np.tanh(c)
        slope = (arr[-1] - arr[0]) / max(1, len(arr) - 1)
        preds = []
        curr_val = norm_seq[-1]
        for step in range(1, horizon + 1):
            z = np.vstack([h, [[curr_val]]])
            f_t = self.sigmoid(self.Wf @ z + self.bf)
            i_t = self.sigmoid(self.Wi @ z + self.bi)
            c = f_t * c + i_t * np.tanh(self.Wc @ z + self.bc)
            o_t = self.sigmoid(self.Wo @ z + self.bo)
            h = o_t * np.tanh(c)
            delta = float((self.Wy @ h + self.by)[0, 0]) * 0.15
            curr_val += delta
            preds.append(round(max(0.0, arr[-1] + slope * step + delta * std), 1))
        return {"forecast_units": preds}

def match_sku_attention(query_emb: np.ndarray, catalog_embs: np.ndarray, catalog_names: list):
    d_k = query_emb.shape[-1]
    scores = (query_emb @ catalog_embs.T) / np.sqrt(d_k)
    exp_s = np.exp(scores - np.max(scores))
    attn_weights = exp_s / np.sum(exp_s)
    ranked = sorted(zip(catalog_names, attn_weights.tolist()), key=lambda x: x[1], reverse=True)
    return [{"sku": name, "attention_weight": round(float(w), 4)} for name, w in ranked]
