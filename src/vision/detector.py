"""
Real-Time Image Patch & Planogram Stockout Detector (NumPy Tensor Vision Engine)
"""
import numpy as np

def compute_iou(boxA, boxB) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    return float(inter / (areaA + areaB - inter + 1e-8))

def analyze_uploaded_shelf_patches(patches: list, conf_threshold: float = 0.55, gap_threshold_px: float = 45.0):
    W1 = np.array([
        [ 1.65, -1.20,  0.85, -1.40],
        [ 0.90, -0.75,  1.10, -0.60],
        [ 1.50, -1.35,  0.95, -1.10],
        [-1.70,  1.80, -1.25,  1.55],
        [ 1.20, -0.95,  0.80, -0.90],
        [ 0.70, -0.50,  0.65, -0.45],
        [ 1.35, -1.10,  1.05, -0.85],
        [ 1.45, -1.25,  0.90, -1.15]
    ])
    b1 = np.array([-0.35, 0.25, -0.20, 0.30])
    W2 = np.array([[1.85], [-1.95], [1.40], [-1.60]])
    b2 = np.array([-0.15])

    sku_boxes, stockout_boxes = [], []
    for p in patches:
        feats = np.array(p["features"], dtype=float)
        z1 = feats @ W1 + b1
        a1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -30, 30)))
        z2 = float((a1 @ W2 + b2)[0])
        sku_prob = float(1.0 / (1.0 + np.exp(-np.clip(z2, -30, 30))))
        stockout_prob = 1.0 - sku_prob

        if sku_prob >= conf_threshold:
            sku_boxes.append({
                "row": p["row"],
                "box": [p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]],
                "confidence": round(sku_prob, 3),
                "label": p.get("sku_hint", "Product SKU")
            })
        elif stockout_prob >= conf_threshold and p["w"] >= gap_threshold_px * 0.5:
            stockout_boxes.append({
                "row": p["row"],
                "box": [p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]],
                "confidence": round(stockout_prob, 3),
                "severity": "CRITICAL" if stockout_prob > 0.78 else "MODERATE"
            })
    return {"sku_detections": sku_boxes, "stockout_gaps": stockout_boxes}

def detect_shelf_stockouts(detected_boxes, shelf_rows, gap_threshold_px=55.0):
    stockout_zones = []
    for row_idx, row in enumerate(shelf_rows):
        ry1, ry2 = row["y_min"], row["y_max"]
        row_items = sorted(
            [b for b in detected_boxes if 0.5 * (b["box"][1] + b["box"][3]) >= ry1 and 0.5 * (b["box"][1] + b["box"][3]) <= ry2],
            key=lambda item: item["box"][0]
        )
        cursor_x = row["x_min"]
        for item in row_items:
            gap = item["box"][0] - cursor_x
            if gap >= gap_threshold_px:
                stockout_zones.append({
                    "row": row_idx + 1, "box": [round(cursor_x, 1), ry1, round(item["box"][0], 1), ry2],
                    "gap_width_px": round(gap, 1), "severity": "CRITICAL" if gap > 110 else "MODERATE"
                })
            cursor_x = max(cursor_x, item["box"][2])
        if row["x_max"] - cursor_x >= gap_threshold_px:
            gap = row["x_max"] - cursor_x
            stockout_zones.append({
                "row": row_idx + 1, "box": [round(cursor_x, 1), ry1, row["x_max"], ry2],
                "gap_width_px": round(gap, 1), "severity": "CRITICAL" if gap > 110 else "MODERATE"
            })
    return stockout_zones
