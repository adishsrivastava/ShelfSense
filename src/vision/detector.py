"""
Retail Shelf Vision Pipeline: IoU Computation & Planogram Horizontal Sweep Stockout Detector.
"""
import numpy as np

def compute_iou(boxA, boxB) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    return float(inter / (areaA + areaB - inter + 1e-8))

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