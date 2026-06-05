"""
utils.py — AgroVision Pro
Shared utilities: preprocessing, report export, severity helpers
"""

import io
import csv
import json
import time
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2


# ─── Advanced Preprocessing ───────────────────────────────────────────────────
def preprocess_image(source, target_size=(224, 224)):

    if isinstance(source, Image.Image):
        img = source.convert("RGB")
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source).convert("RGB")
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        img = Image.open(source).convert("RGB")

    img = img.resize(target_size)

    arr = np.array(img).astype(np.float32) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr


# ─── Severity Estimation ──────────────────────────────────────────────────────
def estimate_severity_from_mask(mask: np.ndarray, confidence: float) -> dict:
    """
    Estimate disease severity from infected area + confidence.

    Levels: LOW / MEDIUM / HIGH / CRITICAL

    Parameters
    ----------
    mask        : uint8 binary mask from leaf segmentation (255 = leaf)
    confidence  : model confidence 0–1

    Returns
    -------
    dict: level, infected_pct, description, color
    """
    total_leaf = np.sum(mask > 0)
    # Proxy: pixels that are "not green" within the leaf
    # Use confidence as a proxy for infected fraction when no separate mask
    infected_pct = round(confidence * 100 * 0.6, 1)   # heuristic

    if infected_pct >= 60 or confidence >= 0.90:
        level, desc, color = "CRITICAL", "Severe infection — immediate intervention required", "#ef4444"
    elif infected_pct >= 35 or confidence >= 0.75:
        level, desc, color = "HIGH",     "High infection — act within 1–2 days",              "#f97316"
    elif infected_pct >= 15 or confidence >= 0.50:
        level, desc, color = "MEDIUM",   "Moderate infection — monitor and treat promptly",   "#eab308"
    else:
        level, desc, color = "LOW",      "Mild infection — standard preventive care advised", "#22c55e"

    return dict(level=level, infected_pct=infected_pct,
                description=desc, color=color)


# ─── Report Generators ────────────────────────────────────────────────────────
def generate_txt_report(result: dict) -> str:
    """Plain-text report."""
    lines = [
        "=" * 55,
        "        AgroVision Pro — Diagnostic Report",
        "=" * 55,
        f"Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Disease    : {result.get('label', 'N/A')}",
        f"Confidence : {result.get('confidence', 0):.2%}",
        f"Severity   : {result.get('level', 'N/A')}",
        f"Temperature: {result.get('temp', 'N/A')}°C",
        f"Humidity   : {result.get('humidity', 'N/A')}%",
        "-" * 55,
        "ADVICE",
        "-" * 55,
        result.get("advice", "N/A"),
        "-" * 55,
        "FARM PLAN",
        "-" * 55,
        result.get("farm", "N/A"),
        "=" * 55,
    ]
    return "\n".join(lines)


def generate_csv_report(result: dict) -> str:
    """CSV report as a string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Field", "Value"])
    writer.writerow(["Timestamp",   time.strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Disease",     result.get("label",      "N/A")])
    writer.writerow(["Confidence",  f"{result.get('confidence', 0):.4f}"])
    writer.writerow(["Severity",    result.get("level",      "N/A")])
    writer.writerow(["Temperature", result.get("temp",       "N/A")])
    writer.writerow(["Humidity",    result.get("humidity",   "N/A")])
    writer.writerow(["Spray Days",  result.get("spray",      "N/A")])
    return output.getvalue()


def generate_json_report(result: dict) -> str:
    """JSON report, serialisable subset of result dict."""
    safe = {
        "timestamp"  : time.strftime("%Y-%m-%d %H:%M:%S"),
        "disease"    : result.get("label",      "N/A"),
        "confidence" : round(float(result.get("confidence", 0)), 4),
        "severity"   : result.get("level",      "N/A"),
        "temperature": result.get("temp",       "N/A"),
        "humidity"   : result.get("humidity",   "N/A"),
        "spray_days" : result.get("spray",      "N/A"),
        "advice"     : result.get("advice",     "N/A"),
    }
    return json.dumps(safe, ensure_ascii=False, indent=2)


# ─── Miscellaneous ────────────────────────────────────────────────────────────
def pil_to_bytes(pil_image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes for Streamlit display."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def top_k_predictions(output: np.ndarray,
                       class_names: list[str],
                       k: int = 5) -> list[dict]:
    """
    Return top-k predictions sorted by probability.

    Returns list of dicts: {rank, label, probability}
    """
    k        = min(k, len(output))
    top_idx  = np.argsort(output)[-k:][::-1]
    return [
        dict(rank=i + 1,
             label=class_names[idx] if idx < len(class_names) else f"Class {idx}",
             probability=round(float(output[idx]), 4))
        for i, idx in enumerate(top_idx)
    ]
