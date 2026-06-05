"""
image_validation.py — AgroVision Pro
Image Quality Validation Engine
Checks: blur, brightness, contrast, resolution
"""

import cv2
import numpy as np
from PIL import Image
import io


# ─── Thresholds ────────────────────────────────────────────────────────────────
BLUR_THRESHOLD       = 80.0   # Laplacian variance; below → too blurry
BRIGHTNESS_LOW       = 40.0   # 0–255; below → underexposed
BRIGHTNESS_HIGH      = 220.0  # 0–255; above → overexposed
CONTRAST_THRESHOLD   = 30.0   # std-dev; below → low contrast
MIN_RESOLUTION       = 64     # px; both dims must exceed this


def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def check_blur(gray: np.ndarray) -> tuple[float, bool, str]:
    """
    Compute Laplacian variance as blur score.
    Returns (score, is_ok, message)
    """
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ok = score >= BLUR_THRESHOLD
    msg = "" if ok else f"❌ Image too blurry (score {score:.1f}). Please retake the photo."
    return score, ok, msg


def check_brightness(gray: np.ndarray) -> tuple[float, bool, str]:
    """
    Compute mean pixel intensity.
    Returns (mean_brightness, is_ok, message)
    """
    mean_val = float(np.mean(gray))
    if mean_val < BRIGHTNESS_LOW:
        return mean_val, False, f"❌ Image too dark (brightness {mean_val:.1f}). Use better lighting."
    if mean_val > BRIGHTNESS_HIGH:
        return mean_val, False, f"❌ Image too bright/overexposed (brightness {mean_val:.1f}). Reduce direct light."
    return mean_val, True, ""


def check_contrast(gray: np.ndarray) -> tuple[float, bool, str]:
    """
    Compute pixel std-dev as contrast proxy.
    Returns (std_dev, is_ok, message)
    """
    std_val = float(np.std(gray))
    ok = std_val >= CONTRAST_THRESHOLD
    msg = "" if ok else f"❌ Low contrast image (std {std_val:.1f}). Ensure the leaf is clearly visible."
    return std_val, ok, msg


def check_resolution(img: np.ndarray) -> tuple[tuple, bool, str]:
    """
    Verify minimum resolution.
    Returns ((h,w), is_ok, message)
    """
    h, w = img.shape[:2]
    ok = h >= MIN_RESOLUTION and w >= MIN_RESOLUTION
    msg = "" if ok else f"❌ Resolution too low ({w}×{h}px). Use a higher-quality image."
    return (h, w), ok, msg


def compute_quality_score(blur: float, brightness: float, contrast: float) -> float:
    """
    Normalised composite quality score 0–100.
    Weights: blur 50%, brightness 25%, contrast 25%
    """
    blur_norm  = min(blur / 300.0, 1.0) * 100
    bri_norm   = (1.0 - abs(brightness - 128) / 128.0) * 100
    cont_norm  = min(contrast / 80.0, 1.0) * 100
    score = 0.5 * blur_norm + 0.25 * bri_norm + 0.25 * cont_norm
    return round(score, 1)


def validate_image(file_like) -> dict:
    """
    Full image validation pipeline.

    Parameters
    ----------
    file_like : file-like object or PIL.Image

    Returns
    -------
    dict with keys:
        valid       : bool
        quality     : float (0–100)
        resolution  : tuple (h, w)
        blur_score  : float
        brightness  : float
        contrast    : float
        errors      : list[str]
        warnings    : list[str]
    """
    errors, warnings = [], []

    # ── Load image ──────────────────────────────────────────────────────────
    try:
        if isinstance(file_like, Image.Image):
            pil_img = file_like.convert("RGB")
        else:
            if hasattr(file_like, "seek"):
                file_like.seek(0)
            pil_img = Image.open(file_like).convert("RGB")
    except Exception as exc:
        return dict(valid=False, quality=0, resolution=(0, 0),
                    blur_score=0, brightness=0, contrast=0,
                    errors=[f"❌ Could not read image: {exc}"], warnings=[])

    cv_img = pil_to_cv(pil_img)
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # ── Run checks ──────────────────────────────────────────────────────────
    (h, w), res_ok,  res_msg  = check_resolution(cv_img)
    blur_score, blur_ok,  blur_msg  = check_blur(gray)
    brightness, bri_ok,  bri_msg   = check_brightness(gray)
    contrast,   cont_ok, cont_msg  = check_contrast(gray)

    for ok, msg in [(res_ok, res_msg), (blur_ok, blur_msg),
                    (bri_ok,  bri_msg),  (cont_ok, cont_msg)]:
        if not ok:
            errors.append(msg)

    # ── Soft warnings (doesn't block) ───────────────────────────────────────
    if blur_score < 150 and blur_ok:
        warnings.append("⚠ Slightly blurry — results may be less accurate.")
    if brightness < 60 and bri_ok:
        warnings.append("⚠ Slightly dark image.")

    quality = compute_quality_score(blur_score, brightness, contrast)
    valid   = len(errors) == 0

    return dict(
        valid      = valid,
        quality    = quality,
        resolution = (h, w),
        blur_score = round(blur_score, 2),
        brightness = round(brightness, 2),
        contrast   = round(contrast,   2),
        errors     = errors,
        warnings   = warnings,
    )
