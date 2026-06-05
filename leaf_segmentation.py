"""
leaf_segmentation.py — AgroVision Pro
Leaf Coverage Analysis + Background Removal
Uses OpenCV HSV segmentation, GrabCut, contour analysis
"""

import cv2
import numpy as np
from PIL import Image


# ─── Coverage threshold ────────────────────────────────────────────────────────
MIN_COVERAGE_PCT = 25.0   # Below this → reject ("move closer")


# ─── HSV ranges for green/yellow/brown vegetation ─────────────────────────────
# Each entry: (lower_hsv, upper_hsv)
LEAF_HSV_RANGES = [
    # healthy green
    (np.array([25,  30,  30],  dtype=np.uint8), np.array([95,  255, 255], dtype=np.uint8)),
    # yellowish / diseased
    (np.array([15,  40,  40],  dtype=np.uint8), np.array([35,  255, 255], dtype=np.uint8)),
    # brownish / dry leaf
    (np.array([5,   20,  20],  dtype=np.uint8), np.array([25,  200, 200], dtype=np.uint8)),
]


def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ─── HSV leaf mask ─────────────────────────────────────────────────────────────
def build_leaf_mask_hsv(cv_img: np.ndarray) -> np.ndarray:
    """
    Build a binary mask of leaf pixels using HSV colour ranges.
    Returns uint8 mask (255 = leaf, 0 = background).
    """
    blurred = cv2.GaussianBlur(cv_img, (7, 7), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in LEAF_HSV_RANGES:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

    # Morphological clean-up
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel, iterations=1)
    return cleaned


# ─── GrabCut refinement ────────────────────────────────────────────────────────
def refine_with_grabcut(cv_img: np.ndarray, hint_mask: np.ndarray) -> np.ndarray:
    """
    Use GrabCut seeded from the HSV hint mask to produce a refined leaf mask.
    Falls back to hint_mask if GrabCut fails.
    """
    try:
        h, w = cv_img.shape[:2]
        gc_mask = np.where(hint_mask > 0,
                           cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

        # Definite background = image border ring
        border = 10
        gc_mask[:border,  :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:,  :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(cv_img, gc_mask, None, bgd_model, fgd_model,
                    iterCount=3, mode=cv2.GC_INIT_WITH_MASK)

        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        return refined
    except Exception:
        return hint_mask


# ─── Coverage analysis ─────────────────────────────────────────────────────────
def analyse_coverage(mask: np.ndarray) -> dict:
    """
    Compute leaf coverage percentage and largest contour info.
    Returns dict: coverage_pct, contour_count, largest_area_px, ok, message
    """
    total_px   = mask.size
    leaf_px    = int(np.sum(mask > 0))
    coverage   = round(100.0 * leaf_px / total_px, 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest     = max((cv2.contourArea(c) for c in contours), default=0)

    ok  = coverage >= MIN_COVERAGE_PCT
    msg = "" if ok else f"❌ Leaf coverage too low ({coverage}%). Move the camera closer to the leaf."

    return dict(
        coverage_pct    = coverage,
        contour_count   = len(contours),
        largest_area_px = int(largest),
        ok              = ok,
        message         = msg,
    )


# ─── Background removal ────────────────────────────────────────────────────────
def remove_background(pil_image: Image.Image,
                       use_grabcut: bool = True) -> tuple[Image.Image, np.ndarray]:
    """
    Remove background from a PIL image.

    Returns
    -------
    (cleaned_pil, mask)
        cleaned_pil : PIL Image with non-leaf pixels set to white
        mask        : uint8 numpy array (255 = leaf)
    """
    cv_img    = pil_to_cv(pil_image)
    hint_mask = build_leaf_mask_hsv(cv_img)

    if use_grabcut:
        mask = refine_with_grabcut(cv_img, hint_mask)
    else:
        mask = hint_mask

    # Apply mask: keep leaf, white-out background
    result = cv_img.copy()
    result[mask == 0] = [255, 255, 255]

    return cv_to_pil(result), mask


# ─── Full pipeline ─────────────────────────────────────────────────────────────
def segment_leaf(file_like, use_grabcut: bool = True) -> dict:
    """
    Complete leaf segmentation pipeline.

    Parameters
    ----------
    file_like : file-like, PIL.Image, or numpy array

    Returns
    -------
    dict:
        ok              : bool  (False → coverage too low)
        coverage_pct    : float
        contour_count   : int
        message         : str   (error if not ok)
        cleaned_image   : PIL.Image (background removed)
        mask            : np.ndarray
    """
    # Load
    if isinstance(file_like, Image.Image):
        pil_img = file_like.convert("RGB")
    elif isinstance(file_like, np.ndarray):
        pil_img = Image.fromarray(file_like).convert("RGB")
    else:
        if hasattr(file_like, "seek"):
            file_like.seek(0)
        pil_img = Image.open(file_like).convert("RGB")

    # Resize to manageable size for speed
    max_side = 512
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale    = max_side / max(w, h)
        pil_img  = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    cleaned_pil, mask = remove_background(pil_img, use_grabcut=use_grabcut)
    coverage_info     = analyse_coverage(mask)

    return dict(
        ok            = coverage_info["ok"],
        coverage_pct  = coverage_info["coverage_pct"],
        contour_count = coverage_info["contour_count"],
        message       = coverage_info["message"],
        cleaned_image = cleaned_pil,
        mask          = mask,
    )
