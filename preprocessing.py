"""
preprocessing.py — Underwater Image Preprocessing for DeepBlue AI
==================================================================
Drop this file in the same folder as app.py

Techniques applied (in order):
    1. CLAHE        — contrast boost in murky/low-light footage
    2. White Balance — removes colour cast (blue/green tint)
    3. Sharpening   — recovers edge detail lost by scattering
    4. Denoising    — reduces sensor / compression noise
    5. Gamma Corr.  — brightens dark regions non-linearly

Each step can be toggled on/off via keyword arguments.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. CLAHE  (already in app.py — kept here for the unified pipeline)
# ──────────────────────────────────────────────────────────────────────────────
def apply_clahe(image_rgb: np.ndarray,
                clip_limit: float = 3.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation.
    Works in LAB colour space so only luminance is stretched.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


# ──────────────────────────────────────────────────────────────────────────────
# 2. White Balance  (Grey-World Assumption)
# ──────────────────────────────────────────────────────────────────────────────
def apply_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """
    Corrects the blue/green colour cast common in underwater images.
    Uses the Grey-World assumption: average of each channel → 128.
    """
    result = image_rgb.astype(np.float32)
    mean_r = np.mean(result[:, :, 0])
    mean_g = np.mean(result[:, :, 1])
    mean_b = np.mean(result[:, :, 2])

    gray_world = (mean_r + mean_g + mean_b) / 3.0

    # Scale each channel so its mean matches the grey-world average
    result[:, :, 0] = np.clip(result[:, :, 0] * (gray_world / (mean_r + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (gray_world / (mean_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (gray_world / (mean_b + 1e-6)), 0, 255)

    return result.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Sharpening  (Unsharp Mask)
# ──────────────────────────────────────────────────────────────────────────────
def apply_sharpening(image_rgb: np.ndarray,
                     strength: float = 1.2) -> np.ndarray:
    """
    Unsharp mask: sharpened = original + strength * (original − blurred).
    Recovers edge detail lost through water scattering.
    """
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image_rgb, 1.0 + strength,
                                blurred,   -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Denoising  (Non-Local Means)
# ──────────────────────────────────────────────────────────────────────────────
def apply_denoising(image_rgb: np.ndarray,
                    h: int = 7) -> np.ndarray:
    """
    Non-Local Means denoising.
    h controls filter strength (higher → smoother but may blur detail).
    Typical range: 5–15.
    """
    return cv2.fastNlMeansDenoisingColored(image_rgb, None, h, h, 7, 21)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Gamma Correction
# ──────────────────────────────────────────────────────────────────────────────
def apply_gamma_correction(image_rgb: np.ndarray,
                            gamma: float = 1.4) -> np.ndarray:
    """
    Brightens dark underwater regions with a power-law transform.
    gamma < 1 → brighter; gamma > 1 → darker.
    Recommended: 1.2–1.6 for dim footage.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        (i / 255.0) ** inv_gamma * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(image_rgb, table)


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline — call this from app.py
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_underwater_image(
    image_rgb: np.ndarray,
    use_clahe:         bool  = True,
    use_white_balance: bool  = True,
    use_sharpening:    bool  = True,
    use_denoising:     bool  = False,   # off by default — slows inference
    use_gamma:         bool  = True,
    clahe_clip:        float = 3.0,
    sharpen_strength:  float = 1.2,
    denoise_h:         int   = 7,
    gamma:             float = 1.4,
) -> np.ndarray:
    """
    Full underwater preprocessing pipeline.

    Parameters
    ----------
    image_rgb        : H×W×3 NumPy array in RGB order
    use_clahe        : CLAHE contrast enhancement
    use_white_balance: Grey-world colour cast removal
    use_sharpening   : Unsharp-mask edge recovery
    use_denoising    : Non-local means noise reduction (slow!)
    use_gamma        : Gamma brightness correction
    clahe_clip       : CLAHE clip limit (2–5 typical)
    sharpen_strength : Unsharp mask strength (0.5–2.0)
    denoise_h        : NLM filter strength (5–15)
    gamma            : Gamma value for correction (1.0 = no change)

    Returns
    -------
    Preprocessed RGB image as np.ndarray uint8
    """
    img = image_rgb.copy()

    if use_white_balance:
        img = apply_white_balance(img)

    if use_gamma:
        img = apply_gamma_correction(img, gamma=gamma)

    if use_clahe:
        img = apply_clahe(img, clip_limit=clahe_clip)

    if use_denoising:
        img = apply_denoising(img, h=denoise_h)

    if use_sharpening:
        img = apply_sharpening(img, strength=sharpen_strength)

    return img