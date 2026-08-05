"""
Histopathological Tissue Segmentation Module for H&E.

This module provides functions dedicated exclusively to adaptively isolating
histological tissue from the bright glass background in low-resolution slide images.
"""

import numpy as np
from skimage import color, filters


def get_raw_tissue_mask(
    thumb_rgb: np.ndarray, 
    sigma: float = 1.0, 
    threshold_method: str = "triangle"
) -> np.ndarray:
    """
    Generates a raw binary mask of histological tissue from an RGB thumbnail.

    This function assumes that the slide background is bright (white) and that
    tissue regions possess darker intensity values due to H&E staining.

    Args:
        thumb_rgb (np.ndarray): WSI thumbnail in RGB format (H, W, 3).
        sigma (float, optional): Standard deviation for Gaussian smoothing.
            Helps reduce high-frequency noise and artificial granularity. Defaults to 1.0.
        threshold_method (str, optional): Adaptive thresholding method to use.
            Supports "triangle" and "otsu". Defaults to "triangle".

    Returns:
        np.ndarray: Boolean mask of shape (H, W) where True represents the
            preliminary presence of histological tissue.
    """
    # 1. Ensure 3-channel RGB format (exclude alpha channel if present)
    if thumb_rgb.ndim == 3 and thumb_rgb.shape[-1] == 4:
        thumb_rgb = thumb_rgb[:, :, :3]

    # 2. Grayscale conversion and intensity inversion (complement)
    # Inverting makes tissue (dark) bright and background (white) dark.
    # This helps thresholding algorithms detect the tissue as the foreground.
    gray_image = color.rgb2gray(thumb_rgb)
    gray_smoothed = filters.gaussian(gray_image, sigma=sigma)
    complement_image = 1.0 - gray_smoothed

    # 3. Apply adaptive thresholding
    method_lower = threshold_method.lower()
    if method_lower == "triangle":
        threshold = filters.threshold_triangle(complement_image)
    elif method_lower == "otsu":
        threshold = filters.threshold_otsu(complement_image)
    else:
        raise ValueError(f"Unsupported thresholding method: {threshold_method}")

    # 4. Generate binary mask
    raw_tissue_mask = complement_image > threshold

    return raw_tissue_mask
