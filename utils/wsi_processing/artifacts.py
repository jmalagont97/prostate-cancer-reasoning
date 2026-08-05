"""
Histopathological Artifact Filtering Module.

This module provides functions to identify and filter out non-tissue artifacts
(e.g., ink marks/pen strokes) and apply morphological operations to clean H&E tissue masks,
as well as tile-level image quality filters (blur and low-contrast checks).
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from skimage import color, exposure, filters, morphology
import large_image
from wsi_processing.utils import get_wsi_metadata

# RGB thresholds for pen markers commonly used by pathologists on slides
PENS_RGB: Dict[str, List[Tuple[int, int, int]]] = {
    "red": [
        (120, 80, 90), (110, 20, 30), (185, 65, 105), 
        (195, 85, 125), (220, 115, 145), (125, 40, 70), 
        (200, 120, 150), (100, 50, 65), (85, 25, 45)
    ],
    "green": [
        (150, 160, 140), (70, 110, 110), (45, 115, 100), 
        (30, 75, 60), (195, 220, 210), (225, 230, 225), 
        (170, 210, 200), (20, 30, 20), (50, 60, 40),
        (30, 50, 35), (65, 70, 60), (100, 110, 105), 
        (165, 180, 180), (140, 140, 150), (185, 195, 195)
    ],
    "blue": [
        (60, 120, 190), (120, 170, 200), (120, 170, 200), 
        (175, 210, 230), (145, 210, 210), (37, 95, 160), 
        (30, 65, 130), (130, 155, 180), (40, 35, 85),
        (30, 20, 65), (90, 90, 140), (60, 60, 120), 
        (110, 110, 175)
    ],
    "black": [
        (70, 70, 70)
    ]
}


def marker_detection(image: np.ndarray, pen_color: str) -> np.ndarray:
    """
    Detects strokes of a specific pen marker color on a slide thumbnail.

    Args:
        image (np.ndarray): Thumbnail image in RGB format (H, W, 3).
        pen_color (str): Target color name ('red', 'green', 'blue', 'black').

    Returns:
        np.ndarray: Boolean mask of shape (H, W) where True marks the detected pen strokes.
    """
    if pen_color not in PENS_RGB:
        raise ValueError(f"Unsupported pen color for detection: {pen_color}")

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    thresholds = PENS_RGB[pen_color]
    mask = np.zeros_like(r, dtype=bool)

    if pen_color == "red":
        for t in thresholds:
            mask |= (r > t[0]) & (g < t[1]) & (b < t[2])
    elif pen_color == "green":
        for t in thresholds:
            mask |= (r < t[0]) & (g > t[1]) & (b > t[2])
    elif pen_color == "blue":
        for t in thresholds:
            mask |= (r < t[0]) & (g < t[1]) & (b > t[2])
    elif pen_color == "black":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b < t[2])

    return mask


def filter_tissue_artifacts(
    raw_tissue_mask: np.ndarray,
    thumb_rgb: np.ndarray,
    disk_size_opening: int = 1,
    disk_size_erosion: int = 1,
    filter_markers: bool = True,
    colors_to_filter: List[str] = ["red", "green", "blue", "black"]
) -> np.ndarray:
    """
    Cleans a raw tissue mask by excluding pen markings and applying morphological filters.

    Args:
        raw_tissue_mask (np.ndarray): The raw boolean tissue mask (H, W).
        thumb_rgb (np.ndarray): The slide thumbnail in RGB format (H, W, 3).
        disk_size_opening (int, optional): Radius of the disk structuring element
            for binary opening to remove small noise objects. Defaults to 1.
        disk_size_erosion (int, optional): Radius of the disk structuring element
            for binary erosion to smooth out boundary fragments. Defaults to 1.
        filter_markers (bool, optional): Whether to detect and exclude pen strokes. Defaults to True.
        colors_to_filter (List[str], optional): Colors to check for pen marks.
            Defaults to ["red", "green", "blue", "black"].

    Returns:
        np.ndarray: The finalized clean tissue mask (H, W).
    """
    # 1. Handle pen marker filtering if enabled
    if filter_markers:
        # Exclude alpha channel if present
        if thumb_rgb.ndim == 3 and thumb_rgb.shape[-1] == 4:
            thumb_rgb = thumb_rgb[:, :, :3]

        # Accumulate pen marker strokes from all selected colors
        pen_mask = np.zeros(raw_tissue_mask.shape[:2], dtype=bool)
        for color_name in colors_to_filter:
            pen_mask |= marker_detection(thumb_rgb, color_name)

        # Exclude markers from the raw tissue mask
        clean_mask = raw_tissue_mask & (~pen_mask)
    else:
        clean_mask = raw_tissue_mask.copy()

    # 2. Apply morphological operations for cleaning and smoothing boundaries
    if disk_size_opening > 0:
        clean_mask = morphology.opening(
            clean_mask, 
            morphology.disk(disk_size_opening)
        )
        
    if disk_size_erosion > 0:
        clean_mask = morphology.erosion(
            clean_mask, 
            morphology.disk(disk_size_erosion)
        )


    return clean_mask


def artifact_filtering(
    tile_source: large_image.tilesource.FileTileSource,
    raw_tissue_mask: np.ndarray,
    rx: float,
    ry: float,
    patch_width: int = 256,
    patch_height: int = 256,
    target_magnification: float = 20.0,
    overlap: float = 0.0,
    tissue_threshold: float = 0.8,
    sharpness_threshold: float = 0.0005,
    contrast_threshold: float = 0.05
) -> np.ndarray:
    """
    Filters out blurred or low-contrast patches and maps valid tiles back to the low-res mask.

    This function iterates through grid coords of the WSI at a target magnification,
    verifies tissue coverage against the raw mask, reads candidate tiles dynamically into RAM,
    evaluates micro-level quality parameters (sharpness and contrast), and outputs a finalized
    low-resolution QC mask where valid tile regions are flagged as True (1).

    Args:
        tile_source (large_image.tilesource.FileTileSource): Open WSI source.
        raw_tissue_mask (np.ndarray): Low-resolution binary tissue mask (H_mask, W_mask).
        rx (float): Scale factor in X axis (width_mask / width_native).
        ry (float): Scale factor in Y axis (height_mask / height_native).
        patch_width (int, optional): Patch width in pixels at target magnification. Defaults to 256.
        patch_height (int, optional): Patch height in pixels at target magnification. Defaults to 256.
        target_magnification (float, optional): Optical magnification for patching. Defaults to 20.0.
        overlap (float, optional): Overlap fraction between adjacent tiles [0.0, 1.0). Defaults to 0.0.
        tissue_threshold (float, optional): Minimum required tissue ratio in mask [0.0, 1.0]. Defaults to 0.8.
        sharpness_threshold (float, optional): Minimum variance of the Laplacian. Defaults to 0.0005.
        contrast_threshold (float, optional): Low contrast tolerance parameter. Defaults to 0.05.

    Returns:
        np.ndarray: Refined QC mask of identical shape to raw_tissue_mask.
    """
    # 1. Fetch native metadata and compute downsample scaling factor
    meta = get_wsi_metadata(tile_source)
    native_w = meta["width"]
    native_h = meta["height"]
    native_mag = meta["magnification"]

    if not native_mag:
        raise ValueError("Could not determine WSI native magnification.")

    scale_factor = native_mag / target_magnification

    # 2. Compute patch size and strides at native resolution (Level 0)
    w_native = int(patch_width * scale_factor)
    h_native = int(patch_height * scale_factor)
    
    stride_x = int(w_native * (1.0 - overlap))
    stride_y = int(h_native * (1.0 - overlap))

    # 3. Instantiate empty output quality mask
    quality_mask = np.zeros_like(raw_tissue_mask, dtype=bool)

    # 4. Iterate over spatial grid at native resolution
    for y in range(0, native_h - h_native, stride_y):
        for x in range(0, native_w - w_native, stride_x):
            
            # Map native coordinate bounding box to the low-resolution mask space
            x_start = int(x * rx)
            y_start = int(y * ry)
            x_end = int((x + w_native) * rx)
            y_end = int((y + h_native) * ry)

            # Extract corresponding sub-mask patch
            mask_patch = raw_tissue_mask[y_start:y_end, x_start:x_end]
            if mask_patch.size == 0:
                continue

            # Check tissue ratio (I/O optimization)
            tissue_ratio = np.sum(mask_patch) / mask_patch.size
            if tissue_ratio < tissue_threshold:
                continue

            # Read patch dynamically in RAM at the requested magnification
            try:
                region, _ = tile_source.getRegion(
                    region=dict(left=x, top=y, width=w_native, height=h_native),
                    scale=dict(magnification=target_magnification),
                    format=large_image.constants.TILE_FORMAT_NUMPY
                )
            except Exception as e:
                # Handle occasional reading errors gracefully
                continue

            # Exclude alpha channel if present
            if region.ndim == 3 and region.shape[-1] == 4:
                region = region[:, :, :3]

            if region.shape != (patch_height, patch_width, 3):
                h_act, w_act = region.shape[:2]
                if abs(h_act - patch_height) <= 2 and abs(w_act - patch_width) <= 2:
                    from PIL import Image
                    pil_img = Image.fromarray(region)
                    pil_img = pil_img.resize((patch_width, patch_height), resample=Image.BILINEAR)
                    region = np.array(pil_img)
                else:
                    continue

            # 5. Evaluate micro-level quality (contrast and blur) in memory
            if exposure.is_low_contrast(region, fraction_threshold=contrast_threshold):
                continue

            gray_patch = color.rgb2gray(region)
            sharpness = filters.laplace(gray_patch).var()
            
            if sharpness < sharpness_threshold:
                continue

            # 6. Flag the corresponding block in the output mask
            quality_mask[y_start:y_end, x_start:x_end] = True

    return quality_mask


def extract_valid_patches(
    tile_source: large_image.tilesource.FileTileSource,
    raw_tissue_mask: np.ndarray,
    rx: float,
    ry: float,
    slide_id: str,
    patch_width: int = 256,
    patch_height: int = 256,
    target_magnification: float = 20.0,
    overlap: float = 0.0,
    tissue_threshold: float = 0.8,
    sharpness_threshold: float = 0.0005,
    contrast_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Filters out blurred or low-contrast patches, returns the valid patch pixels,
    their native coordinates [x, y], slide ID association, and flags them on the QC mask.

    Args:
        tile_source (large_image.tilesource.FileTileSource): Open WSI source.
        raw_tissue_mask (np.ndarray): Low-resolution binary tissue mask (H_mask, W_mask).
        rx (float): Scale factor in X axis (width_mask / width_native).
        ry (float): Scale factor in Y axis (height_mask / height_native).
        slide_id (str): The filename/identifier of the slide.
        patch_width (int, optional): Patch width in pixels at target magnification. Defaults to 256.
        patch_height (int, optional): Patch height in pixels at target magnification. Defaults to 256.
        target_magnification (float, optional): Optical magnification for patching. Defaults to 20.0.
        overlap (float, optional): Overlap fraction between adjacent tiles [0.0, 1.0). Defaults to 0.0.
        tissue_threshold (float, optional): Minimum required tissue ratio in mask [0.0, 1.0]. Defaults to 0.8.
        sharpness_threshold (float, optional): Minimum variance of the Laplacian. Defaults to 0.0005.
        contrast_threshold (float, optional): Low contrast tolerance parameter. Defaults to 0.05.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
            - coordinates: Array of shape (P, 2) [x, y] at native resolution (int32).
            - patches: Array of shape (P, patch_height, patch_width, 3) of uint8.
            - slide_ids: List of slide filenames of length P.
            - quality_mask: Refined QC mask of identical shape to raw_tissue_mask.
    """
    # 1. Fetch native metadata and compute downsample scaling factor
    meta = get_wsi_metadata(tile_source)
    native_w = meta["width"]
    native_h = meta["height"]
    native_mag = meta["magnification"]

    if not native_mag:
        raise ValueError("Could not determine WSI native magnification.")

    scale_factor = native_mag / target_magnification

    # 2. Compute patch size and strides at native resolution (Level 0)
    w_native = int(patch_width * scale_factor)
    h_native = int(patch_height * scale_factor)
    
    stride_x = int(w_native * (1.0 - overlap))
    stride_y = int(h_native * (1.0 - overlap))

    # 3. Instantiate empty output quality mask and collections
    quality_mask = np.zeros_like(raw_tissue_mask, dtype=bool)
    coords_list = []
    patches_list = []
    slide_ids_list = []

    # 4. Iterate over spatial grid at native resolution
    x_coords = list(range(0, native_w - w_native, stride_x))
    y_coords = list(range(0, native_h - h_native, stride_y))
    grid_coords = [(x, y) for y in y_coords for x in x_coords]

    from tqdm import tqdm
    import os
    tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
    for x, y in tqdm(grid_coords, desc="Fine QC Filtering", leave=False, position=tqdm_pos):
        
        # Map native coordinate bounding box to the low-resolution mask space
        x_start = int(x * rx)
        y_start = int(y * ry)
        x_end = int((x + w_native) * rx)
        y_end = int((y + h_native) * ry)

        # Extract corresponding sub-mask patch
        mask_patch = raw_tissue_mask[y_start:y_end, x_start:x_end]
        if mask_patch.size == 0:
            continue

        # Check tissue ratio (I/O optimization)
        tissue_ratio = np.sum(mask_patch) / mask_patch.size
        if tissue_ratio < tissue_threshold:
            continue

        # Read patch dynamically in RAM at the requested magnification
        try:
            region, _ = tile_source.getRegion(
                region=dict(left=x, top=y, width=w_native, height=h_native),
                scale=dict(magnification=target_magnification),
                format=large_image.constants.TILE_FORMAT_NUMPY
            )
        except Exception as e:
            # Handle occasional reading errors gracefully
            continue

        # Exclude alpha channel if present
        if region.ndim == 3 and region.shape[-1] == 4:
            region = region[:, :, :3]

        if region.shape != (patch_height, patch_width, 3):
            h_act, w_act = region.shape[:2]
            if abs(h_act - patch_height) <= 2 and abs(w_act - patch_width) <= 2:
                from PIL import Image
                pil_img = Image.fromarray(region)
                pil_img = pil_img.resize((patch_width, patch_height), resample=Image.BILINEAR)
                region = np.array(pil_img)
            else:
                continue

        # 5. Evaluate micro-level quality (contrast and blur) in memory
        if exposure.is_low_contrast(region, fraction_threshold=contrast_threshold):
            continue

        gray_patch = color.rgb2gray(region)
        sharpness = filters.laplace(gray_patch).var()
        
        if sharpness < sharpness_threshold:
            continue

        # 6. Store valid patch info
        coords_list.append([x, y])
        patches_list.append(region)
        slide_ids_list.append(slide_id)
        
        # Flag the corresponding block in the output mask
        quality_mask[y_start:y_end, x_start:x_end] = True

    # 7. Convert collections to arrays
    if len(coords_list) > 0:
        coordinates = np.array(coords_list, dtype=np.int32)
        patches = np.array(patches_list, dtype=np.uint8)
    else:
        coordinates = np.empty((0, 2), dtype=np.int32)
        patches = np.empty((0, patch_height, patch_width, 3), dtype=np.uint8)

    return coordinates, patches, slide_ids_list, quality_mask

