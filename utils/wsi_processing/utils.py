"""
WSI Reading and Scaling Utilities Module.

This module encapsulates the physical image loader (large_image) to abstract
medical image Input/Output (I/O) dependencies from the downstream data processing.
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Any
import numpy as np
import large_image


def open_wsi(image_path: Union[str, Path]) -> large_image.tilesource.FileTileSource:
    """
    Opens a Whole Slide Image (WSI) file safely.

    Args:
        image_path (Union[str, Path]): Path to the WSI file (e.g. .svs, .tiff).

    Returns:
        large_image.tilesource.FileTileSource: Tile source object.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        RuntimeError: If large_image fails to parse the image format.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"WSI file not found at: {image_path}")

    try:
        tile_source = large_image.getTileSource(str(path_obj))
        return tile_source
    except Exception as e:
        raise RuntimeError(f"Error opening WSI with large_image at {image_path}: {e}")


def get_wsi_metadata(tile_source: large_image.tilesource.FileTileSource) -> Dict[str, Any]:
    """
    Extracts standardized metadata from an open WSI.

    Args:
        tile_source (large_image.tilesource.FileTileSource): Open WSI object.

    Returns:
        Dict[str, Any]: Dictionary containing key metadata:
            - "width" (int): Native width in pixels.
            - "height" (int): Native height in pixels.
            - "magnification" (float): Nominal objective power (e.g. 20.0 or 40.0).
            - "mpp_x" (float): Horizontal pixel spacing in micrometers.
            - "mpp_y" (float): Vertical pixel spacing in micrometers.
    """
    metadata = tile_source.getMetadata()
    
    # Extract native dimensions
    width = metadata.get("sizeX", 0)
    height = metadata.get("sizeY", 0)
    
    # Extract nominal magnification if available
    magnification = metadata.get("magnification", None)
    
    # Extract micrometers per pixel (MPP)
    raw_mm_x = metadata.get("mm_x", None)
    mpp_x = raw_mm_x * 1000.0 if raw_mm_x is not None else None
    
    raw_mm_y = metadata.get("mm_y", None)
    mpp_y = raw_mm_y * 1000.0 if raw_mm_y is not None else None

    # Fallback to internal metadata properties if mm_x/mm_y are missing
    if mpp_x is None or mpp_y is None:
        try:
            internal_meta = tile_source.getInternalMetadata()
            openslide_meta = internal_meta.get("openslide", {})
            
            # 1. Try to extract standard openslide MPP properties if available (format-agnostic)
            mpp_x_val = openslide_meta.get("openslide.mpp-x", None)
            mpp_y_val = openslide_meta.get("openslide.mpp-y", None)
            if mpp_x_val is not None:
                mpp_x = float(mpp_x_val)
            if mpp_y_val is not None:
                mpp_y = float(mpp_y_val)
                
            # 2. Reconstruct from generic TIFF resolution tags if still missing
            if mpp_x is None or mpp_y is None:
                res_unit = openslide_meta.get("tiff.ResolutionUnit", "").lower()
                x_res = float(openslide_meta.get("tiff.XResolution", 0))
                y_res = float(openslide_meta.get("tiff.YResolution", 0))
                
                if x_res > 0 and y_res > 0:
                    if "centimeter" in res_unit:
                        mpp_x = 10000.0 / x_res
                        mpp_y = 10000.0 / y_res
                    elif "inch" in res_unit:
                        mpp_x = 25400.0 / x_res
                        mpp_y = 25400.0 / y_res
        except Exception as e:
            print(f"Warning: Failed to parse internal metadata properties: {e}")

    # Attempt to estimate magnification from MPP if missing in metadata
    if magnification is None and mpp_x is not None:
        # 0.25 MPP approx = 40X, 0.50 MPP approx = 20X, 1.0 MPP approx = 10X
        if 0.20 <= mpp_x <= 0.30:
            magnification = 40.0
        elif 0.40 <= mpp_x <= 0.60:
            magnification = 20.0
        elif 0.80 <= mpp_x <= 1.20:
            magnification = 10.0

    # Handle cases where both magnification and MPP are completely missing
    if magnification is None:
        # Try to retrieve from openslide internal objective-power property
        try:
            internal_meta = tile_source.getInternalMetadata()
            openslide_meta = internal_meta.get("openslide", {})
            obj_power = openslide_meta.get("openslide.objective-power", None)
            if obj_power is not None:
                magnification = float(obj_power)
        except Exception:
            pass

        # Fallback to standard 20X if still not determined
        if magnification is None:
            magnification = 20.0
            print("WARNING: Magnification missing in WSI metadata. Assuming default 20.0X.")

    # Re-estimate MPP if missing
    if mpp_x is None:
        mpp_x = 0.50 if magnification == 20.0 else 0.25
        print(f"WARNING: mm_x missing in WSI metadata. Assuming {mpp_x} MPP based on {magnification}X.")
    if mpp_y is None:
        mpp_y = 0.50 if magnification == 20.0 else 0.25
        print(f"WARNING: mm_y missing in WSI metadata. Assuming {mpp_y} MPP based on {magnification}X.")

    return {
        "width": width,
        "height": height,
        "magnification": magnification,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y
    }



def get_wsi_thumbnail(
    tile_source: large_image.tilesource.FileTileSource, 
    target_size: int = 2048
) -> Tuple[np.ndarray, float, float]:
    """
    Extracts WSI RGB thumbnail and computes the geometric scale factors.

    The thumbnail size is constrained such that its largest dimension fits
    within a target_size x target_size bounding box, preserving the aspect ratio.

    Args:
        tile_source (large_image.tilesource.FileTileSource): Open WSI object.
        target_size (int, optional): Maximum dimension allowed for the largest
            side of the thumbnail. Defaults to 2048.

    Returns:
        Tuple[np.ndarray, float, float]: A tuple containing:
            - np.ndarray: Thumbnail image in RGB format (H_thumb, W_thumb, 3).
            - float: Scale factor in the X axis (rx = width_thumbnail / width_native).
            - float: Scale factor in the Y axis (ry = height_thumbnail / height_native).
    """
    # 1. Fetch thumbnail as a numpy array
    thumb_data, _ = tile_source.getThumbnail(
        width=target_size, 
        height=target_size, 
        format=large_image.constants.TILE_FORMAT_NUMPY
    )
    
    # 2. Exclude alpha channel if present (convert RGBA to RGB)
    if thumb_data.ndim == 3 and thumb_data.shape[-1] == 4:
        thumb_rgb = thumb_data[:, :, :3]
    else:
        thumb_rgb = thumb_data

    # 3. Fetch native and thumbnail dimensions to compute scale factors
    native_meta = get_wsi_metadata(tile_source)
    native_w = native_meta["width"]
    native_h = native_meta["height"]

    thumb_h, thumb_w = thumb_rgb.shape[:2]

    # Calculate reduction scale factors
    rx = thumb_w / native_w if native_w > 0 else 0.0
    ry = thumb_h / native_h if native_h > 0 else 0.0

    return thumb_rgb, rx, ry
