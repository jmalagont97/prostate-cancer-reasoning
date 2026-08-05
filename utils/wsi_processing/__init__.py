"""
wsi_processing package for WSI preprocessing and pipeline operations.

Modules:
    - segmentation: Raw tissue isolation routines.
    - utils: Safe file opening, metadata parsing, and scaling utilities.
"""

from wsi_processing.segmentation import (
    get_raw_tissue_mask,
)
from wsi_processing.artifacts import (
    marker_detection,
    filter_tissue_artifacts,
    artifact_filtering,
    extract_valid_patches,
)
from wsi_processing.utils import (
    open_wsi,
    get_wsi_metadata,
    get_wsi_thumbnail,
)
from wsi_processing.storage import (
    save_qc_report,
    save_case_data_to_h5,
    save_embeddings_to_h5,
)
from wsi_processing.models import (
    UNI2FeatureExtractor,
)

__all__ = [
    "get_raw_tissue_mask",
    "marker_detection",
    "filter_tissue_artifacts",
    "artifact_filtering",
    "extract_valid_patches",
    "open_wsi",
    "get_wsi_metadata",
    "get_wsi_thumbnail",
    "save_qc_report",
    "save_case_data_to_h5",
    "save_embeddings_to_h5",
    "UNI2FeatureExtractor",
]




