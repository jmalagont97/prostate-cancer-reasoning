"""
WSI Data Storage and QC Visualization Module.

This module provides routines to persist WSI quality control reports as images,
and to store extracted features, patches, and native coordinates in HDF5 format
consolidated at the case (patient) level.
"""

import os
from pathlib import Path
from typing import List, Union
import numpy as np
import h5py
import matplotlib.pyplot as plt


def save_qc_report(
    thumb_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    fine_mask: np.ndarray,
    output_dir: Union[str, Path],
    wsi_filename: str
) -> None:
    """
    Generates and saves a 4-panel visual report for slide QC.

    The filename is determined by stripping the extension from wsi_filename
    and appending '_qc.png'.

    Args:
        thumb_rgb (np.ndarray): Original WSI RGB thumbnail.
        coarse_mask (np.ndarray): Tissue mask after macro filtering.
        fine_mask (np.ndarray): Refined mask after micro quality filtering.
        output_dir (Union[str, Path]): Destination directory for the report.
        wsi_filename (str): Original filename of the WSI (e.g. 'case_name.svs').
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Strip WSI file extension (e.g. .svs, .ome.tiff)
    wsi_stem = Path(wsi_filename).stem
    if wsi_stem.endswith(".ome"):
        wsi_stem = Path(wsi_stem).stem  # Strip double extension if .ome.tiff
        
    report_filename = f"{wsi_stem}.png"
    report_path = out_dir_path / report_filename

    # Ensure 3-channel RGB for thumbnail overlay
    if thumb_rgb.ndim == 3 and thumb_rgb.shape[-1] == 4:
        thumb_rgb = thumb_rgb[:, :, :3]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Original image thumbnail
    axes[0].imshow(thumb_rgb)
    axes[0].set_title("1. Original Thumbnail")
    axes[0].axis('off')

    # Panel 2: Coarse tissue mask (after marker/pen removal)
    axes[1].imshow(coarse_mask, cmap='gray')
    axes[1].set_title("2. Coarse Tissue Mask")
    axes[1].axis('off')

    # Panel 3: Fine QC mask (after sharpness/contrast filtering)
    axes[2].imshow(fine_mask, cmap='gray')
    axes[2].set_title("3. Fine Quality Mask")
    axes[2].axis('off')

    # Panel 4: Green overlay showing only valid accepted patch regions
    overlay = thumb_rgb.copy()
    overlay[fine_mask == 1] = (
        overlay[fine_mask == 1] * 0.5 + 
        np.array([0, 255, 0], dtype=np.uint8) * 0.5
    ).astype(np.uint8)
    
    axes[4-1].imshow(overlay)
    axes[4-1].set_title("4. Accepted Patch Regions (Overlay)")
    axes[4-1].axis('off')

    plt.tight_layout()
    plt.savefig(str(report_path), dpi=150)
    plt.close(fig)


def save_case_data_to_h5(
    output_path: Union[str, Path],
    coordinates: np.ndarray,
    slide_ids: List[str],
    patches: np.ndarray = None,
    embeddings: np.ndarray = None
) -> None:
    """
    Stores or appends coordinates, slide IDs, and either raw patches or embeddings (or both)
    in a case-level HDF5 file.

    If the HDF5 file already exists (e.g. from an alternative slide of the same case),
    the datasets are resized and the new data is appended. Native compression is
    applied to patch images to optimize disk space usage.

    Args:
        output_path (Union[str, Path]): Target path of the case HDF5 file.
        coordinates (np.ndarray): Coordinate matrix (P, 2) of int32 [x, y].
        slide_ids (List[str]): List of slide filenames of length P.
        patches (np.ndarray, optional): Raw patch image arrays (P, patch_h, patch_w, 3) of uint8. Defaults to None.
        embeddings (np.ndarray, optional): Feature matrix (P, embed_dim) of float32. Defaults to None.
    """
    dest_path = Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_new_patches = coordinates.shape[0]
    if len(slide_ids) != num_new_patches:
        raise ValueError(f"Mismatch between number of slide_ids ({len(slide_ids)}) and coordinates ({num_new_patches})")
    if patches is not None and len(patches) != num_new_patches:
        raise ValueError(f"Mismatch between number of patches ({len(patches)}) and coordinates ({num_new_patches})")
    if embeddings is not None and embeddings.shape[0] != num_new_patches:
        raise ValueError(f"Mismatch between number of embeddings ({embeddings.shape[0]}) and coordinates ({num_new_patches})")

    # Convert slide IDs to bytes for HDF5 compatibility
    slide_ids_bytes = [sid.encode('utf-8') for sid in slide_ids]

    CHUNK_SIZE = 500
    from tqdm import tqdm
    import os
    tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))

    # Open HDF5 file in append/read-write mode ('a')
    with h5py.File(str(dest_path), 'a') as h5f:
        
        # Check if the file already contains dataset structures
        if "coords" in h5f:
            # 1. Update existing datasets (Append Mode)
            h5_coords = h5f["coords"]
            h5_slides = h5f["slide_ids"]

            current_size = h5_coords.shape[0]
            new_size = current_size + num_new_patches

            # Resize datasets
            h5_coords.resize(new_size, axis=0)
            h5_slides.resize(new_size, axis=0)

            # Write appended data
            h5_coords[current_size:new_size] = coordinates
            h5_slides[current_size:new_size] = slide_ids_bytes

            # Handle patches
            if patches is not None:
                if "patches" in h5f:
                    h5_patches = h5f["patches"]
                    h5_patches.resize(new_size, axis=0)
                else:
                    patch_h, patch_w, channels = patches.shape[1:]
                    h5_patches = h5f.create_dataset(
                        "patches",
                        shape=(new_size, patch_h, patch_w, channels),
                        maxshape=(None, patch_h, patch_w, channels),
                        dtype='uint8',
                        compression="gzip",
                        compression_opts=4
                    )
                # Chunked write
                for start_idx in tqdm(range(0, num_new_patches, CHUNK_SIZE), desc="Appending Patches to H5", unit="chunk", leave=False, position=tqdm_pos):
                    end_idx = min(start_idx + CHUNK_SIZE, num_new_patches)
                    h5_patches[current_size + start_idx : current_size + end_idx] = patches[start_idx:end_idx]
            else:
                if "patches" in h5f:
                    h5_patches = h5f["patches"]
                    h5_patches.resize(new_size, axis=0)

            # Handle embeddings
            if embeddings is not None:
                if "embeddings" in h5f:
                    h5_emb = h5f["embeddings"]
                    h5_emb.resize(new_size, axis=0)
                else:
                    embedding_dim = embeddings.shape[1]
                    h5_emb = h5f.create_dataset(
                        "embeddings",
                        shape=(new_size, embedding_dim),
                        maxshape=(None, embedding_dim),
                        dtype='float32'
                    )
                # Chunked write
                for start_idx in tqdm(range(0, num_new_patches, CHUNK_SIZE), desc="Appending Embeddings to H5", unit="chunk", leave=False, position=tqdm_pos):
                    end_idx = min(start_idx + CHUNK_SIZE, num_new_patches)
                    h5_emb[current_size + start_idx : current_size + end_idx] = embeddings[start_idx:end_idx]
            else:
                if "embeddings" in h5f:
                    h5_emb = h5f["embeddings"]
                    h5_emb.resize(new_size, axis=0)
            
        else:
            # 2. Initialize new datasets (Write Mode)
            h5_coords = h5f.create_dataset(
                "coords", 
                shape=(num_new_patches, 2), 
                maxshape=(None, 2), 
                dtype='int32'
            )
            h5_slides = h5f.create_dataset(
                "slide_ids", 
                shape=(num_new_patches,), 
                maxshape=(None,), 
                dtype=h5py.string_dtype()
            )

            # Write initial data
            h5_coords[:] = coordinates
            h5_slides[:] = slide_ids_bytes

            if patches is not None:
                patch_h, patch_w, channels = patches.shape[1:]
                h5_patches = h5f.create_dataset(
                    "patches", 
                    shape=(num_new_patches, patch_h, patch_w, channels), 
                    maxshape=(None, patch_h, patch_w, channels), 
                    dtype='uint8',
                    compression="gzip",
                    compression_opts=4
                )
                # Chunked write
                for start_idx in tqdm(range(0, num_new_patches, CHUNK_SIZE), desc="Writing Patches to H5", unit="chunk", leave=False, position=tqdm_pos):
                    end_idx = min(start_idx + CHUNK_SIZE, num_new_patches)
                    h5_patches[start_idx:end_idx] = patches[start_idx:end_idx]

            if embeddings is not None:
                embedding_dim = embeddings.shape[1]
                h5_emb = h5f.create_dataset(
                    "embeddings", 
                    shape=(num_new_patches, embedding_dim), 
                    maxshape=(None, embedding_dim), 
                    dtype='float32'
                )
                # Chunked write
                for start_idx in tqdm(range(0, num_new_patches, CHUNK_SIZE), desc="Writing Embeddings to H5", unit="chunk", leave=False, position=tqdm_pos):
                    end_idx = min(start_idx + CHUNK_SIZE, num_new_patches)
                    h5_emb[start_idx:end_idx] = embeddings[start_idx:end_idx]


def save_embeddings_to_h5(
    output_path: Union[str, Path],
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    patches: np.ndarray,
    slide_ids: List[str]
) -> None:
    """
    Backwards-compatible wrapper that calls save_case_data_to_h5 with embeddings.
    """
    save_case_data_to_h5(
        output_path=output_path,
        coordinates=coordinates,
        slide_ids=slide_ids,
        patches=patches,
        embeddings=embeddings
    )


