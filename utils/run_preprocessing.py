#!/usr/bin/env python3
"""
TCGA-BRCA WSI Preprocessing and UNI2 Feature Extraction Master CLI Script.

This script processes a directory of WSI (.svs) files:
1. Performs coarse tissue segmentation on thumbnails.
2. Applies fine quality control filtering (blur/contrast check) on CPU.
3. Saves a visual 4-panel quality control report (.png) under qc_reports/.
4. Passes accepted patches to the UNI2 foundation model on GPU to extract 1536-D embeddings.
5. Saves coordinates, slide IDs, and embeddings directly to case-level HDF5 files (.h5)
   under data/preprocessed/pathology/. Handles multi-slide cases by appending dynamically.
6. Implements slide-level checkpointing to prevent redundant processing.
7. Supports multi-GPU parallel processing by spawning concurrent worker processes (one per GPU).
"""

import os
import sys
import argparse
import time
import h5py
import torch
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Ensure the project root and src directory are in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from wsi_processing import (
    open_wsi,
    get_wsi_metadata,
    get_wsi_thumbnail,
    get_raw_tissue_mask,
    filter_tissue_artifacts,
    extract_valid_patches,
    save_case_data_to_h5,
    save_qc_report,
    UNI2FeatureExtractor
)

# Load environment variables (HF_TOKEN)
load_dotenv()


def check_slide_processed(h5_path: str, slide_id: str) -> bool:
    """
    Checks if a specific slide has already been processed and stored in the case H5 file.
    """
    if not os.path.exists(h5_path):
        return False
    try:
        with h5py.File(h5_path, 'r') as h5f:
            if "slide_ids" in h5f:
                # Decodes bytes to string if stored as HDF5 variable-length string bytes
                processed_slides = [
                    s.decode('utf-8') if isinstance(s, bytes) else s 
                    for s in h5f["slide_ids"][:]
                ]
                return slide_id in processed_slides
    except Exception as e:
        # File might be locked by another process or corrupted
        return False
    return False


def worker_process(queue, device, worker_idx, args):
    """
    Worker process running on a designated GPU device.
    Pulls WSI files from a shared queue and processes them.
    """
    # Set tqdm position for this worker to prevent terminal logs overlap
    os.environ["TQDM_POSITION"] = str(worker_idx)
    if "TQDM_DISABLE" in os.environ:
        del os.environ["TQDM_DISABLE"]
    
    tqdm.write(f"[{device}] Worker process started successfully.")
    
    # Initialize extractor inside worker process to isolate CUDA contexts
    try:
        extractor = UNI2FeatureExtractor(device=device)
    except Exception as e:
        tqdm.write(f"[{device}][Fatal Error] Failed to initialize UNI2 model: {e}")
        return

    output_path = Path(args.output_dir)
    qc_path = Path(args.qc_dir)

    while True:
        try:
            wsi_file_str = queue.get_nowait()
        except Exception:
            # Queue is empty
            break

        wsi_file = Path(wsi_file_str)
        wsi_filename = wsi_file.name
        slide_id = wsi_file.stem
        case_id = wsi_filename[:12] if wsi_filename.startswith("TCGA") else slide_id
        h5_file_path = output_path / f"{case_id}.h5"

        # Checkpoint check
        if check_slide_processed(str(h5_file_path), wsi_filename):
            tqdm.write(f"[{device}][Checkpoint] Slide {wsi_filename} already processed. Skipping.")
            continue

        tqdm.write(f"[{device}] Processing slide: {wsi_filename} (Case: {case_id})")
        slide_start = time.time()

        try:
            # 1. Open WSI
            tile_source = open_wsi(str(wsi_file))
            meta = get_wsi_metadata(tile_source)

            # 2. Extract WSI Thumbnail
            thumb_rgb, rx, ry = get_wsi_thumbnail(tile_source, target_size=1024)

            # 3. Coarse segmentation
            raw_mask = get_raw_tissue_mask(thumb_rgb, sigma=1.0, threshold_method="triangle")
            coarse_mask = filter_tissue_artifacts(raw_mask, thumb_rgb)

            # 4. CPU QC Filtering
            coords, patches, slide_ids_list, fine_mask = extract_valid_patches(
                tile_source=tile_source,
                raw_tissue_mask=coarse_mask,
                rx=rx,
                ry=ry,
                slide_id=wsi_filename,
                patch_width=256,
                patch_height=256,
                target_magnification=20.0,
                overlap=args.overlap,
                tissue_threshold=args.tissue_thresh,
                sharpness_threshold=args.sharpness_thresh,
                contrast_threshold=args.contrast_thresh
            )
            num_valid = len(coords)
            tqdm.write(f"[{device}]   Fine QC completed: {num_valid} valid patches extracted.")

            if num_valid == 0:
                tqdm.write(f"[{device}]   [Warning] No tissue patches accepted. Skipping H5 save.")
                continue

            # 5. Save visual QC report PNG
            save_qc_report(
                thumb_rgb=thumb_rgb,
                coarse_mask=coarse_mask,
                fine_mask=fine_mask,
                output_dir=str(qc_path),
                wsi_filename=wsi_filename
            )

            # 6. GPU Feature Extraction
            embeddings = extractor.extract_features(patches, batch_size=args.batch_size)

            # 7. Save coordinates and embeddings to extensible case H5
            save_case_data_to_h5(
                output_path=str(h5_file_path),
                coordinates=coords,
                slide_ids=slide_ids_list,
                patches=None,
                embeddings=embeddings
            )

            tqdm.write(f"[{device}] Successfully processed {wsi_filename} in {time.time() - slide_start:.2f} seconds.")

        except Exception as e:
            tqdm.write(f"[{device}][Error] Failed to process slide {wsi_filename}: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue


def process_sequentially(wsi_files, device, args):
    """
    Processes all WSI files sequentially inside the main process.
    Supports interactive tqdm progress bars.
    """
    os.environ["TQDM_POSITION"] = "0"
    if "TQDM_DISABLE" in os.environ:
        del os.environ["TQDM_DISABLE"]

    output_path = Path(args.output_dir)
    qc_path = Path(args.qc_dir)
    total_files = len(wsi_files)

    tqdm.write("\n[Init] Initializing UNI2 Feature Extractor (ViT-Giant-SwiGLU) sequentially...")
    extractor = UNI2FeatureExtractor(device=device)

    start_time = time.time()
    num_processed = 0
    num_skipped = 0

    for idx, wsi_file in enumerate(wsi_files, 1):
        wsi_filename = wsi_file.name
        slide_id = wsi_file.stem
        case_id = wsi_filename[:12] if wsi_filename.startswith("TCGA") else slide_id
        h5_file_path = output_path / f"{case_id}.h5"

        tqdm.write(f"\n[{idx}/{total_files}] Processing slide: {wsi_filename} (Case: {case_id})")

        if check_slide_processed(str(h5_file_path), wsi_filename):
            tqdm.write(f"  [Checkpoint] Slide already processed. Skipping.")
            num_skipped += 1
            continue

        slide_start = time.time()
        try:
            # 1. Open WSI
            tile_source = open_wsi(str(wsi_file))
            meta = get_wsi_metadata(tile_source)
            tqdm.write(f"    Scale: {meta['width']}x{meta['height']} pixels, Magnification: {meta['magnification']}X")

            # 2. Extract Thumbnail
            thumb_rgb, rx, ry = get_wsi_thumbnail(tile_source, target_size=1024)

            # 3. Coarse mask
            raw_mask = get_raw_tissue_mask(thumb_rgb, sigma=1.0, threshold_method="triangle")
            coarse_mask = filter_tissue_artifacts(raw_mask, thumb_rgb)

            # 4. Fine QC
            coords, patches, slide_ids_list, fine_mask = extract_valid_patches(
                tile_source=tile_source,
                raw_tissue_mask=coarse_mask,
                rx=rx,
                ry=ry,
                slide_id=wsi_filename,
                patch_width=256,
                patch_height=256,
                target_magnification=20.0,
                overlap=args.overlap,
                tissue_threshold=args.tissue_thresh,
                sharpness_threshold=args.sharpness_thresh,
                contrast_threshold=args.contrast_thresh
            )
            num_valid = len(coords)
            tqdm.write(f"    Fine QC: {num_valid} valid patches.")

            if num_valid == 0:
                tqdm.write("    [Warning] No valid tissue patches. Skipping H5 save.")
                continue

            # 5. Save visual report
            save_qc_report(
                thumb_rgb=thumb_rgb,
                coarse_mask=coarse_mask,
                fine_mask=fine_mask,
                output_dir=str(qc_path),
                wsi_filename=wsi_filename
            )

            # 6. GPU Inference
            tqdm.write(f"    Running UNI2 inference on device {device}...")
            embeddings = extractor.extract_features(patches, batch_size=args.batch_size)

            # 7. Save to H5
            tqdm.write(f"    Saving coordinates and embeddings to case H5...")
            save_case_data_to_h5(
                output_path=str(h5_file_path),
                coordinates=coords,
                slide_ids=slide_ids_list,
                patches=None,
                embeddings=embeddings
            )

            num_processed += 1
            tqdm.write(f"    Processed in {time.time() - slide_start:.2f} seconds.")

        except Exception as e:
            tqdm.write(f"    [Error] Failed to process slide {wsi_filename}: {e}")
            import traceback
            tqdm.write(traceback.format_exc())
            continue

    total_time = time.time() - start_time
    print(f"\n=== Sequential Preprocessing Pipeline Completed ===")
    print(f"  Processed: {num_processed}, Skipped: {num_skipped}, Total time: {total_time:.2f}s")


def main():
    # Set start method for multiprocessing to spawn (mandatory for CUDA safety)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Parallel WSI Preprocessing and UNI2 Embedding Extraction Pipeline")
    parser.add_argument(
        "--raw_dir", 
        type=str, 
        default="data/tcga-brca/raw/pathology", 
        help="Path to the directory containing raw WSI (.svs) files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/preprocessed/pathology", 
        help="Path to save the processed case-level H5 files"
    )
    parser.add_argument(
        "--qc_dir", 
        type=str, 
        default="data/preprocessed/pathology/qc_reports", 
        help="Path to save the visual QC reports"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Batch size for GPU model inference"
    )
    parser.add_argument(
        "--devices", 
        type=str, 
        nargs="+", 
        default=None, 
        help="List of CUDA devices to use in parallel (e.g. cuda:0 cuda:1). If None, auto-detects all GPUs."
    )
    parser.add_argument(
        "--stride_factor",
        type=float,
        default=2.0,
        help="Stride factor relative to patch size (default: 2.0). If specified, overlap is computed as 1.0 - stride_factor."
    )
    parser.add_argument(
        "--overlap", 
        type=float, 
        default=None, 
        help="Overlap fraction between tiles [0.0, 1.0). If specified, overrides stride_factor."
    )
    parser.add_argument(
        "--tissue_thresh", 
        type=float, 
        default=0.8, 
        help="Minimum tissue ratio required in coarse mask to accept a tile"
    )
    parser.add_argument(
        "--sharpness_thresh", 
        type=float, 
        default=0.0005, 
        help="Minimum Laplace sharpness variance required"
    )
    parser.add_argument(
        "--contrast_thresh", 
        type=float, 
        default=0.05, 
        help="Minimum contrast fraction threshold"
    )
    args = parser.parse_args()

    # Compute overlap from stride_factor if overlap is not explicitly provided
    if args.overlap is None:
        args.overlap = 1.0 - args.stride_factor

    # Create target directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    qc_path = Path(args.qc_dir)
    qc_path.mkdir(parents=True, exist_ok=True)

    # Scan for WSI files
    raw_dir_path = Path(args.raw_dir)
    if not raw_dir_path.exists():
        print(f"Error: Raw WSI directory not found at {args.raw_dir}")
        sys.exit(1)
        
    supported_extensions = {".svs", ".tiff", ".tif", ".ndpi"}
    wsi_files = sorted([
        f for f in raw_dir_path.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ])
    total_files = len(wsi_files)
    print(f"=== Batch Preprocessing Pipeline ===")
    print(f"  Found {total_files} WSI files to process.")
    
    if total_files == 0:
        print("Nothing to process. Exiting.")
        sys.exit(0)

    # Determine GPU device(s)
    if args.devices is None:
        devices = ["cuda:0" if torch.cuda.is_available() else "cpu"]
    else:
        devices = args.devices

    print(f"  Target Devices: {devices}")
    start_time = time.time()

    # Determine execution scheme based on devices list
    if len(devices) == 1:
        # Single device -> Process sequentially to preserve clean progress bars
        process_sequentially(wsi_files, devices[0], args)
    else:
        # Multi-device -> Process concurrently via worker process pool
        print(f"\n[Parallel] Initializing {len(devices)} parallel worker processes...")
        
        # Populate multiprocessing queue
        queue = multiprocessing.Queue()
        for wsi_file in wsi_files:
            queue.put(str(wsi_file))

        # Spawn workers
        workers = []
        for worker_idx, device in enumerate(devices):
            p = multiprocessing.Process(
                target=worker_process,
                args=(queue, device, worker_idx, args)
            )
            workers.append(p)
            p.start()

        # Wait for completion
        for p in workers:
            p.join()

        print(f"\n=== Parallel Preprocessing Pipeline Completed ===")
        print(f"  Processed cohorte using GPUs: {devices}")
        print(f"  Total Execution Time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
