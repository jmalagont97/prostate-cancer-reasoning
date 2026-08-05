"""
Pathology Foundation Models Module.

This module provides wrappers and utilities for loading histopathology foundation
models (like UNI and UNI2) via the timm library, preparing image patches according
to model-specific pre-processing guidelines (e.g., resizing to 224x224 and ImageNet
normalization), and extracting features.
"""

import os
import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
from typing import Union, List, Optional
from dotenv import load_dotenv

# Load environment variables (such as HF_TOKEN) from a .env file if present
load_dotenv()



def get_uni2_kwargs() -> dict:
    """
    Returns the specific architecture kwargs required to initialize the UNI2-h model.
    """
    # SiLU act_layer is standard for SwiGLU architectures
    return {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }


class UNI2FeatureExtractor:
    """
    Feature extractor class exclusively for the UNI2 (UNI2-h) pathology foundation model.
    Loads the ViT-Giant-SwiGLU model (embed_dim=1536) and handles batch inference.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            weights_path (str, optional): Path to local PyTorch checkpoint (.pth or .pt).
                If None, downloads/loads from Hugging Face Hub (requires login).
            device (str, optional): Target device ('cuda', 'cpu', etc.). Auto-detected if None.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing UNI2 (UNI2-h) on device: {self.device}")

        # UNI2 architecture parameters
        timm_kwargs = get_uni2_kwargs()
        model_name = "hf_hub:MahmoodLab/UNI2-h"
        
        # Initialize model
        pretrained = True if weights_path is None else False
        self.model = timm.create_model(model_name, pretrained=pretrained, **timm_kwargs)

        # Load custom local weights if provided
        if weights_path:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Local weights path not found: {weights_path}")
            print(f"Loading custom UNI2 weights from checkpoint: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Standardized UNI2 transforms: resize 256x256 tiles to 224x224 and apply ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    @torch.no_grad()
    def extract_features(self, patches: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Extracts features (embeddings) from a collection of raw image patches.

        Args:
            patches (np.ndarray): Array of shape (P, H, W, 3) of uint8.
            batch_size (int, optional): Batch size for inference to prevent GPU OOM. Defaults to 64.

        Returns:
            np.ndarray: Matrix of embeddings of shape (P, embed_dim) of float32.
        """
        num_patches = patches.shape[0]
        if num_patches == 0:
            # Detect embedding dimension dynamically
            dummy_input = torch.zeros((1, 3, 224, 224), device=self.device)
            embed_dim = self.model(dummy_input).shape[1]
            return np.empty((0, embed_dim), dtype=np.float32)

        from tqdm import tqdm
        import os
        tqdm_pos = int(os.environ.get("TQDM_POSITION", "0"))
        embeddings_list = []

        # Process in batches to avoid GPU out-of-memory errors
        for idx in tqdm(range(0, num_patches, batch_size), desc="UNI2 Inference", unit="batch", leave=False, position=tqdm_pos):
            batch_patches_np = patches[idx : idx + batch_size]
            
            # Apply PyTorch transformations patch by patch
            transformed_tensors = []
            for patch in batch_patches_np:
                pil_img = Image.fromarray(patch)
                transformed_tensors.append(self.transform(pil_img))

            # Stack into a single batch tensor and move to device
            batch_tensor = torch.stack(transformed_tensors).to(self.device)
            
            # Forward pass
            batch_embeddings = self.model(batch_tensor)
            embeddings_list.append(batch_embeddings.cpu().numpy())

        return np.concatenate(embeddings_list, axis=0)
