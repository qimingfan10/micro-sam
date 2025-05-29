"""Dataset utilities for TLS finetuning."""

import os
import random
from glob import glob
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TLSDataset(Dataset):
    """Random patch dataset for SVS images with instance masks."""

    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        patch_shape: Tuple[int, int] = (1024, 1024),
        n_samples: int = 100,
    ) -> None:
        assert len(image_paths) == len(mask_paths), "Images and masks mismatch"
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.patch_shape = patch_shape
        self.n_samples = n_samples

        try:
            import openslide  # lazy import
        except ImportError as e:
            raise RuntimeError(
                "openslide-python must be installed to read SVS files"
            ) from e

        self.slides = [openslide.OpenSlide(p) for p in self.image_paths]
        self.masks = [np.array(Image.open(p)) for p in self.mask_paths]
        self.sizes = [slide.dimensions for slide in self.slides]

    def __len__(self) -> int:
        return self.n_samples

    def _sample_location(self, width: int, height: int) -> Tuple[int, int]:
        ph, pw = self.patch_shape
        x0 = random.randint(0, width - pw)
        y0 = random.randint(0, height - ph)
        return x0, y0

    def __getitem__(self, index: int):
        slide_idx = index % len(self.slides)
        slide = self.slides[slide_idx]
        mask = self.masks[slide_idx]
        width, height = self.sizes[slide_idx]

        x0, y0 = self._sample_location(width, height)
        ph, pw = self.patch_shape

        patch = slide.read_region((x0, y0), 0, (pw, ph))
        patch = np.asarray(patch)[..., :3]
        label = mask[y0 : y0 + ph, x0 : x0 + pw]

        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
        label = torch.from_numpy(label.astype(np.int64))
        return patch, label


def get_dataloader(
    image_dir: str,
    mask_dir: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    n_samples: int,
    split: str,
) -> DataLoader:
    """Create dataloader for training or validation."""
    assert split in {"train", "val"}
    def pair_paths(image_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        images: List[str] = []
        masks: List[str] = []
        for img_path in sorted(image_dir.glob("*.svs")):
            base = img_path.stem
            mask_path = mask_dir / f"{base}.tif"
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask for {img_path.name} not found")
            images.append(str(img_path))
            masks.append(str(mask_path))
        return images, masks

    image_paths, mask_paths = pair_paths(image_dir, mask_dir)

    # simple split based on file order
    val_fraction = 0.1
    n_val = max(1, int(len(image_paths) * val_fraction))

    if split == "train":
        image_paths = image_paths[:-n_val]
        mask_paths = mask_paths[:-n_val]
    else:
        image_paths = image_paths[-n_val:]
        mask_paths = mask_paths[-n_val:]

    dataset = TLSDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        patch_shape=patch_shape,
        n_samples=n_samples * batch_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader
