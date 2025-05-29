"""Finetune Segment Anything on TLS data."""

import argparse
import os
from typing import Optional

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from .dataset import get_dataloader


DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "data")


def get_paths(data_root: str) -> tuple[str, str]:
    """Return the image and mask folder for ``data_root``."""
    images = os.path.join(data_root, "images")
    masks = os.path.join(data_root, "masks")
    return images, masks


def run_training(
    data_root: str,
    checkpoint_name: str,
    model_type: str,
    checkpoint: Optional[str],
    device: torch.device,
    n_epochs: int,
    train_decoder: bool,
) -> None:
    """Run SAM finetuning."""
    patch_shape = (1024, 1024)
    batch_size = 1

    images, masks = get_paths(data_root)

    train_loader = get_dataloader(
        images, masks, patch_shape, batch_size, n_samples=50, split="train"
    )
    val_loader = get_dataloader(
        images, masks, patch_shape, batch_size, n_samples=4, split="val"
    )

    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint,
        patch_shape=patch_shape,
        with_segmentation_decoder=train_decoder,
        device=device,
        n_epochs=n_epochs,
    )


def export_model(checkpoint_name: str, model_type: str) -> None:
    export_path = os.path.join(os.path.dirname(__file__), "finetuned_tls_model.pth")
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune SAM on TLS data")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_ROOT,
        help="Folder containing 'images' and 'masks' subdirectories",
    )
    parser.add_argument("--checkpoint", required=True, help="Initial SAM checkpoint")
    parser.add_argument(
        "--model-type", default="vit_b", help="SAM model type used for finetuning"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for training",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--no-decoder",
        action="store_true",
        help="Disable training of the segmentation decoder",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="sam_tls",
        help="Name for the output checkpoint",
    )
    args = parser.parse_args()

    run_training(
        data_root=args.data_root,
        checkpoint_name=args.checkpoint_name,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=torch.device(args.device),
        n_epochs=args.epochs,
        train_decoder=not args.no_decoder,
    )
    export_model(args.checkpoint_name, args.model_type)


if __name__ == "__main__":
    main()
