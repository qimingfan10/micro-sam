"""Finetune SAM on TLS data."""

import argparse
import os

from typing import Optional

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from .dataset import get_dataloader


DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "data")


def get_paths(data_root: str):
    images = os.path.join(data_root, "images")
    masks = os.path.join(data_root, "masks")
    return images, masks


def run_training(
    checkpoint_name: str,
    model_type: str,
    train_decoder: bool,
    checkpoint: Optional[str],
    device: str,
    data_root: str,
    n_epochs: int,
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

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        patch_shape=patch_shape,
        checkpoint=checkpoint,
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
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained SAM weights")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda or auto")
    parser.add_argument("--data-root", type=str, default=DEFAULT_ROOT, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--model-type", type=str, default="vit_b", help="SAM model type")
    parser.add_argument("--name", type=str, default="sam_tls", help="Checkpoint name")
    parser.add_argument("--train-decoder", action="store_true", help="Also train UNETR decoder")
    args = parser.parse_args()

    run_training(
        args.name,
        args.model_type,
        args.train_decoder,
        args.checkpoint,
        args.device,
        args.data_root,
        args.epochs,
    )
    export_model(args.name, args.model_type)


if __name__ == "__main__":
    main()
