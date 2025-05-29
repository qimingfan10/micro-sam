"""Finetune SAM on TLS data."""

import os
import argparse

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from .dataset import get_dataloader


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
IMAGES = os.path.join(DATA_ROOT, "images")
MASKS = os.path.join(DATA_ROOT, "masks")


def run_training(
    checkpoint_name: str,
    model_type: str,
    checkpoint_path: str,
    train_decoder: bool,
    device: str,
) -> None:
    """Run SAM finetuning."""
    patch_shape = (1024, 1024)
    batch_size = 1

    train_loader = get_dataloader(
        IMAGES, MASKS, patch_shape, batch_size, n_samples=50, split="train"
    )
    val_loader = get_dataloader(
        IMAGES, MASKS, patch_shape, batch_size, n_samples=4, split="val"
    )

    device = torch.device(device)

    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        patch_shape=patch_shape,
        checkpoint=checkpoint_path,
        with_segmentation_decoder=train_decoder,
        device=device,
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
        "--checkpoint",
        required=True,
        help="Path to the pretrained SAM weights (*.pth)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for training",
    )
    parser.add_argument("--model-type", default="vit_b", help="SAM model type")
    parser.add_argument(
        "--checkpoint-name",
        default="sam_tls",
        help="Name for the training run",
    )
    parser.add_argument(
        "--train-decoder",
        action="store_true",
        help="Train with instance segmentation decoder",
    )

    args = parser.parse_args()

    run_training(
        checkpoint_name=args.checkpoint_name,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        train_decoder=args.train_decoder,
        device=args.device,
    )
    export_model(args.checkpoint_name, args.model_type)


if __name__ == "__main__":
    main()
