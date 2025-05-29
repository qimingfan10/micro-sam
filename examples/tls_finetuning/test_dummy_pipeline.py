"""Run a minimal end-to-end test of the TLS finetuning pipeline."""

from pathlib import Path

from create_dummy_dataset import create_geojson, create_slide
from prepare_data import convert_dataset
from dataset import get_dataloader


def main() -> None:
    base = Path(__file__).parent / "newdata"
    img_dir = base / "images"
    geo_dir = base / "geojson"
    mask_dir = base / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    slide_path = img_dir / "sample.svs"
    label_path = geo_dir / "sample.geojson"
    if not slide_path.exists():
        create_slide(slide_path)
    if not label_path.exists():
        create_geojson(label_path)

    convert_dataset(img_dir, geo_dir, mask_dir)

    loader = get_dataloader(img_dir, mask_dir, patch_shape=(512, 512), batch_size=1, n_samples=1, split="train")
    sample = next(iter(loader))
    print("Loaded patch shape:", sample[0].shape, "label shape:", sample[1].shape)


if __name__ == "__main__":
    main()
