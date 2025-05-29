"""Utilities to convert GeoJSON annotations to instance masks."""

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    from shapely.geometry import shape
except ImportError:  # fallback if shapely is not installed
    shape = None


def _draw_polygon(draw: ImageDraw.ImageDraw, coords, obj_id: int) -> None:
    """Draw a polygon into ``draw`` with ``obj_id``."""
    xy = [(x, y) for x, y in coords]
    draw.polygon(xy, outline=obj_id, fill=obj_id)


def geojson_to_mask(geojson_path: str, slide_size: Tuple[int, int]) -> np.ndarray:
    """Convert a single GeoJSON file to an instance mask.

    Args:
        geojson_path: Path to the GeoJSON annotation.
        slide_size: Tuple ``(width, height)`` of the slide.

    Returns:
        ``H x W`` array with ``uint32`` labels.
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)

    mask_img = Image.new("I", slide_size, 0)  # 32 bit integer mask
    draw = ImageDraw.Draw(mask_img)

    for obj_id, feature in enumerate(data.get("features", []), start=1):
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            _draw_polygon(draw, coords, obj_id)
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                coords = poly[0]
                _draw_polygon(draw, coords, obj_id)
        else:
            if shape is not None:
                polygon = shape(geom)
                coords = list(polygon.exterior.coords)
                _draw_polygon(draw, coords, obj_id)
            else:
                raise ValueError(f"Unsupported geometry type: {geom['type']}")

    mask = np.array(mask_img, dtype=np.uint32)
    return mask


def convert_dataset(image_dir: str, label_dir: str, output_dir: str) -> None:
    """Convert all GeoJSON files in ``label_dir`` to masks and save them.

    The output masks will have the same base name as the GeoJSON files but with
    ``.tif`` extension.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    for label_file in sorted(label_dir.glob("*.geojson")):
        slide_name = label_file.stem
        svs_path = image_dir / f"{slide_name}.svs"
        if not svs_path.exists():
            raise FileNotFoundError(f"Slide {svs_path} not found")

        try:
            import openslide
        except ImportError as e:
            raise RuntimeError(
                "openslide-python must be installed to read SVS files"
            ) from e

        slide = openslide.OpenSlide(str(svs_path))
        width, height = slide.dimensions
        mask = geojson_to_mask(str(label_file), (width, height))

        mask_path = Path(output_dir) / f"{slide_name}.tif"
        Image.fromarray(mask).save(mask_path)
        print(f"Saved mask {mask_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert TLS GeoJSON to masks")
    parser.add_argument("image_dir", help="Directory with SVS images")
    parser.add_argument("label_dir", help="Directory with GeoJSON files")
    parser.add_argument(
        "output_dir", help="Directory to store the generated masks"
    )

    args = parser.parse_args()
    convert_dataset(args.image_dir, args.label_dir, args.output_dir)
