import json
from pathlib import Path
import numpy as np
from PIL import Image


def create_slide(path: Path, size=(1024, 1024)) -> None:
    data = np.random.randint(0, 255, size + (3,), dtype=np.uint8)
    img = Image.fromarray(data)
    # save as tiff but use .svs suffix to mimic a slide file
    img.save(path)


def create_geojson(path: Path) -> None:
    # simple square polygon
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[100, 100], [400, 100], [400, 400], [100, 400], [100, 100]]],
        },
        "properties": {},
    }
    geo = {"type": "FeatureCollection", "features": [feature]}
    with open(path, "w") as f:
        json.dump(geo, f)


if __name__ == "__main__":
    base = Path(__file__).parent / "newdata"
    img_dir = base / "images"
    geo_dir = base / "geojson"
    img_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    slide_path = img_dir / "sample.svs"
    create_slide(slide_path)

    label_path = geo_dir / "sample.geojson"
    create_geojson(label_path)
    print("Dummy data created in", base)
