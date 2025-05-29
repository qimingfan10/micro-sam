# TLS Finetuning Example

This example demonstrates how to finetune `micro_sam` on whole-slide images
(SVS) with cell annotations in GeoJSON format. It prepares instance masks from
GeoJSON, builds a data loader using `torch_em`, and runs the training with
`micro_sam.training.train_sam`.

Files
-----
- `prepare_data.py`: Convert GeoJSON annotations to instance masks.
- `dataset.py`: Defines `TLSDataset` for loading SVS images and masks.
- `finetune_tls.py`: Script to start the finetuning and export the model.

### Dataset Layout

The code assumes that slides and annotations have matching base names, e.g.
```
XNYY-LUAD-0001.svs
XNYY-LUAD-0001.geojson
```
After converting the GeoJSON files with `prepare_data.py` the folder structure
should look like

```
examples/tls_finetuning/data/
    images/   # contains *.svs files
    geojson/  # original annotations
    masks/    # generated instance masks (*.tif)
```

### Running the Example

1. **Prepare the masks**

   Convert all GeoJSON annotations to instance masks. Replace `PATH_TO_IMAGES`
   and `PATH_TO_GEOJSON` with your directories containing the slides and labels
   and choose an output folder for the masks.

   ```bash
   python prepare_data.py PATH_TO_IMAGES PATH_TO_GEOJSON data/masks
   ```

2. **Start finetuning**

   Place the SVS files in `data/images` and run:

   ```bash
   python finetune_tls.py
   ```

   This will train SAM together with the instance decoder and export the
   resulting weights to `finetuned_tls_model.pth`.
