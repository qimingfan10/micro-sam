# TLS Finetuning Example

This example demonstrates how to finetune `micro_sam` on whole-slide images
(SVS) with cell annotations in GeoJSON format. It prepares instance masks from
GeoJSON, builds a data loader using `torch_em`, and runs the training with
`micro_sam.training.train_sam`.  The training logic follows the **joint**
finetuning strategy, i.e. both the SAM model and the additional UNETR decoder
are optimized.  This setup is suitable for histopathology data such as
pancreatic cancer slides.

The scripts do not download any checkpoints automatically.  Place a pretrained
SAM weight file locally and pass it to `finetune_tls.py` via `--checkpoint` to
run completely offline.  Use `--device cpu` to force CPU execution or leave it
unset to automatically use CUDA when available.

Files
-----
- `prepare_data.py`: Convert GeoJSON annotations to instance masks.
- `dataset.py`: Defines `TLSDataset` for loading SVS images and masks. The
  reader falls back to PIL if `openslide-python` is unavailable.
- `create_dummy_dataset.py`: Utility to generate a small synthetic SVS slide and
  matching GeoJSON annotation for testing.
- `test_dummy_pipeline.py`: Runs a minimal pipeline on the dummy data to verify
  that mask generation and data loading work.
- `finetune_tls.py`: Script to start the finetuning and export the model.

To quickly test the pipeline without real data run `create_dummy_dataset.py`
and then `test_dummy_pipeline.py`.

### Setup

Create the conda environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate sam
```

Download a pretrained SAM checkpoint (e.g. `sam_vit_b_01ec64.pth`) manually and
place it somewhere locally.  When running the training script specify the path
via `--checkpoint`.

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
   python finetune_tls.py --checkpoint /path/to/sam_vit_b_01ec64.pth --device cpu
   ```

   This trains SAM together with the instance decoder and exports the weights to
   `finetuned_tls_model.pth`.
