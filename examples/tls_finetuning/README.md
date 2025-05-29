# TLS Finetuning Example

This example demonstrates how to finetune `micro_sam` on whole-slide images
(SVS) with cell annotations in GeoJSON format. It prepares instance masks from
GeoJSON, builds a data loader using `torch_em`, and runs the training with

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
  It accepts a `--device` parameter to choose between CPU and GPU.

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

   Download the official SAM checkpoint (e.g. `sam_vit_b_01ec64.pth`) and place
   it in a location accessible from this folder. Then run

   ```bash
   python finetune_tls.py --checkpoint PATH_TO_SAM_CKPT --data-root data --device cuda
   ```

   The exported weights will be written to `finetuned_tls_model.pth`.

### Training Approach

The dataset is split in a 90/10 ratio based on slide order. During training
random `1024x1024` patches are sampled from the slides. This patch based
finetuning strategy is suitable for whole-slide pancreas cancer images where
individual cells are relatively small compared to the slide size.
