## SARATR-X HiViT Grounding (Minimal Export)

- **Model**: Faster R-CNN with custom `HiViT` backbone (registered via `saratrx_hivit`).
- **Script**: `ground.py` – runs mmdet detection and draws boxes for a queried class.
- **Config**: `config.yaml` – points to the detector config and checkpoint within this repo.

### Install

```bash
pip install -r requirements.txt
```

Make sure your CUDA/PyTorch setup is compatible with `torch==1.8.2+cu111` and `mmcv-full==1.6.0` (these versions match original SARATR-X detection env and avoid the mmdet/transformers issues).

### Checkpoint

1. Put your trained detector checkpoint as:
   - `checkpoints/epoch_36.pth`
2. `config.yaml` already points to `configs/hivit_base_SARDet.py` and this checkpoint.

### Usage

```bash
python ground.py \
  --image path/to/image.png \
  --query ship \
  --config config.yaml
```

The output will be written under `./outputs` by default.


