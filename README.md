# Point Cloud Classification Benchmark

Clean PyTorch project skeleton for benchmarking `PointNet++`, `DGCNN`, and an improved `PointNet++` on `ModelNet40` and `ScanObjectNN`.

This repository is intentionally scaffolded as an initial course-project codebase:

- The project structure is modular and ready to extend.
- Dataset, model, training, evaluation, and visualization modules are wired together.
- The `ModelNet40` loader now supports the common h5 format used by PointNet-family projects.
- A real `PointNet++` classification model is implemented for the current pipeline.
- `DGCNN`, `ScanObjectNN`, and the improved `PointNet++` path are still placeholders with `TODO` markers.

## Project Structure

```text
.
├── checkpoints/
├── configs/
├── datasets_processed/
├── datasets_raw/
├── experiments/
├── figures/
├── logs/
├── pointcloud_benchmark/
│   ├── configs/
│   ├── datasets/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── visualization/
├── results/
├── scripts/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Expected dataset locations:

- `datasets_raw/modelnet40/`
- `datasets_raw/scanobjectnn/`
- `datasets_processed/modelnet40/`
- `datasets_processed/scanobjectnn/`

### ModelNet40 Folder Structure

The loader expects the common `modelnet40_ply_hdf5_2048` layout. The recommended directory tree is:

```text
datasets_raw/
└── modelnet40/
    └── modelnet40_ply_hdf5_2048/
        ├── shape_names.txt
        ├── train_files.txt
        ├── test_files.txt
        ├── ply_data_train0.h5
        ├── ply_data_train1.h5
        ├── ply_data_train2.h5
        ├── ply_data_train3.h5
        ├── ply_data_train4.h5
        ├── ply_data_test0.h5
        └── ...
```

`root` in the config should usually remain:

```yaml
dataset:
  root: datasets_raw/modelnet40
```

The loader will automatically detect the nested `modelnet40_ply_hdf5_2048/` directory. It also supports the case where the h5 files are placed directly inside `datasets_raw/modelnet40/`.

### ModelNet40 Features

- automatic `train` and `test` split discovery
- configurable `num_points` sampling
- zero-mean, unit-scale normalization
- training-only augmentation:
  - random point dropout
  - random scaling
  - random shifting
  - gaussian jitter
  - optional random rotation around the configured upright axis

### Smoke Test

To sanity-check the dataset pipeline after downloading ModelNet40:

```bash
python scripts/smoke_test_modelnet40.py --config configs/pointnet2_modelnet40_debug.yaml
```

This loads a small batch and prints the point tensor shape, label shape, and min/max values.

### PointNet++ Forward Smoke Test

To verify the PointNet++ classifier builds and runs a real forward pass:

```bash
python scripts/smoke_test_pointnet2.py --config configs/pointnet2_modelnet40_debug.yaml
```

This constructs the model and runs random input of shape `[B, N, 3]` through the network.

### ScanObjectNN

`ScanObjectNN` is not implemented yet and remains a placeholder.

## Training

Example training runs:

```bash
python scripts/train.py --config configs/pointnet2_modelnet40_debug.yaml
python scripts/train.py --config configs/pointnet2_modelnet40_train.yaml
```

What the current training pipeline does:

- loads the YAML config
- builds the configured dataset and model
- runs classification training with cross-entropy loss
- saves config snapshots, logs, checkpoints, and metrics
- writes both `latest.pt` and `best.pt` checkpoints for supported runs

Note:

- `configs/pointnet2_modelnet40_debug.yaml` is the explicit lightweight sanity-check config
- it uses a small subset of `ModelNet40`, batch size `4`, and `1` epoch so the first real run stays fast
- `configs/pointnet2_modelnet40_train.yaml` is the real baseline training config for full-split experiments
- `configs/pointnet2_modelnet40.yaml` remains as a backward-compatible alias of the debug setup

Generated artifacts are written to:

- `experiments/<experiment_name>/`
- `logs/<experiment_name>/`
- `checkpoints/<experiment_name>/`
- `results/<experiment_name>/`

## Evaluation

Run evaluation with a config:

```bash
python scripts/evaluate.py --config configs/pointnet2_modelnet40_debug.yaml
```

If `--checkpoint` is omitted, the evaluator automatically looks for:

- `checkpoints/<experiment_name>/best.pt`
- `checkpoints/<experiment_name>/latest.pt`

Optional checkpoint override:

```bash
python scripts/evaluate.py \
  --config configs/pointnet2_modelnet40_train.yaml \
  --checkpoint checkpoints/pointnet2_modelnet40_train/latest.pt
```

The current evaluator works with the real `PointNet++` + `ModelNet40` path. Metrics for other placeholder model paths are not meaningful yet.

## Visualization

Generate a training-curve figure:

```bash
python scripts/visualize.py --config configs/pointnet2_modelnet40_debug.yaml
```

The visualization script reads `results/<experiment_name>/train_history.json` and writes an SVG figure to `figures/<experiment_name>/training_curves.svg`, which keeps the placeholder workflow dependency-light.

## Configuration Files

Available starter configs:

- `configs/pointnet2_modelnet40.yaml`
- `configs/pointnet2_modelnet40_debug.yaml`
- `configs/pointnet2_modelnet40_train.yaml`
- `configs/dgcnn_modelnet40.yaml`
- `configs/improved_pointnet2_modelnet40.yaml`
- `configs/pointnet2_scanobjectnn.yaml`
- `configs/dgcnn_scanobjectnn.yaml`
- `configs/improved_pointnet2_scanobjectnn.yaml`

Each config defines:

- experiment metadata
- dataset settings
- model selection
- training and evaluation hyperparameters
- output directories

## Next Implementation Tasks

1. Implement the `ScanObjectNN` dataset pipeline.
2. Implement full `DGCNN` and improved `PointNet++` architectures.
3. Add checkpoint resume support and richer metrics.
4. Expand visualization for confusion matrices, class-wise accuracy, and point cloud previews.
