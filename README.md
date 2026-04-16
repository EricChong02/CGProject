# Point Cloud Classification Benchmark

Clean PyTorch project skeleton for benchmarking `PointNet++`, `DGCNN`, and an improved `PointNet++` on `ModelNet40` and `ScanObjectNN`.

This repository is intentionally scaffolded as an initial course-project codebase:

- The project structure is modular and ready to extend.
- Dataset, model, training, evaluation, and visualization modules are wired together.
- The `ModelNet40` loader now supports the common h5 format used by PointNet-family projects.
- The `ScanObjectNN` loader now supports the official h5 release layout.
- A real `PointNet++` classification model is implemented for the current pipeline.
- A real `DGCNN` classification model is implemented for the current pipeline.
- The improved `PointNet++` path is still a placeholder with `TODO` markers.

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

The loader supports the official h5 release used in the ICCV 2019 benchmark. The recommended directory tree is:

```text
datasets_raw/
└── scanobjectnn/
    └── h5_files/
        └── main_split/
            ├── training_objectdataset_augmentedrot_scale75.h5
            ├── test_objectdataset_augmentedrot_scale75.h5
            ├── training_objectdataset_augmentedrot.h5
            ├── test_objectdataset_augmentedrot.h5
            └── ...
```

`root` in the config should usually remain:

```yaml
dataset:
  root: datasets_raw/scanobjectnn
  split_name: main_split
  variant: pb_t50_rs
```

Supported variant keys:

- `obj_bg`
- `obj_only`
- `pb_t25`
- `pb_t25_r`
- `pb_t50_r`
- `pb_t50_rs`

Current ScanObjectNN features:

- automatic detection of `main_split/` and `h5_files/main_split/` layouts
- support for the official h5 naming convention
- configurable point sampling and normalization
- optional foreground-only filtering using the released `mask` field
- the same training-time augmentations used by the current ModelNet40 pipeline

To sanity-check the ScanObjectNN dataloader:

```bash
python scripts/smoke_test_scanobjectnn.py --config configs/pointnet2_scanobjectnn_debug.yaml
```

## Training

Example training runs:

```bash
python scripts/train.py --config configs/pointnet2_modelnet40_debug.yaml
python scripts/train.py --config configs/pointnet2_modelnet40_fast.yaml
python scripts/train.py --config configs/pointnet2_modelnet40_train.yaml
python scripts/train.py --config configs/pointnet2_scanobjectnn_debug.yaml
python scripts/train.py --config configs/pointnet2_scanobjectnn_fast.yaml
python scripts/train.py --config configs/pointnet2_scanobjectnn_fast_light.yaml
python scripts/train.py --config configs/pointnet2_scanobjectnn_train.yaml
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
- `configs/pointnet2_modelnet40_fast.yaml` is a full-split local-development baseline
- it keeps the full `ModelNet40` train/test split, but reduces runtime with `512` points per shape and `25` epochs
- `configs/pointnet2_modelnet40_train.yaml` is the real baseline training config for full-split experiments
- `configs/pointnet2_modelnet40.yaml` remains as a backward-compatible alias of the debug setup
- `configs/pointnet2_scanobjectnn_debug.yaml` is the lightweight ScanObjectNN sanity-check config
- `configs/pointnet2_scanobjectnn_fast.yaml` is the local-development ScanObjectNN config
- `configs/pointnet2_scanobjectnn_fast_light.yaml` is a local-compute ScanObjectNN config tuned for a same-day result on weaker machines
- `configs/pointnet2_scanobjectnn_train.yaml` is the longer ScanObjectNN baseline config
- `configs/pointnet2_scanobjectnn.yaml` remains as a backward-compatible alias of the debug setup

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

The current evaluator works with the real `PointNet++` and `DGCNN` classification paths. Metrics for placeholder model paths are not meaningful yet.

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
- `configs/pointnet2_modelnet40_fast.yaml`
- `configs/pointnet2_modelnet40_train.yaml`
- `configs/dgcnn_modelnet40.yaml`
- `configs/dgcnn_modelnet40_debug.yaml`
- `configs/dgcnn_modelnet40_fast25.yaml`
- `configs/dgcnn_modelnet40_train.yaml`
- `configs/improved_pointnet2_modelnet40.yaml`
- `configs/pointnet2_scanobjectnn.yaml`
- `configs/pointnet2_scanobjectnn_debug.yaml`
- `configs/pointnet2_scanobjectnn_fast.yaml`
- `configs/pointnet2_scanobjectnn_fast_light.yaml`
- `configs/pointnet2_scanobjectnn_train.yaml`
- `configs/dgcnn_scanobjectnn_debug.yaml`
- `configs/dgcnn_scanobjectnn_fast_light.yaml`
- `configs/dgcnn_scanobjectnn_test.yaml`
- `configs/dgcnn_scanobjectnn_train.yaml`
- `configs/improved_pointnet2_scanobjectnn.yaml`

Each config defines:

- experiment metadata
- dataset settings
- model selection
- training and evaluation hyperparameters
- output directories

## Config Naming

The project uses two naming layers:

- repo-facing config and experiment names
- report-facing experiment labels

Repo-facing config families:

- `debug`: smallest sanity-check runs
- `fast` or `fast_light`: reduced-cost local-development runs
- `train`: longer training configs
- bare names like `pointnet2_modelnet40.yaml`, `pointnet2_scanobjectnn.yaml`, `dgcnn_modelnet40.yaml`, and `dgcnn_scanobjectnn.yaml`
  - backward-compatible aliases of the corresponding `debug` setup

Important result-linked exceptions:

- `configs/dgcnn_modelnet40_fast25.yaml`
  - kept because it matches the stored result artifact name `dgcnn_modelnet40_fast25_fullsplit`
- `configs/dgcnn_scanobjectnn_test.yaml`
  - kept because it matches the stored result artifact name `dgcnn_scanobjectnn_test`
  - in the report/PPT, treat this run as the `DGCNN on ScanObjectNN lightweight` result

## Final Status

- This repository is kept as the final archived course-project state.
- The final Canvas-ready package is organized under `submission/`.
- The reported benchmark results are the four experiments summarized in `submission/results_txt/` and backed by `submission/report_data/`.
- `PointNet++` and `DGCNN` are the completed report-facing methods in this repo.
- `improved_pointnet2.py` remains an unfinished placeholder and is not used in the final report or PPT.
