# Point Cloud Classification Benchmark

Clean PyTorch project skeleton for benchmarking `PointNet++`, `DGCNN`, and an improved `PointNet++` on `ModelNet40` and `ScanObjectNN`.

This repository is intentionally scaffolded as an initial course-project codebase:

- The project structure is modular and ready to extend.
- Dataset, model, training, evaluation, and visualization modules are wired together.
- Current model and dataset implementations are placeholders with `TODO` markers.
- Placeholder commands run without import errors and generate mock outputs so the workflow can be tested early.

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

Recommended next step:

1. Download the official `ModelNet40` files into `datasets_raw/modelnet40/`.
2. Download the official `ScanObjectNN` files into `datasets_raw/scanobjectnn/`.
3. Implement the actual parsing and preprocessing logic in:
   - `pointcloud_benchmark/datasets/modelnet40.py`
   - `pointcloud_benchmark/datasets/scanobjectnn.py`

Right now the dataset loaders fall back to synthetic samples so the pipeline can be smoke-tested before the real loaders are written.

## Training

Example placeholder training runs:

```bash
python scripts/train.py --config configs/pointnet2_modelnet40.yaml
python scripts/train.py --config configs/dgcnn_scanobjectnn.yaml
```

What the placeholder trainer does today:

- loads the YAML config
- builds a placeholder dataset and model
- runs a lightweight training loop
- saves config snapshots, logs, checkpoints, and metrics

Generated artifacts are written to:

- `experiments/<experiment_name>/`
- `logs/<experiment_name>/`
- `checkpoints/<experiment_name>/`
- `results/<experiment_name>/`

## Evaluation

Run evaluation with a config:

```bash
python scripts/evaluate.py --config configs/pointnet2_modelnet40.yaml
```

Optional checkpoint override:

```bash
python scripts/evaluate.py \
  --config configs/pointnet2_modelnet40.yaml \
  --checkpoint checkpoints/pointnet2_modelnet40/latest.pt
```

The current evaluator supports the pipeline and output format, but metrics are only meaningful once the real models and datasets are implemented.

## Visualization

Generate a training-curve figure:

```bash
python scripts/visualize.py --config configs/pointnet2_modelnet40.yaml
```

The visualization script reads `results/<experiment_name>/train_history.json` and writes an SVG figure to `figures/<experiment_name>/training_curves.svg`, which keeps the placeholder workflow dependency-light.

## Configuration Files

Available starter configs:

- `configs/pointnet2_modelnet40.yaml`
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

1. Replace synthetic dataset generation with real dataset parsing.
2. Implement full `PointNet++`, `DGCNN`, and improved `PointNet++` architectures.
3. Add real augmentation, checkpoint resume support, and richer metrics.
4. Expand visualization for confusion matrices, class-wise accuracy, and point cloud previews.
