# Project Master Plan

## 1. Course Project Requirements Summary

- Build a complete Computer Graphics course project around a clear technical topic.
- Use a structured experimental workflow with reproducible training and evaluation.
- Compare multiple methods rather than reporting a single model result.
- Present implementation progress, experiments, analysis, and conclusions clearly.
- Deliver both a written report and a presentation.

## 2. Selected Topic

**Point Cloud Classification**

Goal: benchmark and improve deep learning methods for 3D point cloud object classification.

## 3. Main Technical Route

- Baseline: `PointNet++`
- Comparison model: `DGCNN`
- Second dataset: `ScanObjectNN`
- Advanced method: `Robust Augmented PointNet++`

Working interpretation of the advanced method:

- start from the `PointNet++` baseline
- add stronger data augmentation and robustness-oriented training ideas
- optionally add lightweight architectural improvements if time permits

Note:

- no full real model implementation is completed yet
- the repository currently contains a runnable project skeleton with placeholders and TODOs

## 4. Planned Development Phases

### Phase 1: Project Scaffold

- finalize modular directory structure
- add configs, scripts, logging, checkpoints, results, and figures folders
- make placeholder training, evaluation, and visualization commands runnable

### Phase 2: Dataset Pipeline

- implement `ModelNet40` dataset loading
- implement `ScanObjectNN` dataset loading
- add preprocessing, normalization, point sampling, and basic augmentation

### Phase 3: Baseline Models

- implement `PointNet++`
- implement `DGCNN`
- verify training and evaluation on `ModelNet40`

### Phase 4: Robust Advanced Method

- design `Robust Augmented PointNet++`
- test robustness-oriented augmentations
- compare accuracy and stability against baselines

### Phase 5: Analysis and Deliverables

- run final benchmark experiments
- generate tables and figures
- complete report and presentation materials

## 5. Directory Structure Conventions

- `pointcloud_benchmark/datasets/`: dataset classes and loaders
- `pointcloud_benchmark/models/`: model definitions
- `pointcloud_benchmark/training/`: training loops and training logic
- `pointcloud_benchmark/evaluation/`: metrics and evaluation
- `pointcloud_benchmark/visualization/`: plots and figure generation
- `pointcloud_benchmark/utils/`: config, logging, seed, and I/O helpers
- `configs/`: YAML experiment configs
- `scripts/`: entry-point scripts for train, evaluate, and visualize
- `datasets_raw/`: original downloaded datasets
- `datasets_processed/`: cached or preprocessed dataset files
- `experiments/`: config snapshots and experiment metadata
- `logs/`: training and evaluation logs
- `checkpoints/`: saved model weights
- `results/`: metrics and exported JSON results
- `figures/`: plots for analysis and report writing

## 6. Experiment Plan

### Core comparisons

- `PointNet++` on `ModelNet40`
- `DGCNN` on `ModelNet40`
- `PointNet++` on `ScanObjectNN`
- `DGCNN` on `ScanObjectNN`
- `Robust Augmented PointNet++` on both datasets

### Main metrics

- overall classification accuracy
- mean class accuracy
- training stability
- robustness under augmentation or noise settings

### Controlled variables

- same train/test split protocol per dataset
- matched number of points when possible
- consistent optimizer and training budget for fair comparison

## 7. Report Plan

Suggested sections:

1. Introduction
2. Background and related methods
3. Datasets and preprocessing
4. Methodology
5. Experimental setup
6. Results and comparison
7. Robustness analysis
8. Conclusion and future work

Planned report outputs:

- method overview diagram
- experiment table
- accuracy comparison plots
- qualitative or robustness visualization if useful

## 8. Presentation Plan

- motivation and problem statement
- why point clouds are challenging
- method lineup: `PointNet++`, `DGCNN`, and robust improved method
- datasets used
- benchmark setup
- key results and visualizations
- lessons learned and future improvements

Target style:

- short and visual
- focus on comparisons and project decisions
- include one slide on implementation progress and one slide on limitations

## 9. Current Status Checklist

- [x] Modular PyTorch project structure created
- [x] Placeholder training, evaluation, and visualization scripts created
- [x] Initial YAML config files added for all planned model and dataset combinations
- [x] Output directories for logs, checkpoints, results, figures, and experiments added
- [x] README, requirements, and gitignore added
- [ ] Real `ModelNet40` loader implemented
- [ ] Real `ScanObjectNN` loader implemented
- [ ] Real `PointNet++` implemented
- [ ] Real `DGCNN` implemented
- [ ] Real `Robust Augmented PointNet++` implemented
- [ ] Final experiments completed
- [ ] Report drafted
- [ ] Presentation slides drafted

## 10. Pending Tasks

- implement actual dataset loading and preprocessing
- implement baseline `PointNet++`
- implement comparison `DGCNN`
- define and implement `Robust Augmented PointNet++`
- add richer evaluation metrics and plots
- run benchmark experiments on both datasets
- summarize findings for the report
- prepare final presentation slides

