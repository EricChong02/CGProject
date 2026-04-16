# Unified Project Plan and Collaboration Context

## 1. Project Identity

This repository is a **CS5182 Computer Graphics course project** on **3D point cloud classification**.

Core topic:
- benchmark deep learning methods for point cloud object classification
- build a reproducible training and evaluation pipeline
- compare at least two methods
- analyze performance, limitations, and generalization
- prepare a presentation-first deliverable before the final written report

Current project direction:
- baseline model: `PointNet++`
- comparison model: `DGCNN`
- second dataset target: `ScanObjectNN`
- optional extension if time permits: a lightweight improved or robustness-oriented `PointNet++` variant

This file is the **single source of truth** for:
- Member A and Member B collaboration
- default project context for Codex
- current project status and short-term execution plan

## 2. Delivery Strategy

The project must be **presentation-ready by 2026-04-17 19:00**.

Because training time is the main bottleneck, the working strategy is:
- prioritize a stable and explainable baseline over ambitious extra features
- finish a complete `PointNet++` path first
- add `DGCNN` as the comparison method in parallel
- treat the second dataset as required, but keep the implementation path pragmatic
- avoid large refactors, speculative redesigns, and risky scope expansion before the pre

Minimum presentation-ready deliverable:
- one fully working `PointNet++` pipeline on `ModelNet40`
- one comparison method path for `DGCNN`, ideally with at least a first runnable result
- one second-dataset analysis path, preferably `ScanObjectNN`
- result tables, logs, checkpoints or metrics, and a clear explanation of limitations

## 3. Current Verified Repository Status

The repository already contains:
- modular project structure
- PyTorch environment and entry-point scripts
- config, logs, checkpoints, results, figures, and experiments directories
- a working `ModelNet40` dataset loader
- a working `ScanObjectNN` dataset loader
- a working `PointNet++` classification model
- working train and evaluate scripts for the `PointNet++ + ModelNet40` path
- working train and evaluate scripts for the `PointNet++ + ScanObjectNN` path

Verified working pieces:
- `ModelNet40` smoke test
- `ScanObjectNN` smoke test
- `PointNet++` forward smoke test
- `PointNet++` training pipeline
- `PointNet++` evaluation pipeline
- `debug` config
- `fast` config

Important current config files:
- `configs/pointnet2_modelnet40_debug.yaml`
- `configs/pointnet2_modelnet40_fast.yaml`
- `configs/pointnet2_modelnet40_train.yaml`
- `configs/pointnet2_scanobjectnn_debug.yaml`
- `configs/pointnet2_scanobjectnn_fast.yaml`
- `configs/pointnet2_scanobjectnn_train.yaml`

Current known placeholders:
- `pointcloud_benchmark/models/dgcnn.py`
- `pointcloud_benchmark/models/improved_pointnet2.py`

## 4. Current Baseline Evidence

Verified local baseline evidence on `2026-04-15` using:
- config: `configs/pointnet2_modelnet40_fast.yaml`
- model: `PointNet++`
- dataset: `ModelNet40`

Observed verified outputs:
- epoch 1 training completed successfully
- checkpoint save path works
- evaluation on saved checkpoint works

Recorded metrics:
- epoch 1 train loss: `2.2407`
- epoch 1 train accuracy: `0.4127`
- epoch 1 validation loss: `1.7027`
- epoch 1 validation accuracy: `0.4951`
- evaluation accuracy from `best.pt`: `0.4984`

Artifacts verified to exist locally:
- `checkpoints/pointnet2_modelnet40_fast/latest.pt`
- `checkpoints/pointnet2_modelnet40_fast/best.pt`
- `results/pointnet2_modelnet40_fast/evaluation_metrics.json`

Important note:
- training artifacts are intentionally ignored by Git in `.gitignore`
- do not claim longer training or stronger results than what has actually been run

## 5. Team Structure and Ownership

### Member A

Primary role:
- baseline owner and project integrator

Main responsibilities:
1. `PointNet++` baseline
2. main experiment pipeline
3. baseline training and evaluation
4. paper reading and method understanding
5. report main structure
6. final result integration
7. main presentation storyline

Primary ownership:
- `configs/pointnet2_*`
- `pointcloud_benchmark/models/pointnet2.py`
- baseline experiment interpretation
- report main logic

### Member B

Primary role:
- comparison-method owner and support contributor

Main responsibilities:
1. `DGCNN` implementation
2. `DGCNN` smoke test
3. `DGCNN` train and evaluate integration
4. `DGCNN` debug and fast configs
5. comparison results and tables
6. support for figures, PPT, and comparison writeup

Primary ownership:
- `pointcloud_benchmark/models/dgcnn.py`
- `configs/dgcnn_*`
- `DGCNN` smoke tests and comparison outputs

## 6. Branch Strategy

Stable branch:
- `main`

Feature branches:
- `feat/pointnet2-baseline`: Member A working branch
- `feat/dgcnn-compare`: Member B working branch

Branch rules:
- do not develop directly on `main`
- all new work should happen on feature branches
- merge to `main` only after the relevant smoke test or train/evaluate path is confirmed working
- keep PointNet++ baseline work isolated from DGCNN comparison work when possible

Recommended workflow:
1. Member A stabilizes and pushes `feat/pointnet2-baseline`
2. Member B branches from the latest stable baseline commit into `feat/dgcnn-compare`
3. shared files are edited minimally and carefully
4. merge to `main` only after both sides have a stable checkpointed milestone

## 7. Shared Files and Collaboration Rules

Shared files that require extra care:
- `pointcloud_benchmark/training/trainer.py`
- `pointcloud_benchmark/datasets/builder.py`
- `pointcloud_benchmark/models/builder.py`
- `README.md`

Rules for shared files:
- keep changes minimal
- do not break the working `PointNet++` path
- do not remove existing configs
- do not rename important paths during the sprint
- if a shared change is needed, prefer the smallest robust fix

Project-wide rules:
- preserve current project structure
- avoid uncontrolled refactors
- prefer incremental, testable changes
- do not make fake claims about training, accuracy, or completeness
- prioritize presentation-ready deliverables over ambitious extra engineering

## 8. Immediate Execution Plan

### Member A immediate plan

Priority order:
1. preserve the working `PointNet++` baseline branch
2. run longer `fast` training when compute is available
3. collect baseline metrics, logs, checkpoints, and curves
4. prepare comparison against the PointNet++ paper
5. write the baseline section for the pre and final report

### Member B immediate plan

Priority order:
1. implement real `DGCNN`
2. add a `DGCNN` smoke test
3. connect `DGCNN` to the existing training and evaluation pipeline
4. create `DGCNN` debug and fast configs
5. produce a first comparison result on `ModelNet40`

### Joint immediate plan before the pre

Must finish:
- one stable baseline path
- one comparison path
- one presentation-ready explanation of task, dataset, method, and results
- one limitations slide
- one team-contribution explanation

Should finish if possible:
- second dataset experiment or at least a credible external reproduction and analysis path
- training curves and result figures
- concise comparison table

Can be deferred if needed:
- improved or robust PointNet++ variant
- richer visualization features
- broader refactors or cleanup

## 9. Experiment Plan

### Required comparison line

Primary experiments:
- `PointNet++` on `ModelNet40`
- `DGCNN` on `ModelNet40`

Second-dataset target:
- `PointNet++` on `ScanObjectNN`
- `DGCNN` on `ScanObjectNN`

Fallback rule for the second dataset:
- if a full in-repo `ScanObjectNN` path is not ready in time, use a credible external implementation or reference result and clearly label the source in the presentation

### Main metrics

- overall classification accuracy
- validation accuracy
- training stability
- comparison to reported paper performance
- qualitative discussion of failure cases and limitations

### Controlled variables

- same train/test split protocol per dataset
- matched point count when feasible
- consistent training budget where fair comparison is possible
- explicit reporting of any mismatched conditions

## 10. Report and Presentation Structure

Recommended report sections:
1. Introduction
2. Task definition and motivation
3. Dataset and preprocessing
4. Methods
5. Experimental setup
6. Results and comparison
7. Limitations and failure analysis
8. Conclusion and future work

Recommended presentation flow:
1. project topic and motivation
2. problem definition and dataset
3. baseline method: `PointNet++`
4. comparison method: `DGCNN`
5. implementation progress and pipeline
6. experiment results
7. limitations, lessons learned, and next steps

Presentation style target:
- short
- visual
- honest about scope and status
- focused on decisions, experiments, and results rather than excessive code detail

## 11. Default Codex Instructions

Codex should treat the following as default project context for this repository:

1. prioritize Member A's `PointNet++` baseline line unless the user explicitly switches focus
2. do not distract Member A into `DGCNN` work unless asked
3. preserve collaboration compatibility with Member B
4. keep configs clear: `debug`, `fast`, and `full-train`
5. avoid large risky refactors
6. prefer minimal, robust, testable changes
7. do not invent training results or pretend unfinished work is complete
8. when editing shared files, be conservative and protect the working baseline path
9. optimize for presentation-ready progress under time pressure

## 12. Current Status Checklist

- [x] modular project structure created
- [x] output directories and experiment folders created
- [x] README, requirements, and gitignore added
- [x] `ModelNet40` loader implemented
- [x] real `PointNet++` implemented
- [x] `ModelNet40` smoke test passes
- [x] `PointNet++` forward smoke test passes
- [x] `PointNet++` training path verified
- [x] `PointNet++` evaluation path verified
- [x] `pointnet2_modelnet40_fast.yaml` added and validated
- [ ] real `DGCNN` implemented
- [ ] `DGCNN` smoke test implemented
- [ ] `DGCNN` train/eval path verified
- [x] real `ScanObjectNN` loader implemented
- [ ] second-dataset experiment completed
- [ ] improved or robust `PointNet++` variant implemented
- [ ] presentation finalized
- [ ] final report drafted

## 13. Canonical Usage Note

Use this file as the default project context for:
- project planning
- team coordination
- task delegation
- future Codex sessions in this repository

If future plans change, update this file first so both human collaborators and AI collaborators stay aligned.
