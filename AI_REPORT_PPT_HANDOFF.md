# AI Handoff for Report and Presentation

This document is a handoff summary for another AI that will help produce the final `report` and `presentation PPT` for the CS5182 course project.

Use this document together with:

- the course project PDF: `/Users/zhuangyi/Downloads/CS5182_Course_Project_25-26B.pdf`
- the repository at `/Users/zhuangyi/Desktop/CGProject`
- the user's final experiment data, tables, and figures for the four target experiments

Important rule:

- The user will separately provide the final data and charts for the four main experiments.
- Those user-provided final results should be treated as the source of truth for the report and PPT.
- This repository contains some verified local artifacts, but not all four final experiment packages are fully present in the current working tree.

## 1. Project Identity

Course: `CS5182 Computer Graphics`

Project topic:

- benchmark deep learning methods for `3D point cloud classification`
- build a reproducible training and evaluation pipeline
- compare at least two methods
- analyze performance, limitations, and second-dataset behavior

Current project direction from `PROJECT_MASTER_PLAN.md`:

- baseline model: `PointNet++`
- comparison model: `DGCNN`
- second dataset target: `ScanObjectNN`
- optional extension if time permits: improved or robustness-oriented `PointNet++`

Current intended final experiment set:

1. `PointNet++ on ModelNet40 fast`
2. `DGCNN on ModelNet40 fast`
3. `PointNet++ on ScanObjectNN fast_light`
4. `DGCNN on ScanObjectNN fast_light`

### 1.1 Methods and Reference Papers

Baseline method:

- `PointNet++`
- current implementation in this repo is a `single-scale grouping` classification model
- local paper file: `references/pointnet2.pdf`
- standard citation target: `PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space`

Comparison method:

- `DGCNN`
- key idea: dynamic graph construction with `EdgeConv`-style neighborhood feature extraction
- standard citation target: `Dynamic Graph CNN for Learning on Point Clouds`

Use the user's preferred exact citation format if they already have it. If not, the above paper titles are the correct method references to start from.

## 2. What the Course Requires

From the official course PDF:

- `Basic Requirements`: `50%`
- `Advanced Requirements`: `20%`
- `Presentation`: `30%`

This means the report and PPT should mainly make the `basic requirements` look complete and coherent. If the advanced part is weak or missing, do not fake it. It is acceptable to mention a proposed improvement or future work as long as it is clearly labeled as not fully implemented.

### 2.1 Basic Requirements to Cover

The report should explicitly address all of these:

1. `Introduction of the project`
   - task definition
   - input/output
   - dataset
   - reference paper
   - objective and motivation
   - technical challenges
2. `Deploy deep learning environment`
   - PyTorch environment
   - dependencies
   - local or server compute setup
3. `Run demo code`
   - explain what each major step in the code/pipeline does
4. `Train model`
   - dataset download
   - training configuration
   - training process
5. `Compare experimental results with the paper`
   - compare quantitative results with reported paper numbers
   - discuss gap and possible reasons
   - show visualized results when available
6. `Analyze performance on another dataset`
   - use a second dataset to discuss generality
7. `Analyze drawbacks`
   - failure cases
   - limitations
   - why errors happen
   - possible fixes
8. `Implement and compare with other state-of-the-art methods`
   - compare `PointNet++` and `DGCNN`

### 2.2 Advanced Requirement

The advanced part asks for:

- an extension or improvement to the algorithm
- possible changes to network architecture, loss, or dataset
- clear rationale and final results

Current honest status:

- there is no confirmed fully implemented advanced method in the current local branch
- `improved_pointnet2.py` is still a placeholder in the current worktree

So the report/PPT should:

- not claim a completed advanced method unless the user explicitly provides real evidence
- if needed, mention a reasonable extension idea in `future work` or `possible improvement`

### 2.3 Presentation Requirements

From the PDF:

- presentation during lecture/tutorial time
- each project gets `7 minutes presentation + 3 minutes Q&A`
- tentative presentation dates: `April 17, 2026` and `April 24, 2026`

From `PROJECT_MASTER_PLAN.md`:

- the practical internal target was to be `presentation-ready by April 17, 2026 at 19:00`
- because of this, the project strategy prioritized a stable baseline and comparison results over risky scope expansion

### 2.4 Submission Requirements

From the PDF:

- submit via Canvas
- submission deadline: `April 26, 2026`

Required submission items:

- source subdirectory with source files
- text files containing training/evaluation results
- data evaluated in the report
- report with cover page containing name(s) and student ID(s)
- report must describe the results required in the previous section

## 3. Recommended Report Structure

This structure already aligns well with both the project plan and the course rubric:

1. `Introduction`
2. `Task Definition and Motivation`
3. `Dataset and Preprocessing`
4. `Methods`
5. `Experimental Setup`
6. `Results and Comparison`
7. `Limitations and Failure Analysis`
8. `Conclusion and Future Work`

Good mapping to the rubric:

- Sections 1-2 cover the project introduction and motivation
- Section 3 covers dataset and preprocessing
- Section 4 covers method explanation and demo-code understanding
- Section 5 covers environment and training setup
- Section 6 covers training results, second-dataset analysis, and model comparison
- Section 7 covers drawback analysis
- Section 8 can mention possible advanced ideas without overclaiming implementation

## 4. Recommended Presentation Flow

This is also from the project master plan and matches the `7-minute` format:

1. project topic and motivation
2. problem definition and datasets
3. baseline method: `PointNet++`
4. comparison method: `DGCNN`
5. implementation pipeline
6. experiment results
7. limitations, lessons learned, and next steps

Presentation style target:

- short
- visual
- honest about scope and status
- focused on decisions, experiments, and results, not too much code detail

## 5. Repository Overview

Repository root:

- `/Users/zhuangyi/Desktop/CGProject`

Main scripts:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/visualize.py`
- `scripts/smoke_test_modelnet40.py`
- `scripts/smoke_test_scanobjectnn.py`
- `scripts/smoke_test_pointnet2.py`

Model code:

- `pointcloud_benchmark/models/pointnet2.py`
- `pointcloud_benchmark/models/dgcnn.py`
- `pointcloud_benchmark/models/improved_pointnet2.py`

Dataset loaders:

- `pointcloud_benchmark/datasets/modelnet40.py`
- `pointcloud_benchmark/datasets/scanobjectnn.py`

Important configs:

- `configs/pointnet2_modelnet40_fast.yaml`
- `configs/pointnet2_scanobjectnn_fast_light.yaml`
- `configs/dgcnn_modelnet40.yaml`
- `configs/dgcnn_scanobjectnn_fast_light.yaml`

Other useful project files:

- `README.md`
- `PROJECT_MASTER_PLAN.md`
- `requirements.txt`
- `references/pointnet2.pdf`

Environment/dependency basics from `requirements.txt`:

- `torch>=2.2.0`
- `numpy>=1.24.0`
- `PyYAML>=6.0`
- `h5py>=3.10.0`
- `tqdm>=4.66.0`

Framework:

- this is a `PyTorch` project

## 6. Codebase Status Snapshot

Current checked-out branch when this handoff was prepared:

- `feat/pointnet2-baseline`

Important branch situation:

- current local worktree is strongest on the `PointNet++` baseline
- the current local `pointcloud_benchmark/models/dgcnn.py` file is still a `placeholder`
- however, the remote branch `origin/feat/dgcnn-compare` contains a real DGCNN implementation and some DGCNN outputs

This distinction matters a lot for honesty:

- do not describe the current local branch as if it already contains a complete DGCNN training pipeline
- if the final DGCNN results came from the user's separate work or another branch, say so consistently

### 6.1 Current Local Verified Capabilities

Verified in the current repository state:

- modular project structure exists
- `ModelNet40` loader works
- `ScanObjectNN` loader works
- real `PointNet++` classifier is implemented
- `PointNet++ + ModelNet40` training and evaluation path works
- `PointNet++ + ScanObjectNN` training path exists
- output directories, logs, checkpoints, and result JSON files are used consistently

### 6.2 Current Local Placeholder Areas

In the current checked-out branch:

- `pointcloud_benchmark/models/dgcnn.py` is still a placeholder
- `pointcloud_benchmark/models/improved_pointnet2.py` is still a placeholder

## 7. Dataset Setup Used by the Project

### 7.1 ModelNet40

Expected root:

- `datasets_raw/modelnet40`

Typical data layout:

- `modelnet40_ply_hdf5_2048`

Task:

- point cloud object classification

Configured class count:

- `40`

### 7.2 ScanObjectNN

Expected root:

- `datasets_raw/scanobjectnn`

Configured variant in the fast_light configs:

- `pb_t50_rs`

This is the hardest/noisier variant and is often treated as a stronger robustness test than ModelNet40.

Configured class count:

- `15`

Important note for the report:

- `ModelNet40` is relatively clean and synthetic
- `ScanObjectNN` is more realistic and difficult
- that difference should be used in the explanation of why accuracy may drop on `ScanObjectNN`

## 8. Experimental Configuration Summary

### 8.1 PointNet++ on ModelNet40 fast

Config:

- `configs/pointnet2_modelnet40_fast.yaml`

Key settings:

- model: `pointnet2`
- dataset: `modelnet40`
- `512` points
- `25` epochs
- batch size `16`
- full split, not debug subset
- runtime device in config: `cpu`

Outputs go to:

- `experiments/pointnet2_modelnet40_fast/`
- `logs/pointnet2_modelnet40_fast/`
- `checkpoints/pointnet2_modelnet40_fast/`
- `results/pointnet2_modelnet40_fast/`
- `figures/pointnet2_modelnet40_fast/`

### 8.2 PointNet++ on ScanObjectNN fast_light

Config:

- `configs/pointnet2_scanobjectnn_fast_light.yaml`

Key settings:

- model: `pointnet2`
- dataset: `scanobjectnn`
- variant: `pb_t50_rs`
- `512` points
- `6` epochs
- batch size `16`
- debug subset:
  - max train samples: `4096`
  - max test samples: `1024`
- runtime device in config: `cpu`

Outputs go to:

- `experiments/pointnet2_scanobjectnn_fast_light/`
- `logs/pointnet2_scanobjectnn_fast_light/`
- `checkpoints/pointnet2_scanobjectnn_fast_light/`
- `results/pointnet2_scanobjectnn_fast_light/`
- `figures/pointnet2_scanobjectnn_fast_light/`

### 8.3 DGCNN on ModelNet40

Current local config present:

- `configs/dgcnn_modelnet40.yaml`

But note:

- in the current local branch this path is not enough by itself, because local `dgcnn.py` is placeholder

Important remote-branch artifact:

- `origin/feat/dgcnn-compare` contains `configs/dgcnn_modelnet40_fast25.yaml`
- experiment name there: `dgcnn_modelnet40_fast25_fullsplit`

Key settings from that remote config:

- model: `dgcnn`
- dataset: `modelnet40`
- `512` points
- `25` epochs
- batch size `16`
- runtime device: `cpu`

### 8.4 DGCNN on ScanObjectNN fast_light

Current local config present:

- `configs/dgcnn_scanobjectnn_fast_light.yaml`

Key settings in that config:

- model: `dgcnn`
- dataset: `scanobjectnn`
- variant: `pb_t50_rs`
- `512` points
- `6` epochs
- batch size `16`
- debug subset:
  - max train samples: `4096`
  - max test samples: `1024`
- runtime device: `cpu`

But again:

- current local `dgcnn.py` is placeholder
- final DGCNN ScanObjectNN results should be taken from the user's final experiment package, not inferred from current local code

## 9. Verified Local and Branch-Accessible Results

This section is very important. It separates what is already verifiable in the repo from what should be replaced or confirmed by the user's final experiment files.

### 9.1 Verified Local Final-Quality Result

`PointNet++ on ModelNet40 fast`

Local verified artifacts:

- `results/pointnet2_modelnet40_fast/evaluation_metrics.json`
- `results/pointnet2_modelnet40_fast/train_history.json`
- `figures/pointnet2_modelnet40_fast/training_curves.svg`
- `logs/pointnet2_modelnet40_fast/train.log`
- `logs/pointnet2_modelnet40_fast/evaluate.log`
- `checkpoints/pointnet2_modelnet40_fast/best.pt`

Key verified metrics:

- final evaluation accuracy: `0.8723663091659546`
- best validation accuracy: `0.8723663091659546`
- best epoch: `25`

Useful training log milestones:

- epoch 1: train acc `0.4073`, val acc `0.4417`
- epoch 25: train acc `0.8088`, val acc `0.8724`

There is also an earlier partial evaluation entry with `0.4984` in the log history, but the final saved evaluation for this experiment is `0.8724`. Use the final number, not the early intermediate one.

### 9.2 Local Partial / Non-Final ScanObjectNN Evidence

`PointNet++ on ScanObjectNN fast_light`

Current local evidence:

- `logs/pointnet2_scanobjectnn_fast_light/train.log`
- `checkpoints/pointnet2_scanobjectnn_fast_light/best.pt`
- `checkpoints/pointnet2_scanobjectnn_fast_light/latest.pt`

What is visible locally in the training log:

- epoch 1 of 6 reached:
  - train loss `1.2754`
  - train acc `0.5403`
  - val loss `1.1185`
  - val acc `0.5811`

Important warning:

- there is no matching final `results/pointnet2_scanobjectnn_fast_light/evaluation_metrics.json` in the current worktree
- so this local folder should be treated as `in progress` unless the user separately provides the final results

### 9.3 Local Verification Artifact That Should Not Be Used as Final Result

There is a separate experiment:

- `pointnet2_scanobjectnn_fast_light_verify`

Its local evaluation file shows:

- accuracy `0.0`

This looks like a verification or setup artifact, not a valid final report result. Do not use it as the final ScanObjectNN result unless the user explicitly says otherwise.

### 9.4 Branch-Accessible DGCNN Result

From `origin/feat/dgcnn-compare`, there is a real DGCNN implementation and a recorded ModelNet40 experiment:

- config: `configs/dgcnn_modelnet40_fast25.yaml`
- result file: `results/dgcnn_modelnet40_fast25_fullsplit/evaluation_metrics.json`
- history file: `results/dgcnn_modelnet40_fast25_fullsplit/train_history.json`
- figure: `figures/dgcnn_modelnet40_fast25_fullsplit/training_curves.svg`

Verified metric from that remote branch artifact:

- final evaluation accuracy: `0.8703403472900391`
- best epoch: `25`

This is very close to the local PointNet++ ModelNet40 fast result and is useful for the comparison table.

### 9.5 Summary Table of Repo-Verified Facts

| Experiment | Status in current worktree / branch access | Accuracy / note |
|---|---|---:|
| PointNet++ on ModelNet40 fast | locally verified complete | `0.8724` |
| DGCNN on ModelNet40 fast-like run | available from `origin/feat/dgcnn-compare` as `fast25_fullsplit` | `0.8703` |
| PointNet++ on ScanObjectNN fast_light | local run evidence exists, but final eval file missing | partial local epoch-1 val acc `0.5811` |
| DGCNN on ScanObjectNN fast_light | config exists locally, but final result not verifiable in current branch | use user-provided final data |

## 10. Artifact Formats Used by the Project

Training history format:

- file: `results/<experiment_name>/train_history.json`
- typical keys:
  - `train_loss`
  - `train_acc`
  - `val_loss`
  - `val_acc`
  - `best_val_acc`
  - `best_epoch`

Evaluation metric format:

- file: `results/<experiment_name>/evaluation_metrics.json`
- typical keys:
  - `accuracy`
  - `dataset`
  - `model`

Training curves:

- generated as `figures/<experiment_name>/training_curves.svg`

Logs:

- `logs/<experiment_name>/train.log`
- `logs/<experiment_name>/evaluate.log`

Checkpoints:

- `checkpoints/<experiment_name>/best.pt`
- `checkpoints/<experiment_name>/latest.pt`

## 11. Honest Narrative the Report/PPT Should Use

The safest and strongest overall narrative is:

- this project benchmarks deep learning methods for `3D point cloud classification`
- `PointNet++` is the baseline
- `DGCNN` is the comparison method
- the evaluation is done on two datasets:
  - `ModelNet40`
  - `ScanObjectNN`
- `ModelNet40` gives a cleaner benchmark
- `ScanObjectNN` is used to discuss behavior on a more difficult dataset
- the main report value comes from:
  - reproducing a working baseline
  - comparing two models
  - analyzing second-dataset performance
  - discussing limitations and failure causes

Avoid saying:

- that a novel model was fully implemented if it was not
- that all numbers came from the exact same branch unless that is true
- that `ScanObjectNN fast_light` is a full benchmark if it actually used the `4096 / 1024` debug subset

Safer phrasing:

- `a lightweight local-compute configuration was used for same-day experimentation`
- `the ScanObjectNN fast_light runs use a reduced subset for practicality`
- `the advanced improvement direction is proposed as future work`

## 12. What the Other AI Should Pay Attention To

### 12.1 Priority of Evidence

Use evidence in this order:

1. the user's final experiment tables, metrics, and charts
2. this handoff document
3. local verified repo artifacts
4. remote-branch accessible artifacts like `origin/feat/dgcnn-compare`

### 12.2 What Must Be Explicitly Stated in the Report

- task: point cloud object classification
- input: 3D point clouds
- output: class prediction
- datasets: `ModelNet40` and `ScanObjectNN`
- methods: `PointNet++` and `DGCNN`
- environment: PyTorch-based codebase
- experiment settings:
  - point count
  - epoch count
  - batch size
  - whether full split or reduced subset
  - CPU local-compute setting if that was used
- final accuracy table
- comparison with reference paper numbers
- limitations and failure analysis

### 12.3 Important Caveats

- `pointnet2_scanobjectnn_fast_light_verify` with `0.0` accuracy is not a final result
- local debug artifacts like `pointnet2_scanobjectnn_debug` with `1.0` accuracy are smoke-test artifacts and should not be used as serious benchmark numbers
- current local `dgcnn.py` is placeholder, even though another branch contains a real implementation
- if the user gives four final result packages, those should override any partial local evidence

## 13. Suggested Report Content Focus

If the other AI needs to optimize for a realistic `basic + presentation` score rather than an ambitious innovation score, the report should emphasize:

- complete experimental pipeline
- solid explanation of the methods
- fair comparison table
- clear statement that `ScanObjectNN` is harder
- honest discussion of why results differ from papers
- limitations:
  - local compute constraints
  - CPU training
  - lightweight configs
  - reduced subset on `fast_light`
  - possible implementation and hyperparameter gaps

Strong analysis points that are likely useful:

- why `ScanObjectNN` is harder than `ModelNet40`
- differences between `PointNet++` and `DGCNN`:
  - PointNet++ uses hierarchical set abstraction and local grouping
  - DGCNN uses dynamic graph / EdgeConv style neighborhood feature learning
- why reduced points and lighter configs may reduce accuracy
- why paper numbers may not be matched exactly:
  - compute budget
  - training epochs
  - hyperparameters
  - implementation details
  - CPU constraints
  - subset vs full benchmark

## 14. Suggested PPT Content Focus

A compact PPT could be structured like this:

1. Title slide
   - course, topic, names, student IDs
2. Motivation and task
   - why point cloud classification matters
3. Datasets
   - `ModelNet40` vs `ScanObjectNN`
4. Methods
   - `PointNet++`
   - `DGCNN`
5. Project pipeline
   - environment, data loader, training, evaluation, visualization
6. Main results table
   - the four experiments
7. Training curves / charts
   - at least the most representative ones
8. Analysis
   - why some settings work better
   - why ScanObjectNN is harder
9. Limitations and future work
10. Q&A backup slide

For a 7-minute talk, the PPT should stay visual and not overfill with code.

## 15. Concrete Files Another AI May Want to Quote or Inspect

Course and planning:

- `/Users/zhuangyi/Downloads/CS5182_Course_Project_25-26B.pdf`
- `PROJECT_MASTER_PLAN.md`
- `README.md`

Local baseline result:

- `results/pointnet2_modelnet40_fast/evaluation_metrics.json`
- `results/pointnet2_modelnet40_fast/train_history.json`
- `figures/pointnet2_modelnet40_fast/training_curves.svg`
- `logs/pointnet2_modelnet40_fast/train.log`
- `logs/pointnet2_modelnet40_fast/evaluate.log`

Local ScanObjectNN progress:

- `logs/pointnet2_scanobjectnn_fast_light/train.log`
- `checkpoints/pointnet2_scanobjectnn_fast_light/best.pt`

Local artifact to avoid using as final:

- `results/pointnet2_scanobjectnn_fast_light_verify/evaluation_metrics.json`

Branch-accessible DGCNN comparison artifact:

- `origin/feat/dgcnn-compare:results/dgcnn_modelnet40_fast25_fullsplit/evaluation_metrics.json`
- `origin/feat/dgcnn-compare:results/dgcnn_modelnet40_fast25_fullsplit/train_history.json`
- `origin/feat/dgcnn-compare:figures/dgcnn_modelnet40_fast25_fullsplit/training_curves.svg`
- `origin/feat/dgcnn-compare:pointcloud_benchmark/models/dgcnn.py`

## 16. Final Instructions for the Other AI

Please generate the report and PPT with these rules:

- use the user's final four experiment data and charts as the main evidence
- use this document to understand the project context and honesty constraints
- align the writeup tightly with the official course rubric
- do not invent advanced-method implementation results
- do not treat smoke-test artifacts as final benchmark results
- if a result came from a different branch or separate run package, label it consistently and honestly

The safest final positioning is:

- this is a solid benchmark-and-analysis course project centered on `PointNet++` and `DGCNN`
- the work demonstrates environment setup, pipeline construction, training, evaluation, comparison, second-dataset analysis, and reflection on limitations
- the main strength is completeness and analysis, not a novel research contribution
