# Submission Notes

This directory is the final Canvas-ready package for the `CS5182 Course Project`.

## Official Requirement Checklist

From `CS5182_Course_Project_25-26B.pdf`, the required submission items are:

1. `A source subdirectory containing all the source files, do not include the platform source code and training dataset.`
   - satisfied by `source/`
2. `Txt files which contain the results for training and evaluation periods.`
   - satisfied by `results_txt/`
3. `Data that are evaluated in your report.`
   - satisfied by `report_data/`
4. `A cover that indicates your name(s) and student ID(s).`
   - satisfied by `report&PPT/CGFinalReport.pdf`
5. `Describe the results listed in requirements in last section.`
   - satisfied by `report&PPT/CGFinalReport.pdf`

The presentation PDF used for the course presentation is included as:

- `report&PPT/CGFinalPPT.pdf`

## Directory Layout

- `source/`
  - final source-only snapshot
  - includes code, configs, scripts, tests, `README.md`, and `requirements.txt`
  - excludes datasets, checkpoints, logs, and platform/runtime artifacts
- `results_txt/`
  - final plain-text summaries for the four report-facing experiments
  - includes training-period and evaluation-period metrics in a consistent text format
- `report_data/`
  - JSON metrics, training histories, selected logs, and training-curve figures used by the report and PPT
- `report&PPT/`
  - final report PDF and final presentation PDF

## Final Experiment Mapping

The submission is organized around these four final report-facing experiments:

1. `PointNet++ on ModelNet40 fast`
2. `DGCNN on ModelNet40 fast`
3. `PointNet++ on ScanObjectNN lightweight`
4. `DGCNN on ScanObjectNN lightweight`

Repo experiment name mapping:

- `PointNet++ on ModelNet40 fast` -> `pointnet2_modelnet40_fast`
- `DGCNN on ModelNet40 fast` -> `dgcnn_modelnet40_fast25_fullsplit`
- `PointNet++ on ScanObjectNN lightweight` -> `pointnet2_scanobjectnn_fast_light`
- `DGCNN on ScanObjectNN lightweight` -> `dgcnn_scanobjectnn_test`

Submission folder naming:

- `01_pointnet2_modelnet40_fast`
- `02_dgcnn_modelnet40_fast`
- `03_pointnet2_scanobjectnn_lightweight`
- `04_dgcnn_scanobjectnn_lightweight`

## Final Notes

- This package contains only completed final material.
- The previously kept in-progress `pointnet2_scanobjectnn_fast` supplemental folder has been removed.
- Debug or verification-only artifacts are not treated as final benchmark results.
