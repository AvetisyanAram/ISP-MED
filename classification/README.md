# ECG Classification

This package provides the current PTB-XL to ISP classification workflow used for:

- PTB-XL training
- ISP and PTB-XL test evaluation
- probability calibration analysis
- ISP confidence-vs-disagreement analysis
- expert-specific threshold analysis

## Installation

```bash
cd classification
python -m venv classification_venv
source classification_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For `s4` and `ecg_cpc`, build the vendored Cauchy extension in the same environment:

```bash
cd extensions/cauchy
python setup.py install
cd ../..
```

## Data

By default, prepared PTB-XL files live under:

```text
classification_data/
└── ptbxl/
    ├── raw/
    └── prepared/
        ├── npz/
        ├── train_map_ptbxl.csv
        ├── val_map_ptbxl.csv
        └── test_map_ptbxl.csv
```

Override the root if needed:

```bash
export CLASSIFICATION_DATA_ROOT=/path/to/classification_data
```

Prepare PTB-XL:

```bash
classification-prepare-ptbxl
```

## Current Baselines

The current final baselines are:

- `resnet1d18`
- `densenet1d121`
- `s4_scratch`
- `densenet1d121_tis_linear_probe`
- `ecgfounder_linear_probe`
- `ecgjepa_linear_probe`
- `stmem_linear_probe`
- `merl_linear_probe`
- `hubert_ecg_linear_probe`
- `ecg_cpc_block_s4_1`

`ecg_cpc_block_s4_1` is the only retained non-linear-probe foundation-model baseline.

## Classification

Use the unified runner:

```bash
python run_classification.py \
  --device-id 0 \
  --batch-size 64 \
  --num-runs 5
```

Run only a subset of baselines:

```bash
python run_classification.py \
  --device-id 0 \
  --batch-size 64 \
  --num-runs 5 \
  --experiments \
    resnet1d18 \
    densenet1d121 \
    s4_scratch \
    ecgfounder_linear_probe
```

Write to a custom output root:

```bash
python run_classification.py \
  --device-id 0 \
  --batch-size 64 \
  --num-runs 5 \
  --output-root outputs/custom_classification_log
```

Outputs are written as:

```text
outputs/classification_log/
└── <baseline>/
    ├── run_01/
    │   ├── train/
    │   ├── isp/
    │   └── ptbxl_test/
    ├── run_02/
    ├── ...
    ├── final_summary.csv
    └── panel_weighted_summary.csv
```

## Calibration

Run calibration on completed baselines:

```bash
python analyze_probability_calibration.py \
  resnet1d18 \
  densenet1d121 \
  s4_scratch \
  ecgfounder_linear_probe \
  ecgjepa_linear_probe \
  stmem_linear_probe \
  merl_linear_probe \
  hubert_ecg_linear_probe \
  ecg_cpc_block_s4_1 \
  densenet1d121_tis_linear_probe \
  --output-root outputs/classification_log \
  --n-bins 10
```

This creates per-baseline calibration outputs and global summaries under:

```text
outputs/classification_log/probability_calibration_summary/
```

The script reports:

- quantile ECE
- uniform ECE
- Brier score
- log loss
- calibration intercept
- calibration slope
- calibration curve tables

Use `--merge-global` when adding one new baseline into an existing calibration summary directory.

## Confidence Analysis

This analysis is not classical probability calibration. It measures how confidence and task metrics
change across disagreement-driven difficulty slices (`easy`, `medium`, `hard`) defined from the exact
active expert panel.

Run:

```bash
python analyze_isp_confidence.py \
  resnet1d18 \
  densenet1d121 \
  s4_scratch \
  ecgfounder_linear_probe \
  ecgjepa_linear_probe \
  stmem_linear_probe \
  merl_linear_probe \
  hubert_ecg_linear_probe \
  ecg_cpc_block_s4_1 \
  densenet1d121_tis_linear_probe \
  --output-root outputs/classification_log \
  --min-count 30
```

Outputs are written under:

```text
outputs/classification_log/confidence_analysis/
```

Key exported artifacts include:

- `all_baselines_final_summary.csv`
- per-baseline pathology highlight tables
- easy/medium/hard confidence summaries
- easy/medium consensus performance summaries
- hard highest-expert summaries

## Expert-Specific Threshold Analysis

This script now performs both stages:

1. doctor-specific threshold tuning and raw result export into `threshold_log`
2. automatic final aggregation into ranked pathology-level threshold trade-off tables

Run:

```bash
python threshold_expert_analysis.py \
  --source-root outputs/classification_log \
  --output-root outputs/threshold_log \
  --baselines \
    resnet1d18 \
    densenet1d121 \
    s4_scratch \
    ecgfounder_linear_probe \
    ecgjepa_linear_probe \
    stmem_linear_probe \
    merl_linear_probe \
    hubert_ecg_linear_probe \
    ecg_cpc_block_s4_1 \
    densenet1d121_tis_linear_probe \
  --optimization-metric G-mean \
  --val-fraction 0.35 \
  --seed 1729 \
  --threshold-grid-step 0.01 \
  --min-test-fraction 0.6
```

By default, this also writes final ranked outputs under:

```text
outputs/threshold_analysis/
```

You can override the final aggregation directory with:

```bash
--final-output-root /path/to/threshold_analysis
```

Important threshold-analysis outputs:

- `ranked_pathology_results.csv`
- `ranked_pathology_results_G_mean.csv`
- `ranked_pathology_results_f1_score.csv`
- `ranked_pathology_results_fbeta2_score.csv`
- `ranked_pathology_tradeoffs_only.csv`
- `ranked_pathology_aligned_only.csv`
