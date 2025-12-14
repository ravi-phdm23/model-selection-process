# Model Selection Pipeline - CLI Guide

Complete end-to-end machine learning pipeline for credit risk modeling without Streamlit.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Output Structure](#output-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

This CLI pipeline automates the complete machine learning workflow:

1. **Data Loading** - Load multiple datasets with automatic preprocessing
2. **EDA** - Exploratory data analysis with feature importance
3. **Model Training** - Train multiple model groups (LR, RF, XGB, etc.)
4. **Benchmarking** - Compare and rank models by performance
5. **SHAP Analysis** - Global interpretability with visualizations
6. **Reliability Testing** - Bootstrap confidence intervals
7. **Local Analysis** - Instance-level SHAP explanations
8. **Report Generation** - LaTeX reports with all results
9. **Finalization** - Summary and cleanup

**Key Features:**
- ✅ Multi-dataset support
- ✅ Automatic categorical encoding
- ✅ SMOTE class balancing
- ✅ 8 performance metrics (AUC, F1, KS, PCC, BS, PG, H, Recall)
- ✅ Checkpoint/resume capability
- ✅ Parallel processing
- ✅ Comprehensive logging

---

## Installation

### Requirements

```bash
# Python 3.8+
pip install -r requirements.txt
```

**Required packages:**
```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.44.0
matplotlib>=3.7.0
pyyaml>=6.0
imbalanced-learn>=0.11.0
```

**Optional:**
```
pdflatex  # For PDF compilation (TeX Live or MiKTeX)
```

---

## Quick Start

### 1. Basic Run (Single Dataset)

```bash
# Run complete pipeline with test configuration
python model_pipeline.py --config test_complete_config.yaml
```

**Output:** All 9 stages complete in ~1 minute, results in `output_complete_test/`

### 2. Multi-Dataset Run

```bash
# Run with 3 datasets, 102 models
python model_pipeline.py --config multi_dataset_config.yaml
```

**Output:** Complete analysis in ~5-10 minutes, results in `output_multi_dataset/`

### 3. Check Results

```bash
# View summary
cat output_complete_test/PIPELINE_SUMMARY.txt

# View best models
cat output_complete_test/results/cross_dataset_best_models.csv

# List all outputs
ls -R output_complete_test/results/
```

---

## Configuration

### Configuration File Structure

Create a YAML file (e.g., `my_config.yaml`):

```yaml
# Project Settings
project:
  name: "My Analysis"
  output_dir: "output_my_analysis"
  random_seed: 42
  n_jobs: -1  # Use all CPU cores

# Data Configuration
data:
  datasets:
    - path: "data/dataset1.csv"
      name: "Dataset 1"
      target_column: "target"

    - path: "data/dataset2.csv"
      name: "Dataset 2"
      target_column: "outcome"

  preprocessing:
    handle_missing: true        # Drop rows with missing values
    test_size: 0.3              # 70% train, 30% test
    smote: true                 # Apply SMOTE to training data
    smote_k_neighbors: 5        # SMOTE k_neighbors parameter

# Model Configuration
models:
  groups:
    - lr          # Logistic Regression (3 variants)
    - RF          # Random Forest (25 variants)
    - XGB         # XGBoost (6 variants)
    # Available: lr, lr_reg, adaboost, Bag-CART, BagNN,
    #            Boost-DT, RF, SGB, KNN, XGB, LGBM, DL

  benchmark:
    primary_metric: "AUC"       # Metric for best model selection
    secondary_metrics:          # Additional metrics to report
      - "F1"
      - "KS"

# Analysis Configuration
analysis:
  # Exploratory Data Analysis
  eda:
    enabled: true
    generate_profile: false     # Full profiling (slow)
    feature_importance: true    # RF + LR importance

  # SHAP Analysis
  shap:
    enabled: true
    global_plots: true
    plot_types:
      - bar                     # Bar plot of mean |SHAP|
      - dot                     # Beeswarm plot
    max_display: 20             # Top N features

  # Reliability Testing
  reliability:
    enabled: true
    n_permutations: 30          # Bootstrap iterations

  # Local Instance Analysis
  local_shap:
    enabled: true
    instances: []               # Specific row indices (empty = random)
    n_random: 5                 # Number of random instances

# Report Generation
report:
  latex:
    enabled: true
    compile_pdf: false          # Compile to PDF (requires pdflatex)
    include_narratives: false   # Add narrative text

# Logging
logging:
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  log_file: "pipeline.log"
  console_output: true
```

### Example Configurations

**Minimal (Fast Testing):**
```yaml
project:
  name: "Quick Test"
  output_dir: "output_quick"

data:
  datasets:
    - path: "data/Australian Credit.csv"
      name: "Australian Credit"
      target_column: "target"
  preprocessing:
    test_size: 0.3
    smote: true

models:
  groups: [lr]
  benchmark:
    primary_metric: "AUC"

analysis:
  eda: {enabled: true}
  shap: {enabled: false}
  reliability: {enabled: false}
  local_shap: {enabled: false}

report:
  latex: {enabled: false}

logging:
  level: "INFO"
```

**Production (Complete Analysis):**
```yaml
# Use multi_dataset_config.yaml as template
# Enable all analysis stages
# Use multiple model groups
# Set n_permutations: 100 for robust reliability testing
```

---

## Usage

### Basic Command

```bash
python model_pipeline.py --config <config_file.yaml>
```

### Command-Line Options

```bash
# Run specific stages
python model_pipeline.py --config my_config.yaml --stages data,eda,models

# Resume from checkpoint
python model_pipeline.py --config my_config.yaml --resume

# Dry run (validate config without execution)
python model_pipeline.py --config my_config.yaml --dry-run

# Verbose logging
python model_pipeline.py --config my_config.yaml --verbose
```

### Available Stages

| Stage | Description | Dependencies |
|-------|-------------|--------------|
| `init` | Initialize pipeline | None |
| `data` | Load and preprocess data | init |
| `eda` | Exploratory data analysis | data |
| `models` | Train and benchmark models | data |
| `shap` | SHAP analysis | models |
| `reliability` | Reliability testing | models |
| `local` | Local instance analysis | models |
| `report` | Generate reports | All previous |
| `finalize` | Finalize and cleanup | All previous |

### Stage Selection Examples

```bash
# Only data preprocessing and EDA
python model_pipeline.py --config my_config.yaml --stages data,eda

# Skip SHAP and local analysis
python model_pipeline.py --config my_config.yaml --stages init,data,eda,models,reliability,report,finalize

# Only model training (assuming data already loaded)
python model_pipeline.py --config my_config.yaml --stages models --resume
```

---

## Pipeline Stages

### Stage 1: INIT
- Validate configuration
- Check dependencies (pdflatex, libraries)
- Create output directories

### Stage 2: DATA
- Load CSV datasets
- Handle missing values (drop rows)
- Encode categorical features (LabelEncoder)
- Train/test split (stratified)
- Apply SMOTE to training data
- Save preprocessed data

**Outputs:**
- `data/preprocessed/<dataset>/train.csv`
- `data/preprocessed/<dataset>/test.csv`
- `data/preprocessed/<dataset>/metadata.json`

### Stage 3: EDA
- Generate summary statistics
- Analyze target distribution
- Compute feature importance (RF + LR)

**Outputs:**
- `results/eda/<dataset>_summary_stats.csv`
- `results/eda/<dataset>_target_distribution.csv`
- `results/eda/<dataset>_feature_importance_merged.csv`

### Stage 4: MODELS
- Train all models in selected groups
- Evaluate with 8 metrics (AUC, PCC, F1, Recall, BS, KS, PG, H)
- Benchmark and rank models
- Identify best model per dataset
- Save trained models in memory

**Outputs:**
- `results/model_training_summary.csv` (all models)
- `results/model_summary_<dataset>.csv` (per dataset)
- `results/cross_dataset_best_models.csv`
- `results/benchmarks/<dataset>_rankings.csv`
- `results/benchmarks/<dataset>_top10.csv`

### Stage 5: SHAP
- Compute SHAP values for best models
- Generate global SHAP plots (bar, beeswarm)
- Calculate SHAP-based feature importance

**Outputs:**
- `results/shap/<dataset>/<model>_shap_values.csv`
- `results/figures/shap/<dataset>/<model>_shap_bar.png`
- `results/figures/shap/<dataset>/<model>_shap_beeswarm.png`

### Stage 6: RELIABILITY
- Bootstrap resampling (n_permutations iterations)
- Compute mean AUC and standard deviation
- Calculate 95% confidence intervals

**Outputs:**
- `results/reliability_summary.csv`

### Stage 7: LOCAL
- Analyze specific instances or random sample
- Compute instance-level SHAP values
- Generate waterfall plots (if possible)
- Save detailed JSON analysis

**Outputs:**
- `results/local_analyses/<dataset>_row<N>.json`
- `results/figures/shap/<dataset>/<model>_waterfall_row<N>.png` (if generated)

### Stage 8: REPORT
- Generate LaTeX document
- Create text summary
- Optionally compile to PDF

**Outputs:**
- `PIPELINE_SUMMARY.txt`
- `results/report.tex` (if enabled)
- `results/report.pdf` (if compile_pdf: true)

### Stage 9: FINALIZE
- Generate final summary
- Save execution metadata
- Clean up temporary files

**Outputs:**
- `summary.txt`

---

## Output Structure

```
output_<project_name>/
│
├── config_used.yaml                    # Copy of configuration
├── PIPELINE_SUMMARY.txt                # Executive summary
├── summary.txt                         # Execution summary
├── pipeline.log                        # Full logs
│
├── data/preprocessed/                  # Preprocessed datasets
│   └── <dataset_name>/
│       ├── train.csv
│       ├── test.csv
│       └── metadata.json
│
├── results/
│   ├── model_training_summary.csv      # All models, all metrics
│   ├── model_summary_<dataset>.csv     # Per-dataset summary
│   ├── cross_dataset_best_models.csv   # Best model per dataset
│   ├── reliability_summary.csv         # Bootstrap results
│   │
│   ├── eda/                            # EDA outputs
│   │   ├── <dataset>_summary_stats.csv
│   │   ├── <dataset>_target_distribution.csv
│   │   ├── <dataset>_feature_importance_rf.csv
│   │   ├── <dataset>_feature_importance_lr.csv
│   │   └── <dataset>_feature_importance_merged.csv
│   │
│   ├── benchmarks/                     # Model rankings
│   │   ├── <dataset>_rankings.csv
│   │   └── <dataset>_top10.csv
│   │
│   ├── shap/<dataset>/                 # SHAP values
│   │   └── <model>_shap_values.csv
│   │
│   ├── local_analyses/                 # Instance analyses
│   │   ├── <dataset>_row<N>.json
│   │   └── ...
│   │
│   ├── figures/shap/<dataset>/         # SHAP plots
│   │   ├── <model>_shap_bar.png
│   │   ├── <model>_shap_beeswarm.png
│   │   └── <model>_waterfall_row<N>.png
│   │
│   └── models/<dataset>/<group>/       # Trained models (in memory)
│
└── checkpoints/                        # Resume checkpoints
    └── checkpoint_<timestamp>.txt
```

---

## Examples

### Example 1: Single Dataset, Quick Test

```bash
# Create config
cat > quick_test.yaml << EOF
project:
  name: "Quick Test"
  output_dir: "output_quick"

data:
  datasets:
    - path: "data/Australian Credit.csv"
      name: "Australian Credit"
      target_column: "target"
  preprocessing:
    test_size: 0.3
    smote: true

models:
  groups: [lr]
  benchmark:
    primary_metric: "AUC"

analysis:
  eda: {enabled: true, feature_importance: true}
  shap: {enabled: true, plot_types: [bar]}
  reliability: {enabled: true, n_permutations: 20}
  local_shap: {enabled: true, n_random: 2}

report:
  latex: {enabled: true, compile_pdf: false}

logging:
  level: "INFO"
EOF

# Run
python model_pipeline.py --config quick_test.yaml

# Check results
cat output_quick/PIPELINE_SUMMARY.txt
```

### Example 2: Multi-Dataset Production Run

```bash
# Use existing config
python model_pipeline.py --config multi_dataset_config.yaml

# Monitor progress
tail -f output_multi_dataset/pipeline.log

# After completion, check best models
cat output_multi_dataset/results/cross_dataset_best_models.csv
```

### Example 3: Resume After Interruption

```bash
# Start long-running pipeline
python model_pipeline.py --config multi_dataset_config.yaml

# Press Ctrl+C to interrupt after some stages

# Resume from last checkpoint
python model_pipeline.py --config multi_dataset_config.yaml --resume

# Check which stages completed
ls output_multi_dataset/checkpoints/
```

### Example 4: Custom Model Groups

```yaml
# custom_models.yaml
models:
  groups:
    - lr        # 3 Logistic Regression variants
    - RF        # 25 Random Forest variants
    - XGB       # 6 XGBoost variants
    - SGB       # 7 Stochastic Gradient Boosting variants
    - KNN       # 4 K-Nearest Neighbors variants
```

```bash
python model_pipeline.py --config custom_models.yaml
```

### Example 5: Stage-by-Stage Execution

```bash
# Phase 1: Data preparation
python model_pipeline.py --config my_config.yaml --stages init,data,eda

# Review EDA outputs
cat output_*/results/eda/*_feature_importance_merged.csv

# Phase 2: Model training
python model_pipeline.py --config my_config.yaml --stages models --resume

# Phase 3: Analysis
python model_pipeline.py --config my_config.yaml --stages shap,reliability,local --resume

# Phase 4: Reporting
python model_pipeline.py --config my_config.yaml --stages report,finalize --resume
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```
ImportError: No module named 'shap'
```

**Solution:**
```bash
pip install shap matplotlib
```

#### 2. Data File Not Found

```
Failed to load data/my_dataset.csv: File not found
```

**Solution:**
- Check file path in config (use absolute or relative to script location)
- Verify file exists: `ls data/my_dataset.csv`

#### 3. SHAP Errors with Pipelines

```
InvalidModelError: Model type not yet supported by TreeExplainer
```

**Solution:** Already fixed in current version. Update to latest code.

#### 4. Memory Issues with Large Datasets

```
MemoryError: Unable to allocate array
```

**Solutions:**
- Reduce `n_permutations` in reliability testing
- Use smaller `n_random` for local analysis
- Process fewer datasets at once
- Use fewer model variants

#### 5. Categorical Encoding Errors

```
ValueError: could not convert string to float
```

**Solution:** Already fixed in current version - automatic categorical encoding applied.

#### 6. pdflatex Not Found

```
Warning: pdflatex not found - PDF compilation will be skipped
```

**Solutions:**
- Install TeX Live (Linux/Mac): `sudo apt install texlive-full`
- Install MiKTeX (Windows): https://miktex.org/download
- Or set `compile_pdf: false` in config

### Debugging

**Enable verbose logging:**
```yaml
logging:
  level: "DEBUG"
```

**Check logs:**
```bash
tail -f output_*/pipeline.log
```

**Validate config:**
```bash
python model_pipeline.py --config my_config.yaml --dry-run
```

**Test with minimal config:**
```bash
python model_pipeline.py --config test_complete_config.yaml
```

---

## Performance Tips

### 1. Parallel Processing

```yaml
project:
  n_jobs: -1  # Use all CPU cores
```

### 2. Reduce Bootstrap Iterations

```yaml
analysis:
  reliability:
    n_permutations: 20  # Instead of 100
```

### 3. Disable Slow Stages

```yaml
analysis:
  eda:
    generate_profile: false  # Skip full profiling
  reliability:
    enabled: false           # Skip if not needed
  local_shap:
    n_random: 2              # Analyze fewer instances
```

### 4. Use Smaller Test Set

```yaml
data:
  preprocessing:
    test_size: 0.2  # Instead of 0.3
```

### 5. Select Fewer Models

```yaml
models:
  groups:
    - lr  # 3 models instead of RF (25 models)
```

---

## Advanced Usage

### Custom Metrics Priority

```yaml
models:
  benchmark:
    primary_metric: "F1"      # Use F1 instead of AUC
    secondary_metrics: ["AUC", "Recall", "KS"]
```

### Specific Instance Analysis

```yaml
analysis:
  local_shap:
    instances: [0, 5, 10, 42, 100]  # Specific row indices
```

### Multiple Datasets with Different Targets

```yaml
data:
  datasets:
    - path: "data/credit.csv"
      name: "Credit Risk"
      target_column: "default"

    - path: "data/churn.csv"
      name: "Customer Churn"
      target_column: "churned"
```

### Resume from Specific Stage

```bash
# Resume but skip to shap stage
python model_pipeline.py --config my_config.yaml --stages shap,reliability,local,report,finalize --resume
```

---

## Comparison with Streamlit

| Feature | Streamlit UI | CLI Pipeline |
|---------|-------------|--------------|
| **Interaction** | Web UI | Command line |
| **Speed** | Slower (UI overhead) | Faster (no UI) |
| **Automation** | Manual clicks | Fully automated |
| **Batch Processing** | One at a time | Multiple configs |
| **Scheduling** | Manual | Cron/scheduled |
| **Reproducibility** | Session-dependent | Config-driven |
| **Output** | results/ folder | Structured output_*/ |
| **Resume** | Not supported | Checkpoint resume |
| **Multi-dataset** | Sequential | Parallel |

---

## Support

### Getting Help

1. Check this README
2. Review example configs: `test_complete_config.yaml`, `multi_dataset_config.yaml`
3. Check logs: `output_*/pipeline.log`
4. Review test results: `output_complete_test/`

### Reporting Issues

When reporting issues, include:
1. Config file used
2. Command executed
3. Error message
4. Log file (`pipeline.log`)
5. Python version and package versions

---

## Version History

**v1.0.0** (Current)
- Complete 9-stage pipeline
- Multi-dataset support
- Automatic categorical encoding
- SHAP analysis with multiple plot types
- Bootstrap reliability testing
- Local instance analysis
- LaTeX report generation
- Checkpoint/resume capability
- Comprehensive logging

---

## License

[Your License Here]

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{model_selection_pipeline,
  title = {Model Selection Pipeline - CLI},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/model-selection-pipeline}
}
```

---

**Happy Modeling!** 🚀

For questions or issues, contact: [your-email@example.com]
