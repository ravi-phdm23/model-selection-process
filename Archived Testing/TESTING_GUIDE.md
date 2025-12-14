# Pipeline Testing Guide

Complete guide for testing the Model Selection Pipeline with different configurations.

## Quick Reference

### Test Configurations Available

| Config File | Datasets | Models | Output Directory | Time |
|-------------|----------|--------|------------------|------|
| `test_single_australian.yaml` | 1 (Australian) | LR, RF | `test_results/single_australian/` | ~1-2 min |
| `test_single_german.yaml` | 1 (German) | LR, RF | `test_results/single_german/` | ~1-2 min |
| `test_multi_all.yaml` | 3 (All) | LR, RF, XGB | `test_results/multi_all_datasets/` | ~8-12 min |
| `test_complete_config.yaml` | 1 (Australian) | LR only | `output_complete_test/` | ~1 min |
| `multi_dataset_config.yaml` | 3 (All) | LR, RF, XGB | `output_multi_dataset/` | ~8-12 min |

---

## Testing Scenarios

### Scenario 1: Single Dataset - Quick Test

**Purpose:** Verify pipeline works end-to-end

```bash
# Test with Australian Credit
python model_pipeline.py --config test_single_australian.yaml

# Test with German Credit (includes categorical features)
python model_pipeline.py --config test_single_german.yaml
```

**Expected Results:**
- Training summary: 2 model groups × ~3-5 models each
- SHAP analysis completed
- Reliability testing with confidence intervals
- LaTeX + PDF reports generated
- Output in `test_results/single_*/`

---

### Scenario 2: Multi-Dataset Test

**Purpose:** Test parallel processing of multiple datasets

```bash
# Test with all 3 datasets
python model_pipeline.py --config test_multi_all.yaml
```

**Expected Results:**
- Training summary: 3 datasets × 3 groups × ~34 models = ~102 models
- Best model identified per dataset
- Cross-dataset comparison tables
- SHAP analysis for each dataset
- Output in `test_results/multi_all_datasets/`

---

### Scenario 3: Partial Pipeline Testing

**Purpose:** Test specific stages without running the entire pipeline

```bash
# Test only data loading and EDA
python model_pipeline.py --config test_single_australian.yaml --stages data,eda

# Test only model training
python model_pipeline.py --config test_single_australian.yaml --stages models --resume

# Test only report generation
python model_pipeline.py --config test_multi_all.yaml --stages report --resume
```

---

### Scenario 4: Resume After Interruption

**Purpose:** Continue from where the pipeline stopped

```bash
# Run pipeline (may be interrupted)
python model_pipeline.py --config test_multi_all.yaml

# Resume from last checkpoint
python model_pipeline.py --config test_multi_all.yaml --resume
```

---

## Creating Custom Output Directories

### Method 1: Edit YAML Config

Edit any config file and change the `output_dir`:

```yaml
project:
  name: "My Custom Test"
  output_dir: "my_results/experiment_001"  # Custom path
  random_seed: 42
```

### Method 2: Use Python to Generate Config

```bash
# Create a new config with custom output directory
python -c "
import yaml

# Load existing config
with open('test_single_australian.yaml') as f:
    config = yaml.safe_load(f)

# Modify output directory
config['project']['output_dir'] = 'my_experiments/test_001'
config['project']['name'] = 'Custom Experiment 001'

# Save new config
with open('my_custom_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

# Run with custom config
python model_pipeline.py --config my_custom_config.yaml
```

### Method 3: Organize by Date/Experiment

```yaml
project:
  output_dir: "experiments/2025-12-09/run_001"
  # or
  output_dir: "results/baseline_models"
  # or
  output_dir: "comparison/tuned_vs_default"
```

---

## Output Directory Structure

Every output directory contains:

```
test_results/single_australian/
├── checkpoints/              # Resume points
├── config_used.yaml         # Copy of config used
├── data/                    # Preprocessed data
├── eda/                     # EDA outputs
├── local_analyses/          # Instance-level SHAP
├── logs/                    # Execution logs
├── models/                  # Trained model metadata
├── pipeline.log            # Main log file
├── reliability/            # Bootstrap test results
├── reports/
│   ├── report_*.tex        # LaTeX source
│   └── report_*.pdf        # Compiled PDF
├── results/                # CSV summaries
│   ├── model_training_summary.csv
│   ├── cross_dataset_best_models.csv
│   └── reliability_summary.csv
├── shap/                   # SHAP plots and values
├── PIPELINE_SUMMARY.txt   # High-level summary
└── summary.txt            # Stage completion status
```

---

## Verification Steps

After running a test, verify the results:

### 1. Check Pipeline Completion
```bash
cat test_results/single_australian/summary.txt
```

Should show all stages marked as `[OK]`

### 2. Check Best Models
```bash
cat test_results/single_australian/results/cross_dataset_best_models.csv
```

### 3. View LaTeX Report
```bash
# Open PDF
start test_results/single_australian/reports/report_*.pdf  # Windows

# Or view LaTeX source
cat test_results/single_australian/reports/report_*.tex
```

### 4. Check SHAP Plots
```bash
ls test_results/single_australian/shap/
# Should contain: *_shap_bar.png, *_shap_beeswarm.png, *_shap_values.csv
```

### 5. Check Logs for Errors
```bash
grep -i "error\|warning" test_results/single_australian/pipeline.log
```

---

## Performance Benchmarks

Based on typical hardware (4-8 cores, 16GB RAM):

| Configuration | Datasets | Models | Time | Disk Space |
|--------------|----------|--------|------|------------|
| Single (LR only) | 1 | 3 | ~1 min | ~5 MB |
| Single (LR+RF) | 1 | 20 | ~2 min | ~10 MB |
| Multi (3 datasets) | 3 | 102 | ~10 min | ~30 MB |

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution:** Check data paths in YAML config. Paths are relative to project root.

```yaml
data:
  datasets:
    - path: "data/Australian Credit.csv"  # Correct
    # NOT: "C:/full/path/to/data.csv"
```

### Issue: "PDF compilation failed"
**Solution:** Install pdflatex or disable PDF:

```yaml
report:
  latex:
    compile_pdf: false  # LaTeX only, no PDF
```

### Issue: "Out of memory"
**Solution:** Reduce workload:

```yaml
models:
  groups:
    - lr  # Start with just LR

analysis:
  reliability:
    n_permutations: 10  # Reduce from 30
  local_shap:
    n_random: 1  # Reduce from 3
```

### Issue: "Takes too long"
**Solution:** Test specific stages:

```bash
# Run only data + models, skip SHAP
python model_pipeline.py --config test_single_australian.yaml --stages data,eda,models
```

---

## Clean Up Test Results

```bash
# Remove all test results
rm -rf test_results/

# Remove specific test
rm -rf test_results/single_australian/

# Keep results but remove large files
rm -rf test_results/*/models/
rm -rf test_results/*/shap/
```

---

## Advanced: Batch Testing

Test multiple configurations in sequence:

```bash
# Create a test script
cat > run_all_tests.sh << 'EOF'
#!/bin/bash

echo "Running Single Dataset Tests..."
python model_pipeline.py --config test_single_australian.yaml
python model_pipeline.py --config test_single_german.yaml

echo "Running Multi-Dataset Test..."
python model_pipeline.py --config test_multi_all.yaml

echo "All tests complete!"
ls -lh test_results/*/reports/*.pdf
EOF

chmod +x run_all_tests.sh
./run_all_tests.sh
```

---

## Next Steps

1. **Start Simple:** Run `test_single_australian.yaml` first
2. **Add Complexity:** Try `test_single_german.yaml` to test categorical features
3. **Full Test:** Run `test_multi_all.yaml` for comprehensive test
4. **Customize:** Create your own config with your datasets

For more details, see `README_CLI.md` for complete documentation.
