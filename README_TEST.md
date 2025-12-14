# Automated Test Workflow

This directory contains an automated test workflow that runs the entire model selection and SHAP analysis pipeline.

## Quick Start

1. **Edit the configuration file** (`test_config.yaml`):
   ```yaml
   datasets:
     - path: "path/to/your/dataset.csv"
       name: "your_dataset.csv"
   
   target_column: "target"  # Your target column name
   
   local_analysis:
     dataset: "your_dataset.csv"
     row_index: 0  # Row to analyze
   ```

2. **Run the workflow**:
   ```bash
   python test_workflow.py
   ```

3. **Check results** in the `results/` directory

## Configuration File (`test_config.yaml`)

### Dataset Configuration
```yaml
datasets:
  - path: "data/sample_dataset.csv"
    name: "sample_dataset.csv"
  - path: "data/another_dataset.csv"
    name: "another_dataset.csv"

target_column: "target"
```

### Model Selection
```yaml
model_groups:
  - "Logistic Regression"
  - "Random Forest"
  - "XGBoost"
  - "LightGBM"
```

Available model groups:
- Logistic Regression
- Random Forest
- Bagging
- Boosting
- Gradient Boosting
- KNN
- Neural Network
- XGBoost
- LightGBM

### Global SHAP Settings
```yaml
global_shap:
  enabled: true
  use_stable_shap: true
  stable_shap_trials: 10
  stable_shap_bg_size: 200
  stable_shap_explain_size: 200
```

### Reliability Test Settings
```yaml
reliability_test:
  enabled: true
  preset: "Balanced"  # Quick, Balanced, or Thorough
  n_trials: 10
  n_bg: 200
```

Presets:
- **Quick**: 3 trials, 50 background samples (~1-2 min)
- **Balanced**: 10 trials, 200 background samples (~5-10 min)
- **Thorough**: 30 trials, 500 background samples (~30-60 min)

### Local SHAP Analysis
```yaml
local_analysis:
  enabled: true
  dataset: "sample_dataset.csv"  # Must match dataset name
  row_index: 0  # Row to analyze (0-based index)
```

## Workflow Steps

The workflow runs the following steps automatically:

1. **Step 1: Load Datasets**
   - Loads CSV files specified in config
   - Validates target column
   - Shows class distribution

2. **Step 2: Select Models**
   - Selects model groups from config
   - Lists all models to be trained

3. **Step 3: Run Experiment**
   - Trains all selected models
   - Performs train-test split (80/20)
   - Calculates metrics (AUC, F1, Recall, etc.)

4. **Step 4: Benchmark Analysis**
   - Identifies best model per group
   - Creates benchmark results table
   - Saves to `results/benchmark_results.csv`

5. **Step 5: Global SHAP Analysis**
   - Generates SHAP importance for benchmark models
   - Calculates rank stability metrics (if enabled)
   - Saves to `results/shap_stability_*.csv`

6. **Step 5.5: Reliability Test**
   - Runs rank stability analysis
   - Performs randomization sanity check
   - Computes sanity ratio
   - Saves to `results/reliability_*.csv` and `results/reliability_*.txt`

7. **Step 6: Local SHAP Analysis**
   - Analyzes specified row
   - Generates waterfall plot
   - Creates AI explanation with reliability metrics
   - Saves to `results/explanation_*.txt` and `results/figures/shap_*_waterfall_*.png`

## Output Files

All outputs are saved to the `results/` directory:

### CSV Files
- `benchmark_results.csv` - Benchmark model performance
- `shap_stability_*.csv` - Global SHAP with rank stability
- `reliability_*.csv` - Reliability test results

### Text Files
- `reliability_*.txt` - Reliability test summary
- `explanation_*_row*.txt` - AI explanation for analyzed row

### Figures
- `figures/shap_*_waterfall_row*.png` - SHAP waterfall plots

## Advanced Usage

### Custom Configuration File
```bash
python test_workflow.py --config my_custom_config.yaml
```

### Running Individual Steps

You can also import and use the `WorkflowRunner` class directly:

```python
from test_workflow import WorkflowRunner

runner = WorkflowRunner(config_path="test_config.yaml")
runner.step_1_load_datasets()
runner.step_2_select_models()
runner.step_3_run_experiment()
# ... etc
```

## Requirements

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- shap
- matplotlib
- pyyaml
- openai
- python-dotenv

## OpenAI API Key

The workflow requires an OpenAI API key for AI-generated explanations in Step 6.

Create a `.env` file with:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL_NAME=gpt-4o-mini
```

## Troubleshooting

### "Dataset not found"
- Check that the `path` in `test_config.yaml` is correct
- Use absolute paths or paths relative to the script location

### "Target column not found"
- Verify the `target_column` name matches your CSV exactly
- Check for typos or extra spaces

### "No benchmark model found"
- Ensure Step 4 completed successfully
- Check that models were trained without errors

### "Reliability metrics unavailable"
- Make sure Step 5.5 is enabled and completed
- Check that the dataset name matches exactly

### Long execution time
- Reduce `n_trials` in reliability test
- Use fewer model groups
- Disable Global SHAP or Reliability Test if not needed

## Example Configuration for Quick Testing

```yaml
datasets:
  - path: "data/sample.csv"
    name: "sample.csv"

target_column: "target"

model_groups:
  - "Logistic Regression"
  - "Random Forest"

global_shap:
  enabled: true
  use_stable_shap: false  # Faster

reliability_test:
  enabled: true
  preset: "Quick"
  n_trials: 3
  n_bg: 50

local_analysis:
  enabled: true
  dataset: "sample.csv"
  row_index: 0
```

This configuration will run in ~2-5 minutes depending on dataset size.
