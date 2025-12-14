"""
Test script to verify auto-save functionality for dataset upload.
This simulates what happens in streamlit_app.py when a dataset is uploaded.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from result_manager import get_result_manager

# Create a sample dataset
np.random.seed(42)
sample_data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randint(0, 10, 100),
    'feature3': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.randint(0, 2, 100)
})

# Save to a temporary CSV file
test_csv_path = "test_dataset.csv"
sample_data.to_csv(test_csv_path, index=False)
print(f"Created test dataset: {test_csv_path}")

# Initialize ResultManager (this is what happens in streamlit_app.__init__)
result_mgr = get_result_manager("results")
print(f"ResultManager initialized")
print(f"EDA directory: {result_mgr.eda_dir}")
print(f"EDA directory exists: {result_mgr.eda_dir.exists()}")

# Simulate the auto-save operations from streamlit_app.py
dataset_name = "test_dataset.csv"

print("\n=== Testing Auto-Save Operations ===\n")

# 1. Save sample preview (first 5 rows)
try:
    df_head = pd.read_csv(test_csv_path, nrows=5)
    safe_name = result_mgr.sanitize_name(dataset_name)
    sample_path = result_mgr.eda_dir / f"{safe_name}_sample_preview.csv"
    df_head.to_csv(sample_path, index=False)
    print(f"[OK] Saved sample preview to: {sample_path}")
    print(f"  File exists: {sample_path.exists()}")
except Exception as e:
    print(f"[FAIL] Failed to save sample preview: {e}")

# 2. Save metadata (shape info)
try:
    df = pd.read_csv(test_csv_path)
    n_rows, n_cols = df.shape
    safe_name = result_mgr.sanitize_name(dataset_name)
    metadata = pd.DataFrame({
        'metric': ['rows', 'columns'],
        'value': [n_rows, n_cols]
    })
    metadata_path = result_mgr.eda_dir / f"{safe_name}_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    print(f"[OK] Saved metadata to: {metadata_path}")
    print(f"  File exists: {metadata_path.exists()}")
except Exception as e:
    print(f"[FAIL] Failed to save metadata: {e}")

# 3. Save column info
try:
    df = pd.read_csv(test_csv_path)
    n_rows = len(df)
    cols = df.columns.tolist()

    info_df = pd.DataFrame({
        "column": cols,
        "non_null": [df[c].notna().sum() for c in cols],
        "nulls": [df[c].isna().sum() for c in cols],
        "%_non_null": [(df[c].notna().sum() / n_rows * 100.0) for c in cols],
        "dtype": [str(df[c].dtype) for c in cols],
    })

    safe_name = result_mgr.sanitize_name(dataset_name)
    info_path = result_mgr.eda_dir / f"{safe_name}_column_info.csv"
    info_df.to_csv(info_path, index=False)
    print(f"[OK] Saved column info to: {info_path}")
    print(f"  File exists: {info_path.exists()}")

    # Save missing values summary using ResultManager method
    missing_summary = info_df[['column', 'nulls', '%_non_null']].copy()
    missing_summary.columns = ['column', 'missing_count', 'percent_non_null']
    result_mgr.save_eda_summary(
        dataset_name=dataset_name,
        summary_stats=None,
        target_distribution=None,
        correlation_matrix=None,
        missing_summary=missing_summary
    )
    missing_path = result_mgr.eda_dir / f"{safe_name}_missing_values.csv"
    print(f"[OK] Saved missing values to: {missing_path}")
    print(f"  File exists: {missing_path.exists()}")
except Exception as e:
    print(f"[FAIL] Failed to save column info: {e}")

# 4. Save describe() statistics
try:
    df = pd.read_csv(test_csv_path)
    describe_df = df.describe(include='number').round(6)

    result_mgr.save_eda_summary(
        dataset_name=dataset_name,
        summary_stats=describe_df,
        target_distribution=None,
        correlation_matrix=None,
        missing_summary=None
    )

    safe_name = result_mgr.sanitize_name(dataset_name)
    stats_path = result_mgr.eda_dir / f"{safe_name}_summary_stats.csv"
    print(f"[OK] Saved summary stats to: {stats_path}")
    print(f"  File exists: {stats_path.exists()}")
except Exception as e:
    print(f"[FAIL] Failed to save summary stats: {e}")

# 5. Save target distribution
try:
    df = pd.read_csv(test_csv_path)
    target_col = 'target'
    if target_col in df.columns:
        counts_series = df[target_col].value_counts()

        result_mgr.save_eda_summary(
            dataset_name=dataset_name,
            summary_stats=None,
            target_distribution=counts_series,
            correlation_matrix=None,
            missing_summary=None
        )

        safe_name = result_mgr.sanitize_name(dataset_name)
        target_path = result_mgr.eda_dir / f"{safe_name}_target_distribution.csv"
        print(f"[OK] Saved target distribution to: {target_path}")
        print(f"  File exists: {target_path.exists()}")
except Exception as e:
    print(f"[FAIL] Failed to save target distribution: {e}")

# Summary
print("\n=== Summary ===")
print(f"\nFiles in results/eda/:")
eda_files = list(result_mgr.eda_dir.glob("*"))
for f in sorted(eda_files):
    print(f"  - {f.name}")

print(f"\nTotal files created: {len(eda_files)}")

# Clean up
import os
os.remove(test_csv_path)
print(f"\nCleaned up test dataset: {test_csv_path}")
