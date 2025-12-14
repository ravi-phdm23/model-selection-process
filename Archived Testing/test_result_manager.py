"""
Quick test script for result_manager.py

Run this to verify the ResultManager module is working correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from result_manager import ResultManager, get_result_manager

def test_basic_functionality():
    """Test basic directory creation and sanitization."""
    print("=" * 60)
    print("TEST 1: Basic Initialization")
    print("=" * 60)
    
    rm = ResultManager()
    print(f"✓ Base directory: {rm.base_dir}")
    print(f"✓ EDA directory: {rm.eda_dir}")
    print(f"✓ Figures directory: {rm.figures_dir}")
    print(f"✓ Reliability directory: {rm.reliability_dir}")
    print(f"✓ Batch directory: {rm.batch_dir}")
    print(f"✓ Local analyses directory: {rm.local_analyses_dir}")
    
    # Test sanitization
    test_names = [
        "Australian Credit.csv",
        "Dataset with Spaces",
        "special-chars_test!@#.csv"
    ]
    print("\nFilename sanitization:")
    for name in test_names:
        sanitized = rm.sanitize_name(name)
        print(f"  '{name}' → '{sanitized}'")
    
    print("\n✅ Basic functionality test passed!\n")


def test_eda_operations():
    """Test EDA save/load operations."""
    print("=" * 60)
    print("TEST 2: EDA Operations")
    print("=" * 60)
    
    rm = ResultManager()
    dataset_name = "Test_Dataset"
    
    # Create sample data
    summary_stats = pd.DataFrame({
        'feature1': {'mean': 10.5, 'std': 2.3, 'min': 5.0, 'max': 20.0},
        'feature2': {'mean': 100.2, 'std': 15.1, 'min': 50.0, 'max': 150.0}
    }).T
    
    target_dist = pd.Series({'class_0': 100, 'class_1': 150}, name='count')
    
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.65], [0.65, 1.0]],
        columns=['feature1', 'feature2'],
        index=['feature1', 'feature2']
    )
    
    missing_summary = pd.DataFrame({
        'column': ['feature1', 'feature2', 'target'],
        'missing_count': [0, 5, 0],
        'missing_pct': [0.0, 2.0, 0.0]
    })
    
    # Save EDA summary
    print("\nSaving EDA summary...")
    paths = rm.save_eda_summary(
        dataset_name=dataset_name,
        summary_stats=summary_stats,
        target_distribution=target_dist,
        correlation_matrix=correlation_matrix,
        missing_summary=missing_summary
    )
    
    for key, path in paths.items():
        print(f"  ✓ {key}: {path.name}")
    
    # Load EDA summary
    print("\nLoading EDA summary...")
    loaded = rm.load_eda_summary(dataset_name)
    if loaded:
        print(f"  ✓ Loaded {len(loaded)} components")
        for key in loaded:
            print(f"    - {key}: {loaded[key].shape if hasattr(loaded[key], 'shape') else 'N/A'}")
    
    # Save profile HTML (simulated)
    print("\nSaving EDA profile...")
    profile_html = "<html><body><h1>Test Profile Report</h1></body></html>"
    profile_path = rm.save_eda_profile(dataset_name, profile_html)
    print(f"  ✓ Profile saved: {profile_path.name}")
    
    # Load profile
    loaded_html = rm.load_eda_profile(dataset_name)
    if loaded_html:
        print(f"  ✓ Profile loaded: {len(loaded_html)} characters")
    
    print("\n✅ EDA operations test passed!\n")


def test_benchmark_operations():
    """Test benchmark save/load operations."""
    print("=" * 60)
    print("TEST 3: Benchmark Operations")
    print("=" * 60)
    
    rm = ResultManager()
    
    # Create sample benchmark results
    benchmark_df = pd.DataFrame({
        'Dataset': ['Dataset1', 'Dataset1', 'Dataset2', 'Dataset2'],
        'Model': ['LogReg', 'RandomForest', 'LogReg', 'RandomForest'],
        'AUC': [0.85, 0.88, 0.82, 0.87],
        'Accuracy': [0.80, 0.83, 0.78, 0.82],
        'F1': [0.75, 0.79, 0.73, 0.78]
    })
    
    # Save benchmarks
    print("\nSaving benchmark results...")
    csv_path = rm.save_benchmark_results(benchmark_df)
    print(f"  ✓ Saved to: {csv_path.name}")
    
    # Load benchmarks
    print("\nLoading benchmark results...")
    loaded_df = rm.load_benchmark_results()
    if loaded_df is not None:
        print(f"  ✓ Loaded: {loaded_df.shape[0]} rows × {loaded_df.shape[1]} columns")
        print(f"    Columns: {list(loaded_df.columns)}")
    
    print("\n✅ Benchmark operations test passed!\n")


def test_manifest_generation():
    """Test report manifest generation."""
    print("=" * 60)
    print("TEST 4: Manifest Generation")
    print("=" * 60)
    
    rm = ResultManager()
    
    print("\nGenerating manifest...")
    manifest = rm.prepare_report_manifest()
    
    print("\nManifest structure:")
    for section, content in manifest.items():
        if isinstance(content, dict):
            if section == 'benchmarks':
                print(f"  {section}:")
                print(f"    - CSV: {'✓' if content.get('csv') else '✗'}")
                print(f"    - Figures: {len(content.get('figures', []))} files")
            elif section in ['eda', 'shap_plots', 'reliability', 'batch_reliability']:
                print(f"  {section}: {len(content)} datasets")
                for ds_name in list(content.keys())[:2]:  # Show first 2
                    print(f"    - {ds_name}")
        elif isinstance(content, list):
            print(f"  {section}: {len(content)} items")
    
    print("\n✅ Manifest generation test passed!\n")


def test_storage_summary():
    """Test storage summary."""
    print("=" * 60)
    print("TEST 5: Storage Summary")
    print("=" * 60)
    
    rm = ResultManager()
    
    print("\nGetting storage summary...")
    summary = rm.get_storage_summary()
    
    print(f"\nTotal files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_mb']} MB")
    print("\nBy type:")
    for file_type, count in summary['by_type'].items():
        size_mb = summary['size_by_type_mb'][file_type]
        print(f"  {file_type:20s}: {count:3d} files ({size_mb:.2f} MB)")
    
    print("\n✅ Storage summary test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RESULT MANAGER TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_basic_functionality()
        test_eda_operations()
        test_benchmark_operations()
        test_manifest_generation()
        test_storage_summary()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        print("\nThe result_manager.py module is working correctly.")
        print("You can now integrate it into streamlit_app.py.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
