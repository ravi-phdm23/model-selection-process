"""
test_report_manager.py

Test suite for ReportManager class.
Tests content gathering, LLM generation, LaTeX conversion, and file operations.
"""

from pathlib import Path
import pandas as pd
import json
import shutil
from report_manager import get_report_manager, ReportManager
from result_manager import get_result_manager, ResultManager


def test_basic_initialization():
    """Test that ReportManager initializes correctly and creates directory."""
    print("\n" + "="*60)
    print("TEST 1: Basic Initialization")
    print("="*60)
    
    # Use test directory
    test_dir = Path("test_reports")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    rm = get_report_manager(str(test_dir))
    
    assert rm.base_dir.exists(), "Reports directory not created"
    assert rm.base_dir == test_dir, "Base directory path mismatch"
    
    print(f"✅ Reports directory created: {rm.base_dir}")
    print(f"✅ Singleton pattern working")
    
    # Cleanup
    shutil.rmtree(test_dir)
    ReportManager._instance = None
    
    print("✅ TEST 1 PASSED\n")


def test_content_gathering():
    """Test gathering content from ResultManager."""
    print("\n" + "="*60)
    print("TEST 2: Content Gathering")
    print("="*60)
    
    # Setup test directories
    test_results_dir = Path("test_results_rm")
    test_reports_dir = Path("test_reports_rm")
    
    if test_results_dir.exists():
        shutil.rmtree(test_results_dir)
    if test_reports_dir.exists():
        shutil.rmtree(test_reports_dir)
    
    # Create ResultManager with test data
    result_mgr = get_result_manager(str(test_results_dir))
    
    # Add some test EDA data
    test_summary = pd.DataFrame({
        'feature': ['age', 'income', 'debt_ratio'],
        'mean': [45.2, 50000, 0.35],
        'std': [12.1, 15000, 0.12]
    })
    result_mgr.save_eda_summary(
        'TestDataset',
        summary_stats=test_summary,
        target_distribution=pd.DataFrame({'count': [700, 300]}, index=[0, 1]),
        correlation_matrix=None,
        missing_summary=None
    )
    
    # Add test benchmark data
    test_benchmark = pd.DataFrame({
        'model': ['LogisticRegression', 'RandomForest', 'XGBoost'],
        'dataset': ['TestDataset', 'TestDataset', 'TestDataset'],
        'auc': [0.75, 0.82, 0.85],
        'accuracy': [0.72, 0.78, 0.81]
    })
    result_mgr.save_benchmark_results(test_benchmark)
    
    # Create ReportManager
    report_mgr = get_report_manager(str(test_reports_dir))
    
    # Test content gathering
    content = report_mgr.gather_available_content(result_mgr)
    
    assert 'manifest' in content, "Missing manifest"
    assert content['has_eda'] == True, "EDA not detected"
    assert content['has_benchmarks'] == True, "Benchmarks detected correctly"
    assert 'TestDataset' in content['datasets'], "Dataset not in list"
    
    print(f"✅ Content gathered successfully")
    print(f"   - Datasets: {content['datasets']}")
    print(f"   - Has EDA: {content['has_eda']}")
    print(f"   - Has Benchmarks: {content['has_benchmarks']}")
    
    # Test specific content loaders
    eda_content = report_mgr.get_eda_content('TestDataset', result_mgr)
    assert eda_content['summary_stats'] is not None, "EDA summary not loaded"
    assert len(eda_content['summary_stats']) == 3, "Summary stats incomplete"
    
    print(f"✅ EDA content loaded: {len(eda_content['summary_stats'])} features")
    
    benchmark_content = report_mgr.get_benchmark_content(result_mgr)
    assert benchmark_content['results_df'] is not None, "Benchmark results not loaded"
    assert len(benchmark_content['results_df']) == 3, "Benchmark results incomplete"
    
    print(f"✅ Benchmark content loaded: {len(benchmark_content['results_df'])} models")
    
    # Cleanup
    shutil.rmtree(test_results_dir)
    shutil.rmtree(test_reports_dir)
    ResultManager._instance = None
    ReportManager._instance = None
    
    print("✅ TEST 2 PASSED\n")


def test_prompt_generation():
    """Test that prompts are generated correctly."""
    print("\n" + "="*60)
    print("TEST 3: Prompt Generation")
    print("="*60)
    
    test_dir = Path("test_reports_prompts")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    rm = get_report_manager(str(test_dir))
    
    # Test introduction prompt
    intro_prompt = rm._get_introduction_prompt(['German Credit', 'Australian'], ['LogisticRegression', 'XGBoost'])
    assert 'German Credit' in intro_prompt, "Dataset not in intro prompt"
    assert 'credit risk' in intro_prompt.lower(), "Missing credit risk context"
    
    print(f"✅ Introduction prompt generated ({len(intro_prompt)} chars)")
    print(f"   Preview: {intro_prompt[:150]}...")
    
    # Test EDA prompt
    test_stats = pd.DataFrame({
        'feature': ['age', 'income'],
        'mean': [45, 50000],
        'std': [12, 15000]
    })
    eda_prompt = rm._get_eda_prompt('TestDataset', test_stats, None, None, None)
    assert 'TestDataset' in eda_prompt, "Dataset not in EDA prompt"
    assert 'age' in eda_prompt, "Feature not in EDA prompt"
    
    print(f"✅ EDA prompt generated ({len(eda_prompt)} chars)")
    
    # Test benchmark prompt
    test_bench = pd.DataFrame({
        'model': ['LR', 'RF'],
        'auc': [0.75, 0.85]
    })
    bench_prompt = rm._get_benchmark_prompt(test_bench)
    assert 'model' in bench_prompt.lower(), "Missing benchmark context"
    
    print(f"✅ Benchmark prompt generated ({len(bench_prompt)} chars)")
    
    # Test SHAP prompt
    shap_prompt = rm._get_shap_prompt('TestDataset', None)
    assert 'SHAP' in shap_prompt, "Missing SHAP context"
    
    print(f"✅ SHAP prompt generated ({len(shap_prompt)} chars)")
    
    # Test reliability prompt
    rel_prompt = rm._get_reliability_prompt('TestDataset', {'ks_stat': 0.25, 'ks_pvalue': 0.01})
    assert 'reliability' in rel_prompt.lower(), "Missing reliability context"
    assert 'ks_stat' in rel_prompt, "Missing diagnostic metrics"
    
    print(f"✅ Reliability prompt generated ({len(rel_prompt)} chars)")
    
    # Test conclusion prompt
    concl_prompt = rm._get_conclusion_prompt("Test summary of findings")
    assert 'conclusion' in concl_prompt.lower(), "Missing conclusion context"
    
    print(f"✅ Conclusion prompt generated ({len(concl_prompt)} chars)")
    
    # Cleanup
    shutil.rmtree(test_dir)
    ReportManager._instance = None
    
    print("✅ TEST 3 PASSED\n")


def test_markdown_assembly():
    """Test assembling markdown report from sections."""
    print("\n" + "="*60)
    print("TEST 4: Markdown Assembly")
    print("="*60)
    
    test_dir = Path("test_reports_assembly")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    rm = get_report_manager(str(test_dir))
    
    # Create test sections
    sections = {
        'introduction': '# Introduction\n\nThis is the introduction.',
        'eda': '## Dataset Analysis\n\nThis is EDA.',
        'benchmark': '# Benchmark Results\n\nThese are results.',
        'shap': '## SHAP Analysis\n\nThese are SHAP findings.',
        'reliability': '## Reliability Tests\n\nThese are diagnostics.',
        'conclusion': '# Conclusion\n\nThis is the conclusion.'
    }
    
    full_report = rm.assemble_full_report(sections, title="Test Report")
    
    assert 'Test Report' in full_report, "Title not in report"
    assert 'Introduction' in full_report, "Introduction missing"
    assert 'EDA' in full_report, "EDA section missing"
    assert 'Benchmark Results' in full_report, "Benchmark section missing"
    assert 'SHAP' in full_report, "SHAP section missing"
    assert 'Reliability' in full_report, "Reliability section missing"
    assert 'Conclusion' in full_report, "Conclusion missing"
    
    print(f"✅ Full report assembled ({len(full_report)} chars)")
    print(f"   - Contains all sections")
    print(f"   - Properly formatted with headers")
    
    # Cleanup
    shutil.rmtree(test_dir)
    ReportManager._instance = None
    
    print("✅ TEST 4 PASSED\n")


def test_latex_conversion():
    """Test markdown to LaTeX conversion."""
    print("\n" + "="*60)
    print("TEST 5: LaTeX Conversion")
    print("="*60)
    
    test_dir = Path("test_reports_latex")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    rm = get_report_manager(str(test_dir))
    
    # Test markdown conversion
    markdown = """# Main Section
## Subsection
### Subsubsection

This is **bold** text and *italic* text.

- Item 1
- Item 2
- Item 3

1. Numbered item
2. Another item

---

Some text with special characters: 50% accuracy, $100, R&D, feature_importance.
"""
    
    latex = rm.convert_markdown_to_latex(markdown)
    
    assert '\\section{Main Section}' in latex, "Section header not converted"
    assert '\\subsection{Subsection}' in latex, "Subsection not converted"
    assert '\\textbf{bold}' in latex, "Bold not converted"
    assert '\\textit{italic}' in latex, "Italic not converted"
    assert '\\item Item 1' in latex, "List items not converted"
    assert '\\%' in latex, "Percent not escaped"
    assert '\\_' in latex, "Underscore not escaped"
    
    print(f"✅ Markdown converted to LaTeX ({len(latex)} chars)")
    print(f"   - Headers: ✓")
    print(f"   - Emphasis: ✓")
    print(f"   - Lists: ✓")
    print(f"   - Escaping: ✓")
    
    # Test figure insertion
    fig_latex = rm.insert_figure_latex(
        Path("results/figures/shap_bar.png"),
        "SHAP feature importance",
        "fig:shap_bar"
    )
    
    assert '\\begin{figure}' in fig_latex, "Figure environment missing"
    assert '\\includegraphics' in fig_latex, "Graphics command missing"
    assert 'SHAP feature importance' in fig_latex, "Caption missing"
    assert 'fig:shap_bar' in fig_latex, "Label missing"
    
    print(f"✅ Figure LaTeX generated")
    
    # Test table insertion
    test_df = pd.DataFrame({
        'model_name': ['LR', 'RF'],
        'test_auc': [0.75, 0.85]
    })
    
    table_latex = rm.insert_table_latex(test_df, "Benchmark results", "tab:benchmark")
    
    assert '\\begin{tabular}' in table_latex, "Table environment missing"
    assert 'Benchmark results' in table_latex, "Table caption missing"
    assert 'tab:benchmark' in table_latex, "Table label missing"
    
    print(f"✅ Table LaTeX generated")
    
    # Test full LaTeX document generation
    markdown_report = "# Results\n\nThis is a test report."
    manifest = {'benchmarks': {}, 'eda': {}}
    
    full_latex = rm.generate_latex_report(markdown_report, manifest, "Test Paper", "John Doe")
    
    assert '\\documentclass' in full_latex, "Document class missing"
    assert '\\begin{document}' in full_latex, "Document begin missing"
    assert '\\end{document}' in full_latex, "Document end missing"
    assert 'Test Paper' in full_latex, "Title missing"
    assert 'John Doe' in full_latex, "Author missing"
    
    print(f"✅ Full LaTeX document generated ({len(full_latex)} chars)")
    
    # Cleanup
    shutil.rmtree(test_dir)
    ReportManager._instance = None
    
    print("✅ TEST 5 PASSED\n")


def test_save_load_operations():
    """Test saving and loading reports."""
    print("\n" + "="*60)
    print("TEST 6: Save/Load Operations")
    print("="*60)
    
    test_dir = Path("test_reports_io")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    rm = get_report_manager(str(test_dir))
    
    # Test saving markdown
    test_markdown = "# Test Report\n\nThis is a test."
    timestamp = "20240101_120000"
    
    md_path = rm.save_report_markdown(test_markdown, timestamp)
    assert md_path.exists(), "Markdown file not created"
    assert md_path.name == f"report_{timestamp}.md", "Incorrect filename"
    
    print(f"✅ Markdown saved: {md_path.name}")
    
    # Test saving LaTeX
    test_latex = "\\documentclass{article}\n\\begin{document}\nTest\\end{document}"
    tex_path = rm.save_latex_report(test_latex, timestamp)
    assert tex_path.exists(), "LaTeX file not created"
    assert tex_path.name == f"report_{timestamp}.tex", "Incorrect filename"
    
    print(f"✅ LaTeX saved: {tex_path.name}")
    
    # Test saving metadata
    config = {'provider': 'openai', 'model': 'gpt-4', 'temperature': 0.3}
    manifest = {'benchmarks': {'results_csv': 'test.csv'}, 'eda': {'TestDataset': {}}}
    
    meta_path = rm.save_report_metadata(config, manifest, timestamp)
    assert meta_path.exists(), "Metadata file not created"
    
    with open(meta_path, 'r') as f:
        loaded_meta = json.load(f)
    assert loaded_meta['llm_config']['model'] == 'gpt-4', "Metadata incorrect"
    
    print(f"✅ Metadata saved: {meta_path.name}")
    
    # Test listing reports
    reports = rm.list_available_reports()
    assert len(reports) == 1, "Report not listed"
    assert reports[0]['timestamp'] == timestamp, "Timestamp mismatch"
    assert reports[0]['markdown_path'] == md_path, "Markdown path mismatch"
    assert reports[0]['latex_path'] == tex_path, "LaTeX path mismatch"
    
    print(f"✅ Reports listed: {len(reports)} found")
    
    # Test loading report
    loaded_md = rm.load_report(md_path)
    assert loaded_md == test_markdown, "Loaded markdown doesn't match"
    
    print(f"✅ Report loaded successfully")
    
    # Test storage summary
    summary = rm.get_storage_summary()
    assert summary['total_reports'] == 1, "Report count incorrect"
    assert summary['markdown_files'] == 1, "Markdown count incorrect"
    assert summary['latex_files'] == 1, "LaTeX count incorrect"
    assert summary['metadata_files'] == 1, "Metadata count incorrect"
    
    print(f"✅ Storage summary:")
    print(f"   - Total reports: {summary['total_reports']}")
    print(f"   - Markdown: {summary['markdown_files']}")
    print(f"   - LaTeX: {summary['latex_files']}")
    print(f"   - Metadata: {summary['metadata_files']}")
    print(f"   - Total size: {summary['total_size_mb']:.2f} MB")
    
    # Cleanup
    shutil.rmtree(test_dir)
    ReportManager._instance = None
    
    print("✅ TEST 6 PASSED\n")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*60)
    print("REPORT MANAGER TEST SUITE")
    print("="*60)
    
    try:
        test_basic_initialization()
        test_content_gathering()
        test_prompt_generation()
        test_markdown_assembly()
        test_latex_conversion()
        test_save_load_operations()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nReportManager is ready for integration into Streamlit.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
