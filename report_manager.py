"""
report_manager.py

Manages automated research report generation from analysis results using LLMs.
Creates both markdown and LaTeX versions of comprehensive research reports.

Key Features:
- Gathers content from results/ folder using ResultManager
- Generates narrative sections using LLM (OpenAI, Anthropic, etc.)
- Converts markdown to LaTeX with proper formatting
- Manages report versioning and metadata

Directory Structure:
    reports/
    ├── report_YYYYMMDD_HHMMSS.md       (markdown version)
    ├── report_YYYYMMDD_HHMMSS.tex      (LaTeX version)
    └── metadata_YYYYMMDD_HHMMSS.json   (generation config)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import json
from datetime import datetime
import re

# Optional LLM clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


class ReportManager:
    """
    Singleton class for managing research report generation.
    Interacts with ResultManager to gather analysis outputs and generate
    comprehensive research reports using LLMs.
    """
    
    _instance = None
    
    def __new__(cls, base_dir: str = "reports"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: str = "reports"):
        if self._initialized:
            return
        
        self.base_dir = Path(base_dir)
        self.ensure_directory_structure()
        self._initialized = True
    
    def ensure_directory_structure(self) -> None:
        """Create reports directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def sanitize_name(name: str) -> str:
        """Convert dataset/model names to safe filenames."""
        return re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
    
    # ==================== CONTENT GATHERING ====================
    
    def gather_available_content(self, result_manager) -> Dict[str, Any]:
        """
        Scan results/ folder and gather all available analysis content.
        
        Args:
            result_manager: Instance of ResultManager
            
        Returns:
            Dict with keys:
                - manifest: Complete result manifest
                - datasets: List of dataset names
                - has_benchmarks: bool
                - has_eda: bool
                - has_shap: bool
                - has_reliability: bool
        """
        manifest = result_manager.prepare_report_manifest()
        
        content = {
            'manifest': manifest,
            'datasets': list(manifest.get('eda', {}).keys()),
            'has_benchmarks': bool(manifest.get('benchmarks', {}).get('csv')),
            'has_eda': bool(manifest.get('eda')),
            'has_shap': bool(manifest.get('shap_plots')),
            'has_reliability': bool(manifest.get('reliability')),
            'has_batch': bool(manifest.get('batch_reliability')),
            'has_local': bool(manifest.get('local_analyses'))
        }
        
        return content
    
    def get_eda_content(self, dataset_name: str, result_manager) -> Dict[str, Any]:
        """
        Load EDA content for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            result_manager: Instance of ResultManager
            
        Returns:
            Dict with keys:
                - profile_html: HTML string of profiling report (or None)
                - summary_stats: DataFrame with summary statistics
                - target_distribution: DataFrame with target variable distribution
                - correlation_matrix: DataFrame with correlations
                - missing_summary: DataFrame with missing value info
                - visualizations: List of visualization paths
        """
        # Load EDA summary - returns dict with DataFrames
        eda_data = result_manager.load_eda_summary(dataset_name)
        
        # Load profile HTML - returns HTML string
        profile_html = result_manager.load_eda_profile(dataset_name)
        
        # Load visualization paths
        viz_paths = result_manager.get_eda_visualizations(dataset_name)
        
        # Extract DataFrames from eda_data dict
        summary_stats = eda_data.get('summary') if eda_data else None
        target_dist = eda_data.get('target') if eda_data else None
        correlation = eda_data.get('correlation') if eda_data else None
        missing = eda_data.get('missing') if eda_data else None
        smote_dist = eda_data.get('smote') if eda_data else None

        # Load feature selection metadata
        feature_selection = result_manager.load_feature_selection(dataset_name)

        return {
            'profile_html': profile_html,
            'summary_stats': summary_stats,
            'target_distribution': target_dist,
            'correlation_matrix': correlation,
            'missing_summary': missing,
            'smote_distribution': smote_dist,
            'feature_selection': feature_selection,
            'visualizations': viz_paths
        }
    
    def get_benchmark_content(self, result_manager) -> Dict[str, Any]:
        """
        Load benchmark analysis content.
        
        Returns:
            Dict with keys:
                - results_df: DataFrame with benchmark results
                - all_results_df: DataFrame with all model results (optional)
                - comparison_figures: List of paths to comparison plots
        """
        results_df = result_manager.load_benchmark_results()
        
        # Get comparison figures from manifest
        manifest = result_manager.prepare_report_manifest()
        comparison_figs = manifest.get('benchmarks', {}).get('figures', [])
        
        # Load all model results if available
        all_results_df = None
        try:
            all_results_df = result_manager.load_all_model_results()
        except FileNotFoundError:
            # All model results not available - this is optional
            pass
        
        return {
            'results_df': results_df,
            'all_results_df': all_results_df,
            'comparison_figures': comparison_figs
        }
    
    def get_shap_content(self, dataset_name: str, result_manager) -> Dict[str, Any]:
        """
        Load SHAP analysis content for a dataset.
        
        Returns:
            Dict with keys:
                - shap_plots: Dict with plot types (bar, dot, waterfall, pdp) each containing list of paths
                - reliability_results: Dict with rank stability info
                - diagnostics: Dict with diagnostic test results
        """
        manifest = result_manager.prepare_report_manifest()
        # shap_plots is a dict: {dataset: {bar: [paths], dot: [paths], ...}}
        shap_plots_dict = manifest.get('shap_plots', {}).get(dataset_name, {})
        
        reliability = result_manager.load_reliability_results(dataset_name)
        diagnostics = result_manager.load_diagnostics(dataset_name)
        
        return {
            'shap_plots': shap_plots_dict,
            'reliability_results': reliability,
            'diagnostics': diagnostics
        }
    
    def get_reliability_content(self, dataset_name: str, result_manager) -> Dict[str, Any]:
        """
        Load reliability analysis content for a dataset.
        
        Returns:
            Dict with keys:
                - batch_results: DataFrame with batch reliability
                - batch_buckets: DataFrame with bucket definitions
                - diagnostics: Dict with statistical tests
        """
        batch_data = result_manager.load_batch_reliability(dataset_name)
        diagnostics = result_manager.load_diagnostics(dataset_name)
        
        batch_df = None
        buckets_df = None
        
        if batch_data:
            if batch_data.get('results_csv'):
                try:
                    batch_df = pd.read_csv(batch_data['results_csv'])
                except Exception:
                    pass
            if batch_data.get('buckets_csv'):
                try:
                    buckets_df = pd.read_csv(batch_data['buckets_csv'])
                except Exception:
                    pass
        
        return {
            'batch_results': batch_df,
            'batch_buckets': buckets_df,
            'diagnostics': diagnostics
        }
    
    # ==================== TWO-PHASE RESULTS GENERATION ====================
    
    def assemble_results_artifacts(self, result_manager, include_placeholders: bool = True) -> Dict[str, Any]:
        """
        Phase 1: Assemble all available artifacts into structured LaTeX Results section.
        NO LLM calls - just gathering and formatting artifacts.
        
        Args:
            result_manager: ResultManager instance
            include_placeholders: If True, add %PLACEHOLDER% markers for narrative text
            
        Returns:
            Dict with:
                - latex: Complete LaTeX Results section with artifacts
                - markdown: Markdown preview version
                - placeholders: List of placeholder IDs for Phase 2
                - artifact_summary: Count of tables/figures per subsection
        """
        manifest = result_manager.prepare_report_manifest()
        
        latex_parts = []
        markdown_parts = []
        placeholders = []
        artifact_summary = {
            'eda': {'tables': 0, 'figures': 0},
            'feature_importance': {'tables': 0, 'figures': 0},
            'benchmark': {'tables': 0, 'figures': 0},
            'paired_comparisons': {'tables': 0, 'figures': 0},
            'shap': {'tables': 0, 'figures': 0},
            'reliability': {'tables': 0, 'figures': 0},
            'local': {'tables': 0, 'figures': 0}
        }
        
        latex_parts.append("\\section{Results}\n")
        markdown_parts.append("# Results\n")
        
        # 4.1 Exploratory Data Analysis
        if manifest.get('eda'):
            latex_parts.append("\\subsection{Exploratory Data Analysis}\n")
            markdown_parts.append("## Exploratory Data Analysis\n")
            
            for dataset_name in manifest['eda'].keys():
                eda_latex, eda_md, eda_count = self._create_eda_subsection_latex(
                    dataset_name, 
                    result_manager,
                    include_placeholders
                )
                latex_parts.append(eda_latex)
                markdown_parts.append(eda_md)
                artifact_summary['eda']['tables'] += eda_count['tables']
                artifact_summary['eda']['figures'] += eda_count['figures']
                
                if include_placeholders:
                    placeholders.append({
                        'placeholder': f'%PLACEHOLDER_EDA_{dataset_name}%',
                        'type': 'eda',
                        'dataset': dataset_name
                    })

                    # Add feature selection placeholder if feature selection data exists
                    eda_content = self.get_eda_content(dataset_name, result_manager)
                    if eda_content.get('feature_selection') is not None:
                        safe_name = result_manager.sanitize_name(dataset_name)
                        placeholders.append({
                            'placeholder': f'%PLACEHOLDER_FEATURE_SELECTION_{safe_name}%',
                            'type': 'feature_selection',
                            'dataset': dataset_name
                        })

        # 4.2 Supervised Feature Importance Analysis
        # Check if any dataset has feature importance results
        has_feature_importance = False
        if manifest.get('eda'):
            for dataset_name, eda_content in manifest['eda'].items():
                fi_content = eda_content.get('feature_importance', {})
                if fi_content.get('merged'):
                    has_feature_importance = True
                    break
        
        if has_feature_importance:
            latex_parts.append("\\subsection{Supervised Feature Importance Analysis}\n")
            markdown_parts.append("## Supervised Feature Importance Analysis\n")
            
            for dataset_name, eda_content in manifest['eda'].items():
                fi_content = eda_content.get('feature_importance', {})
                if fi_content.get('merged'):
                    fi_latex, fi_md, fi_count = self._create_feature_importance_subsection_latex(
                        dataset_name,
                        fi_content,
                        include_placeholders
                    )
                    latex_parts.append(fi_latex)
                    markdown_parts.append(fi_md)
                    artifact_summary['feature_importance']['tables'] += fi_count['tables']
                    artifact_summary['feature_importance']['figures'] += fi_count['figures']
                    
                    if include_placeholders:
                        placeholders.append({
                            'placeholder': f'%PLACEHOLDER_FI_{dataset_name}%',
                            'type': 'feature_importance',
                            'dataset': dataset_name
                        })
        
        # 4.3 Model Performance Comparison
        if manifest.get('benchmarks', {}).get('csv'):
            bench_latex, bench_md, bench_count = self._create_benchmark_subsection_latex(
                result_manager,
                include_placeholders
            )
            latex_parts.append(bench_latex)
            markdown_parts.append(bench_md)
            artifact_summary['benchmark'] = bench_count
            
            if include_placeholders:
                # Get datasets for dataset-specific placeholders
                bench_content = self.get_benchmark_content(result_manager)
                datasets = []
                if bench_content.get('all_results_df') is not None:
                    datasets = bench_content['all_results_df']['Dataset'].unique().tolist()
                elif bench_content.get('results_df') is not None:
                    datasets = bench_content['results_df']['Dataset'].unique().tolist()
                
                # Add placeholder for each dataset
                for dataset_name in datasets:
                    placeholders.append({
                        'placeholder': f'%PLACEHOLDER_BENCHMARK_{result_manager.sanitize_name(dataset_name)}%',
                        'type': 'benchmark',
                        'dataset': dataset_name
                    })
                
                # Add cross-dataset analysis placeholder only if multiple datasets
                if len(datasets) > 1:
                    placeholders.append({
                        'placeholder': '%PLACEHOLDER_BENCHMARK_CROSS_DATASET%',
                        'type': 'benchmark_cross_dataset',
                        'dataset': None
                    })
        
        # 4.3.5 Paired Statistical Comparisons (new section)
        if manifest.get('paired_comparisons'):
            comparisons_latex, comparisons_md, comparisons_count = self._create_paired_comparisons_subsection_latex(
                result_manager,
                include_placeholders
            )
            if comparisons_latex:  # Only add if there are comparisons
                latex_parts.append(comparisons_latex)
                markdown_parts.append(comparisons_md)
                artifact_summary['paired_comparisons'] = comparisons_count
                
                if include_placeholders:
                    # Add placeholder for each comparison
                    for idx, comparison in enumerate(manifest['paired_comparisons'], 1):
                        placeholders.append({
                            'placeholder': f'%PLACEHOLDER_COMPARISON_{idx}%',
                            'type': 'paired_comparison',
                            'comparison_id': idx,
                            'dataset': comparison.get('dataset')
                        })
        
        # 4.4 SHAP-Based Explainability Analysis
        if manifest.get('shap_plots'):
            latex_parts.append("\\subsection{SHAP-Based Explainability Analysis}\n")
            markdown_parts.append("## SHAP-Based Explainability Analysis\n")
            
            for dataset_name in manifest['shap_plots'].keys():
                shap_latex, shap_md, shap_count = self._create_shap_subsection_latex(
                    dataset_name,
                    result_manager,
                    include_placeholders
                )
                latex_parts.append(shap_latex)
                markdown_parts.append(shap_md)
                artifact_summary['shap']['tables'] += shap_count['tables']
                artifact_summary['shap']['figures'] += shap_count['figures']
                
                if include_placeholders:
                    placeholders.append({
                        'placeholder': f'%PLACEHOLDER_SHAP_{dataset_name}%',
                        'type': 'shap',
                        'dataset': dataset_name
                    })
        
        # 4.5 Model Reliability and Stability
        if manifest.get('reliability') or manifest.get('batch_reliability'):
            latex_parts.append("\\subsection{Model Reliability and Stability}\n")
            markdown_parts.append("## Model Reliability and Stability\n")
            
            # Combine reliability and batch data
            all_datasets = set()
            if manifest.get('reliability'):
                all_datasets.update(manifest['reliability'].keys())
            if manifest.get('batch_reliability'):
                all_datasets.update(manifest['batch_reliability'].keys())
            
            for dataset_name in all_datasets:
                rel_latex, rel_md, rel_count = self._create_reliability_subsection_latex(
                    dataset_name,
                    result_manager,
                    include_placeholders
                )
                latex_parts.append(rel_latex)
                markdown_parts.append(rel_md)
                artifact_summary['reliability']['tables'] += rel_count['tables']
                artifact_summary['reliability']['figures'] += rel_count['figures']
                
                if include_placeholders:
                    placeholders.append({
                        'placeholder': f'%PLACEHOLDER_RELIABILITY_{dataset_name}%',
                        'type': 'reliability',
                        'dataset': dataset_name
                    })
        
        # 4.5 Local Explanations (if available)
        if manifest.get('local_analyses'):
            local_latex, local_md, local_count = self._create_local_subsection_latex(
                result_manager,
                include_placeholders
            )
            if local_latex:  # Only add if there are local analyses
                latex_parts.append(local_latex)
                markdown_parts.append(local_md)
                artifact_summary['local'] = local_count
                
                if include_placeholders:
                    placeholders.append({
                        'placeholder': '%PLACEHOLDER_LOCAL%',
                        'type': 'local',
                        'dataset': None
                    })

        # =====================================================================
        # APPENDIX: Comprehensive Model Performance Tables
        # =====================================================================
        # Add appendix with comprehensive all-models tables (moved from main body)
        bench_content = self.get_benchmark_content(result_manager)
        if bench_content.get('all_results_df') is not None:
            all_df = bench_content['all_results_df']
            datasets = all_df['Dataset'].unique().tolist()

            if datasets:
                # Add appendix section
                latex_parts.append("\\newpage\n")
                latex_parts.append("\\appendix\n")
                latex_parts.append("\\section{Comprehensive Model Performance Tables}\n")
                latex_parts.append("\\label{appendix:comprehensive}\n\n")
                latex_parts.append("This appendix contains detailed performance metrics for all models tested on each dataset.\n\n")
                markdown_parts.append("\n## Appendix: Comprehensive Model Performance Tables\n")

                appendix_count = {'tables': 0, 'figures': 0}

                # Add comprehensive table for each dataset
                for dataset_name in datasets:
                    dataset_all_df = all_df[all_df['Dataset'] == dataset_name]
                    if not dataset_all_df.empty:
                        latex_parts.append(self._create_all_models_table_latex_academic(dataset_all_df, dataset_name))
                        markdown_parts.append(f"**Comprehensive Model Results for {dataset_name}** (see LaTeX)\n")
                        appendix_count['tables'] += 1

                # Update artifact summary
                artifact_summary['appendix'] = appendix_count

        return {
            'latex': '\n'.join(latex_parts),
            'markdown': '\n\n'.join(markdown_parts),
            'placeholders': placeholders,
            'artifact_summary': artifact_summary,
            'total_artifacts': sum(
                s['tables'] + s['figures'] 
                for s in artifact_summary.values()
            )
        }
    
    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters in text.
        
        Args:
            text: Text that may contain LaTeX special characters
            
        Returns:
            Text with special characters properly escaped for LaTeX
        """
        # First handle Unicode mathematical symbols and special characters
        unicode_replacements = {
            '≥': '$\\geq$',
            '≤': '$\\leq$',
            '≠': '$\\neq$',
            '±': '$\\pm$',
            '×': '$\\times$',
            '÷': '$\\div$',
            '∞': '$\\infty$',
            '√': '$\\sqrt{}$',
            '∑': '$\\sum$',
            '∏': '$\\prod$',
            '∫': '$\\int$',
            '∂': '$\\partial$',
            '∇': '$\\nabla$',
            '∈': '$\\in$',
            '∉': '$\\notin$',
            '⊂': '$\\subset$',
            '⊃': '$\\supset$',
            '∩': '$\\cap$',
            '∪': '$\\cup$',
            '∅': '$\\emptyset$',
            '→': '$\\rightarrow$',
            '←': '$\\leftarrow$',
            '↔': '$\\leftrightarrow$',
            '⇒': '$\\Rightarrow$',
            '⇐': '$\\Leftarrow$',
            '⇔': '$\\Leftrightarrow$',
            '∀': '$\\forall$',
            '∃': '$\\exists$',
            'α': '$\\alpha$',
            'β': '$\\beta$',
            'γ': '$\\gamma$',
            'δ': '$\\delta$',
            'ε': '$\\epsilon$',
            'θ': '$\\theta$',
            'λ': '$\\lambda$',
            'μ': '$\\mu$',
            'π': '$\\pi$',
            'σ': '$\\sigma$',
            'τ': '$\\tau$',
            'φ': '$\\phi$',
            'ω': '$\\omega$',
            '°': '$^\\circ$',
            '′': '$^\\prime$',
            '″': '$^{\\prime\\prime}$',
        }
        
        for unicode_char, latex_code in unicode_replacements.items():
            text = text.replace(unicode_char, latex_code)
        
        # Then handle standard LaTeX special characters
        replacements = {
            '_': '\\_',
            '%': '\\%',
            '$': '\\$',  # This won't affect our math mode $ above since we do it after
            '&': '\\&',
            '#': '\\#',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
        
        # But we need to be careful not to escape $ that we added for math mode
        # So we temporarily replace our math $ with a placeholder
        text = text.replace('$', '\x00MATHMODE\x00')
        
        for char, escaped in replacements.items():
            if char != '$':  # Skip $ since we're handling it specially
                text = text.replace(char, escaped)
        
        # Restore math mode $
        text = text.replace('\x00MATHMODE\x00', '$')
        
        return text
    
    @staticmethod
    def _wrap_table_in_adjustbox(table_latex: str, standard_width: float = 1.0) -> str:
        """Wrap a LaTeX table's tabular environment in adjustbox to prevent overflow.

        Uses max width to ensure tables fit within page margins while maintaining
        readability. Tables will be displayed at natural size unless they exceed
        the specified width.

        Args:
            table_latex: LaTeX table code containing \\begin{tabular}...\\end{tabular}
            standard_width: Maximum width as fraction of \\textwidth (default 1.0).

        Returns:
            Table with tabular wrapped in adjustbox to prevent overflow
        """
        import re
        # Find the tabular environment
        pattern = r'(\\begin\{tabular\}.*?\\end\{tabular\})'
        match = re.search(pattern, table_latex, re.DOTALL)
        if match:
            tabular_block = match.group(1)
            # Use max width to prevent overflow while keeping natural size when possible
            wrapped = f"\\begin{{adjustbox}}{{max width={standard_width}\\textwidth}}\n" + tabular_block + "\n\\end{adjustbox}"
            return table_latex.replace(tabular_block, wrapped)
        return table_latex
    
    def _create_eda_subsection_latex(self, dataset_name: str, result_manager, 
                                      include_placeholders: bool) -> Tuple[str, str, Dict]:
        """Create EDA subsection with tables and figures."""
        latex_parts = []
        md_parts = []
        count = {'tables': 0, 'figures': 0}
        
        safe_name = result_manager.sanitize_name(dataset_name)
        latex_parts.append(f"\\subsubsection{{{self._escape_latex(dataset_name)}}}\n")
        md_parts.append(f"### {dataset_name}\n")
        
        if include_placeholders:
            latex_parts.append(f"%PLACEHOLDER_EDA_{safe_name}%\n")
            md_parts.append(f"*[Commentary placeholder for {dataset_name} EDA]*\n")
        
        # Load EDA content
        eda_content = self.get_eda_content(dataset_name, result_manager)
        
        # Add summary statistics table
        if eda_content.get('summary_stats') is not None:
            df = eda_content['summary_stats']
            latex_parts.append(self._create_summary_stats_table_latex(df, dataset_name))
            md_parts.append(f"**Summary Statistics Table** (see LaTeX)\n")
            count['tables'] += 1
        
        # Target distribution table removed - not required in LaTeX output
        # (Target distribution information is shown in the bar chart instead)
        
        # Add correlation matrix (top correlations) if available
        if eda_content.get('correlation_matrix') is not None:
            corr_df = eda_content['correlation_matrix']
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                # Show only top 10 correlations with target if available
                if 'target' in corr_df.columns or len(corr_df) > 0:
                    top_corr = corr_df.head(10) if len(corr_df) > 10 else corr_df
                    latex_parts.append("\\begin{table}[H]\n")
                    latex_parts.append("\\centering\n")
                    latex_parts.append(f"\\caption{{Top correlations for {self._escape_latex(dataset_name)}}}\n")
                    latex_parts.append(f"\\label{{tab:corr_{safe_name}}}\n")
                    table_content = top_corr.to_latex(index=True, escape=True)
                    table_content = self._wrap_table_in_adjustbox(table_content)
                    latex_parts.append(table_content)
                    latex_parts.append("\\end{table}\n")
                    latex_parts.append("\\FloatBarrier\n")
                    md_parts.append(f"**Correlation Table** (see LaTeX)\n")
                    count['tables'] += 1
        
        # Add SMOTE distribution table if available
        if eda_content.get('smote_distribution') is not None:
            smote_df = eda_content['smote_distribution']
            if isinstance(smote_df, pd.DataFrame) and not smote_df.empty:
                latex_parts.append("\\begin{table}[H]\n")
                latex_parts.append("\\centering\n")
                latex_parts.append(f"\\caption{{SMOTE class distribution comparison for {self._escape_latex(dataset_name)}}}\n")
                latex_parts.append(f"\\label{{tab:smote_{safe_name}}}\n")
                table_content = smote_df.to_latex(index=False, escape=True)
                table_content = self._wrap_table_in_adjustbox(table_content, standard_width=0.6)
                latex_parts.append(table_content)
                latex_parts.append("\\end{table}\n")
                latex_parts.append("\\FloatBarrier\n")
                md_parts.append(f"**SMOTE Distribution Table** (see LaTeX)\n")
                count['tables'] += 1

        # Add feature selection information if available
        if eda_content.get('feature_selection') is not None:
            fs_info = eda_content['feature_selection']
            latex_parts.append("\\paragraph{Feature Selection (Step 1.4)}\n")

            # Build feature selection description
            selection_desc = f"Selected {fs_info['num_selected']} out of {fs_info['total_features']} available features"

            if fs_info.get('quick_select_option') and fs_info['quick_select_option'] != "—":
                selection_desc += f" using \\textbf{{{fs_info['quick_select_option']}}}"
                if fs_info.get('fi_source'):
                    selection_desc += f" from \\textbf{{{fs_info['fi_source']}}}"
                selection_desc += " feature importance ranking"
            else:
                selection_desc += " (manual selection)"

            selection_desc += ".\n\n"
            latex_parts.append(selection_desc)

            # Add placeholder for commentary
            if include_placeholders:
                latex_parts.append(f"%PLACEHOLDER_FEATURE_SELECTION_{safe_name}%\n\n")

            # List selected features (limit to 20 for readability)
            features = fs_info['selected_features']
            if len(features) <= 20:
                features_str = ", ".join([self._escape_latex(f) for f in features])
                latex_parts.append(f"\\textit{{Selected features:}} {features_str}.\n\n")
            else:
                features_str = ", ".join([self._escape_latex(f) for f in features[:20]])
                latex_parts.append(f"\\textit{{Selected features (first 20):}} {features_str}, and {len(features) - 20} more.\n\n")

            md_parts.append(f"**Feature Selection:** {fs_info['num_selected']}/{fs_info['total_features']} features\n")

        # Add all available EDA visualizations
        viz_paths = eda_content.get('visualizations', [])
        if viz_paths:
            # Group by type (target_dist, correlation, missing_values, etc.)
            for viz_path in viz_paths:
                # Extract figure type from filename
                fig_name = Path(viz_path).stem.replace(safe_name + '_', '')
                latex_parts.append(self._create_figure_latex_academic(
                    viz_path,
                    f"{fig_name.replace('_', ' ').title()} for {self._escape_latex(dataset_name)}",
                    f"fig:eda_{safe_name}_{fig_name}",
                    width="0.7"
                ))
                md_parts.append(f"![{fig_name}]({viz_path})\n")
                count['figures'] += 1
        
        latex_parts.append("\n")
        return '\n'.join(latex_parts), '\n'.join(md_parts), count
    
    def _create_feature_importance_subsection_latex(
        self,
        dataset_name: str,
        fi_content: Dict[str, Any],
        include_placeholders: bool
    ) -> Tuple[str, str, Dict]:
        """
        Create feature importance subsection from Step 1.2 results.
        
        Args:
            dataset_name: Dataset name
            fi_content: Feature importance content dict with 'rf', 'lr', 'merged' paths
            include_placeholders: Whether to include LLM commentary placeholders
            
        Returns:
            Tuple of (latex_str, markdown_str, count_dict)
        """
        latex_parts = []
        md_parts = []
        count = {'tables': 0, 'figures': 0}
        
        safe_name = self.sanitize_name(dataset_name)
        
        latex_parts.append(f"\\subsubsection{{{self._escape_latex(dataset_name)}}}\n")
        md_parts.append(f"### {dataset_name}\n")
        
        if include_placeholders:
            latex_parts.append(f"%PLACEHOLDER_FI_{safe_name}%\n")
            md_parts.append("*[Commentary placeholder for feature importance analysis]*\n")
        
        # Load merged feature importance table (primary table)
        merged_path = fi_content.get('merged')
        if merged_path and Path(merged_path).exists():
            try:
                merged_df = pd.read_csv(merged_path)
                
                # Show top 20 features by average score
                top_features = merged_df.head(20)
                
                # Create academic-style table
                latex_parts.append(self._create_feature_importance_table_latex(
                    top_features,
                    dataset_name,
                    table_type="merged"
                ))
                md_parts.append("**Top 20 Features by Combined Importance** (see LaTeX)\n")
                count['tables'] += 1
                
            except Exception as e:
                latex_parts.append(f"% Error loading feature importance: {e}\n")
                md_parts.append(f"*Error loading feature importance: {e}*\n")
        
        # Optionally load metadata for context
        meta_path = fi_content.get('meta')
        if meta_path and Path(meta_path).exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    n_rows = meta.get('n_rows', 'N/A')
                    n_cols = meta.get('n_cols', 'N/A')
                    kept_cols = len(meta.get('kept_columns_after_missing_drop', []))
                    
                    latex_parts.append(f"% Dataset: {n_rows} rows, {n_cols} columns, {kept_cols} retained after missing-value filter\n")
                    md_parts.append(f"*Dataset: {n_rows} rows, {n_cols} columns, {kept_cols} retained after missing-value filter*\n")
            except Exception:
                pass
        
        latex_parts.append("\n")
        return '\n'.join(latex_parts), '\n'.join(md_parts), count
    
    def _create_benchmark_subsection_latex(self, result_manager, 
                                            include_placeholders: bool) -> Tuple[str, str, Dict]:
        """Create benchmark results subsection with table and comparison figures."""
        latex_parts = []
        md_parts = []
        count = {'tables': 0, 'figures': 0}
        
        latex_parts.append("\\subsection{Model Performance Comparison}\n")
        md_parts.append("## Model Performance Comparison\n")
        
        # Load benchmark content
        bench_content = self.get_benchmark_content(result_manager)
        
        # Get unique datasets from the dataframes
        datasets = []
        if bench_content.get('all_results_df') is not None:
            datasets = bench_content['all_results_df']['Dataset'].unique().tolist()
        elif bench_content.get('results_df') is not None:
            datasets = bench_content['results_df']['Dataset'].unique().tolist()
        
        # Check if SMOTE column is missing and add a note
        if bench_content.get('results_df') is not None:
            df = bench_content['results_df']
            if 'SMOTE' not in df.columns:
                latex_parts.append("\\textit{Note: SMOTE information not available in this benchmark run. Please re-run Step 4 to include preprocessing details.}\n\n")
        
        # Create subsubsection for each dataset with comprehensive results, benchmarks, and comparison charts
        for dataset_name in datasets:
            # Add subsubsection header
            safe_name = result_manager.sanitize_name(dataset_name)
            latex_parts.append(f"\\subsubsection{{{self._escape_latex(dataset_name)}}}\n")
            md_parts.append(f"### {dataset_name}\n")
            
            # Add placeholder for dataset-specific commentary
            if include_placeholders:
                latex_parts.append(f"%PLACEHOLDER_BENCHMARK_{safe_name}%\n")
                md_parts.append(f"*[Commentary placeholder for {dataset_name} benchmark results]*\n")

            # Add benchmark results table for this dataset
            if bench_content.get('results_df') is not None:
                df = bench_content['results_df']
                dataset_bench_df = df[df['Dataset'] == dataset_name]
                if not dataset_bench_df.empty:
                    latex_parts.append(self._create_benchmark_table_latex_academic(dataset_bench_df, dataset_name))
                    md_parts.append(f"**Benchmark Results** (see LaTeX)\n")
                    count['tables'] += 1

        # Add Model Comparison Charts section (for both single and multiple datasets)
        comp_figs = bench_content.get('comparison_figures', [])
        if comp_figs:
            if len(datasets) > 1:
                latex_parts.append("\\subsubsection{Cross-Dataset Model Performance Analysis}\n")
                md_parts.append("### Cross-Dataset Model Performance Analysis\n")
            else:
                latex_parts.append("\\subsubsection{Model Comparison Charts}\n")
                md_parts.append("### Model Comparison Charts\n")

            if include_placeholders and len(datasets) > 1:
                latex_parts.append("%PLACEHOLDER_BENCHMARK_CROSS_DATASET%\n")
                md_parts.append("*[Commentary placeholder for cross-dataset analysis]*\n")

            # Add comparison figures (grid layout)
            latex_parts.append(self._create_comparison_figures_latex_academic(comp_figs))
            md_parts.append(f"**Model Comparison Figures:** {len(comp_figs)} metrics\n")
            count['figures'] += len(comp_figs)
        elif len(datasets) > 1:
            # Add cross-dataset section even if no figures (for placeholder)
            latex_parts.append("\\subsubsection{Cross-Dataset Model Performance Analysis}\n")
            md_parts.append("### Cross-Dataset Model Performance Analysis\n")

            if include_placeholders:
                latex_parts.append("%PLACEHOLDER_BENCHMARK_CROSS_DATASET%\n")
                md_parts.append("*[Commentary placeholder for cross-dataset analysis]*\n")
        
        latex_parts.append("\n")
        return '\n'.join(latex_parts), '\n'.join(md_parts), count
    
    def _create_shap_subsection_latex(self, dataset_name: str, result_manager,
                                       include_placeholders: bool) -> Tuple[str, str, Dict]:
        """Create SHAP analysis subsection with plots."""
        latex_parts = []
        md_parts = []
        count = {'tables': 0, 'figures': 0}
        
        safe_name = result_manager.sanitize_name(dataset_name)
        latex_parts.append(f"\\subsubsection{{Global SHAP --- {self._escape_latex(dataset_name)}}}\n")
        md_parts.append(f"### Global SHAP - {dataset_name}\n")
        
        if include_placeholders:
            latex_parts.append(f"%PLACEHOLDER_SHAP_{safe_name}%\n")
            md_parts.append(f"*[Commentary placeholder for {dataset_name} SHAP]*\n")
        
        # Load SHAP content
        shap_content = self.get_shap_content(dataset_name, result_manager)
        
        # Add reliability rank table if available
        if shap_content.get('reliability_results'):
            rank_path = shap_content['reliability_results'].get('rank_csv')
            if rank_path and Path(rank_path).exists():
                try:
                    rank_df = pd.read_csv(rank_path)
                    latex_parts.append(self._create_feature_rank_table_latex(rank_df, dataset_name))
                    md_parts.append("**Feature Importance Ranking** (see LaTeX)\n")
                    count['tables'] += 1
                except Exception:
                    pass
        
        # Add SHAP plots (bar + dot side by side, like in example)
        shap_plots_dict = shap_content.get('shap_plots', {})
        bar_plots = shap_plots_dict.get('bar', [])
        dot_plots = shap_plots_dict.get('dot', [])
        waterfall_plots = shap_plots_dict.get('waterfall', [])
        pdp_plots = shap_plots_dict.get('pdp', [])
        
        # Add bar + dot plots side by side (primary visualization)
        if bar_plots and dot_plots:
            latex_parts.append(self._create_shap_dual_plot_latex_academic(
                bar_plots[0], dot_plots[0], dataset_name
            ))
            md_parts.append(f"**SHAP Feature Importance:** Bar + Dot plots\n")
            count['figures'] += 2
        
        # Add any additional bar plots (different models)
        for bar_plot in bar_plots[1:]:
            latex_parts.append(self._create_figure_latex_academic(
                bar_plot,
                f"SHAP bar plot for {self._escape_latex(dataset_name)}",
                f"fig:shap_bar_{safe_name}_{count['figures']}",
                width="0.8"
            ))
            md_parts.append(f"![SHAP Bar]({bar_plot})\n")
            count['figures'] += 1
        
        # Add any additional dot plots
        for dot_plot in dot_plots[1:]:
            latex_parts.append(self._create_figure_latex_academic(
                dot_plot,
                f"SHAP dot plot for {self._escape_latex(dataset_name)}",
                f"fig:shap_dot_{safe_name}_{count['figures']}",
                width="0.8"
            ))
            md_parts.append(f"![SHAP Dot]({dot_plot})\n")
            count['figures'] += 1
        
        # Add PDP/dependence plots if available
        for pdp_plot in pdp_plots[:5]:  # Limit PDPs to 5 to avoid overload
            latex_parts.append(self._create_figure_latex_academic(
                pdp_plot,
                f"Partial dependence plot for {self._escape_latex(dataset_name)}",
                f"fig:shap_pdp_{safe_name}_{count['figures']}",
                width="0.7"
            ))
            md_parts.append(f"![PDP]({pdp_plot})\n")
            count['figures'] += 1
        
        latex_parts.append("\n")
        return '\n'.join(latex_parts), '\n'.join(md_parts), count
    
    def _create_reliability_subsection_latex(self, dataset_name: str, result_manager,
                                              include_placeholders: bool) -> Tuple[str, str, Dict]:
        """Create reliability diagnostics subsection."""
        latex_parts = []
        md_parts = []
        count = {'tables': 0, 'figures': 0}
        
        safe_name = result_manager.sanitize_name(dataset_name)
        latex_parts.append(f"\\subsubsection{{Reliability Diagnostics --- {self._escape_latex(dataset_name)}}}\n")
        md_parts.append(f"### Reliability Diagnostics - {dataset_name}\n")
        
        if include_placeholders:
            latex_parts.append(f"%PLACEHOLDER_RELIABILITY_{safe_name}%\n")
            md_parts.append(f"*[Commentary placeholder for {dataset_name} reliability]*\n")
        
        # Load reliability content
        rel_content = self.get_reliability_content(dataset_name, result_manager)
        
        # Add batch reliability table if available
        if rel_content.get('batch_results') is not None:
            df = rel_content['batch_results']
            latex_parts.append(self._create_reliability_table_latex(df, dataset_name))
            md_parts.append("**Batch Reliability Results** (see LaTeX)\n")
            count['tables'] += 1
        
        # Add diagnostics summary table
        if rel_content.get('diagnostics'):
            latex_parts.append(self._create_diagnostics_table_latex(
                rel_content['diagnostics'], dataset_name
            ))
            md_parts.append("**Statistical Diagnostics** (see LaTeX)\n")
            count['tables'] += 1
        
        latex_parts.append("\n")
        return '\n'.join(latex_parts), '\n'.join(md_parts), count
    
    def _create_local_subsection_latex(self, result_manager,
                                        include_placeholders: bool) -> Tuple[str, str, Dict]:
        """Create local SHAP analysis subsection with waterfall plots."""
        latex_parts = []
        md_parts = []
        count = {'tables': 0, 'figures': 0}

        manifest = result_manager.prepare_report_manifest()
        local_analyses = manifest.get('local_analyses', [])

        if not local_analyses:
            return "", "", count

        latex_parts.append("\\subsection{Local SHAP Analysis Examples}\n")
        md_parts.append("## Local SHAP Analysis Examples\n")

        if include_placeholders:
            latex_parts.append("%PLACEHOLDER_LOCAL%\n")
            md_parts.append("*[Commentary placeholder for local SHAP examples]*\n")

        # Show first 5 local analyses as examples
        for data in local_analyses[:5]:
            try:
                # Data is already a dictionary (not a file path)
                dataset = data.get('dataset', 'Unknown')
                row_idx = data.get('row_index', 0)
                model_name = data.get('model_name', 'Unknown')
                pred_prob = data.get('predicted_prob', 0.0)
                actual = data.get('actual_target', 'N/A')
                ai_commentary = data.get('ai_commentary', '')
                waterfall_png = data.get('waterfall_png', None)
                reliability_metrics = data.get('reliability_metrics', None)
                analysis_id = data.get('analysis_id', 'N/A')

                safe_ds = result_manager.sanitize_name(dataset)

                latex_parts.append(f"\\subsubsection{{{self._escape_latex(dataset)} (Row {row_idx}) - {self._escape_latex(analysis_id)}}}\n")
                latex_parts.append(f"Model: {self._escape_latex(model_name)}\\\\")
                latex_parts.append(f"Actual Target: {actual}, Predicted Probability (Class 1): {pred_prob:.4f}\\\\\n\n")

                md_parts.append(f"### {dataset} (Row {row_idx}) - {analysis_id}\n")
                md_parts.append(f"Model: {model_name}, Actual: {actual}, Predicted: {pred_prob:.4f}\n")

                # Add reliability metrics if available
                if reliability_metrics:
                    score = reliability_metrics.get('reliability_score', 0)
                    bucket = reliability_metrics.get('reliability_bucket', 'Unknown')
                    latex_parts.append(f"Reliability Score: {score:.3f} ({bucket})\\\\")
                    md_parts.append(f"Reliability: {score:.3f} ({bucket})\n")

                # Add AI commentary if available
                if ai_commentary:
                    latex_parts.append("\n\\textbf{AI-Generated Explanation:}\\\\\n")
                    latex_parts.append(f"{self._escape_latex(ai_commentary)}\n\n")
                    md_parts.append(f"\n**AI Explanation:** {ai_commentary}\n\n")

                # Add waterfall plot if path is provided
                if waterfall_png:
                    from pathlib import Path
                    # Convert relative path to absolute if needed
                    if not Path(waterfall_png).is_absolute():
                        waterfall_path = Path(result_manager.base_dir).parent / waterfall_png
                    else:
                        waterfall_path = Path(waterfall_png)

                    if waterfall_path.exists():
                        # Use the same helper function as other figures (uses absolute paths)
                        latex_parts.append(self._create_figure_latex_academic(
                            waterfall_path,
                            f"Waterfall plot for {self._escape_latex(dataset)}, row {row_idx}",
                            f"fig:waterfall_{safe_ds}_row{row_idx}",
                            width="0.75"
                        ))
                        md_parts.append(f"![Waterfall Plot]({waterfall_path})\n")
                        count['figures'] += 1

            except Exception as e:
                # Skip analyses that cause errors
                continue

        latex_parts.append("\n")
        return '\n'.join(latex_parts), '\n'.join(md_parts), count
    
    def _create_paired_comparisons_subsection_latex(self, result_manager,
                                                     include_placeholders: bool) -> Tuple[str, str, Dict]:
        """Create paired statistical comparisons subsection."""
        latex_parts = []
        md_parts = []
        count = {'tables': 0, 'figures': 0}
        
        comparisons = result_manager.get_paired_comparisons()
        
        if not comparisons:
            return "", "", count
        
        latex_parts.append("\\subsection{Paired Statistical Comparisons}\n")
        md_parts.append("## Paired Statistical Comparisons\n")
        
        latex_parts.append(
            "This section presents pairwise statistical comparisons between models using "
            "Wilcoxon signed-rank test (for absolute errors), McNemar's test (for classification agreement), "
            "and DeLong's test (for AUC differences).\\\\[0.5em]\n\n"
        )
        md_parts.append(
            "Statistical tests comparing model pairs using Wilcoxon, McNemar, and DeLong methods.\n\n"
        )
        
        for idx, comparison in enumerate(comparisons, 1):
            dataset = comparison.get('dataset', 'Unknown')
            model_a = comparison.get('model_a', 'Model A')
            model_b = comparison.get('model_b', 'Model B')
            
            latex_parts.append(f"\\subsubsection{{Comparison {idx}: Model A vs Model B}}\n")
            md_parts.append(f"### Comparison {idx}: {model_a} vs {model_b}\n")

            # Create compact table with model details
            latex_parts.append("\\noindent\n")
            latex_parts.append("\\begin{tabular}{@{}ll@{}}\n")
            latex_parts.append(f"\\textbf{{Dataset:}} & {self._escape_latex(dataset)} \\\\\n")
            latex_parts.append(f"\\textbf{{Model A:}} & {self._escape_latex(model_a)} \\\\\n")
            latex_parts.append(f"\\textbf{{Model B:}} & {self._escape_latex(model_b)} \\\\\n")
            latex_parts.append("\\end{tabular}\n\n")
            latex_parts.append("\\vspace{0.5em}\n\n")
            
            if include_placeholders:
                latex_parts.append(f"%PLACEHOLDER_COMPARISON_{idx}%\n")
                md_parts.append(f"*[Commentary placeholder for comparison {idx}]*\n")
            
            # Create results table for this comparison
            latex_parts.append(self._create_comparison_results_table_latex(comparison, idx))
            md_parts.append(f"**Comparison {idx} Results** (see LaTeX)\n")
            count['tables'] += 1
            
            latex_parts.append("\n")
        
        return '\n'.join(latex_parts), '\n'.join(md_parts), count
    
    def _create_comparison_results_table_latex(self, comparison: Dict, comparison_id: int) -> str:
        """Create LaTeX table for a single paired comparison."""
        latex_parts = []

        latex_parts.append("\\begin{table}[H]\n")
        latex_parts.append("\\centering\n")
        latex_parts.append(f"\\caption{{Statistical Test Results for Comparison {comparison_id}}}\n")
        latex_parts.append(f"\\label{{tab:comparison_{comparison_id}}}\n")
        latex_parts.append("\\begin{adjustbox}{max width=0.6\\textwidth}\n")
        latex_parts.append("\\begin{tabular}{lll}\n")
        latex_parts.append("\\toprule\n")
        latex_parts.append("Test & Metric & Value \\\\\n")
        latex_parts.append("\\midrule\n")
        
        # Wilcoxon results
        wilcoxon = comparison.get('wilcoxon', {})
        if wilcoxon.get('statistic') is not None:
            latex_parts.append(f"Wilcoxon & Statistic & {wilcoxon.get('statistic', 'N/A'):.4f} \\\\\n")
            latex_parts.append(f"Wilcoxon & p-value & {wilcoxon.get('p_value', 'N/A'):.4f} \\\\\n")
            med_diff = wilcoxon.get('median_abs_error_diff')
            if med_diff is not None:
                latex_parts.append(f"Wilcoxon & Median $|$Error$|$ Diff (A-B) & {med_diff:.4f} \\\\\n")
        else:
            latex_parts.append("Wilcoxon & Status & Not computed \\\\\n")
        
        latex_parts.append("\\midrule\n")
        
        # McNemar results
        mcnemar = comparison.get('mcnemar', {})
        if mcnemar.get('chi2') is not None:
            latex_parts.append(f"McNemar & b (A correct, B wrong) & {mcnemar.get('b', 'N/A')} \\\\\n")
            latex_parts.append(f"McNemar & c (A wrong, B correct) & {mcnemar.get('c', 'N/A')} \\\\\n")
            latex_parts.append(f"McNemar & $\\chi^2$ & {mcnemar.get('chi2', 'N/A'):.4f} \\\\\n")
            latex_parts.append(f"McNemar & p-value & {mcnemar.get('p_value', 'N/A'):.4f} \\\\\n")
        else:
            latex_parts.append("McNemar & Status & Not computed \\\\\n")
        
        latex_parts.append("\\midrule\n")
        
        # DeLong results
        delong = comparison.get('delong')
        if delong and isinstance(delong, dict):
            for key, value in delong.items():
                clean_key = self._escape_latex(str(key))
                if isinstance(value, (int, float)):
                    latex_parts.append(f"DeLong & {clean_key} & {value:.4f} \\\\\n")
                else:
                    latex_parts.append(f"DeLong & {clean_key} & {self._escape_latex(str(value))} \\\\\n")
        else:
            latex_parts.append("DeLong & Status & Not computed \\\\\n")
        
        latex_parts.append("\\bottomrule\n")
        latex_parts.append("\\end{tabular}\n")
        latex_parts.append("\\end{adjustbox}\n")
        latex_parts.append("\\end{table}\n\n")

        return ''.join(latex_parts)
    
    # ==================== PHASE 2: NARRATIVE GENERATION ====================
    
    def generate_results_narratives(
        self,
        assembled_result: Dict[str, Any],
        result_manager: Any,
        llm_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate AI narratives for each Results subsection (Phase 2).
        
        This function takes the assembled artifacts from Phase 1 and generates
        concise academic commentary for each subsection using LLM.
        
        Args:
            assembled_result: Output from assemble_results_artifacts()
            result_manager: ResultManager instance for accessing data
            llm_config: LLM configuration dict with:
                - provider: "openai" or "anthropic"
                - model: model name
                - api_key: API key
                - temperature: 0.0-1.0 (default 0.7)
                - max_tokens: max response tokens (default 500)
        
        Returns:
            Dict containing:
                - narratives: Dict[str, str] mapping placeholder → narrative text
                - success: List of successfully generated narratives
                - failed: List of failed narrative generations
                - latex_with_narratives: Complete LaTeX with narratives inserted
                - markdown_with_narratives: Complete markdown with narratives inserted
        """
        
        if not assembled_result or not assembled_result.get('placeholders'):
            return {
                'narratives': {},
                'success': [],
                'failed': [],
                'latex_with_narratives': assembled_result.get('latex', ''),
                'markdown_with_narratives': assembled_result.get('markdown', '')
            }
        
        narratives = {}
        success = []
        failed = []
        
        placeholders = assembled_result['placeholders']
        
        # Get benchmark results for context
        benchmark_df = None
        try:
            benchmark_results = result_manager.get_benchmark_results()
            if benchmark_results.get('results_df') is not None:
                benchmark_df = benchmark_results['results_df']
        except Exception:
            pass
        
        # Generate narratives for each placeholder
        for placeholder_info in placeholders:
            placeholder = placeholder_info['placeholder']
            ptype = placeholder_info['type']
            dataset = placeholder_info.get('dataset')
            
            try:
                # Generate appropriate prompt based on type
                if ptype == 'eda':
                    prompt = self._get_eda_narrative_prompt(dataset, result_manager)
                elif ptype == 'benchmark':
                    # Get dataset-specific benchmark data if dataset is specified
                    if dataset and benchmark_df is not None:
                        dataset_bench_df = benchmark_df[benchmark_df['Dataset'] == dataset]
                        prompt = self._get_benchmark_narrative_prompt(dataset_bench_df, dataset)
                    else:
                        prompt = self._get_benchmark_narrative_prompt(benchmark_df)
                elif ptype == 'benchmark_cross_dataset':
                    # Load all results for cross-dataset analysis
                    all_results_df = None
                    try:
                        all_results_df = result_manager.load_all_model_results()
                    except:
                        pass
                    prompt = self._get_benchmark_cross_dataset_narrative_prompt(benchmark_df, all_results_df)
                elif ptype == 'feature_importance':
                    prompt = self._get_feature_importance_narrative_prompt(dataset, result_manager)
                elif ptype == 'feature_selection':
                    # Load feature selection metadata for this dataset
                    fs_info = result_manager.load_feature_selection(dataset)
                    if fs_info:
                        prompt = self._get_feature_selection_narrative_prompt(dataset, fs_info)
                    else:
                        failed.append(f"{placeholder} (no feature selection data)")
                        continue
                elif ptype == 'shap':
                    prompt = self._get_shap_narrative_prompt(dataset, result_manager)
                elif ptype == 'reliability':
                    prompt = self._get_reliability_narrative_prompt(dataset, result_manager)
                elif ptype == 'local':
                    prompt = self._get_local_narrative_prompt(result_manager)
                elif ptype == 'paired_comparison':
                    comparison_id = placeholder_info.get('comparison_id', 1)
                    prompt = self._get_paired_comparison_narrative_prompt(comparison_id, result_manager)
                else:
                    failed.append(f"{placeholder} (unknown type)")
                    continue
                
                # Generate narrative with LLM
                narrative = self.generate_section_with_llm(prompt, llm_config)
                
                if narrative:
                    narratives[placeholder] = narrative
                    success.append(placeholder)
                else:
                    failed.append(f"{placeholder} (LLM returned empty)")
                    
            except Exception as e:
                failed.append(f"{placeholder} (error: {str(e)})")
        
        # Insert narratives into templates
        latex_with_narratives = self._insert_narratives_into_template(
            assembled_result['latex'], 
            narratives
        )
        markdown_with_narratives = self._insert_narratives_into_template(
            assembled_result['markdown'], 
            narratives
        )
        
        return {
            'narratives': narratives,
            'success': success,
            'failed': failed,
            'latex_with_narratives': latex_with_narratives,
            'markdown_with_narratives': markdown_with_narratives
        }
    
    @staticmethod
    def _insert_narratives_into_template(template: str, narratives: Dict[str, str]) -> str:
        """
        Replace placeholder markers with actual narrative text.
        
        Args:
            template: LaTeX or markdown string with %PLACEHOLDER_*% markers
            narratives: Dict mapping placeholder → narrative text
        
        Returns:
            Template with narratives inserted
        """
        result = template
        
        for placeholder, narrative in narratives.items():
            # Clean up narrative (remove extra whitespace, ensure ends with period)
            clean_narrative = narrative.strip()
            if clean_narrative and not clean_narrative.endswith('.'):
                clean_narrative += '.'
            
            # Escape LaTeX special characters in the narrative text
            clean_narrative = ReportManager._escape_latex(clean_narrative)
            
            # Wrap in paragraph environment to prevent text overflow
            clean_narrative = f"\n{clean_narrative}\n"
            
            # Replace placeholder with narrative
            result = result.replace(placeholder, clean_narrative)
        
        return result
    
    def _get_eda_narrative_prompt(self, dataset_name: str, result_manager: Any) -> str:
        """Generate prompt for EDA subsection narrative (concise academic commentary)."""
        
        # Get EDA data for context
        eda_content = self.get_eda_content(dataset_name, result_manager)
        
        summary_text = ""
        if eda_content.get('summary_stats') is not None:
            try:
                df = eda_content['summary_stats']
                summary_text = f"\nSummary Statistics:\n{df.to_string()}"
            except Exception:
                pass
        
        # Check if SMOTE distribution table is available
        smote_info = ""
        if eda_content.get('smote_distribution') is not None:
            try:
                smote_df = eda_content['smote_distribution']
                smote_info = f"\n\nSMOTE Class Distribution:\n{smote_df.to_string()}\n\nNote: SMOTE (Synthetic Minority Over-sampling Technique) was applied to address class imbalance."
            except Exception:
                pass
        
        prompt = f"""You are writing a concise academic commentary for the EDA subsection of dataset: {dataset_name}.

Context:
{summary_text}{smote_info}

Write a brief {"150-200" if smote_info else "100-150"} word narrative that:
1. Summarizes key dataset characteristics (number of features, observations, target distribution)
2. Highlights notable patterns (class imbalance, missing values, key correlations)
3. Notes any data quality considerations"""
        
        if smote_info:
            prompt += "\n4. Mentions the application of SMOTE to address class imbalance (refer to the SMOTE distribution table showing before/after class counts)"
        
        prompt += """

Use academic language. DO NOT explain methodology - focus only on interpreting the results shown in the tables and figures.
Reference figures/tables naturally (e.g., "As shown in Table X..." or "Figure X illustrates...").
Keep it concise and focused on insights."""
        
        return prompt
    
    def _get_feature_importance_narrative_prompt(self, dataset_name: str, result_manager: Any) -> str:
        """Generate prompt for Feature Importance subsection narrative (Step 1.2 results)."""
        
        # Get feature importance data for context
        fi_text = ""
        try:
            manifest = result_manager.prepare_report_manifest()
            eda_content = manifest.get('eda', {}).get(dataset_name, {})
            fi_content = eda_content.get('feature_importance', {})
            
            merged_path = fi_content.get('merged')
            if merged_path and Path(merged_path).exists():
                df = pd.read_csv(merged_path)
                fi_text = f"\nTop 15 Features by Combined RF/LR Importance:\n{df.head(15).to_string()}"
        except Exception:
            pass
        
        return f"""You are writing a concise academic commentary for the Supervised Feature Importance Analysis subsection of dataset: {dataset_name}.

This analysis uses RandomForest (impurity-based) and LogisticRegression (L1 coefficient magnitude) to rank features BEFORE model training.

Context:
{fi_text}

Write a brief 100-150 word narrative that:
1. Identifies the top 5-10 most important predictive features
2. Compares insights from RF vs LR (do they agree on top features?)
3. Interprets what these features reveal about credit risk drivers
4. Notes any domain-relevant patterns (e.g., payment history, debt ratios, demographics)

Use academic language. DO NOT explain RF/LR methodology - focus only on interpreting which features matter and why.
Reference the table naturally (e.g., "Table X shows..." or "As indicated in Table X...").
Keep it concise and insight-focused."""

    def _get_feature_selection_narrative_prompt(self, dataset_name: str, fs_info: Dict[str, Any]) -> str:
        """Generate prompt for Feature Selection (Step 1.4) narrative."""

        selection_context = f"""Dataset: {dataset_name}
Total available features: {fs_info['total_features']}
Selected features: {fs_info['num_selected']} ({fs_info['num_selected']/fs_info['total_features']*100:.1f}%)
Selection method: {fs_info.get('selection_method', 'Manual')}"""

        if fs_info.get('quick_select_option') and fs_info['quick_select_option'] != "—":
            selection_context += f"\nQuick-select option used: {fs_info['quick_select_option']}"
            if fs_info.get('fi_source'):
                selection_context += f"\nFeature importance source: {fs_info['fi_source']}"

        # Add feature names if reasonable number
        if fs_info['num_selected'] <= 15:
            features_list = ", ".join(fs_info['selected_features'])
            selection_context += f"\n\nSelected features: {features_list}"

        return f"""You are writing a concise academic commentary for the Feature Selection subsection of dataset: {dataset_name}.

Context:
{selection_context}

Write a brief 80-120 word narrative that:
1. Explains the feature selection approach (manual vs. top-N from feature importance)
2. If using top-N selection, briefly notes the rationale (focusing computational resources on most predictive variables)
3. Comments on the selection ratio (e.g., aggressive dimensionality reduction vs. conservative inclusion)
4. If a very small subset was chosen, note potential benefits (reduced overfitting, faster training) and risks (information loss)

Use academic language. Be concise and focused on the practical implications of the feature selection strategy."""

    def _get_benchmark_narrative_prompt(
        self,
        benchmark_df: Optional[pd.DataFrame],
        dataset_name: Optional[str] = None
    ) -> str:
        """
        Generate prompt for Benchmark subsection narrative (dataset-specific if provided).

        This version injects domain knowledge about model families and evaluation metrics
        so that the LLM can give richer, more informed commentary while keeping the
        existing infrastructure unchanged.
        """
        results_text = ""
        smote_info = ""
        if benchmark_df is not None and not benchmark_df.empty:
            results_text = f"\nPerformance Results:\n{benchmark_df.to_string(max_rows=30)}"
            # Check if SMOTE column exists and was used
            if "SMOTE" in benchmark_df.columns:
                smote_values = benchmark_df["SMOTE"].dropna().unique().tolist()
                if "Yes" in smote_values:
                    smote_info = (
                        "\nNote: SMOTE (Synthetic Minority Over-sampling Technique) "
                        "was applied in at least one benchmark configuration to address "
                        "class imbalance in the default/non-default labels."
                    )

        dataset_context = (
            f"for dataset: {dataset_name}"
            if dataset_name is not None
            else "for the current benchmark table"
        )

        # The prompt below encodes (a) model-family naming conventions and
        # (b) metric semantics, but still leaves the actual commentary for the LLM
        # so the rest of the reporting pipeline stays exactly the same.
        return f"""You are writing a concise academic commentary for the **Model Benchmark** subsection {dataset_context} in a credit risk modeling paper.

The benchmark table reports performance for many models. Model IDs follow these naming conventions:

- Logistic regression:
  - `lr_*` = plain logistic regression (e.g. `lr_lbfgs`, `lr_saga`, `lr_newton_cg`).
  - `lr_reg_*` = regularised logistic regression (L1/L2/elastic-net; e.g. `lr_reg_saga`, `lr_reg_lbfgs`, `lr_reg_liblinear`).
- Bagging and tree ensembles:
  - `bag_cart_*` = bagged CART decision trees with different numbers of trees.
  - `bagnn_*` = bagged neural networks.
  - `rf_*` = random forests, with pattern `rf_<n_estimators>_mf_<max_features>`
    where `mf` is `sqrt`, `0p1`, `0p25`, `0p5`, or `1p0`.
- Boosting methods:
  - `adaboost_*` = AdaBoost with different estimator counts (10, 20, 30).
  - `boost_dt_*` = boosted decision-tree models, pattern `boost_dt_<n_estimators>x<learning_rate>`.
  - `sgb_*` = stochastic gradient boosting with varying numbers of estimators (10–1000).
  - `xgb_*` = XGBoost models (estimators and learning rates vary).
  - `lgbm_*` = LightGBM models (estimators and learning rates vary).
- Instance-based and deep models:
  - `knn_*` and `knn_tuned` = k-nearest-neighbour classifiers.
  - `torch_mlp`, `torch_tcn`, `torch_transformer` = neural models (MLP, temporal convolutional network, transformer-style tabular model).

Each row in the benchmark table includes the following evaluation metrics:

- **AUC** – Area Under the ROC Curve (primary discrimination metric; higher is better).
- **PCC** – Prevalence-corrected classification accuracy.
- **F1** – Harmonic mean of precision and recall.
- **Recall** – Sensitivity to the default class (true-positive rate for defaults).
- **BS** – Brier score (mean squared error of predicted probabilities; lower is better and reflects calibration).
- **KS** – Kolmogorov–Smirnov statistic (maximum separation between score distributions of defaults vs non-defaults).
- **PG** – Partial Gini, capturing discrimination in conservative decision regions.
- **H**  – Hand's H-measure, a cost-sensitive alternative to AUC that incorporates misclassification costs.

Context (raw results from Python):
{results_text}{smote_info}

Write a **120–180 word** narrative that:

1. Identifies the best-performing model or models for this dataset, using **AUC** as the primary criterion but commenting also on **KS** and **H** for discrimination and cost-sensitive performance.
2. Compares **model families** (logistic regression vs tree ensembles vs boosting vs deep vs KNN) based on the metrics in the table, not just individual IDs.
3. Discusses trade-offs between discrimination and calibration: for example, where a model slightly sacrifices AUC but improves **BS**, **PG**, or **Recall** in a way that is relevant for default detection.
4. Briefly notes any effect of SMOTE on handling class imbalance if the table indicates that SMOTE was used.

Use precise academic language. Do **not** explain the experimental pipeline or the general definition of each metric; instead, interpret what the **pattern of numbers** says about the behaviour of the different model families on this dataset. Refer to the benchmark as "Table X" and to any accompanying plots as "Figure X" without inventing exact table or figure numbers. Avoid over-claiming; treat small performance differences as marginal unless they are clearly substantial.
"""
    
    def _get_benchmark_cross_dataset_narrative_prompt(
        self,
        benchmark_df: Optional[pd.DataFrame],
        all_results_df: Optional[pd.DataFrame]
    ) -> str:
        """
        Generate prompt for cross-dataset benchmark analysis.

        This version encodes model-family conventions and metric semantics so the LLM
        can write a deeper, cross-dataset explanation without changing the rest of
        the pipeline.
        """
        results_text = ""
        patterns_text = ""

        if benchmark_df is not None and not benchmark_df.empty:
            results_text = (
                "\nBenchmark Results Across Datasets:\n"
                f"{benchmark_df.to_string(max_rows=50)}"
            )
            if "Benchmark Model" in benchmark_df.columns:
                model_freq = benchmark_df["Benchmark Model"].value_counts()
                patterns_text += (
                    "\nFrequency with which each model appears as a per-dataset benchmark:\n"
                    f"{model_freq.to_string()}\n"
                )

        if all_results_df is not None and not all_results_df.empty:
            if "Model" in all_results_df.columns and "AUC" in all_results_df.columns:
                avg_perf = (
                    all_results_df.groupby("Model")["AUC"]
                    .agg(["mean", "std"])
                    .sort_values("mean", ascending=False)
                )
                patterns_text += (
                    "\nAverage AUC (mean and std) by model across all datasets "
                    "(top 15 models shown):\n"
                    f"{avg_perf.head(15).to_string()}\n"
                )

        return f"""You are writing the **Cross-Dataset Model Performance Analysis** subsection of a credit risk modeling paper. Your goal is to go beyond simple description and provide a genuine explanation of the patterns in the benchmark results.

Model IDs follow these families and conventions (same as in the dataset-level benchmark prompt):

- Logistic regression: `lr_*` (plain LR) and `lr_reg_*` (regularised LR).
- Bagging / ensembles: `bag_cart_*` (bagged CART), `bagnn_*` (bagged neural nets), `rf_*` (random forests with different numbers of trees and max-features settings).
- Boosting: `adaboost_*`, `boost_dt_*`, `sgb_*`, `xgb_*`, `lgbm_*` with varying estimators and learning rates.
- Instance-based / deep: `knn_*`, `knn_tuned`, `torch_mlp`, `torch_tcn`, `torch_transformer`.

Evaluation metrics are as defined earlier:
AUC (discrimination), PCC (prevalence-corrected accuracy), F1, Recall, BS (calibration error), KS, PG, and H (Hand's H-measure).

Context from Python:
{results_text}
{patterns_text}

Write a **200–300 word** narrative that:

1. **Identifies invariant patterns across datasets:** Which model families or specific models consistently appear as benchmarks or have the highest average AUC across datasets? Use the frequency and average-AUC summaries as evidence.
2. **Explains mechanisms, not just rankings:** For example, why might tree-based ensembles (RF, boosted trees, gradient boosting, XGBoost, LightGBM) generalise better than plain logistic regression or KNN across heterogeneous credit datasets? Relate this to how these models represent decision boundaries, handle interactions, and manage bias–variance trade-offs.
3. **Analyzes failure modes and trade-offs:** Where do high-AUC models show weaknesses (e.g., poorer BS or H, or unstable performance across datasets)? What does this suggest about calibration, cost sensitivity, and robustness in credit risk modelling?
4. **Synthesizes implications for practice:** What do these cross-dataset patterns imply for choosing modelling families under Basel-style model-risk governance, where both predictive power and stability/interpretability matter?

Use precise academic language, and ground every claim either in the numeric patterns shown in the tables or in well-known properties of these model families. Do **not** invent datasets or metrics that do not exist in the inputs. Refer to the tables and figures generically as "Table X" and "Figure X". The goal is to produce a hard-to-vary explanation of *why* certain model families perform the way they do across datasets, not just a restatement of the scores.
"""
    
    def _get_shap_narrative_prompt(self, dataset_name: str, result_manager: Any) -> str:
        """Generate prompt for SHAP subsection narrative."""
        
        # Get SHAP results for context
        try:
            shap_content = self.get_shap_content(dataset_name, result_manager)
            rank_text = ""
            if shap_content.get('reliability_results') and shap_content['reliability_results'].get('rank_csv'):
                rank_path = shap_content['reliability_results']['rank_csv']
                df = pd.read_csv(rank_path)
                rank_text = f"\nTop Features by SHAP:\n{df.head(10).to_string()}"
        except Exception:
            rank_text = ""
        
        return f"""You are writing a concise academic commentary for the SHAP Analysis subsection of dataset: {dataset_name}.

Context:
{rank_text}

Write a brief 100-150 word narrative that:
1. Identifies the most influential features for credit risk prediction
2. Interprets feature importance patterns (what drives predictions?)
3. Notes feature stability (low rank std = stable, high = unstable)
4. Relates findings to credit risk domain knowledge

Use academic language. DO NOT explain SHAP methodology - focus only on interpreting the feature importance results.
Reference figures/tables naturally.
Keep it concise and actionable."""
    
    def _get_reliability_narrative_prompt(self, dataset_name: str, result_manager: Any) -> str:
        """Generate prompt for Reliability subsection narrative."""
        
        # Get diagnostics for context
        diag_text = ""
        try:
            rel_content = self.get_reliability_content(dataset_name, result_manager)
            if rel_content.get('diagnostics'):
                diag_text = f"\nDiagnostics:\n{json.dumps(rel_content['diagnostics'], indent=2)}"
        except Exception:
            pass
        
        return f"""You are writing a concise academic commentary for the Reliability subsection of dataset: {dataset_name}.

Context:
{diag_text}

Write a brief 100-150 word narrative that interprets the available reliability metrics. Focus on: (1) reliability bucket ranges and distribution of predictions across confidence levels, (2) reliability summary statistics showing model discrimination ability (ROC AUC, KS statistic), (3) statistical test results (Mann-Whitney, logistic regression) indicating separation between correct and incorrect predictions, and (4) implications for model trustworthiness and deployment confidence.

Use academic language. Analyze the actual metrics provided - do not mention missing data or lack of reliability information. Reference tables naturally."""
    
    def _get_local_narrative_prompt(self, result_manager: Any) -> str:
        """Generate prompt for Local SHAP subsection narrative."""
        
        # Count available local analyses
        try:
            manifest = result_manager.prepare_report_manifest()
            local_results = manifest.get('local_analyses', [])
            n_examples = len(local_results) if local_results else 0
        except Exception:
            n_examples = 0
        
        return f"""You are writing a concise academic commentary for the Local SHAP Analysis subsection.

Context:
{n_examples} example predictions with local explanations available.

Write a brief 80-120 word narrative that:
1. Explains the purpose of local SHAP (individual prediction explanations)
2. Describes what waterfall plots reveal (feature contributions to specific predictions)
3. Notes insights from example cases (which features drove these particular decisions)
4. Discusses value for model transparency and trust

Use academic language. DO NOT explain SHAP methodology - focus only on interpreting the example explanations.
Reference figures naturally.
Keep it concise and interpretability-focused."""
    
    def _get_paired_comparison_narrative_prompt(self, comparison_id: int, result_manager: Any) -> str:
        """Generate prompt for a specific paired comparison narrative."""
        
        # Load the comparison data
        try:
            comparisons = result_manager.get_paired_comparisons()
            if comparison_id <= len(comparisons):
                comparison = comparisons[comparison_id - 1]
            else:
                comparison = {}
        except Exception:
            comparison = {}
        
        dataset = comparison.get('dataset', 'Unknown')
        model_a = comparison.get('model_a', 'Model A')
        model_b = comparison.get('model_b', 'Model B')
        
        wilcoxon = comparison.get('wilcoxon', {})
        mcnemar = comparison.get('mcnemar', {})
        delong = comparison.get('delong', {})
        
        # Build results summary
        results_summary = f"""
Dataset: {dataset}
Model A: {model_a}
Model B: {model_b}

Wilcoxon Signed-Rank Test (absolute error):
- Statistic: {wilcoxon.get('statistic', 'N/A')}
- p-value: {wilcoxon.get('p_value', 'N/A')}
- Median |Error| Difference (A-B): {wilcoxon.get('median_abs_error_diff', 'N/A')}

McNemar's Test (classification agreement):
- b (A correct, B wrong): {mcnemar.get('b', 'N/A')}
- c (A wrong, B correct): {mcnemar.get('c', 'N/A')}
- Chi-square: {mcnemar.get('chi2', 'N/A')}
- p-value: {mcnemar.get('p_value', 'N/A')}

DeLong Test (AUC difference):
{delong if delong else 'Not computed'}
"""
        
        return f"""You are writing a concise academic commentary for a paired statistical comparison between two credit risk models.

{results_summary}

Write a brief 100-150 word narrative that:
1. Interprets the Wilcoxon test results (which model has lower absolute errors, statistical significance)
2. Interprets McNemar's test (which model is more accurate on discordant cases, if any)
3. Interprets DeLong test (AUC difference significance if available)
4. Synthesizes the findings: Is one model clearly superior? Do tests agree? What does this mean for model selection?
5. Notes any caveats (e.g., small sample, non-significant differences, agreement between models)

**Important Statistical Interpretation Guidelines:**
- Wilcoxon p < 0.05: significant difference in error distributions
- McNemar p < 0.05: significant difference in classification accuracy
- DeLong p < 0.05: significant AUC difference
- Median error diff < 0 favors Model A; > 0 favors Model B
- b > c in McNemar means Model A more accurate on disagreements; c > b means Model B more accurate

Use academic language. Focus on practical implications for credit risk modeling.
Reference the results table naturally.
Keep it concise and decision-focused."""
    
    # ==================== PROMPT TEMPLATES ====================
    
    @staticmethod
    def _get_introduction_prompt(datasets: List[str], models: List[str]) -> str:
        """Generate prompt for introduction section."""
        return f"""You are writing the Introduction section of a research paper on credit risk modeling and machine learning explainability.

The study investigates:
- Datasets: {', '.join(datasets)}
- Models: {', '.join(models) if models else 'Multiple ML algorithms'}

Write a concise 3-4 paragraph introduction that:
1. Motivates the importance of credit risk modeling and explainability
2. Outlines the research objectives (benchmark models, SHAP analysis, reliability testing)
3. Briefly describes the datasets under investigation
4. States the contribution (comprehensive evaluation framework)

Use academic language suitable for a research paper. Do not include citations (we'll add them later).
Keep it under 400 words."""
    
    @staticmethod
    def _get_eda_prompt(dataset_name: str, summary_stats: Optional[pd.DataFrame], 
                        target_dist: Optional[pd.DataFrame], 
                        correlation: Optional[pd.DataFrame],
                        missing: Optional[pd.DataFrame]) -> str:
        """Generate prompt for EDA section."""
        
        # Prepare data summaries for the prompt
        stats_text = ""
        if summary_stats is not None:
            stats_text = f"\nSummary Statistics:\n{summary_stats.to_string(max_rows=20)}"
        
        target_text = ""
        if target_dist is not None:
            target_text = f"\nTarget Distribution:\n{target_dist.to_string()}"
        
        corr_text = ""
        if correlation is not None:
            # Get top correlations with target
            corr_text = f"\nTop correlations with target (first 10 features):\n{correlation.iloc[:10, 0].to_string()}"
        
        missing_text = ""
        if missing is not None:
            missing_text = f"\nMissing Values:\n{missing.to_string(max_rows=15)}"
        
        return f"""You are writing the Exploratory Data Analysis (EDA) section for the dataset: {dataset_name}.

Available data:
{stats_text}
{target_text}
{corr_text}
{missing_text}

Write a comprehensive 2-3 paragraph EDA narrative that:
1. Describes the dataset structure (number of features, observations)
2. Discusses the target variable distribution (class balance/imbalance)
3. Highlights key feature characteristics (ranges, distributions, missing values)
4. Notes important correlations or patterns
5. Mentions data quality issues if any (missing values, outliers)

Use academic language. Reference "Figure X" and "Table X" as placeholders for visualizations.
Keep it under 300 words per dataset."""
    
    @staticmethod
    def _get_benchmark_prompt(results_df: Optional[pd.DataFrame]) -> str:
        """Generate prompt for benchmark results section."""
        
        results_text = ""
        if results_df is not None:
            # Format results table for prompt
            results_text = f"\nBenchmark Results:\n{results_df.to_string(max_rows=30)}"
        
        return f"""You are writing the Benchmark Results section of a credit risk modeling paper.

Available results:
{results_text}

Write a comprehensive 3-4 paragraph analysis that:
1. Compares model performance across all datasets
2. Identifies best-performing model families (e.g., gradient boosting, neural nets, logistic regression)
3. Discusses trade-offs between model complexity and performance
4. Highlights any surprising results or patterns
5. Relates findings to credit risk modeling context (interpretability vs accuracy)

Reference "Table X" for the full results table and "Figure X" for comparison plots.
Use academic language. Keep it under 400 words."""
    
    @staticmethod
    def _get_shap_prompt(dataset_name: str, reliability_results: Optional[Dict]) -> str:
        """Generate prompt for SHAP analysis section."""
        
        rank_text = ""
        sanity_text = ""
        
        if reliability_results:
            rank_df = None
            if reliability_results.get('rank_csv'):
                try:
                    rank_df = pd.read_csv(reliability_results['rank_csv'])
                    rank_text = f"\nTop Features by SHAP Importance:\n{rank_df.head(10).to_string()}"
                except Exception:
                    pass
            
            if reliability_results.get('summary_txt'):
                try:
                    with open(reliability_results['summary_txt'], 'r') as f:
                        sanity_text = f"\nReliability Summary:\n{f.read()}"
                except Exception:
                    pass
        
        return f"""You are writing the SHAP Analysis section for dataset: {dataset_name}.

Available analysis:
{rank_text}
{sanity_text}

Write a comprehensive 3-4 paragraph narrative that:
1. Explains what SHAP values reveal about feature importance
2. Discusses the top influential features and their direction of impact
3. Interprets rank stability metrics (low std_rank = stable, high = unstable)
4. Explains the sanity ratio and what it indicates about model reliability
5. Provides actionable insights for credit risk decision-making

Reference "Figure X" for SHAP plots (bar plots, waterfall plots, etc.).
Use academic language. Keep it under 400 words."""
    
    @staticmethod
    def _get_reliability_prompt(dataset_name: str, diagnostics: Optional[Dict]) -> str:
        """Generate prompt for reliability diagnostics section."""
        
        diag_text = ""
        if diagnostics:
            diag_text = f"\nDiagnostic Test Results:\n{json.dumps(diagnostics, indent=2)}"
        
        return f"""You are writing the Reliability Diagnostics section for dataset: {dataset_name}.

Available diagnostics:
{diag_text}

Write a comprehensive 2-3 paragraph analysis interpreting the reliability framework. Address: (1) reliability bucket definitions and how predictions are distributed across confidence ranges, (2) discrimination metrics (ROC AUC, KS statistic) showing the model's ability to separate correct from incorrect predictions, (3) statistical test results (Mann-Whitney U, logistic regression) quantifying the relationship between reliability scores and prediction accuracy, and (4) practical implications for model deployment and trust calibration.

Use academic language. Focus on interpreting the available metrics - do not discuss missing data. Keep it under 350 words."""
    
    @staticmethod
    def _get_conclusion_prompt(content_summary: str) -> str:
        """Generate prompt for conclusion section."""
        
        return f"""You are writing the Conclusion section of a credit risk modeling and explainability research paper.

Summary of findings:
{content_summary}

Write a concise 2-3 paragraph conclusion that:
1. Synthesizes the main findings across benchmarking, SHAP analysis, and reliability testing
2. Highlights the most important insights for credit risk practitioners
3. Discusses limitations of the study
4. Suggests directions for future research

Use academic language. Keep it under 300 words."""
    
    # ==================== LLM GENERATION ====================
    
    def generate_section_with_llm(self, prompt: str, llm_config: Dict[str, Any]) -> Optional[str]:
        """
        Generate a report section using LLM.
        
        Args:
            prompt: The prompt for the LLM
            llm_config: Dict with keys:
                - provider: 'openai' or 'anthropic'
                - api_key: API key
                - model: Model name (e.g., 'gpt-4', 'claude-3-sonnet')
                - temperature: Temperature (0.0-1.0)
                - max_tokens: Maximum tokens
        
        Returns:
            Generated text or None on failure
        """
        provider = llm_config.get('provider', 'openai').lower()
        api_key = llm_config.get('api_key', '')
        model = llm_config.get('model', 'gpt-3.5-turbo')
        temperature = llm_config.get('temperature', 0.3)
        max_tokens = llm_config.get('max_tokens', 800)
        
        if not api_key:
            return None
        
        try:
            if provider == 'openai':
                if not OPENAI_AVAILABLE:
                    return None
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert academic writer specializing in machine learning and credit risk modeling research papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            
            elif provider == 'anthropic':
                if not ANTHROPIC_AVAILABLE:
                    return None
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text.strip()
            
            else:
                return None
                
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return None
    
    def generate_introduction(self, datasets: List[str], models: List[str], 
                            llm_config: Dict[str, Any]) -> str:
        """Generate introduction section using LLM."""
        prompt = self._get_introduction_prompt(datasets, models)
        result = self.generate_section_with_llm(prompt, llm_config)
        return result if result else "# Introduction\n\n[Introduction section could not be generated]"
    
    def generate_eda_section(self, dataset_name: str, eda_content: Dict[str, Any],
                            llm_config: Dict[str, Any]) -> str:
        """Generate EDA section using LLM."""
        prompt = self._get_eda_prompt(
            dataset_name,
            eda_content.get('summary_stats'),
            eda_content.get('target_distribution'),
            eda_content.get('correlation_matrix'),
            eda_content.get('missing_summary')
        )
        result = self.generate_section_with_llm(prompt, llm_config)
        return result if result else f"## {dataset_name}\n\n[EDA section could not be generated]"
    
    def generate_benchmark_section(self, benchmark_content: Dict[str, Any],
                                   llm_config: Dict[str, Any]) -> str:
        """Generate benchmark results section using LLM."""
        prompt = self._get_benchmark_prompt(benchmark_content.get('results_df'))
        result = self.generate_section_with_llm(prompt, llm_config)
        return result if result else "# Benchmark Results\n\n[Benchmark section could not be generated]"
    
    def generate_shap_section(self, dataset_name: str, shap_content: Dict[str, Any],
                             llm_config: Dict[str, Any]) -> str:
        """Generate SHAP analysis section using LLM."""
        prompt = self._get_shap_prompt(dataset_name, shap_content.get('reliability_results'))
        result = self.generate_section_with_llm(prompt, llm_config)
        return result if result else f"## {dataset_name}\n\n[SHAP section could not be generated]"
    
    def generate_reliability_section(self, dataset_name: str, reliability_content: Dict[str, Any],
                                    llm_config: Dict[str, Any]) -> str:
        """Generate reliability diagnostics section using LLM."""
        prompt = self._get_reliability_prompt(dataset_name, reliability_content.get('diagnostics'))
        result = self.generate_section_with_llm(prompt, llm_config)
        return result if result else f"## {dataset_name}\n\n[Reliability section could not be generated]"
    
    def generate_conclusion(self, content_summary: str, llm_config: Dict[str, Any]) -> str:
        """Generate conclusion section using LLM."""
        prompt = self._get_conclusion_prompt(content_summary)
        result = self.generate_section_with_llm(prompt, llm_config)
        return result if result else "# Conclusion\n\n[Conclusion could not be generated]"
    
    def assemble_full_report(self, sections: Dict[str, str], title: str = "Credit Risk Modeling and Explainability Analysis") -> str:
        """
        Assemble all sections into a complete markdown report.
        
        Args:
            sections: Dict with keys like 'introduction', 'eda', 'benchmark', etc.
            title: Report title
            
        Returns:
            Complete markdown document
        """
        timestamp = datetime.now().strftime("%B %d, %Y")
        
        report_parts = [
            f"# {title}\n",
            f"*Generated: {timestamp}*\n",
            "---\n",
            sections.get('introduction', ''),
            "\n\n---\n\n",
            "# Exploratory Data Analysis\n",
            sections.get('eda', ''),
            "\n\n---\n\n",
            sections.get('benchmark', ''),
            "\n\n---\n\n",
            "# SHAP Analysis\n",
            sections.get('shap', ''),
            "\n\n---\n\n",
            "# Reliability Analysis\n",
            sections.get('reliability', ''),
            "\n\n---\n\n",
            sections.get('conclusion', ''),
        ]
        
        return "\n".join(report_parts)
    
    # ==================== LATEX CONVERSION ====================
    
    # Academic-style LaTeX formatting helpers (for Results section)
    
    @staticmethod
    def _create_figure_latex_academic(fig_path: Path, caption: str, label: str,
                                       width: str = "0.75") -> str:
        """Create academic-style figure with [H] placement."""
        # Check if file exists
        if not Path(fig_path).exists():
            return f"% Figure not found: {fig_path}\n"
        
        fig_str = str(Path(fig_path).resolve()).replace('\\', '/')
        return f"""\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width={width}\\textwidth]{{{fig_str}}}
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{figure}}
\\FloatBarrier

"""
    
    @staticmethod
    def _create_shap_dual_plot_latex_academic(bar_path: Path, dot_path: Path, 
                                                dataset_name: str) -> str:
        """Create side-by-side SHAP plots (bar + dot) like in the example."""
        bar_str = str(Path(bar_path).resolve()).replace('\\', '/')
        dot_str = str(Path(dot_path).resolve()).replace('\\', '/')
        safe_name = dataset_name.replace('_', '\\_')
        
        return f"""\\begin{{figure}}[H]
\\centering
\\begin{{tabular}}{{cc}}
\\includegraphics[width=0.48\\textwidth]{{{bar_str}}} &
\\includegraphics[width=0.48\\textwidth]{{{dot_str}}} \\\\
Bar Plot & Summary Plot (Dot) \\\\
\\end{{tabular}}
\\caption{{Global SHAP plots for {safe_name}}}
\\label{{fig:shap_{dataset_name.replace('.', '_').replace(' ', '_')}}}
\\end{{figure}}
\\FloatBarrier

"""
    
    @staticmethod
    def _create_comparison_figures_latex_academic(figure_paths: List[Path]) -> str:
        """Create grid of model comparison figures (3x2 or 2x3 layout)."""
        paths_str = [str(Path(p).resolve()).replace('\\', '/') for p in figure_paths]
        
        # Determine grid layout based on number of figures
        if len(paths_str) <= 3:
            cols = len(paths_str)
            rows = 1
        elif len(paths_str) <= 6:
            cols = 3
            rows = 2
        else:
            cols = 3
            rows = (len(paths_str) + 2) // 3
        
        grid_parts = []
        for i, path in enumerate(paths_str):
            grid_parts.append(f"\\includegraphics[width=0.32\\textwidth]{{{path}}}")
            if (i + 1) % cols == 0 and i < len(paths_str) - 1:
                grid_parts.append("\\\\")
            elif i < len(paths_str) - 1:
                grid_parts.append("&")
        
        grid_str = '\n    '.join(grid_parts)
        
        return f"""\\begin{{figure}}[ht]
    \\centering
    \\begin{{tabular}}{{{'c' * cols}}}
        {grid_str}
    \\end{{tabular}}
    \\caption{{Model comparison across metrics.}}
    \\label{{fig:model_comparisons}}
\\end{{figure}}
\\FloatBarrier

"""
    
    @staticmethod
    def _create_summary_stats_table_latex(df: pd.DataFrame, dataset_name: str) -> str:
        """Create summary statistics table."""
        safe_name = dataset_name.replace('_', '\\_')

        # Select key columns if DataFrame is large
        if len(df.columns) > 8:
            key_cols = df.columns[:8].tolist()
            df = df[key_cols]

        # Escape underscores and percent signs in column names
        df_escaped = df.copy()
        df_escaped.columns = [col.replace('_', '\\_').replace('%', '\\%') for col in df_escaped.columns]

        # Escape % in index if it's string-based
        if df_escaped.index.dtype == 'object':
            df_escaped.index = df_escaped.index.astype(str).str.replace('%', '\\%', regex=False)

        # Format count row as integers, other rows with 3 decimal places
        # Convert count row to integers first
        if 'count' in df_escaped.index:
            df_escaped.loc['count'] = df_escaped.loc['count'].astype(float).round(0).astype(int)

        # Create custom formatters for each column to handle count vs other rows
        formatters = {}
        for col in df_escaped.columns:
            formatters[col] = lambda x: "{:.0f}".format(x) if isinstance(x, (int, float)) and x == int(x) else "{:.3f}".format(x) if isinstance(x, (int, float)) else str(x)

        latex_table = df_escaped.to_latex(
            index=True,
            formatters=formatters,
            escape=False,
            caption=f"Summary statistics for {safe_name}",
            label=f"tab:summary_{dataset_name.replace('.', '_')}"
        )

        # Add [H] placement specifier to keep table in place
        latex_table = latex_table.replace(r'\begin{table}', r'\begin{table}[H]')

        # Wrap tabular in adjustbox (only resize if necessary)
        latex_table = ReportManager._wrap_table_in_adjustbox(latex_table, standard_width=0.9)

        return latex_table + "\\FloatBarrier\n"
    
    @staticmethod
    def _create_all_models_table_latex_academic(df: pd.DataFrame, dataset_name: str = None) -> str:
        """Create comprehensive table of all model results for a specific dataset."""
        # Escape underscores and percent signs in string columns and column names
        df_clean = df.copy()
        df_clean.columns = [col.replace('_', '\\_').replace('%', '\\%') for col in df_clean.columns]
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.replace('_', '\\_', regex=False).str.replace('%', '\\%', regex=False)
        
        # Sort by Model Group and Model for better readability
        if 'Model Group' in df_clean.columns and 'Model' in df_clean.columns:
            df_clean = df_clean.sort_values(['Model Group', 'Model'])
        
        # Remove Dataset column if filtering by dataset
        if dataset_name and 'Dataset' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Dataset'])
        
        # Generate LaTeX with longtable for multi-page support
        safe_name = dataset_name.replace('_', '\\_') if dataset_name else 'All Datasets'
        label_name = dataset_name.replace('.', '_').replace(' ', '_') if dataset_name else 'all'

        # Get column specification from dataframe
        num_cols = len(df_clean.columns)
        col_spec = 'l' * num_cols  # left-aligned columns

        # Build header row
        header_row = ' & '.join(df_clean.columns) + ' \\\\'

        # Build data rows
        data_rows = []
        for _, row in df_clean.iterrows():
            row_values = []
            for val in row:
                if isinstance(val, float):
                    row_values.append(f"{val:.4f}")
                else:
                    row_values.append(str(val))
            data_rows.append(' & '.join(row_values) + ' \\\\')

        data_content = '\n'.join(data_rows)

        # Create longtable with repeating headers
        return f"""{{\\small
\\begin{{longtable}}{{@{{}}{col_spec}@{{}}}}
\\caption{{Comprehensive Model Performance Results --- {safe_name}}} \\label{{tab:all_models_{label_name}}} \\\\
\\toprule
{header_row}
\\midrule
\\endfirsthead

\\multicolumn{{{num_cols}}}{{c}}{{\\tablename\\ \\thetable\\ -- continued from previous page}} \\\\
\\toprule
{header_row}
\\midrule
\\endhead

\\midrule
\\multicolumn{{{num_cols}}}{{r}}{{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\endlastfoot

{data_content}
\\end{{longtable}}
}}

"""
    
    @staticmethod
    def _create_benchmark_table_latex_academic(df: pd.DataFrame, dataset_name: str = None) -> str:
        """Create benchmark results table for a specific dataset."""
        # Escape underscores and percent signs in string columns and column names
        df_clean = df.copy()
        df_clean.columns = [col.replace('_', '\\_').replace('%', '\\%') for col in df_clean.columns]
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.replace('_', '\\_', regex=False).str.replace('%', '\\%', regex=False)
        
        # Remove Dataset column if filtering by dataset
        if dataset_name and 'Dataset' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Dataset'])
        
        # Generate LaTeX with toprule/midrule/bottomrule
        latex_table = df_clean.to_latex(
            index=False,
            float_format="{:.4f}".format,
            escape=False
        )
        
        # Wrap in adjustbox for smart resizing (only when necessary)
        lines = latex_table.split('\n')
        
        # Find tabular environment
        tabular_start = next(i for i, line in enumerate(lines) if '\\begin{tabular}' in line)
        tabular_end = next(i for i, line in enumerate(lines) if '\\end{tabular}' in line)
        
        # Replace default rules with booktabs
        for i in range(len(lines)):
            if '\\hline' in lines[i]:
                if i == tabular_start + 1:
                    lines[i] = '\\toprule'
                elif i == tabular_end - 1:
                    lines[i] = '\\bottomrule'
                else:
                    lines[i] = '\\midrule'
        
        # Rebuild with adjustbox (only resize if table exceeds textwidth)
        table_content = '\n'.join(lines[tabular_start:tabular_end+1])
        
        safe_name = dataset_name.replace('_', '\\_') if dataset_name else 'All Datasets'
        label_name = dataset_name.replace('.', '_').replace(' ', '_') if dataset_name else 'all'

        return f"""\\begin{{table}}[H]
\\caption{{Benchmark Results --- {safe_name}}}
\\label{{tab:benchmark_{label_name}}}
\\centering
\\begin{{adjustbox}}{{max width=0.9\\textwidth}}
{table_content}
\\end{{adjustbox}}
\\end{{table}}
\\FloatBarrier

"""
    
    @staticmethod
    def _create_feature_rank_table_latex(df: pd.DataFrame, dataset_name: str) -> str:
        """Create feature importance ranking table."""
        safe_name = dataset_name.replace('_', '\\_')
        
        # Select top 10 features and key columns
        df_top = df.head(10).copy()
        key_cols = [c for c in ['feature', 'abs_mean', 'avg_rank', 'std_rank'] if c in df_top.columns]
        if key_cols:
            df_top = df_top[key_cols]
        
        # Escape underscores and percent signs in column names
        df_top.columns = [col.replace('_', '\\_').replace('%', '\\%') for col in df_top.columns]
        
        # Escape underscores and percent signs in feature names
        if 'feature' in df_top.columns:
            df_top['feature'] = df_top['feature'].astype(str).str.replace('_', '\\_', regex=False).str.replace('%', '\\%', regex=False)
        
        latex_table = df_top.to_latex(
            index=False,
            float_format="{:.4f}".format,
            escape=False,
            caption=f"Top features by SHAP importance for {safe_name}",
            label=f"tab:shap_rank_{dataset_name.replace('.', '_')}"
        )

        # Add [H] placement and wrap in adjustbox
        latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[H]', 1)
        latex_table = ReportManager._wrap_table_in_adjustbox(latex_table, standard_width=0.75)

        return latex_table + "\\FloatBarrier\n"
    
    @staticmethod
    def _create_feature_importance_table_latex(
        df: pd.DataFrame,
        dataset_name: str,
        table_type: str = "merged"
    ) -> str:
        """
        Create feature importance table from Step 1.2 results.
        
        Args:
            df: Feature importance DataFrame (merged, rf, or lr)
            dataset_name: Dataset name
            table_type: Type of table ('merged', 'rf', or 'lr')
            
        Returns:
            LaTeX table string
        """
        safe_name = ReportManager._escape_latex(dataset_name)
        
        # Select columns to display based on table type
        df_display = df.copy()
        
        if table_type == "merged":
            # For merged table: show rank, feature, rf_importance, lr_coef_abs, avg_score
            key_cols = [c for c in ['rank', 'feature', 'rf_importance', 'lr_coef_abs', 'avg_score'] 
                       if c in df_display.columns]
            if key_cols:
                df_display = df_display[key_cols]
            caption = f"Top features by combined RF/LR importance for {safe_name}"
            label = f"tab:fi_merged_{dataset_name.replace('.', '_').replace(' ', '_')}"
        elif table_type == "rf":
            # For RF: show rank, feature, importance
            key_cols = [c for c in ['rank', 'feature', 'importance'] if c in df_display.columns]
            if key_cols:
                df_display = df_display[key_cols]
            caption = f"RandomForest feature importance for {safe_name}"
            label = f"tab:fi_rf_{dataset_name.replace('.', '_').replace(' ', '_')}"
        else:  # lr
            # For LR: show rank, feature, importance (absolute coefficient)
            key_cols = [c for c in ['rank', 'feature', 'importance'] if c in df_display.columns]
            if key_cols:
                df_display = df_display[key_cols]
            caption = f"LogisticRegression L1 feature importance for {safe_name}"
            label = f"tab:fi_lr_{dataset_name.replace('.', '_').replace(' ', '_')}"
        
        # Escape special characters in column names
        df_display.columns = [
            ReportManager._escape_latex(str(col)) for col in df_display.columns
        ]
        
        # Escape special characters in feature names (if 'feature' column exists)
        for col in df_display.columns:
            if 'feature' in col.lower():
                df_display[col] = df_display[col].astype(str).apply(ReportManager._escape_latex)
        
        # Generate LaTeX table
        latex_table = df_display.to_latex(
            index=False,
            float_format="{:.4f}".format,
            escape=False,
            caption=caption,
            label=label
        )

        # Add [H] placement specifier for fixed positioning
        latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[H]', 1)

        # Replace default rules with booktabs
        latex_table = latex_table.replace('\\hline', '\\toprule', 1)
        latex_table = latex_table.replace('\\hline', '\\midrule', 1)
        latex_table = latex_table.replace('\\hline', '\\bottomrule', 1)

        # Wrap in adjustbox for consistent sizing
        latex_table = ReportManager._wrap_table_in_adjustbox(latex_table, standard_width=0.75)

        return latex_table + "\\FloatBarrier\n"
    
    @staticmethod
    def _create_reliability_table_latex(df: pd.DataFrame, dataset_name: str) -> str:
        """Create batch reliability results table."""
        safe_name = dataset_name.replace('_', '\\_')
        
        # Select key columns
        key_cols = [c for c in df.columns if c in [
            'row_index', 'reliability_score', 'pred_default', 
            'actual_default', 'bucket', 'correct'
        ]]
        if key_cols:
            df_show = df[key_cols].head(20).copy()  # Show first 20 rows
        else:
            df_show = df.head(20).copy()
        
        # Escape % in column names and string columns
        df_show.columns = [col.replace('_', '\\_').replace('%', '\\%') for col in df_show.columns]
        for col in df_show.columns:
            if df_show[col].dtype == 'object':
                df_show[col] = df_show[col].astype(str).str.replace('%', '\\%', regex=False)
        
        latex_table = df_show.to_latex(
            index=False,
            float_format="{:.4f}".format,
            escape=False,
            caption=f"Batch reliability results for {safe_name}",
            label=f"tab:batch_rel_{dataset_name.replace('.', '_')}"
        )

        # Add [H] placement and wrap in adjustbox
        latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[H]', 1)
        latex_table = ReportManager._wrap_table_in_adjustbox(latex_table, standard_width=0.75)

        return latex_table + "\\FloatBarrier\n"
    
    @staticmethod
    def _create_diagnostics_table_latex(diagnostics: Dict, dataset_name: str) -> str:
        """Create statistical diagnostics summary table."""
        safe_name = dataset_name.replace('_', '\\_')
        
        # Build summary table from diagnostics dict
        metrics = []
        values = []
        
        for key, val in diagnostics.items():
            if val is not None:
                metrics.append(key.replace('_', ' ').title().replace('%', '\\%'))
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val).replace('%', '\\%'))
        
        df = pd.DataFrame({'Metric': metrics, 'Value': values})
        
        latex_table = df.to_latex(
            index=False,
            escape=False,
            caption=f"Statistical diagnostics for {safe_name}",
            label=f"tab:diagnostics_{dataset_name.replace('.', '_')}"
        )

        # Add [H] placement and wrap in adjustbox
        latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[H]', 1)
        latex_table = ReportManager._wrap_table_in_adjustbox(latex_table, standard_width=0.75)

        return latex_table + "\\FloatBarrier\n"
    
    @staticmethod
    def convert_markdown_to_latex(markdown_text: str) -> str:
        """
        Convert markdown text to LaTeX.
        Handles headers, lists, emphasis, code blocks.
        """
        latex = markdown_text
        
        # Convert headers FIRST (before escaping #)
        latex = re.sub(r'^### (.+)$', r'\\subsubsection{\1}', latex, flags=re.MULTILINE)
        latex = re.sub(r'^## (.+)$', r'\\subsection{\1}', latex, flags=re.MULTILINE)
        latex = re.sub(r'^# (.+)$', r'\\section{\1}', latex, flags=re.MULTILINE)
        
        # Bold and italic (before escaping special chars)
        latex = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', latex)
        latex = re.sub(r'\*(.+?)\*', r'\\textit{\1}', latex)
        
        # Horizontal rules
        latex = latex.replace('---', '\\hrule')
        
        # Escape special characters (but NOT backslashes we just added)
        # We'll do this more carefully
        special_chars = {
            '%': r'\%',
            '$': r'\$',
            '&': r'\&',
            '_': r'\_',
        }
        for char, escaped in special_chars.items():
            latex = latex.replace(char, escaped)
        
        # Lists (after escaping)
        latex = re.sub(r'^\- (.+)$', r'\\item \1', latex, flags=re.MULTILINE)
        latex = re.sub(r'^(\d+)\. (.+)$', r'\\item \2', latex, flags=re.MULTILINE)
        
        return latex
    
    @staticmethod
    def insert_figure_latex(fig_path: Path, caption: str, label: str, width: str = "0.8\\textwidth") -> str:
        """
        Create LaTeX figure environment.
        
        Args:
            fig_path: Path to figure
            caption: Figure caption
            label: Figure label (e.g., 'fig:benchmark_comparison')
            width: Figure width (default: 0.8\textwidth)
        """
        # Convert to relative path if needed
        fig_str = str(fig_path).replace('\\', '/')
        
        return f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width={width}]{{{fig_str}}}
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{figure}}
\\FloatBarrier
"""
    
    @staticmethod
    def insert_table_latex(df: pd.DataFrame, caption: str, label: str) -> str:
        """
        Convert DataFrame to LaTeX table.
        
        Args:
            df: DataFrame to convert
            caption: Table caption
            label: Table label (e.g., 'tab:benchmark_results')
        """
        # Escape underscores in column names and string values
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.replace('_', '\\_', regex=False)
        
        latex_str = df_clean.to_latex(
            index=False,
            caption=caption,
            label=label,
            float_format="{:.4f}".format,
            escape=False
        )

        return latex_str + "\\FloatBarrier\n"
    
    def generate_latex_report(self, markdown_report: str, manifest: Dict[str, Any],
                             title: str = "Credit Risk Modeling and Explainability Analysis",
                             author: str = "Research Team") -> str:
        """
        Generate complete LaTeX document from markdown report and manifest.
        
        Args:
            markdown_report: Markdown version of report
            manifest: Result manifest with paths to figures/tables
            title: Document title
            author: Document author
            
        Returns:
            Complete LaTeX document
        """
        # Convert markdown to LaTeX
        body = self.convert_markdown_to_latex(markdown_report)
        
        # Build preamble
        preamble = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{amsmath}}
\\usepackage{{float}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{adjustbox}}

\\title{{{title}}}
\\author{{{author}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle
\\tableofcontents
\\newpage

"""
        
        # Add body
        full_latex = preamble + body
        
        # TODO: Insert figure references automatically based on manifest
        # For now, users will need to manually reference figures
        
        # Add closing
        full_latex += "\n\n\\end{document}\n"
        
        return full_latex
    
    # ==================== SAVE/LOAD ====================
    
    def save_report_markdown(self, markdown_text: str, timestamp: Optional[str] = None) -> Path:
        """
        Save markdown report to file.
        
        Args:
            markdown_text: The markdown content
            timestamp: Optional timestamp string (default: current time)
            
        Returns:
            Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"report_{timestamp}.md"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        
        return filepath
    
    def save_latex_report(self, latex_text: str, timestamp: Optional[str] = None) -> Path:
        """
        Save LaTeX report to file.
        
        Args:
            latex_text: The LaTeX content
            timestamp: Optional timestamp string (default: current time)
            
        Returns:
            Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"report_{timestamp}.tex"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_text)
        
        return filepath
    
    def save_report_metadata(self, config: Dict[str, Any], manifest: Dict[str, Any],
                            timestamp: Optional[str] = None) -> Path:
        """
        Save report generation metadata.
        
        Args:
            config: LLM configuration used
            manifest: Result manifest
            timestamp: Optional timestamp string
            
        Returns:
            Path to saved metadata file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metadata = {
            'timestamp': timestamp,
            'llm_config': {
                'provider': config.get('provider'),
                'model': config.get('model'),
                'temperature': config.get('temperature')
            },
            'manifest_summary': {
                'datasets': list(manifest.get('eda', {}).keys()),
                'has_benchmarks': bool(manifest.get('benchmarks')),
                'total_shap_plots': sum(len(plots) for plots in manifest.get('shap_plots', {}).values()),
                'total_reliability_analyses': len(manifest.get('reliability', {}))
            }
        }
        
        filename = f"metadata_{timestamp}.json"
        filepath = self.base_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def list_available_reports(self) -> List[Dict[str, Any]]:
        """
        List all previously generated reports.
        
        Returns:
            List of dicts with keys: timestamp, markdown_path, latex_path, metadata_path
        """
        reports = []
        
        # Find all markdown reports
        md_files = sorted(self.base_dir.glob("report_*.md"), reverse=True)
        
        for md_file in md_files:
            # Extract timestamp from filename
            match = re.search(r'report_(\d{8}_\d{6})\.md', md_file.name)
            if match:
                timestamp = match.group(1)
                
                latex_file = self.base_dir / f"report_{timestamp}.tex"
                metadata_file = self.base_dir / f"metadata_{timestamp}.json"
                
                reports.append({
                    'timestamp': timestamp,
                    'markdown_path': md_file,
                    'latex_path': latex_file if latex_file.exists() else None,
                    'metadata_path': metadata_file if metadata_file.exists() else None
                })
        
        return reports
    
    def load_report(self, report_path: Path) -> str:
        """
        Load existing report content.
        
        Args:
            report_path: Path to report file (.md or .tex)
            
        Returns:
            Report content as string
        """
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Failed to load report: {e}")
            return ""
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get summary of reports directory.
        
        Returns:
            Dict with file counts and total size
        """
        md_count = len(list(self.base_dir.glob("*.md")))
        tex_count = len(list(self.base_dir.glob("*.tex")))
        json_count = len(list(self.base_dir.glob("*.json")))
        
        total_size = sum(f.stat().st_size for f in self.base_dir.rglob("*") if f.is_file())
        
        return {
            'total_reports': md_count,
            'markdown_files': md_count,
            'latex_files': tex_count,
            'metadata_files': json_count,
            'total_size_mb': total_size / (1024 * 1024)
        }


# Singleton accessor
def get_report_manager(base_dir: str = "reports") -> ReportManager:
    """
    Get or create the singleton ReportManager instance.
    
    Args:
        base_dir: Base directory for reports (default: 'reports')
        
    Returns:
        ReportManager instance
    """
    return ReportManager(base_dir)
