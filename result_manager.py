"""
Result Management Module for ML Model Selection Process

This module centralizes all file I/O operations for saving and loading
analysis results, SHAP visualizations, reliability metrics, EDA outputs,
and report assets.

Directory Structure:
    results/
    ├── benchmark_results.csv
    ├── eda/
    │   ├── *_profile.html
    │   ├── *_summary_stats.csv
    │   ├── *_target_distribution.csv
    │   ├── *_correlation.csv
    │   ├── *_missing_values.csv
    │   ├── *_pairplot.png
    │   ├── *_histogram_*.png
    │   └── *_correlation_heatmap.png
    ├── figures/
    │   ├── shap_*_bar.png
    │   ├── shap_*_dot.png
    │   ├── shap_*_waterfall_*.png
    │   ├── shap_*_pdp_*.png
    │   └── model_comparison_*.png
    ├── reliability/
    │   ├── *_rank.csv
    │   ├── *_summary.txt
    │   └── *_diagnostics.json
    ├── batch/
    │   ├── batch_reliability_*.xlsx
    │   └── batch_reliability_*.csv
    └── local_analyses/
        └── *_row*_timestamp.json

Author: Generated for Streamlit ML App
Date: December 2025
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np


class ResultManager:
    """
    Centralized manager for all result file operations.
    
    Attributes:
        base_dir: Root directory for all results (default: ./results)
        eda_dir: Directory for exploratory data analysis outputs
        figures_dir: Directory for SHAP and comparison plots
        reliability_dir: Directory for reliability analysis outputs
        batch_dir: Directory for batch reliability computations
        local_analyses_dir: Directory for per-instance SHAP analyses
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize ResultManager with directory structure.
        
        Args:
            base_dir: Root directory for results. If None, uses ./results
        """
        if base_dir is None:
            # Default to results/ in the same directory as this file
            base_dir = Path(__file__).parent / "results"
        
        self.base_dir = Path(base_dir)
        self.figures_dir = self.base_dir / "figures"
        self.reliability_dir = self.base_dir / "reliability"
        self.batch_dir = self.base_dir / "batch"
        self.local_analyses_dir = self.base_dir / "local_analyses"
        self.eda_dir = self.base_dir / "eda"
        
        # Ensure directory structure exists
        self.ensure_directory_structure()
    
    def ensure_directory_structure(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [
            self.base_dir,
            self.figures_dir,
            self.reliability_dir,
            self.batch_dir,
            self.local_analyses_dir,
            self.eda_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def sanitize_name(name: str) -> str:
        """
        Convert dataset/model names to safe filenames.
        
        Args:
            name: Raw name (may contain spaces, special chars, .csv extension)
            
        Returns:
            Sanitized filename-safe string
            
        Example:
            >>> ResultManager.sanitize_name("Australian Credit.csv")
            'Australian_Credit'
        """
        # Remove .csv extension if present
        name = name.replace('.csv', '')
        # Replace spaces and special chars with underscores
        name = re.sub(r'[^\w\-]', '_', name)
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Strip leading/trailing underscores
        name = name.strip('_')
        return name
    
    # =========================================================================
    # BENCHMARK RESULTS
    # =========================================================================
    
    def save_benchmark_results(
        self,
        results_df: pd.DataFrame,
        filename: str = "benchmark_results.csv"
    ) -> Path:
        """
        Save benchmark model comparison results to CSV.
        
        Args:
            results_df: DataFrame with columns like Dataset, Model, AUC, Accuracy, etc.
            filename: Output filename (default: benchmark_results.csv)
            
        Returns:
            Path to saved CSV file
            
        Raises:
            ValueError: If results_df is empty
            IOError: If file cannot be written
        """
        if results_df is None or results_df.empty:
            raise ValueError("Cannot save empty benchmark results")
        
        output_path = self.base_dir / filename
        results_df.to_csv(output_path, index=False)
        return output_path
    
    def save_all_model_results(
        self,
        results_df: pd.DataFrame,
        filename: str = "all_model_results.csv"
    ) -> Path:
        """
        Save comprehensive results for all models (not just benchmarks).
        
        Args:
            results_df: DataFrame with all model results across all datasets
            filename: Output filename (default: all_model_results.csv)
            
        Returns:
            Path to saved CSV file
        """
        if results_df is None or results_df.empty:
            raise ValueError("Cannot save empty model results")
        
        output_path = self.base_dir / filename
        results_df.to_csv(output_path, index=False)
        return output_path
    
    def load_benchmark_results(
        self,
        filename: str = "benchmark_results.csv"
    ) -> Optional[pd.DataFrame]:
        """
        Load benchmark results from CSV.
        
        Args:
            filename: CSV filename to load
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        csv_path = self.base_dir / filename
        if not csv_path.exists():
            return None
        
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    
    def load_all_model_results(
        self,
        filename: str = "all_model_results.csv"
    ) -> Optional[pd.DataFrame]:
        """
        Load all model results from CSV.
        
        Args:
            filename: CSV filename to load
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        csv_path = self.base_dir / filename
        if not csv_path.exists():
            return None
        
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    
    def save_model_comparison_figure(
        self,
        fig,
        metric: str = "AUC",
        dpi: int = 300
    ) -> Path:
        """
        Save model comparison bar chart figure.
        
        Args:
            fig: Matplotlib figure object
            metric: Metric name for filename (e.g., "AUC", "Accuracy")
            dpi: Resolution for saved PNG
            
        Returns:
            Path to saved PNG file
        """
        safe_metric = self.sanitize_name(metric)
        output_path = self.figures_dir / f"model_comparison_{safe_metric}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        return output_path
    
    # =========================================================================
    # SHAP VISUALIZATIONS
    # =========================================================================
    
    def save_shap_bar_plot(
        self,
        fig,
        dataset_name: str,
        model_name: str,
        dpi: int = 150
    ) -> Path:
        """
        Save SHAP bar plot (feature importance).
        
        Args:
            fig: Matplotlib figure
            dataset_name: Dataset identifier
            model_name: Model identifier
            dpi: Resolution
            
        Returns:
            Path to saved PNG
        """
        safe_ds = self.sanitize_name(dataset_name)
        safe_model = self.sanitize_name(model_name)
        output_path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_bar.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        return output_path
    
    def save_shap_dot_plot(
        self,
        fig,
        dataset_name: str,
        model_name: str,
        dpi: int = 150
    ) -> Path:
        """
        Save SHAP dot plot (beeswarm).
        
        Args:
            fig: Matplotlib figure
            dataset_name: Dataset identifier
            model_name: Model identifier
            dpi: Resolution
            
        Returns:
            Path to saved PNG
        """
        safe_ds = self.sanitize_name(dataset_name)
        safe_model = self.sanitize_name(model_name)
        output_path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_dot.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        return output_path
    
    def save_shap_waterfall(
        self,
        fig,
        dataset_name: str,
        model_name: str,
        row_index: int,
        dpi: int = 150
    ) -> Path:
        """
        Save SHAP waterfall plot for a specific instance.
        
        Args:
            fig: Matplotlib figure
            dataset_name: Dataset identifier
            model_name: Model identifier
            row_index: Row/instance index
            dpi: Resolution
            
        Returns:
            Path to saved PNG
        """
        safe_ds = self.sanitize_name(dataset_name)
        safe_model = self.sanitize_name(model_name)
        output_path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_waterfall_row{row_index}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        return output_path
    
    def save_pdp_ice_plot(
        self,
        fig,
        dataset_name: str,
        model_name: str,
        row_index: int,
        feature_name: str,
        dpi: int = 150
    ) -> Path:
        """
        Save PDP/ICE plot for a specific feature and instance.
        
        Args:
            fig: Matplotlib figure
            dataset_name: Dataset identifier
            model_name: Model identifier
            row_index: Row/instance index
            feature_name: Feature being plotted
            dpi: Resolution
            
        Returns:
            Path to saved PNG
        """
        safe_ds = self.sanitize_name(dataset_name)
        safe_model = self.sanitize_name(model_name)
        safe_feat = self.sanitize_name(feature_name)
        output_path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_pdp_row{row_index}_{safe_feat}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        return output_path
    
    def get_shap_plot_path(
        self,
        plot_type: str,
        dataset_name: str,
        model_name: str = None,
        row_index: int = None,
        feature_name: str = None
    ) -> Optional[Path]:
        """
        Retrieve path to existing SHAP plot.
        
        Args:
            plot_type: One of 'bar', 'dot', 'waterfall', 'pdp'
            dataset_name: Dataset identifier
            model_name: Model identifier (required for all types)
            row_index: Row index (required for waterfall, pdp)
            feature_name: Feature name (required for pdp)
            
        Returns:
            Path if file exists, None otherwise
        """
        safe_ds = self.sanitize_name(dataset_name)
        
        if model_name:
            safe_model = self.sanitize_name(model_name)
        else:
            # Try to find any model for this dataset
            pattern = f"shap_{safe_ds}_*_{plot_type}.png"
            candidates = list(self.figures_dir.glob(pattern))
            return candidates[0] if candidates else None
        
        if plot_type == "bar":
            path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_bar.png"
        elif plot_type == "dot":
            path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_dot.png"
        elif plot_type == "waterfall":
            if row_index is None:
                return None
            path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_waterfall_row{row_index}.png"
        elif plot_type == "pdp":
            if row_index is None or feature_name is None:
                return None
            safe_feat = self.sanitize_name(feature_name)
            path = self.figures_dir / f"shap_{safe_ds}_{safe_model}_pdp_row{row_index}_{safe_feat}.png"
        else:
            return None
        
        return path if path.exists() else None
    
    # =========================================================================
    # RELIABILITY ANALYSIS
    # =========================================================================
    
    def save_reliability_results(
        self,
        dataset_name: str,
        rank_df: pd.DataFrame,
        sanity_ratio: float,
        summary_text: str
    ) -> Dict[str, Path]:
        """
        Save reliability analysis outputs (rank table, sanity ratio, summary).
        
        Args:
            dataset_name: Dataset identifier
            rank_df: DataFrame with feature rankings and statistics
            sanity_ratio: Computed sanity check ratio
            summary_text: Human-readable summary text
            
        Returns:
            Dictionary with keys 'csv', 'txt' pointing to saved files
        """
        safe_ds = self.sanitize_name(dataset_name)
        
        # Save rank table
        csv_path = self.reliability_dir / f"{safe_ds}_rank.csv"
        rank_df.to_csv(csv_path, index=False)
        
        # Save summary text (includes sanity_ratio)
        txt_path = self.reliability_dir / f"{safe_ds}_summary.txt"
        full_summary = f"Sanity Ratio: {sanity_ratio:.4f}\n\n{summary_text}"
        txt_path.write_text(full_summary, encoding='utf-8')
        
        return {"csv": csv_path, "txt": txt_path}
    
    def load_reliability_results(
        self,
        dataset_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load reliability analysis results.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            Dictionary with 'rank_df', 'sanity_ratio', 'summary' or None if not found
        """
        safe_ds = self.sanitize_name(dataset_name)
        
        csv_path = self.reliability_dir / f"{safe_ds}_rank.csv"
        txt_path = self.reliability_dir / f"{safe_ds}_summary.txt"
        
        if not csv_path.exists() or not txt_path.exists():
            return None
        
        try:
            rank_df = pd.read_csv(csv_path)
            summary_text = txt_path.read_text(encoding='utf-8')
            
            # Extract sanity_ratio from first line
            sanity_ratio = None
            if summary_text.startswith("Sanity Ratio:"):
                first_line = summary_text.split('\n')[0]
                sanity_ratio = float(first_line.split(':')[1].strip())
                summary_text = '\n'.join(summary_text.split('\n')[2:])  # Remove first 2 lines
            
            return {
                "rank_df": rank_df,
                "sanity_ratio": sanity_ratio,
                "summary": summary_text
            }
        except Exception:
            return None
    
    # =========================================================================
    # BATCH RELIABILITY
    # =========================================================================
    
    def save_batch_reliability(
        self,
        dataset_name: str,
        results_df: pd.DataFrame,
        buckets_df: pd.DataFrame
    ) -> Dict[str, Path]:
        """
        Save batch reliability results to Excel (2 sheets) and CSV.
        
        Args:
            dataset_name: Dataset identifier
            results_df: DataFrame with per-row reliability scores
            buckets_df: DataFrame with bucket definitions
            
        Returns:
            Dictionary with keys 'excel', 'csv' pointing to saved files
        """
        safe_ds = self.sanitize_name(dataset_name)
        
        # Save Excel with two sheets
        excel_path = self.batch_dir / f"batch_reliability_{safe_ds}.xlsx"
        try:
            with pd.ExcelWriter(str(excel_path), engine="xlsxwriter") as writer:
                results_df.to_excel(writer, sheet_name="row_scores", index=False)
                buckets_df.to_excel(writer, sheet_name="buckets", index=False)
        except Exception:
            excel_path = None
        
        # Save CSV fallback
        csv_path = self.batch_dir / f"batch_reliability_{safe_ds}.csv"
        results_df.to_csv(csv_path, index=False)
        
        return {"excel": excel_path, "csv": csv_path}
    
    def load_batch_reliability(
        self,
        dataset_name: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load batch reliability results.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            Dictionary with 'results', 'buckets' DataFrames or None
        """
        safe_ds = self.sanitize_name(dataset_name)
        excel_path = self.batch_dir / f"batch_reliability_{safe_ds}.xlsx"
        
        if excel_path.exists():
            try:
                results_df = pd.read_excel(excel_path, sheet_name="row_scores")
                buckets_df = pd.read_excel(excel_path, sheet_name="buckets")
                return {"results": results_df, "buckets": buckets_df}
            except Exception:
                pass
        
        # Fallback to CSV
        csv_path = self.batch_dir / f"batch_reliability_{safe_ds}.csv"
        if csv_path.exists():
            try:
                results_df = pd.read_csv(csv_path)
                return {"results": results_df, "buckets": None}
            except Exception:
                pass
        
        return None
    
    def save_diagnostics(
        self,
        dataset_name: str,
        diagnostics_dict: Dict[str, Any]
    ) -> Path:
        """
        Save statistical diagnostics to JSON.
        
        Args:
            dataset_name: Dataset identifier
            diagnostics_dict: Dict with KS, Mann-Whitney, ROC AUC, logistic regression stats
            
        Returns:
            Path to saved JSON file
        """
        safe_ds = self.sanitize_name(dataset_name)
        json_path = self.reliability_dir / f"{safe_ds}_diagnostics.json"
        
        # Convert numpy types to native Python for JSON serialization
        clean_dict = self._clean_for_json(diagnostics_dict)
        
        with open(json_path, 'w') as f:
            json.dump(clean_dict, f, indent=2)
        
        return json_path
    
    def load_diagnostics(
        self,
        dataset_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load statistical diagnostics from JSON.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            Diagnostics dictionary or None
        """
        safe_ds = self.sanitize_name(dataset_name)
        json_path = self.reliability_dir / f"{safe_ds}_diagnostics.json"
        
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    # =========================================================================
    # FEATURE IMPORTANCE (Step 1.2)
    # =========================================================================
    
    def save_feature_importance(
        self,
        dataset_name: str,
        rf_df: pd.DataFrame,
        lr_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Save feature importance results from Step 1.2 computation.
        
        Args:
            dataset_name: Dataset identifier
            rf_df: RandomForest feature importance DataFrame
            lr_df: LogisticRegression L1 coefficient importance DataFrame
            merged_df: Merged importance scores DataFrame
            metadata: Optional metadata dict (n_rows, n_cols, kept_columns, etc.)
            
        Returns:
            Dictionary with paths: {'rf': Path, 'lr': Path, 'merged': Path, 'meta': Path}
        """
        safe_ds = self.sanitize_name(dataset_name)
        saved_paths = {}
        
        # Save RandomForest importance
        rf_path = self.eda_dir / f"{safe_ds}_feature_importance_rf.csv"
        rf_df.to_csv(rf_path, index=False)
        saved_paths['rf'] = rf_path
        
        # Save LogisticRegression importance
        lr_path = self.eda_dir / f"{safe_ds}_feature_importance_lr.csv"
        lr_df.to_csv(lr_path, index=False)
        saved_paths['lr'] = lr_path
        
        # Save merged importance (primary table for reports)
        merged_path = self.eda_dir / f"{safe_ds}_feature_importance_merged.csv"
        merged_df.to_csv(merged_path, index=False)
        saved_paths['merged'] = merged_path
        
        # Save metadata if provided
        if metadata:
            meta_path = self.eda_dir / f"{safe_ds}_feature_importance_meta.json"
            clean_meta = self._clean_for_json(metadata)
            with open(meta_path, 'w') as f:
                json.dump(clean_meta, f, indent=2)
            saved_paths['meta'] = meta_path
        
        return saved_paths
    
    def load_feature_importance(
        self,
        dataset_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load feature importance results.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            Dictionary with 'rf', 'lr', 'merged' DataFrames and 'meta' dict, or None if not found
        """
        safe_ds = self.sanitize_name(dataset_name)
        
        rf_path = self.eda_dir / f"{safe_ds}_feature_importance_rf.csv"
        lr_path = self.eda_dir / f"{safe_ds}_feature_importance_lr.csv"
        merged_path = self.eda_dir / f"{safe_ds}_feature_importance_merged.csv"
        meta_path = self.eda_dir / f"{safe_ds}_feature_importance_meta.json"
        
        # Require at least the merged file
        if not merged_path.exists():
            return None
        
        try:
            result = {
                'merged': pd.read_csv(merged_path)
            }
            
            # Load optional files
            if rf_path.exists():
                result['rf'] = pd.read_csv(rf_path)
            if lr_path.exists():
                result['lr'] = pd.read_csv(lr_path)
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    result['meta'] = json.load(f)
            
            return result
        except Exception:
            return None
    
    # =========================================================================
    # FEATURE SELECTION (Step 1.4)
    # =========================================================================

    def save_feature_selection(
        self,
        dataset_name: str,
        selected_features: List[str],
        total_features: int,
        selection_method: str = "Manual",
        fi_source: str = None,
        quick_select_option: str = None
    ) -> Path:
        """
        Save feature selection metadata for Step 1.4.

        Args:
            dataset_name: Dataset identifier
            selected_features: List of selected feature names
            total_features: Total number of available features
            selection_method: "Manual", "Top-N from FI", etc.
            fi_source: Feature importance source if applicable ("Merged", "RandomForest", "L1-LR")
            quick_select_option: Quick select option if used ("Top 5", "Top 10", etc.)

        Returns:
            Path to saved JSON file
        """
        safe_ds = self.sanitize_name(dataset_name)

        metadata = {
            "dataset_name": dataset_name,
            "selected_features": selected_features,
            "num_selected": len(selected_features),
            "total_features": total_features,
            "selection_method": selection_method,
            "fi_source": fi_source,
            "quick_select_option": quick_select_option,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        json_path = self.eda_dir / f"{safe_ds}_feature_selection.json"
        clean_meta = self._clean_for_json(metadata)

        with open(json_path, 'w') as f:
            json.dump(clean_meta, f, indent=2)

        return json_path

    def load_feature_selection(
        self,
        dataset_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load feature selection metadata.

        Args:
            dataset_name: Dataset identifier

        Returns:
            Dictionary with feature selection info or None
        """
        safe_ds = self.sanitize_name(dataset_name)
        json_path = self.eda_dir / f"{safe_ds}_feature_selection.json"

        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    # =========================================================================
    # EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================

    def save_smote_distribution(
        self,
        dataset_name: str,
        original_counts: pd.Series,
        smote_counts: pd.Series
    ) -> Path:
        """
        Save SMOTE class distribution comparison.
        
        Args:
            dataset_name: Dataset identifier
            original_counts: Original class distribution
            smote_counts: Class distribution after SMOTE
            
        Returns:
            Path to saved CSV file
        """
        safe_ds = self.sanitize_name(dataset_name)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Class': original_counts.index,
            'Original_Count': original_counts.values,
            'After_SMOTE': smote_counts.values
        })
        
        # Calculate percentages
        comparison_df['Original_Pct'] = (comparison_df['Original_Count'] / comparison_df['Original_Count'].sum() * 100).round(0).astype(int)
        comparison_df['After_SMOTE_Pct'] = (comparison_df['After_SMOTE'] / comparison_df['After_SMOTE'].sum() * 100).round(0).astype(int)
        
        csv_path = self.eda_dir / f"{safe_ds}_smote_distribution.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def save_eda_profile(
        self,
        dataset_name: str,
        profile_html: str
    ) -> Path:
        """
        Save ydata-profiling/pandas-profiling HTML report.
        
        Args:
            dataset_name: Dataset identifier
            profile_html: HTML content of the profiling report
            
        Returns:
            Path to saved HTML file
        """
        safe_ds = self.sanitize_name(dataset_name)
        html_path = self.eda_dir / f"{safe_ds}_profile.html"
        html_path.write_text(profile_html, encoding='utf-8')
        return html_path
    
    def save_eda_summary(
        self,
        dataset_name: str,
        summary_stats: pd.DataFrame,
        target_distribution: Optional[pd.Series] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        missing_summary: Optional[pd.DataFrame] = None
    ) -> Dict[str, Path]:
        """
        Save comprehensive EDA summary statistics for a dataset.
        
        Args:
            dataset_name: Dataset identifier
            summary_stats: DataFrame.describe() output
            target_distribution: Target variable value counts
            correlation_matrix: Correlation matrix between features
            missing_summary: Missing value statistics per column
            
        Returns:
            Dictionary with paths: {'summary': Path, 'target': Path, 'correlation': Path, 'missing': Path}
        """
        safe_ds = self.sanitize_name(dataset_name)
        saved_paths = {}
        
        # Save summary statistics
        if summary_stats is not None and not summary_stats.empty:
            summary_path = self.eda_dir / f"{safe_ds}_summary_stats.csv"
            summary_stats.to_csv(summary_path)
            saved_paths['summary'] = summary_path
        
        # Save target distribution
        if target_distribution is not None:
            target_path = self.eda_dir / f"{safe_ds}_target_distribution.csv"
            target_distribution.to_csv(target_path, header=['count'])
            saved_paths['target'] = target_path
        
        # Save correlation matrix
        if correlation_matrix is not None and not correlation_matrix.empty:
            corr_path = self.eda_dir / f"{safe_ds}_correlation.csv"
            correlation_matrix.to_csv(corr_path)
            saved_paths['correlation'] = corr_path
        
        # Save missing value summary
        if missing_summary is not None and not missing_summary.empty:
            missing_path = self.eda_dir / f"{safe_ds}_missing_values.csv"
            missing_summary.to_csv(missing_path, index=False)
            saved_paths['missing'] = missing_path
        
        return saved_paths
    
    def save_eda_visualization(
        self,
        fig,
        dataset_name: str,
        plot_type: str,
        feature_name: str = None,
        dpi: int = 150
    ) -> Path:
        """
        Save EDA visualizations (histograms, pairplots, distributions, etc.).
        
        Args:
            fig: Matplotlib/seaborn figure
            dataset_name: Dataset identifier
            plot_type: Type of plot ('histogram', 'pairplot', 'distribution', 'boxplot', 'correlation_heatmap')
            feature_name: Optional feature name for single-feature plots
            dpi: Resolution
            
        Returns:
            Path to saved PNG
        """
        safe_ds = self.sanitize_name(dataset_name)
        
        if feature_name:
            safe_feat = self.sanitize_name(feature_name)
            filename = f"{safe_ds}_{plot_type}_{safe_feat}.png"
        else:
            filename = f"{safe_ds}_{plot_type}.png"
        
        output_path = self.eda_dir / filename
        fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
        return output_path
    
    def load_eda_profile(
        self,
        dataset_name: str
    ) -> Optional[str]:
        """
        Load saved profiling report HTML.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            HTML string or None if not found
        """
        safe_ds = self.sanitize_name(dataset_name)
        html_path = self.eda_dir / f"{safe_ds}_profile.html"
        
        if not html_path.exists():
            return None
        
        try:
            return html_path.read_text(encoding='utf-8')
        except Exception:
            return None
    
    def load_eda_summary(
        self,
        dataset_name: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load EDA summary statistics.
        
        Args:
            dataset_name: Dataset identifier
            
        Returns:
            Dictionary with 'summary', 'target', 'correlation', 'missing' DataFrames or None
        """
        safe_ds = self.sanitize_name(dataset_name)
        result = {}
        
        # Load summary stats
        summary_path = self.eda_dir / f"{safe_ds}_summary_stats.csv"
        if summary_path.exists():
            try:
                result['summary'] = pd.read_csv(summary_path, index_col=0)
            except Exception:
                pass
        
        # Load target distribution
        target_path = self.eda_dir / f"{safe_ds}_target_distribution.csv"
        if target_path.exists():
            try:
                result['target'] = pd.read_csv(target_path, index_col=0)
            except Exception:
                pass
        
        # Load correlation matrix
        corr_path = self.eda_dir / f"{safe_ds}_correlation.csv"
        if corr_path.exists():
            try:
                result['correlation'] = pd.read_csv(corr_path, index_col=0)
            except Exception:
                pass
        
        # Load missing values
        missing_path = self.eda_dir / f"{safe_ds}_missing_values.csv"
        if missing_path.exists():
            try:
                result['missing'] = pd.read_csv(missing_path)
            except Exception:
                pass
        
        # Load SMOTE distribution
        smote_path = self.eda_dir / f"{safe_ds}_smote_distribution.csv"
        if smote_path.exists():
            try:
                result['smote'] = pd.read_csv(smote_path)
            except Exception:
                pass
        
        return result if result else None
    
    def get_eda_visualizations(
        self,
        dataset_name: str = None
    ) -> List[Path]:
        """
        Get all EDA visualization files for a dataset.
        
        Args:
            dataset_name: Dataset identifier (None = all datasets)
            
        Returns:
            List of PNG paths
        """
        if dataset_name:
            safe_ds = self.sanitize_name(dataset_name)
            pattern = f"{safe_ds}_*.png"
        else:
            pattern = "*.png"
        
        return sorted(self.eda_dir.glob(pattern))
    
    # =========================================================================
    # LOCAL ANALYSIS TRACKING
    # =========================================================================
    # =========================================================================
    
    def save_local_analysis(
        self,
        dataset_name: str,
        row_index: int,
        analysis_record: Dict[str, Any]
    ) -> Path:
        """
        Save a single local SHAP analysis to JSON.
        
        Args:
            dataset_name: Dataset identifier
            row_index: Row/instance index
            analysis_record: Dictionary with instance features, SHAP values, commentary, etc.
            
        Returns:
            Path to saved JSON file
        """
        safe_ds = self.sanitize_name(dataset_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_ds}_row{row_index}_{timestamp}.json"
        json_path = self.local_analyses_dir / filename
        
        # Add metadata
        record_with_meta = {
            "dataset": dataset_name,
            "row_index": row_index,
            "timestamp": timestamp,
            **analysis_record
        }
        
        clean_record = self._clean_for_json(record_with_meta)
        
        with open(json_path, 'w') as f:
            json.dump(clean_record, f, indent=2)
        
        return json_path
    
    def get_local_analyses(
        self,
        dataset_name: str = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored local analyses, optionally filtered.

        Args:
            dataset_name: Filter by dataset (None = all datasets)
            limit: Maximum number of results (None = all, most recent first)

        Returns:
            List of analysis record dictionaries
        """
        # Search for both old format (*_row*.json) and new format (local_analysis_*.json)
        json_files = []

        if dataset_name:
            safe_ds = self.sanitize_name(dataset_name)
            pattern = f"{safe_ds}_row*.json"
            json_files.extend(list(self.local_analyses_dir.glob(pattern)))
        else:
            # Search for old format
            pattern = "*_row*.json"
            json_files.extend(list(self.local_analyses_dir.glob(pattern)))

            # Also search for new format (local_analysis_*.json)
            new_pattern = "local_analysis_*.json"
            json_files.extend(list(self.local_analyses_dir.glob(new_pattern)))

        # Sort by modification time (most recent first) and remove duplicates
        json_files = sorted(
            set(json_files),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if limit:
            json_files = json_files[:limit]

        analyses = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    analysis_data = json.load(f)
                    # Filter by dataset if specified
                    if dataset_name and analysis_data.get('dataset') != dataset_name:
                        continue
                    analyses.append(analysis_data)
            except Exception:
                continue

        return analyses
    
    def save_paired_comparison(
        self,
        dataset_name: str,
        model_a_name: str,
        model_b_name: str,
        wilcoxon_results: Dict[str, Any],
        mcnemar_results: Dict[str, Any],
        delong_results: Dict[str, Any],
        timestamp: str = None
    ) -> Path:
        """
        Save paired statistical comparison results.
        
        Args:
            dataset_name: Dataset used for comparison
            model_a_name: Name of model A (format: "group::name")
            model_b_name: Name of model B (format: "group::name")
            wilcoxon_results: Dict with 'statistic', 'p_value', 'median_abs_error_diff'
            mcnemar_results: Dict with 'b', 'c', 'chi2', 'p_value'
            delong_results: Dict with DeLong test results or None
            timestamp: Optional timestamp string
            
        Returns:
            Path to saved JSON file
        """
        import datetime
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create paired_comparisons directory
        comparisons_dir = self.base_dir / "paired_comparisons"
        comparisons_dir.mkdir(exist_ok=True)
        
        # Create filename with counter to avoid conflicts
        safe_ds = self.sanitize_name(dataset_name)
        counter = 1
        while True:
            json_path = comparisons_dir / f"{safe_ds}_comparison_{counter:03d}.json"
            if not json_path.exists():
                break
            counter += 1
        
        record = {
            "comparison_id": counter,
            "timestamp": timestamp,
            "dataset": dataset_name,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "wilcoxon": wilcoxon_results,
            "mcnemar": mcnemar_results,
            "delong": delong_results
        }
        
        clean_record = self._clean_for_json(record)
        
        with open(json_path, 'w') as f:
            json.dump(clean_record, f, indent=2)
        
        return json_path
    
    def get_paired_comparisons(
        self,
        dataset_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored paired comparison results.
        
        Args:
            dataset_name: Filter by dataset (None = all datasets)
            
        Returns:
            List of comparison record dictionaries
        """
        comparisons_dir = self.base_dir / "paired_comparisons"
        if not comparisons_dir.exists():
            return []
        
        if dataset_name:
            safe_ds = self.sanitize_name(dataset_name)
            pattern = f"{safe_ds}_comparison_*.json"
        else:
            pattern = "*_comparison_*.json"
        
        json_files = sorted(
            comparisons_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )
        
        comparisons = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    comparisons.append(json.load(f))
            except Exception:
                continue
        
        return comparisons
    
    # =========================================================================
    # REPORT GENERATION SUPPORT
    # =========================================================================
    
    def prepare_report_manifest(self) -> Dict[str, Any]:
        """
        Create a structured manifest of all available results for report generation.
        
        Returns:
            Dictionary with organized paths to all result files:
            {
                "benchmarks": {"csv": Path, "figures": [Path, ...]},
                "eda": {"dataset": {"profile": Path, "summary": Path, "visualizations": [Path]}},
                "shap_plots": {"dataset": {"bar": [Path], "dot": [Path], ...}},
                "reliability": {"dataset": {"csv": Path, "txt": Path, "diagnostics": Path}},
                "batch_reliability": {"dataset": {"excel": Path, "csv": Path}},
                "local_analyses": [record, ...]
            }
        """
        manifest = {
            "benchmarks": self._manifest_benchmarks(),
            "eda": self._manifest_eda(),
            "shap_plots": self._manifest_shap_plots(),
            "reliability": self._manifest_reliability(),
            "batch_reliability": self._manifest_batch(),
            "local_analyses": self.get_local_analyses(limit=50),
            "paired_comparisons": self.get_paired_comparisons()
        }
        
        return manifest
    
    def _manifest_benchmarks(self) -> Dict[str, Any]:
        """Gather benchmark-related files."""
        csv_path = self.base_dir / "benchmark_results.csv"
        all_results_path = self.base_dir / "all_model_results.csv"
        figures = list(self.figures_dir.glob("model_comparison_*.png"))
        
        return {
            "csv": csv_path if csv_path.exists() else None,
            "all_results_csv": all_results_path if all_results_path.exists() else None,
            "figures": figures
        }
    
    def _manifest_eda(self) -> Dict[str, Dict[str, Any]]:
        """Organize EDA files by dataset."""
        eda_results = {}
        
        # Find all profile HTML files
        for html_file in self.eda_dir.glob("*_profile.html"):
            dataset = html_file.stem.replace('_profile', '')
            
            # Gather all related EDA files for this dataset
            eda_results[dataset] = {
                "profile": html_file,
                "summary_stats": self.eda_dir / f"{dataset}_summary_stats.csv" if (self.eda_dir / f"{dataset}_summary_stats.csv").exists() else None,
                "target_distribution": self.eda_dir / f"{dataset}_target_distribution.csv" if (self.eda_dir / f"{dataset}_target_distribution.csv").exists() else None,
                "correlation": self.eda_dir / f"{dataset}_correlation.csv" if (self.eda_dir / f"{dataset}_correlation.csv").exists() else None,
                "missing_values": self.eda_dir / f"{dataset}_missing_values.csv" if (self.eda_dir / f"{dataset}_missing_values.csv").exists() else None,
                "smote_distribution": self.eda_dir / f"{dataset}_smote_distribution.csv" if (self.eda_dir / f"{dataset}_smote_distribution.csv").exists() else None,
                "feature_selection": self.eda_dir / f"{dataset}_feature_selection.json" if (self.eda_dir / f"{dataset}_feature_selection.json").exists() else None,
                "visualizations": list(self.eda_dir.glob(f"{dataset}_*.png")),
                "feature_importance": {
                    "rf": self.eda_dir / f"{dataset}_feature_importance_rf.csv" if (self.eda_dir / f"{dataset}_feature_importance_rf.csv").exists() else None,
                    "lr": self.eda_dir / f"{dataset}_feature_importance_lr.csv" if (self.eda_dir / f"{dataset}_feature_importance_lr.csv").exists() else None,
                    "merged": self.eda_dir / f"{dataset}_feature_importance_merged.csv" if (self.eda_dir / f"{dataset}_feature_importance_merged.csv").exists() else None,
                    "meta": self.eda_dir / f"{dataset}_feature_importance_meta.json" if (self.eda_dir / f"{dataset}_feature_importance_meta.json").exists() else None,
                }
            }
        
        # Also check for datasets with summary stats but no profile
        for csv_file in self.eda_dir.glob("*_summary_stats.csv"):
            dataset = csv_file.stem.replace('_summary_stats', '')
            if dataset not in eda_results:
                eda_results[dataset] = {
                    "profile": None,
                    "summary_stats": csv_file,
                    "target_distribution": self.eda_dir / f"{dataset}_target_distribution.csv" if (self.eda_dir / f"{dataset}_target_distribution.csv").exists() else None,
                    "correlation": self.eda_dir / f"{dataset}_correlation.csv" if (self.eda_dir / f"{dataset}_correlation.csv").exists() else None,
                    "missing_values": self.eda_dir / f"{dataset}_missing_values.csv" if (self.eda_dir / f"{dataset}_missing_values.csv").exists() else None,
                    "smote_distribution": self.eda_dir / f"{dataset}_smote_distribution.csv" if (self.eda_dir / f"{dataset}_smote_distribution.csv").exists() else None,
                    "feature_selection": self.eda_dir / f"{dataset}_feature_selection.json" if (self.eda_dir / f"{dataset}_feature_selection.json").exists() else None,
                    "visualizations": list(self.eda_dir.glob(f"{dataset}_*.png")),
                    "feature_importance": {
                        "rf": self.eda_dir / f"{dataset}_feature_importance_rf.csv" if (self.eda_dir / f"{dataset}_feature_importance_rf.csv").exists() else None,
                        "lr": self.eda_dir / f"{dataset}_feature_importance_lr.csv" if (self.eda_dir / f"{dataset}_feature_importance_lr.csv").exists() else None,
                        "merged": self.eda_dir / f"{dataset}_feature_importance_merged.csv" if (self.eda_dir / f"{dataset}_feature_importance_merged.csv").exists() else None,
                        "meta": self.eda_dir / f"{dataset}_feature_importance_meta.json" if (self.eda_dir / f"{dataset}_feature_importance_meta.json").exists() else None,
                    }
                }
        
        # Also check for datasets with feature importance but no other EDA files
        for fi_file in self.eda_dir.glob("*_feature_importance_merged.csv"):
            dataset = fi_file.stem.replace('_feature_importance_merged', '')
            if dataset not in eda_results:
                eda_results[dataset] = {
                    "profile": None,
                    "summary_stats": self.eda_dir / f"{dataset}_summary_stats.csv" if (self.eda_dir / f"{dataset}_summary_stats.csv").exists() else None,
                    "target_distribution": self.eda_dir / f"{dataset}_target_distribution.csv" if (self.eda_dir / f"{dataset}_target_distribution.csv").exists() else None,
                    "correlation": self.eda_dir / f"{dataset}_correlation.csv" if (self.eda_dir / f"{dataset}_correlation.csv").exists() else None,
                    "missing_values": self.eda_dir / f"{dataset}_missing_values.csv" if (self.eda_dir / f"{dataset}_missing_values.csv").exists() else None,
                    "smote_distribution": self.eda_dir / f"{dataset}_smote_distribution.csv" if (self.eda_dir / f"{dataset}_smote_distribution.csv").exists() else None,
                    "feature_selection": self.eda_dir / f"{dataset}_feature_selection.json" if (self.eda_dir / f"{dataset}_feature_selection.json").exists() else None,
                    "visualizations": list(self.eda_dir.glob(f"{dataset}_*.png")),
                    "feature_importance": {
                        "rf": self.eda_dir / f"{dataset}_feature_importance_rf.csv" if (self.eda_dir / f"{dataset}_feature_importance_rf.csv").exists() else None,
                        "lr": self.eda_dir / f"{dataset}_feature_importance_lr.csv" if (self.eda_dir / f"{dataset}_feature_importance_lr.csv").exists() else None,
                        "merged": fi_file,
                        "meta": self.eda_dir / f"{dataset}_feature_importance_meta.json" if (self.eda_dir / f"{dataset}_feature_importance_meta.json").exists() else None,
                    }
                }
        
        return eda_results
    
    def _manifest_shap_plots(self) -> Dict[str, Dict[str, List[Path]]]:
        """Organize SHAP plots by dataset and type."""
        plots = {}
        
        for plot_type in ["bar", "dot", "waterfall", "pdp"]:
            for png_file in self.figures_dir.glob(f"shap_*_{plot_type}*.png"):
                # Extract dataset name from filename
                parts = png_file.stem.split('_')
                if len(parts) >= 3:
                    dataset = parts[1]  # e.g., "Australian" from "shap_Australian_Credit_..."
                    
                    if dataset not in plots:
                        plots[dataset] = {"bar": [], "dot": [], "waterfall": [], "pdp": []}
                    
                    plots[dataset][plot_type].append(png_file)
        
        return plots
    
    def _manifest_reliability(self) -> Dict[str, Dict[str, Optional[Path]]]:
        """Organize reliability files by dataset."""
        reliability = {}
        
        for csv_file in self.reliability_dir.glob("*_rank.csv"):
            dataset = csv_file.stem.replace('_rank', '')
            txt_file = self.reliability_dir / f"{dataset}_summary.txt"
            diag_file = self.reliability_dir / f"{dataset}_diagnostics.json"
            
            reliability[dataset] = {
                "csv": csv_file,
                "txt": txt_file if txt_file.exists() else None,
                "diagnostics": diag_file if diag_file.exists() else None
            }
        
        return reliability
    
    def _manifest_batch(self) -> Dict[str, Dict[str, Optional[Path]]]:
        """Organize batch reliability files by dataset."""
        batch = {}
        
        for excel_file in self.batch_dir.glob("batch_reliability_*.xlsx"):
            dataset = excel_file.stem.replace('batch_reliability_', '')
            csv_file = self.batch_dir / f"batch_reliability_{dataset}.csv"
            
            batch[dataset] = {
                "excel": excel_file,
                "csv": csv_file if csv_file.exists() else None
            }
        
        return batch
    
    def get_latex_friendly_paths(
        self,
        paths: List[Path],
        relative_to: str = "results"
    ) -> List[str]:
        """
        Convert absolute paths to LaTeX-friendly relative paths.
        
        Args:
            paths: List of Path objects
            relative_to: Strip this prefix from paths
            
        Returns:
            List of forward-slash string paths suitable for LaTeX
        """
        latex_paths = []
        for path in paths:
            path_str = str(path).replace('\\', '/')
            
            # Remove base prefix if present
            if relative_to in path_str:
                idx = path_str.find(relative_to)
                path_str = path_str[idx + len(relative_to):].lstrip('/')
            
            latex_paths.append(path_str)
        
        return latex_paths
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    @staticmethod
    def _clean_for_json(data: Any) -> Any:
        """
        Recursively convert numpy/pandas types to JSON-serializable types.
        
        Args:
            data: Any data structure
            
        Returns:
            JSON-serializable version
        """
        if isinstance(data, dict):
            return {k: ResultManager._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [ResultManager._clean_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif pd.isna(data):
            return None
        else:
            return data
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get statistics about stored results.
        
        Returns:
            Dictionary with counts and sizes:
            {
                "total_files": int,
                "total_size_mb": float,
                "by_type": {"benchmarks": int, "eda": int, "figures": int, ...}
            }
        """
        def count_files(directory: Path) -> Tuple[int, int]:
            """Count files and total size in bytes."""
            if not directory.exists():
                return 0, 0
            files = list(directory.rglob("*"))
            files = [f for f in files if f.is_file()]
            total_size = sum(f.stat().st_size for f in files)
            return len(files), total_size
        
        eda_count, eda_size = count_files(self.eda_dir)
        fig_count, fig_size = count_files(self.figures_dir)
        rel_count, rel_size = count_files(self.reliability_dir)
        batch_count, batch_size = count_files(self.batch_dir)
        local_count, local_size = count_files(self.local_analyses_dir)
        
        # Count benchmark CSVs in base_dir
        benchmark_files = list(self.base_dir.glob("*.csv"))
        bench_count = len(benchmark_files)
        bench_size = sum(f.stat().st_size for f in benchmark_files)
        benchmark_files = list(self.base_dir.glob("*.csv"))
        bench_count = len(benchmark_files)
        bench_size = sum(f.stat().st_size for f in benchmark_files)
        
        total_files = eda_count + fig_count + rel_count + batch_count + local_count + bench_count
        total_size = eda_size + fig_size + rel_size + batch_size + local_size + bench_size
        
        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_type": {
                "benchmarks": bench_count,
                "eda": eda_count,
                "figures": fig_count,
                "reliability": rel_count,
                "batch": batch_count,
                "local_analyses": local_count
            },
            "size_by_type_mb": {
                "benchmarks": round(bench_size / (1024 * 1024), 2),
                "eda": round(eda_size / (1024 * 1024), 2),
                "figures": round(fig_size / (1024 * 1024), 2),
                "reliability": round(rel_size / (1024 * 1024), 2),
                "batch": round(batch_size / (1024 * 1024), 2),
                "local_analyses": round(local_size / (1024 * 1024), 2)
            }
        }
    
    def cleanup_old_results(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Remove result files older than specified age.
        
        Args:
            max_age_days: Files older than this are deleted
            
        Returns:
            Dictionary with counts of deleted files by type
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cutoff_timestamp = cutoff_time.timestamp()
        
        deleted = {"eda": 0, "figures": 0, "local_analyses": 0, "reliability": 0, "batch": 0}
        
        for directory, key in [
            (self.eda_dir, "eda"),
            (self.figures_dir, "figures"),
            (self.local_analyses_dir, "local_analyses"),
            (self.reliability_dir, "reliability"),
            (self.batch_dir, "batch")
        ]:
            if not directory.exists():
                continue
            
            for file_path in directory.rglob("*"):
                if not file_path.is_file():
                    continue
                
                if file_path.stat().st_mtime < cutoff_timestamp:
                    try:
                        file_path.unlink()
                        deleted[key] += 1
                    except Exception:
                        pass
        
        return deleted


# Convenience function for getting a singleton instance
_result_manager_instance = None

def get_result_manager(base_dir: Optional[Path] = None) -> ResultManager:
    """
    Get or create a singleton ResultManager instance.
    
    Args:
        base_dir: Root directory for results (only used on first call)
        
    Returns:
        ResultManager instance
    """
    global _result_manager_instance
    if _result_manager_instance is None:
        _result_manager_instance = ResultManager(base_dir)
    return _result_manager_instance
