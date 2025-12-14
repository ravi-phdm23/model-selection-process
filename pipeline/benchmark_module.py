"""Model benchmarking and comparison module."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from result_manager import ResultManager


class BenchmarkModule:
    """
    Handles model benchmarking and comparison.

    Responsibilities:
    - Compare models within each dataset
    - Rank models by performance metrics
    - Identify best models
    - Generate comparison tables and rankings
    - Save benchmark results
    """

    def __init__(
        self,
        config: Dict[str, Any],
        result_manager: ResultManager,
        logger: logging.Logger
    ):
        """
        Initialize benchmark module.

        Args:
            config: Pipeline configuration
            result_manager: Result manager instance
            logger: Logger instance
        """
        self.config = config
        self.result_mgr = result_manager
        self.logger = logger

        # Extract benchmark config
        self.benchmark_config = config.get('models', {}).get('benchmark', {})
        self.primary_metric = self.benchmark_config.get('primary_metric', 'AUC')
        self.secondary_metrics = self.benchmark_config.get('secondary_metrics', ['F1', 'KS'])

    def run_benchmarking(
        self,
        training_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run benchmarking analysis on training results.

        Args:
            training_results: Results from model training

        Returns:
            Dictionary with benchmarking results
        """
        self.logger.info("Running model benchmarking...")

        results = {}

        for dataset_name, dataset_results in training_results.items():
            if 'error' in dataset_results:
                self.logger.warning(f"  Skipping {dataset_name} (training failed)")
                continue

            self.logger.info(f"  Benchmarking: {dataset_name}")

            try:
                benchmark_result = self._benchmark_dataset(
                    dataset_name,
                    dataset_results
                )
                results[dataset_name] = benchmark_result

                # Log best model
                best = benchmark_result['best_model']
                self.logger.info(
                    f"    Best model: {best['group']}/{best['model']} "
                    f"({self.primary_metric}={best['score']:.4f})"
                )

            except Exception as e:
                self.logger.error(
                    f"  [ERROR] Benchmarking failed for {dataset_name}: {str(e)}",
                    exc_info=True
                )
                results[dataset_name] = {'status': 'failed', 'error': str(e)}

        # Generate cross-dataset summary
        self._generate_cross_dataset_summary(results, training_results)

        return results

    def _benchmark_dataset(
        self,
        dataset_name: str,
        dataset_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Benchmark models for a single dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_results: Training results for this dataset

        Returns:
            Dictionary with benchmark results
        """
        # Collect all model metrics
        model_metrics = []

        for group_name, group_results in dataset_results.items():
            for model_name, model_result in group_results.items():
                if model_result.get('status') == 'failed':
                    continue

                metrics = model_result.get('metrics', {})
                if not metrics:
                    continue

                model_metrics.append({
                    'group': group_name,
                    'model': model_name,
                    'full_name': f"{group_name}/{model_name}",
                    **metrics
                })

        if not model_metrics:
            raise ValueError(f"No valid model metrics for {dataset_name}")

        # Create DataFrame for analysis
        df = pd.DataFrame(model_metrics)

        # Rank by primary metric
        df_ranked = df.sort_values(self.primary_metric, ascending=False).reset_index(drop=True)
        df_ranked['rank'] = range(1, len(df_ranked) + 1)

        # Get best model
        best_row = df_ranked.iloc[0]
        best_model = {
            'group': best_row['group'],
            'model': best_row['model'],
            'full_name': best_row['full_name'],
            'score': best_row[self.primary_metric],
            'metrics': {col: best_row[col] for col in df.columns if col not in ['group', 'model', 'full_name', 'rank']}
        }

        # Get top 5 models
        top_5 = df_ranked.head(5)[['rank', 'full_name', self.primary_metric] + self.secondary_metrics].to_dict('records')

        # Save results
        self._save_benchmark_results(dataset_name, df_ranked)

        return {
            'status': 'completed',
            'best_model': best_model,
            'top_5': top_5,
            'total_models': len(df),
            'rankings': df_ranked
        }

    def _save_benchmark_results(self, dataset_name: str, rankings: pd.DataFrame):
        """
        Save benchmark results to disk.

        Args:
            dataset_name: Dataset name
            rankings: DataFrame with model rankings
        """
        safe_ds = self.result_mgr.sanitize_name(dataset_name)

        # Create benchmark directory
        benchmark_dir = self.result_mgr.base_dir / 'benchmarks'
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Save full rankings
        rankings_path = benchmark_dir / f'{safe_ds}_rankings.csv'
        rankings.to_csv(rankings_path, index=False)
        self.logger.debug(f"    Saved rankings: {rankings_path}")

        # Save top 10 summary
        top_10 = rankings.head(10)
        summary_path = benchmark_dir / f'{safe_ds}_top10.csv'
        top_10.to_csv(summary_path, index=False)
        self.logger.debug(f"    Saved top 10: {summary_path}")

    def _generate_cross_dataset_summary(
        self,
        benchmark_results: Dict[str, Any],
        training_results: Dict[str, Any]
    ):
        """
        Generate summary comparing best models across datasets.

        Args:
            benchmark_results: Benchmark results for all datasets
            training_results: Training results for all datasets
        """
        try:
            rows = []

            for dataset_name, bench_result in benchmark_results.items():
                if bench_result.get('status') == 'failed':
                    continue

                best = bench_result['best_model']
                row = {
                    'dataset': dataset_name,
                    'best_model': best['full_name'],
                    'group': best['group'],
                    'model': best['model'],
                    self.primary_metric: best['score']
                }

                # Add secondary metrics
                for metric in self.secondary_metrics:
                    row[metric] = best['metrics'].get(metric, np.nan)

                rows.append(row)

            if not rows:
                self.logger.warning("No benchmark results to summarize")
                return

            summary_df = pd.DataFrame(rows)

            # Save cross-dataset summary
            summary_path = self.result_mgr.base_dir / 'cross_dataset_best_models.csv'
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Saved cross-dataset summary: {summary_path}")

            # Log summary table
            self.logger.info("\nBest Models by Dataset:")
            for _, row in summary_df.iterrows():
                self.logger.info(
                    f"  {row['dataset']:20s} -> {row['best_model']:30s} "
                    f"({self.primary_metric}={row[self.primary_metric]:.4f})"
                )

        except Exception as e:
            self.logger.error(f"Failed to generate cross-dataset summary: {str(e)}")

    def generate_comparison_table(
        self,
        training_results: Dict[str, Any],
        metric: str = None
    ) -> pd.DataFrame:
        """
        Generate comparison table for all models across all datasets.

        Args:
            training_results: Training results
            metric: Metric to compare (default: primary metric)

        Returns:
            DataFrame with comparison
        """
        if metric is None:
            metric = self.primary_metric

        rows = []

        for dataset_name, dataset_results in training_results.items():
            if 'error' in dataset_results:
                continue

            for group_name, group_results in dataset_results.items():
                for model_name, model_result in group_results.items():
                    if model_result.get('status') == 'failed':
                        continue

                    metrics = model_result.get('metrics', {})
                    score = metrics.get(metric, np.nan)

                    rows.append({
                        'dataset': dataset_name,
                        'group': group_name,
                        'model': model_name,
                        metric: score
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Pivot for comparison
        pivot_df = df.pivot_table(
            values=metric,
            index=['group', 'model'],
            columns='dataset',
            aggfunc='first'
        )

        return pivot_df
